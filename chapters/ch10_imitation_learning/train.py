"""第10章: 示範データから方策を学習する

使い方:
  python train.py bc                                   # BC を demos_mix で学習 → models/bc_demos_mix.pt
  python train.py chunk                                # チャンク + Temporal Ensembling
  python train.py diffusion                            # Diffusion Policy（DDPM）
  python train.py bc --demos data/demos_left_pause.npz # 示範データを変える
  python train.py diffusion --epochs 50                # エポック数を減らして短時間で試す
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from evaluate import evaluate, print_table
from policies import (ALPHA_BARS, CHUNK, NETS, RUNNERS, T_DIFF, Normalizer, make_chunk_labels,
                      save_checkpoint)

# 既定のエポック数。demos_mix で実測すると、loss は BC が約 20 エポックで下げ止まり、チャンクと Diffusion は
# 100 エポックを過ぎてもゆっくり下がり続ける。CPU 時間（1回 約1分）と比べやすさを優先して、3つとも 100 にした
EPOCHS = {"bc": 100, "chunk": 100, "diffusion": 100}
BATCH, LR = 256, 1e-3


def load_demos(path):
    """示範データを読み込み、成功したエピソードだけを (観測, 行動, エピソード番号) で返す"""
    d = np.load(path)
    ok = d["success"][d["episode_index"]]              # 失敗した示範はまねしない
    return d["obs"][ok], d["actions"][ok], d["episode_index"][ok]


def loss_fn(kind, net, obs, target):
    """1バッチぶんの loss。BC とチャンクは行動の MSE、Diffusion は「乗せたノイズを当てる」MSE"""
    if kind != "diffusion":
        return torch.mean((net(obs) - target) ** 2)
    t = torch.randint(T_DIFF, (len(obs),))                 # バッチの各サンプルにランダムなステップ t
    noise = torch.randn_like(target)
    ab = ALPHA_BARS[t][:, None, None]
    noisy = torch.sqrt(ab) * target + torch.sqrt(1 - ab) * noise   # 正解チャンクに t ステップぶんのノイズを乗せる
    return torch.mean((net(obs, noisy, t) - noise) ** 2)


def train(kind, demos, epochs=None, seed=0):
    """kind の方策を学習して models/<kind>_<示範の名前>.pt に保存し、(実行器, 学習秒数) を返す"""
    torch.set_num_threads(1)       # 小さなネットワークでは1スレッドのほうが速く、結果もコア数に依存しない
    torch.manual_seed(seed); np.random.seed(seed)
    epochs = EPOCHS[kind] if epochs is None else epochs
    obs, actions, episode_index = load_demos(demos)
    obs_norm, act_norm = Normalizer(obs), Normalizer(actions)
    x = torch.as_tensor(obs_norm.norm(obs))
    labels = actions if kind == "bc" else make_chunk_labels(actions, episode_index, CHUNK)
    y = torch.as_tensor(act_norm.norm(labels))         # (M, 2) または (M, CHUNK, 2)

    net = NETS[kind]()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    losses = []
    t0 = time.perf_counter()
    for epoch in range(epochs):
        perm = torch.randperm(len(x))                  # 毎エポック、順番をシャッフルする
        batch_losses = []
        for i in range(0, len(x), BATCH):
            b = perm[i:i + BATCH]
            loss = loss_fn(kind, net, x[b], y[b])
            opt.zero_grad(); loss.backward(); opt.step()
            batch_losses.append(loss.item())
        losses.append(float(np.mean(batch_losses)))
        if (epoch + 1) % max(epochs // 10, 1) == 0 or epoch == 0:
            print(f"epoch {epoch + 1:5d}  loss {losses[-1]:.5f}  {time.perf_counter() - t0:6.1f} 秒", flush=True)
    sec = time.perf_counter() - t0

    net.eval()
    path = Path("models") / f"{kind}_{Path(demos).stem}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(path, kind, net, obs_norm, act_norm,
                    {"demos": str(demos), "epochs": epochs, "seed": seed, "train_sec": sec, "losses": losses})
    return RUNNERS[kind](net, obs_norm, act_norm), sec


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=list(NETS))
    p.add_argument("--demos", default="data/demos_mix.npz")
    p.add_argument("--epochs", type=int, default=None)   # 省略すると EPOCHS[kind]
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    runner, sec = train(args.kind, args.demos, args.epochs, args.seed)
    print(f"\n学習時間 {sec:.0f} 秒 → models/{args.kind}_{Path(args.demos).stem}.pt")
    t0 = time.perf_counter()
    r = evaluate(runner)
    print(f"評価（未知の 1024 環境 × 600 ステップ、{time.perf_counter() - t0:.0f} 秒）")
    print_table([(f"{args.kind}_{Path(args.demos).stem}", r)])
