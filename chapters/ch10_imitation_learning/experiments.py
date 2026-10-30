"""第10章: 実験 ― 同じ示範から BC / チャンク / Diffusion を学習し、同じ条件で評価して並べる

使い方:
  python experiments.py                  # seed 0, 1, 2 で全部（筆者の環境で、E2対照 を含めて 約 12 分。
                                         # CPU がほかの処理で混んでいたときは 約 26 分だった）
  python experiments.py --seeds 0        # seed を1つだけ
  python experiments.py --epochs 5       # エポック数を減らして動作確認

  E1  平均化     : 左右が混ざった示範（demos_mix）で3つを学習。起点と目標は中心の ±2cm でランダム
  E1b 同じ起点   : E1 の方策を、毎回まったく同じ起点・目標から走らせる（spawn_half=0）
  E2  チャンク   : 左だけ通り、経由点で止まる示範（demos_left_pause）で BC とチャンクを学習
  E2対照         : 止まらない示範（demos_left）で BC を学習（E2 の BC の失敗が「停止」のせいかを確かめる）

seed s の行は「学習の乱数 = s、Diffusion のサンプルの乱数 = s」。
"""
import argparse
import io
import shutil
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from collect_demos import collect
from env import EPISODE_STEPS
from evaluate import evaluate, pad, print_table
from policies import load_runner
from train import train

# 実験名 -> 示範ファイル、その示範を作る collect の引数（sides, pause）、学習する方策
EXPERIMENTS = {
    "E1": {"demos": "data/demos_mix.npz", "sides": "mix", "pause": False, "kinds": ["bc", "chunk", "diffusion"]},
    "E2": {"demos": "data/demos_left_pause.npz", "sides": "left", "pause": True, "kinds": ["bc", "chunk"]},
    "E2対照": {"demos": "data/demos_left.npz", "sides": "left", "pause": False, "kinds": ["bc"]},
}


def expert_row(sides, pause):
    """示範者そのものの成績。評価と同じ seed=1 の起点・目標で 1024 本走らせる

    左右は示範者が選んだ経由点の側（示範者は衝突しないので、計画どおりの側を通る）。
    トルク変化率と推論時間は測らないので nan にする。
    """
    d = collect(1024, sides, pause, seed=1)
    s, c = d["success"].mean(), d["collision"].mean()
    return {"success_rate": s, "collision_rate": c, "timeout_rate": 1 - s - c, "episodes": 1024,
            "left": int((d["side"] == -1).sum()), "right": int((d["side"] == 1).sum()), "none": 0,
            "torque_rate": np.nan, "ms_per_call": np.nan}


def train_and_keep(kind, demos, epochs, seed):
    """学習して、seed ごとに別名でモデルを残す（train() は models/<kind>_<示範名>.pt に上書き保存するため）"""
    with redirect_stdout(io.StringIO()):                     # エポックごとの loss 表示は省く
        _, sec = train(kind, demos, epochs, seed)
    path = Path("models") / f"{kind}_{Path(demos).stem}.pt"
    kept = path.with_name(f"{path.stem}_s{seed}.pt")
    shutil.copy(path, kept)
    return kept, sec


def summarize(results):
    """seed をまたいだ平均と（最小〜最大）を出力する。左右・未通過はエピソード数に対する割合 [%]"""
    print(f"{pad('', 16)}" + "".join(f" {pad(h, 21)}" for h in ["成功率", "衝突率", "時間切れ", "左", "右", "未通過"]))
    for name, rs in sorted(results.items(), key=lambda kv: kv[0].split()[0]):   # 実験ごとにまとめる
        cols = []
        for key in ["success_rate", "collision_rate", "timeout_rate", "left", "right", "none"]:
            v = np.array([r[key] if key.endswith("rate") else r[key] / r["episodes"] for r in rs]) * 100
            cols.append(f"{v.mean():5.1f} ({v.min():5.1f}〜{v.max():5.1f})")
        print(f"{pad(name, 16)} " + " ".join(cols))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=None)       # 省略すると train.py の EPOCHS
    args = p.parse_args()
    t_start = time.perf_counter()

    experts, results = {}, {}                                # results["実験 方策"] = [seed ごとの結果]
    for exp, e in EXPERIMENTS.items():
        if not Path(e["demos"]).exists():                    # 示範データがなければここで作る
            Path(e["demos"]).parent.mkdir(parents=True, exist_ok=True)
            np.savez(e["demos"], **collect(200, e["sides"], e["pause"]))
        experts[exp] = expert_row(e["sides"], e["pause"])

    for seed in args.seeds:
        for exp, e in EXPERIMENTS.items():
            demos = e["demos"]
            rows, rows_same = [("示範者", experts[exp])], []
            for kind in e["kinds"]:
                path, sec = train_and_keep(kind, demos, args.epochs, seed)
                t0 = time.perf_counter()
                r = evaluate(load_runner(path, seed))        # 1024 環境 × 600 ステップ（未知の起点・目標）
                rows.append((kind, r))
                results.setdefault(f"{exp} {kind}", []).append(r)
                if exp == "E1":                              # 同じ方策を「全エピソード同じ起点」で（新しい実行器で）
                    r = evaluate(load_runner(path, seed), steps=EPISODE_STEPS, spawn_half=0.0)
                    rows_same.append((kind, r))
                    results.setdefault(f"E1b {kind}", []).append(r)
                print(f"[seed {seed}] {pad(exp, 6)} {kind:9s} 学習 {sec:4.0f} 秒 / 評価 {time.perf_counter() - t0:4.0f} 秒", flush=True)
            print(f"\n=== {exp}（{Path(demos).stem}）seed {seed} ===")
            print_table(rows)
            if rows_same:
                print(f"\n=== E1b（同じ起点・目標、1024 環境 × {EPISODE_STEPS} ステップ）seed {seed} ===")
                print_table(rows_same)
            print(flush=True)

    print(f"=== seed {args.seeds} の平均（最小〜最大）===")
    summarize(results)
    print(f"\n合計 {(time.perf_counter() - t_start) / 60:.1f} 分")
