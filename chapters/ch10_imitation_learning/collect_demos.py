"""第10章: 脚本示範者で示範データを集める

使い方:
  python collect_demos.py                          # 左右 50/50 の示範 200 本 -> data/demos_mix.npz
  python collect_demos.py --sides left             # 左だけ通る示範 -> data/demos_left.npz
  python collect_demos.py --sides left --pause     # 左だけ + 経由点で止まる示範 -> data/demos_left_pause.npz
"""
import argparse
from pathlib import Path

import numpy as np

from env import EPISODE_STEPS, VecObstacleEnv, fk
from expert import plan


def collect(episodes, sides="mix", pause=False, noise=0.01, seed=0):
    """示範者を episodes 本並列に走らせて、(観測, 行動) の組を集める

    記録する行動は、ノイズを足す前の q_des（示範者の本来の意図）。ノイズ入りの指令で動かすので、
    観測には「少しずれた状態」が入り、ラベルは正しいままになる（DART の考え方）。
    各環境は最初の1エピソードだけを記録する（成功・衝突・時間切れのどれかで終わるまで）。
    """
    rng = np.random.default_rng((seed, 1))               # 経路の選択とノイズ用（環境の乱数とは別系列）
    env = VecObstacleEnv(episodes, seed)
    obs = env.reset()
    side = np.full(episodes, -1) if sides == "left" else rng.choice([-1, 1], episodes)
    pause_steps = rng.integers(0, 51, episodes) if pause else np.zeros(episodes, dtype=int)
    start, target = fk(env.q), env.target.copy()
    plans = np.stack([plan(start[i], target[i], side[i], pause_steps[i]) for i in range(episodes)])

    alive = np.ones(episodes, dtype=bool)
    success = np.zeros(episodes, dtype=bool); collision = np.zeros(episodes, dtype=bool)
    obs_log, act_log, idx_log = [], [], []
    for t in range(EPISODE_STEPS):
        ids = np.nonzero(alive)[0]
        obs_log.append(obs[ids]); act_log.append(plans[ids, t]); idx_log.append(ids)
        q_des = plans[:, t] + rng.normal(0.0, noise, (episodes, 2))
        obs, info = env.step(q_des)
        success[ids] = info["success"][ids]; collision[ids] = info["collision"][ids]
        alive &= ~info["done"]                           # 終わった環境はその場で自動リセットされるが、以降は記録しない
        if not alive.any():
            break

    order = np.argsort(np.concatenate(idx_log), kind="stable")   # エピソードごとにまとめる（時間順は保つ）
    return {"obs": np.concatenate(obs_log)[order].astype(np.float32),
            "actions": np.concatenate(act_log)[order].astype(np.float32),
            "episode_index": np.concatenate(idx_log)[order],
            "side": side, "success": success, "collision": collision}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--sides", choices=["mix", "left"], default="mix")
    parser.add_argument("--pause", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = Path(args.out or f"data/demos_{args.sides}{'_pause' if args.pause else ''}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)

    d = collect(args.episodes, args.sides, args.pause, seed=args.seed)
    np.savez(out, **d)
    print(f"{out}: {args.episodes} エピソード, 総ステップ {len(d['actions'])}")
    print(f"  左 {(d['side'] == -1).sum()} / 右 {(d['side'] == 1).sum()}")
    print(f"  成功率 {d['success'].mean():.3f}, 衝突 {d['collision'].sum()} 本")
