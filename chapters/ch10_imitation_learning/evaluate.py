"""第10章: 方策を評価する（成功率・衝突率・時間切れ率・障害物を通った側・トルクの変化率・推論時間）

使い方:
  python evaluate.py models/bc_demos_mix.pt models/chunk_demos_mix.pt   # 学習済みの方策を1行ずつ並べて評価
  （Windows の cmd / PowerShell は models/*.pt のようなワイルドカードを展開しないので、ファイル名を並べて書く）

pad() と トルクの変化率の定義は第9章 evaluate.py から複製した（章ごとに自己完結させるため）。
"""
import sys
import time
from pathlib import Path

import numpy as np

from env import DECIMATION, DT, SPAWN_HALF, VecObstacleEnv


def evaluate(policy, num_envs=1024, steps=600, seed=1, spawn_half=SPAWN_HALF):
    """policy(obs (N,10)) -> q_des (N,2) を num_envs 環境 × steps ステップ動かして統計を取る

    policy は状態を持ってよい。エピソードが終わった環境は env が自動でリセットするので、
    その直後に policy.reset(ids) を呼んで、その環境ぶんの内部状態（バッファなど）だけを捨てさせる。
    学習データ（seed=0）とは別のシードで、まだ見ていない初期位置・目標を使って評価する。
    spawn_half=0 にすると、全エピソードを「まったく同じ起点・目標」から始める（実験 E1b）。
    """
    env = VecObstacleEnv(num_envs, seed=seed, spawn_half=spawn_half)
    obs = env.reset()
    prev_tau = np.zeros((num_envs, 2))
    same_episode = np.zeros(num_envs, dtype=bool)   # 前のステップと同じエピソードか（初回とリセット直後は False）
    dtau_sum, dtau_count = 0.0, 0
    n_success = n_collision = n_timeout = 0
    sides = {-1: 0, 0: 0, 1: 0}                     # 障害物を通った側（-1 = 左、+1 = 右、0 = 通っていない）
    call_time = 0.0
    for _ in range(steps):
        t0 = time.perf_counter()
        q_des = policy(obs)
        call_time += time.perf_counter() - t0       # 推論時間は policy の呼び出しだけを測る
        obs, info = env.step(q_des)
        # トルクの変化率 |Δτ| / Δt [Nm/s]（クリップ後に実際にかかったトルクで計算）
        dtau = np.abs(env.tau - prev_tau)[same_episode] / (DT * DECIMATION)
        dtau_sum += dtau.sum(); dtau_count += dtau.size
        prev_tau = env.tau.copy()
        done = info["done"]
        n_success += int(info["success"].sum()); n_collision += int(info["collision"].sum())
        n_timeout += int(info["timeout"].sum())
        for s in sides:                             # final_side は越線の最初の1回だけを記録している（折り返しても変わらない）
            sides[s] += int((info["final_side"][done] == s).sum())
        if done.any():
            policy.reset(np.nonzero(done)[0])       # 終わった環境のぶんだけ内部状態を捨てる
        same_episode = ~done
    episodes = n_success + n_collision + n_timeout
    return {
        "success_rate": n_success / max(episodes, 1),
        "collision_rate": n_collision / max(episodes, 1),
        "timeout_rate": n_timeout / max(episodes, 1),
        "episodes": episodes,
        "left": sides[-1], "right": sides[1], "none": sides[0],   # 3つの合計 = episodes
        "torque_rate": dtau_sum / max(dtau_count, 1),
        "ms_per_call": call_time / steps * 1000,
    }


def pad(s, width):
    """全角文字を幅 2 として、表示幅が width になるように右側を空白で埋める"""
    return s + " " * (width - sum(2 if ord(c) > 0x2E80 else 1 for c in s))


def print_table(rows):
    """rows = [(方策名, evaluate の結果), ...] を表にして出力する"""
    print(f"{pad('方策', 24)} {pad('成功率', 8)} {pad('衝突率', 8)} {pad('時間切れ', 8)}"
          f" {pad('左', 6)} {pad('右', 6)} {pad('未通過', 6)} 推論[ms]")
    for name, r in rows:
        print(f"{pad(name, 24)} {r['success_rate'] * 100:7.1f}% {r['collision_rate'] * 100:7.1f}%"
              f" {r['timeout_rate'] * 100:7.1f}% {r['left']:6d} {r['right']:6d} {r['none']:6d} {r['ms_per_call']:8.3f}")


if __name__ == "__main__":
    rows = []
    if sys.argv[1:]:
        import torch
        torch.set_num_threads(1)                      # 学習と同じ1スレッドにそろえる（推論[ms] を記事の値と比べられるように）
        from policies import load_runner              # 学習済みモデルを読むときだけ必要（遅延 import）
        rows = [(Path(path).stem, evaluate(load_runner(path))) for path in sys.argv[1:]]
    print_table(rows)
