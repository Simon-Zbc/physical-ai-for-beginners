"""第7章 (1): 「ループで N 環境」vs「一括で N 環境」のスループット比較"""
import time

import numpy as np

from vec_env import SingleReachEnv, VecReachEnv, scripted_policy


def steps_for(n):
    return max(50, 5000 // n)            # 環境数が少ないときは多めに回して計測誤差を減らす


def bench_loop(n):
    STEPS = steps_for(n)
    envs = [SingleReachEnv(seed=i) for i in range(n)]
    obs = [e.env._get_observations() for e in envs]
    t0 = time.perf_counter()
    for _ in range(STEPS):
        for i, e in enumerate(envs):                     # Python のループで1環境ずつ
            obs[i], _, _ = e.step(scripted_policy(obs[i])[0])
    return n * STEPS / (time.perf_counter() - t0)


def bench_vec(n):
    STEPS = steps_for(n)
    env = VecReachEnv(n)
    obs = env._get_observations()
    t0 = time.perf_counter()
    for _ in range(STEPS):
        obs, _, _ = env.step(scripted_policy(obs))       # N 環境を配列演算で一括
    return n * STEPS / (time.perf_counter() - t0)


if __name__ == "__main__":
    bench_loop(4); bench_vec(4)                          # ウォームアップ（初回の遅延を除く）
    print(f"{'環境数':>6s} {'ループ [env-step/s]':>20s} {'一括 [env-step/s]':>18s} {'倍率':>6s}")
    for n in [1, 16, 256, 4096]:
        loop = bench_loop(n) if n <= 256 else float("nan")   # 4096 はループだと遅すぎるので省略
        vec = bench_vec(n)
        ratio = f"{vec / loop:5.1f}x" if not np.isnan(loop) else "   -"
        loop_s = f"{loop:15,.0f}" if not np.isnan(loop) else f"{'（省略）':>14s}"
        print(f"{n:6d} {loop_s:>20s} {vec:18,.0f} {ratio:>6s}")

    print("\n=== 4096 環境で 10 秒分（500 方策ステップ）を回して、エピソードの統計を取る ===")
    env = VecReachEnv(4096, seed=1)
    obs = env._get_observations()
    t0 = time.perf_counter()
    for _ in range(500):
        obs, rew, done = env.step(scripted_policy(obs))
    sec = time.perf_counter() - t0
    print(f"実時間 {sec:.1f} 秒 / シミュレーション時間の合計 {4096 * 10 / 3600:.1f} 時間分"
          f"（実時間の {4096 * 10 / sec:,.0f} 倍）")
    print(f"終了したエピソード {env.episodes:,} 回 / 成功 {env.successes:,} 回"
          f"（成功率 {env.successes / env.episodes * 100:.1f}%）")
