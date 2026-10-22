"""第9章: 報酬設計の実験 ― 報酬だけを変えて、同じ条件で PPO を学習・評価する

使い方:
  python reward_experiments.py                 # 5種類の報酬 × 300万ステップ（筆者の環境で合計 10〜20 分）
  python reward_experiments.py --steps 1000000 # 短時間で試す
"""
import argparse

from evaluate import evaluate, pad, sb3_policy
from train_ppo import TOTAL_STEPS, train
from vec_env import REWARD_MODES

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=TOTAL_STEPS)
    args = p.parse_args()

    rows = []
    for mode in REWARD_MODES:
        model, env, sec = train(mode, args.steps, verbose=False)
        # 成功率は共通の物差し。報酬の合計は「その報酬関数で測った値」なので、モードの間では比べられない
        r = evaluate(sb3_policy(model, env), reward=mode)
        rows.append((mode, r, sec))
        print(f"{mode:18s} 学習 {sec:4.0f} 秒 / 成功率 {r['success_rate'] * 100:5.1f}%", flush=True)

    print(f"\n{pad('報酬', 18)} {pad('成功率', 7)} {pad('平均ステップ', 12)} "
          f"{pad('トルク変化率', 12)} {pad('時間切れ時の距離[cm]', 20)} 1エピソードの報酬")
    for mode, r, sec in rows:
        print(f"{mode:18s} {r['success_rate'] * 100:6.1f}% {r['mean_steps']:12.1f} "
              f"{r['torque_rate']:12.1f} {r['timeout_dist'] * 100:20.1f} {r['mean_return']:17.2f}")
