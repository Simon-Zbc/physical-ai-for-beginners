"""第9章: PPO で Reach タスクを学習する

使い方:
  python train_ppo.py                      # dense 報酬で学習 → models/ppo_dense.zip（+ 正規化の統計 .pkl）
  python train_ppo.py --reward sparse      # 報酬を変えて学習
  python train_ppo.py --no-normalize       # 観測の正規化なしで学習（失敗例の再現）
  python train_ppo.py --steps 300000       # 学習ステップ数を減らして短時間で試す
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

from evaluate import evaluate, sb3_policy
from sb3_env import SB3ReachEnv
from vec_env import GAMMA, REWARD_MODES

torch.set_num_threads(1)     # 小さなネットワークでは、1スレッドのほうが速い（結果もコア数に依存しなくなる）

NUM_ENVS = 64
TOTAL_STEPS = 3_000_000
PPO_KWARGS = dict(
    n_steps=64,              # 1回の更新で 64 環境 × 64 ステップ = 4096 個の経験を集める
    batch_size=1024,
    n_epochs=10,             # 集めた経験を 10 回使い回して方策を更新する
    gamma=GAMMA,
    learning_rate=3e-4,
    clip_range=0.2,          # PPO の「1回で変えすぎない」ための上限
)


class ProgressCallback(BaseCallback):
    """一定間隔で、学習の進み具合（報酬の平均・成功率）を表示する"""

    def __init__(self, every=250_000):
        super().__init__()
        self.every, self.next_print, self.t0 = every, every, time.perf_counter()

    def _on_step(self):
        if self.num_timesteps >= self.next_print:
            self.next_print += self.every
            rew = np.mean([ep["r"] for ep in self.model.ep_info_buffer]) if self.model.ep_info_buffer else float("nan")
            succ = np.mean(self.model.ep_success_buffer) * 100 if self.model.ep_success_buffer else float("nan")
            print(f"{self.num_timesteps:>10,d} steps {time.perf_counter() - self.t0:5.0f} s"
                  f"  エピソード報酬 {rew:8.2f}  成功率 {succ:5.1f}%", flush=True)
        return True


def train(reward="dense", total_steps=TOTAL_STEPS, seed=0, num_envs=NUM_ENVS, normalize=True, verbose=True):
    """PPO で学習し、(モデル, 学習に使った環境, 学習にかかった秒数) を返す"""
    env = VecMonitor(SB3ReachEnv(num_envs, seed=seed, reward=reward))   # エピソードの報酬・長さを記録する
    if normalize:
        env = VecNormalize(env, norm_reward=False)   # 観測を「平均 0・標準偏差 1」に揃える（統計は学習中に更新）
    model = PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0, **PPO_KWARGS)
    t0 = time.perf_counter()
    model.learn(total_steps, callback=ProgressCallback() if verbose else None)
    return model, env, time.perf_counter() - t0


def save(model, env, path):
    """モデル（.zip）と、観測の正規化の統計（同名の .pkl）を保存する"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    if isinstance(env, VecNormalize):
        env.save(path.with_suffix(".pkl"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reward", default="dense", choices=REWARD_MODES)
    p.add_argument("--steps", type=int, default=TOTAL_STEPS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-normalize", action="store_true")
    args = p.parse_args()

    model, env, sec = train(args.reward, args.steps, args.seed, normalize=not args.no_normalize)
    name = f"ppo_{args.reward}" + ("_nonorm" if args.no_normalize else "")
    path = Path("models") / f"{name}.zip"
    save(model, env, path)
    r = evaluate(sb3_policy(model, env))
    print(f"\n学習時間 {sec:.0f} 秒 → {path}")
    print(f"評価（未知の 1024 環境 × 10 秒）: 成功率 {r['success_rate'] * 100:.1f}%"
          f" / 平均ステップ {r['mean_steps']:.1f} / トルク変化率 {r['torque_rate']:.1f} Nm/s")
