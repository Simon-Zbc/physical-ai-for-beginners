"""第9章: 方策を評価する（成功率・平均ステップ数・トルクの変化率・時間切れ時の距離・報酬の合計）

使い方:
  python evaluate.py                         # 比較用のヤコビ転置法だけを評価
  python evaluate.py models/ppo_dense.zip    # 学習済みモデルも並べて評価
"""
import sys
from pathlib import Path

import numpy as np

from vec_env import DECIMATION, DT, VecReachEnv, scripted_policy


def evaluate(policy, num_envs=1024, steps=500, seed=1, reward="dense"):
    """policy(obs (N,10)) -> 行動 (N,2) を num_envs 環境 × steps ステップ動かして統計を取る

    学習（seed=0）とは別のシードで、まだ見ていない初期姿勢・目標を使って評価する。
    """
    env = VecReachEnv(num_envs, seed=seed, reward=reward)
    obs = env.reset()
    ep_return, ep_len = np.zeros(num_envs), np.zeros(num_envs, dtype=int)
    returns, success_steps, timeout_dists = [], [], []
    prev_tau = np.zeros((num_envs, 2))
    same_episode = np.zeros(num_envs, dtype=bool)   # 前のステップと同じエピソードか（初回とリセット直後は False）
    dtau_sum, dtau_count = 0.0, 0
    for _ in range(steps):
        obs, rew, terminated, truncated, final_obs = env.step(policy(obs))
        # トルクの変化率 |Δτ| / Δt [Nm/s]（クリップ後に実際にかかったトルクで計算）
        dtau = np.abs(env.tau - prev_tau)[same_episode] / (DT * DECIMATION)
        dtau_sum += dtau.sum(); dtau_count += dtau.size
        prev_tau = env.tau.copy()
        ep_return += rew; ep_len += 1
        done = terminated | truncated
        returns += ep_return[done].tolist()
        success_steps += ep_len[terminated].tolist()
        timeout_dists += np.linalg.norm(final_obs[truncated, 8:10], axis=1).tolist()   # 時間切れの時点で、目標まで何 m 残っていたか
        ep_return[done] = 0.0; ep_len[done] = 0
        same_episode = ~done
    return {
        "success_rate": env.successes / max(env.episodes, 1),
        "episodes": env.episodes,
        "mean_steps": float(np.mean(success_steps)) if success_steps else float("nan"),
        "mean_return": float(np.mean(returns)) if returns else float("nan"),
        "torque_rate": dtau_sum / max(dtau_count, 1),
        "timeout_dist": float(np.median(timeout_dists)) if timeout_dists else float("nan"),
    }


def sb3_policy(model, vecnorm=None):
    """SB3 のモデルを policy(obs) の形にする（評価では確率的なノイズを入れず、平均の行動を使う）

    学習時に観測を正規化していた場合は、評価でも同じ統計で正規化してから方策に渡す。
    """
    from stable_baselines3.common.vec_env import VecNormalize
    norm = vecnorm.normalize_obs if isinstance(vecnorm, VecNormalize) else (lambda o: o)
    return lambda obs: model.predict(norm(obs.astype(np.float32)), deterministic=True)[0]


def load_model(path):
    """保存したモデル（と、あれば正規化の統計 .pkl）を読み込み、policy(obs) を返す"""
    import pickle

    from stable_baselines3 import PPO, SAC
    path = Path(path)
    model = (SAC if path.name.startswith("sac") else PPO).load(path, device="cpu")
    vecnorm = None
    if path.with_suffix(".pkl").exists():
        with open(path.with_suffix(".pkl"), "rb") as f:
            vecnorm = pickle.load(f)                  # 統計だけを使う（環境は不要）
    return sb3_policy(model, vecnorm)


def pad(s, width):
    """全角文字を幅 2 として、表示幅が width になるように右側を空白で埋める"""
    return s + " " * (width - sum(2 if ord(c) > 0x2E80 else 1 for c in s))


def print_row(name, r):
    print(f"{pad(name, 24)} {r['success_rate'] * 100:6.1f}% {r['mean_steps']:12.1f} {r['torque_rate']:18.1f}"
          f" {r['timeout_dist'] * 100:17.1f}")


if __name__ == "__main__":
    print(f"{pad('方策', 24)} {pad('成功率', 7)} {pad('平均ステップ', 12)} トルク変化率[Nm/s] 時間切れ時の距離[cm]")
    print_row("ヤコビ転置法（第0章）", evaluate(scripted_policy))
    for path in sys.argv[1:]:
        print_row(Path(path).stem, evaluate(load_model(path)))
