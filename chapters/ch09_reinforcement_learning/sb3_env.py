"""第9章: VecReachEnv を Stable-Baselines3 の VecEnv として使うためのアダプタ

SB3 の VecEnv は「N 環境をまとめて step する」インタフェースで、第7章の並列環境と同じ考え方。
中身の計算はすべて VecReachEnv が配列で一括に行い、ここでは型と info の形式を合わせるだけ。
"""
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from vec_env import VecReachEnv


class SB3ReachEnv(VecEnv):
    def __init__(self, num_envs, seed=0, reward="dense"):
        self.raw = VecReachEnv(num_envs, seed=seed, reward=reward)
        self.raw.render_mode = None                           # SB3 が問い合わせる属性（描画はしない）
        super().__init__(
            num_envs,
            spaces.Box(-np.inf, np.inf, (10,), np.float32),   # 観測：cos q, sin q, q̇, 目標, 目標 − 手先
            spaces.Box(-1.0, 1.0, (2,), np.float32),          # 行動：正規化した関節トルク
        )

    def reset(self):
        return self.raw.reset().astype(np.float32)

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        obs, rew, terminated, truncated, final_obs = self.raw.step(self._actions)
        dones = terminated | truncated
        infos = [{} for _ in range(self.num_envs)]
        for i in np.nonzero(dones)[0]:
            infos[i]["terminal_observation"] = final_obs[i].astype(np.float32)   # 最後の状態の価値の計算に使う
            infos[i]["TimeLimit.truncated"] = bool(truncated[i])                 # 時間切れは「失敗で終了」と区別する
            infos[i]["is_success"] = bool(terminated[i])
        return obs.astype(np.float32), rew.astype(np.float32), dones, infos

    def close(self):
        pass

    # ---- 以下は SB3 の VecEnv が要求するメソッド（この章では最小限の実装） ----
    def get_attr(self, attr_name, indices=None):
        return [getattr(self.raw, attr_name)] * len(list(self._get_indices(indices)))

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.raw, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [getattr(self.raw, method_name)(*method_args, **method_kwargs)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(list(self._get_indices(indices)))
