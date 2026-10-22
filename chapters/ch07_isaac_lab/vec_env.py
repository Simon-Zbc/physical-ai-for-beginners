"""第7章: GPU 並列シミュレーションの考え方を NumPy で体験する

2関節アームの Reach タスクを、
  (a) 1環境ずつのクラスを N 個並べてループで回す（従来の書き方）
  (b) N 環境分の状態を (N, ...) の配列にまとめて一括で計算する（Isaac Lab の書き方）
の2通りで実装し、スループットを比べる。

(b) のメソッド構成は Isaac Lab の DirectRLEnv（_apply_action / _get_observations /
_get_rewards / _get_dones / _reset_idx）に合わせてある。NumPy を PyTorch の CUDA テンソルに
置き換えれば、そのまま GPU 上で動く形になる。
"""
import numpy as np

L1, L2 = 0.30, 0.25          # リンク長 [m]
M1, M2 = 1.0, 0.5            # リンク先端の質量 [kg]（質点モデル）
DAMP = 0.5                   # 関節の粘性
DT, DECIMATION = 0.002, 10   # 物理 500Hz、方策 50Hz（物理10ステップごとに行動を1回決める）
EPISODE_STEPS = 150          # 1エピソード = 3秒
TAU_MAX = 5.0


def dynamics(q, qd, tau):
    """平面2関節アームの運動方程式  M(q) q̈ + c(q, q̇) + D q̇ = τ  を一括で解く（形状 (N, 2)）"""
    c2, s2 = np.cos(q[:, 1]), np.sin(q[:, 1])
    m11 = (M1 + M2) * L1**2 + M2 * L2**2 + 2 * M2 * L1 * L2 * c2
    m12 = M2 * L2**2 + M2 * L1 * L2 * c2
    m22 = np.full_like(m11, M2 * L2**2)
    h = M2 * L1 * L2 * s2
    c = np.stack([-h * (2 * qd[:, 0] * qd[:, 1] + qd[:, 1] ** 2), h * qd[:, 0] ** 2], axis=1)
    rhs = tau - c - DAMP * qd
    det = m11 * m22 - m12**2                                  # 2x2 の逆行列を解析的に
    qdd = np.stack([(m22 * rhs[:, 0] - m12 * rhs[:, 1]) / det,
                    (-m12 * rhs[:, 0] + m11 * rhs[:, 1]) / det], axis=1)
    return qdd


def fk(q):
    """手先位置 (N, 2)"""
    return np.stack([L1 * np.cos(q[:, 0]) + L2 * np.cos(q[:, 0] + q[:, 1]),
                     L1 * np.sin(q[:, 0]) + L2 * np.sin(q[:, 0] + q[:, 1])], axis=1)


def jacobian(q):
    """手先のヤコビアン (N, 2, 2)（第4章）"""
    s1, c1 = np.sin(q[:, 0]), np.cos(q[:, 0])
    s12, c12 = np.sin(q[:, 0] + q[:, 1]), np.cos(q[:, 0] + q[:, 1])
    return np.stack([np.stack([-L1 * s1 - L2 * s12, -L2 * s12], axis=1),
                     np.stack([L1 * c1 + L2 * c12, L2 * c12], axis=1)], axis=1)


class VecReachEnv:
    """N 環境を一括で扱う Reach 環境（Isaac Lab の DirectRLEnv と同じ構成）"""

    def __init__(self, num_envs, seed=0):
        self.n = num_envs
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros((num_envs, 2)); self.qd = np.zeros((num_envs, 2))
        self.target = np.zeros((num_envs, 2))
        self.step_count = np.zeros(num_envs, dtype=int)
        self.actions = np.zeros((num_envs, 2))
        self.successes = 0; self.episodes = 0
        self._reset_idx(np.arange(num_envs))

    # ---- 行動を受け取り、物理を DECIMATION ステップ進める ----
    def step(self, actions):
        self.actions = np.clip(actions, -1.0, 1.0)            # 方策の出力は [-1, 1] に正規化
        for _ in range(DECIMATION):
            self._apply_action()
            self.qd += dynamics(self.q, self.qd, self.tau) * DT   # 半陰的オイラー積分
            self.q += self.qd * DT
        self.step_count += 1
        obs = self._get_observations()
        rew = self._get_rewards()
        success, timeout = self._get_dones()
        done = success | timeout
        self.successes += int(success.sum()); self.episodes += int(done.sum())
        if done.any():                                        # 終わった環境だけをその場でリセット
            self._reset_idx(np.nonzero(done)[0])
        return obs, rew, done

    def _apply_action(self):
        self.tau = TAU_MAX * self.actions

    def _get_observations(self):
        return np.concatenate([np.cos(self.q), np.sin(self.q), self.qd,
                               self.target, self.target - fk(self.q)], axis=1)   # (N, 10)

    def _get_rewards(self):
        dist = np.linalg.norm(self.target - fk(self.q), axis=1)
        return -dist - 0.01 * np.square(self.actions).sum(axis=1)

    def _get_dones(self):
        dist = np.linalg.norm(self.target - fk(self.q), axis=1)
        speed = np.linalg.norm(self.qd, axis=1)
        return (dist < 0.01) & (speed < 0.2), self.step_count >= EPISODE_STEPS

    def _reset_idx(self, ids):
        """指定した環境だけ初期化する（関節角と目標位置をランダム化）"""
        k = len(ids)
        self.q[ids] = self.rng.uniform(-np.pi, np.pi, (k, 2))
        self.q[ids, 1] = self.rng.uniform(0.3, 2.5, k) * self.rng.choice([-1, 1], k)  # 特異姿勢を避ける
        self.qd[ids] = 0.0
        r = self.rng.uniform(0.15, 0.50, k); th = self.rng.uniform(-np.pi, np.pi, k)
        self.target[ids] = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
        self.step_count[ids] = 0


class SingleReachEnv:
    """比較用：1環境だけを扱うクラス（中身は VecReachEnv を N=1 で使う）"""

    def __init__(self, seed):
        self.env = VecReachEnv(1, seed)

    def step(self, action):
        return self.env.step(action[None, :])


def scripted_policy(obs):
    """学習の代わりの方策：第0章のヤコビ転置法 + ダンピング（一括で計算）"""
    q = np.arctan2(obs[:, 2:4], obs[:, 0:2])
    qd, err = obs[:, 4:6], obs[:, 8:10]
    tau = 100.0 * np.einsum("nij,ni->nj", jacobian(q), err) - 1.5 * qd   # τ = Kp Jᵀe − Kd q̇
    return tau / TAU_MAX
