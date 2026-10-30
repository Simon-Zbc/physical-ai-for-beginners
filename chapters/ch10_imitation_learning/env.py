"""第10章: 模倣学習で使う、障害物つき 2関節アームの並列環境

第9章の vec_env.py（動力学・fk・jacobian・観測）をコピーし、次の点を変えている。
  - 行動を「関節トルク」から「目標関節角 q_des」に変えた。内部で PD 制御がトルクに変換する
    （示範も方策も「次にどこへ行くか」を出力するだけでよく、データ収集と学習が素直になる）
  - 机の上に円形の障害物を1つ置いた（固定。観測には入れない）。手先が触れたらエピソード終了
  - 報酬はない（模倣学習は示範から学ぶので不要）。代わりに結果 success / collision / timeout と、
    障害物をどちら側から通ったか final_side を info で返す
  - 起点と目標は、障害物をはさんで反対側に置く（まっすぐ進むとぶつかる配置）
ik() は第4章 ik_2link.py の analytic_ik を一括計算にし、肘の向きを q2 > 0 の1本に固定したもの。

衝突を判定するのは手先だけ。これは俯瞰図で、リンクは机より高いところを通り、
机の上の物体に触れるのは垂れ下がった手先だけ、という想定による。
"""
import numpy as np

L1, L2 = 0.30, 0.25          # リンク長 [m]
M1, M2 = 1.0, 0.5            # リンク先端の質量 [kg]（質点モデル）
DAMP = 0.5                   # 関節の粘性
DT, DECIMATION = 0.002, 10   # 物理 500Hz、方策 50Hz（物理10ステップごとに行動を1回決める）
EPISODE_STEPS = 200          # 1エピソード = 4秒
TAU_MAX = 5.0
KP, KD = 100.0, 6.0         # PD ゲイン。関節角 0.3rad のステップ応答が 0.2秒で誤差 1cm 未満に収まる値

OBS_CENTER = np.array([0.35, 0.0])       # 障害物の中心 [m]
OBS_RADIUS = 0.05                        # 障害物の半径
HAND_RADIUS = 0.02                       # 手先の半径
START_CENTER = np.array([0.35, -0.22])   # 起点の中心（障害物の手前側）
TARGET_CENTER = np.array([0.35, 0.22])   # 目標の中心（障害物の向こう側）
SPAWN_HALF = 0.02                        # 起点・目標は中心の ±2cm にランダム
SUCCESS_DIST, SUCCESS_SPEED = 0.01, 0.2  # 成功 = 目標まで 1cm 以内 かつ 関節速度のノルム 0.2 rad/s 未満


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


def ik(xy):
    """解析的 IK（第4章）を一括で計算する (N, 2) -> (N, 2)。肘の向きは q2 > 0 の解に固定"""
    x, y = xy[:, 0], xy[:, 1]
    c2 = np.clip((x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2), -1.0, 1.0)   # 届かない点は端に丸める
    s2 = np.sqrt(1 - c2**2)
    q1 = np.arctan2(y, x) - np.arctan2(L2 * s2, L1 + L2 * c2)
    return np.stack([q1, np.arctan2(s2, c2)], axis=1)


class VecObstacleEnv:
    """N 環境を一括で扱う、障害物つき Reach 環境。行動は目標関節角 q_des (N, 2)"""

    def __init__(self, num_envs, seed=0, spawn_half=SPAWN_HALF):
        self.n = num_envs
        self.rng = np.random.default_rng(seed)
        self.spawn_half = spawn_half                          # 0 にすると、毎回まったく同じ起点・目標から始まる
        self.q = np.zeros((num_envs, 2)); self.qd = np.zeros((num_envs, 2))
        self.target = np.zeros((num_envs, 2))
        self.tau = np.zeros((num_envs, 2))                    # 最後に加えたトルク
        self.step_count = np.zeros(num_envs, dtype=int)
        self.side = np.zeros(num_envs, dtype=int)             # 障害物のどちら側を通ったか（0 = まだ通っていない）
        self.successes = 0; self.collisions = 0; self.episodes = 0

    def reset(self):
        """全環境を初期化して、観測 (N, 10) を返す"""
        self._reset_idx(np.arange(self.n))
        return self._get_observations()

    # ---- 目標関節角を受け取り、物理を DECIMATION ステップ進める ----
    def step(self, q_des):
        collided = np.zeros(self.n, dtype=bool)
        for _ in range(DECIMATION):
            active = ~collided                                # 衝突した環境はその場で止める
            self.tau = np.clip(KP * (q_des - self.q) - KD * self.qd, -TAU_MAX, TAU_MAX)
            self.qd += dynamics(self.q, self.qd, self.tau) * DT * active[:, None]   # 半陰的オイラー積分
            self.q += self.qd * DT * active[:, None]
            hand = fk(self.q)
            # 衝突・通過方向は、方策の1ステップの途中で手先が障害物をかすめても拾えるよう、物理ステップごとに判定する
            collided |= np.linalg.norm(hand - OBS_CENTER, axis=1) < OBS_RADIUS + HAND_RADIUS
            crossed = active & (self.side == 0) & (hand[:, 1] >= OBS_CENTER[1])    # y 線を最初に越えた瞬間だけ記録
            self.side[crossed] = np.where(hand[crossed, 0] < OBS_CENTER[0], -1, 1)
        self.step_count += 1
        obs = self._get_observations()
        dist = np.linalg.norm(self.target - fk(self.q), axis=1)
        speed = np.linalg.norm(self.qd, axis=1)
        collision = collided                                  # 優先順位は 衝突 > 成功 > 時間切れ
        success = ~collision & (dist < SUCCESS_DIST) & (speed < SUCCESS_SPEED)
        timeout = ~collision & ~success & (self.step_count >= EPISODE_STEPS)
        done = collision | success | timeout
        self.successes += int(success.sum()); self.collisions += int(collision.sum())
        self.episodes += int(done.sum())
        info = {"success": success, "collision": collision, "timeout": timeout, "done": done,
                "final_obs": obs.copy(), "final_side": self.side.copy()}   # リセット前の値
        if done.any():                                        # 終わった環境だけをその場でリセット
            ids = np.nonzero(done)[0]
            self._reset_idx(ids)
            obs[ids] = self._get_observations()[ids]
        return obs, info

    def _get_observations(self):
        return np.concatenate([np.cos(self.q), np.sin(self.q), self.qd,
                               self.target, self.target - fk(self.q)], axis=1)   # (N, 10)

    def _reset_idx(self, ids):
        """指定した環境だけ初期化する（起点と目標を中心の近くでランダム化）"""
        k = len(ids)
        start = START_CENTER + self.rng.uniform(-self.spawn_half, self.spawn_half, (k, 2))
        self.q[ids] = ik(start)
        self.qd[ids] = 0.0
        self.target[ids] = TARGET_CENTER + self.rng.uniform(-self.spawn_half, self.spawn_half, (k, 2))
        self.step_count[ids] = 0
        self.side[ids] = 0
