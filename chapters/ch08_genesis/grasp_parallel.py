"""第8章 (2): Franka で箱を掴む ― 16 環境を並列に動かし、箱の位置と把持力を環境ごとに変える

第6章の把持の物理（2μF ≥ m(g+a)）を、実在のロボットモデル（Franka Emika Panda）と
Genesis の並列環境で確かめる。
"""
import time

import numpy as np
import torch

import genesis as gs

N_ENVS = 16
GRIP_FORCES = [0.1, 0.3, 0.4, 2.0]      # 指1本あたりの把持力 [N]（4種類 × 箱の位置 4種類）

gs.init(backend=gs.cpu, logging_level="warning", seed=0)   # GPU があれば gs.gpu
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01),
    rigid_options=gs.options.RigidOptions(box_box_detection=True),
    show_viewer=False,
)
scene.add_entity(gs.morphs.Plane())
cube = scene.add_entity(gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02)))
franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))

t0 = time.perf_counter()
scene.build(n_envs=N_ENVS, env_spacing=(1.0, 1.0))        # 同じシーンを 16 個複製
print(f"build: {time.perf_counter() - t0:.1f} 秒（{N_ENVS} 環境）")

arm = np.arange(7)                          # 腕の関節（7自由度）
fingers = np.arange(7, 9)                   # 指（2自由度）
hand = franka.get_link("hand")

# 第5章の PD ゲイン（Genesis のチュートリアルの値）と、力の上限
franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
franka.set_dofs_force_range(np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
                            np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]))

# ---- 環境ごとに、箱の位置と把持力を変える ----
rng = np.random.default_rng(0)
offsets = rng.uniform(-0.08, 0.08, size=(4, 2))            # 箱の位置のずらし方 4種類
cube_xy = np.array([[0.6 + offsets[i % 4, 0], offsets[i % 4, 1]] for i in range(N_ENVS)])
cube.set_pos(torch.tensor(np.c_[cube_xy, np.full(N_ENVS, 0.02)], dtype=torch.float32))
grip = torch.tensor([GRIP_FORCES[i // 4] for i in range(N_ENVS)], dtype=torch.float32)

home = torch.tensor([0, -0.3, 0, -2.2, 0, 2.0, 0.79, 0.04, 0.04], dtype=torch.float32).repeat(N_ENVS, 1)
franka.set_qpos(home)
franka.control_dofs_position(home)

down = torch.tensor([0.0, 1.0, 0.0, 0.0]).repeat(N_ENVS, 1)   # 手先を真下に向ける姿勢（wxyz）


def ik(z):
    """全環境の手先目標 (N, 3) に対して、一括で IK を解く（第4章）"""
    pos = torch.tensor(np.c_[cube_xy, np.full(N_ENVS, z)], dtype=torch.float32)
    return franka.inverse_kinematics(link=hand, pos=pos, quat=down)


def move(q_goal, seconds, finger=None):
    """第5章の5次多項式で、関節空間をなめらかに補間しながら移動する"""
    q_start = franka.get_dofs_position()[:, arm].clone()
    n = int(seconds / 0.01)
    for k in range(n):
        s = min(1.0, (k + 1) / (0.8 * n))
        s = 10 * s**3 - 15 * s**4 + 6 * s**5
        franka.control_dofs_position(q_start + s * (q_goal[:, arm] - q_start), arm)
        if finger is not None:
            finger()
        scene.step()


t0 = time.perf_counter()
move(ik(0.25), 1.5)                                                      # 1. 箱の上へ
franka.control_dofs_position(torch.full((N_ENVS, 2), 0.04), fingers)
move(ik(0.135), 1.0)                                                     # 2. 下ろす
close = lambda: franka.control_dofs_force(-grip[:, None].repeat(1, 2), fingers)
move(ik(0.135), 0.5, close)                                              # 3. 力で指を閉じる
move(ik(0.30), 1.5, close)                                               # 4. 持ち上げる
sec = time.perf_counter() - t0

z = cube.get_pos()[:, 2].numpy()
print(f"シミュレーション 4.5 秒 × {N_ENVS} 環境 = {4.5 * N_ENVS:.0f} 秒分を、実時間 {sec:.1f} 秒で計算\n")
mu = min(cube.geoms[0].get_friction(), franka.get_link("left_finger").geoms[0].get_friction())
m, g = 0.04**3 * 1000, 9.81                                     # 箱の質量（既定の密度 1000 kg/m³）
a = 10 / np.sqrt(3) * (0.30 - 0.135) / (0.8 * 1.5) ** 2        # 持ち上げ時の最大加速度（5次多項式）
need = m * (g + a)
print(f"箱の質量 {m * 1000:.0f} g、摩擦係数 μ={mu:.2f}、持ち上げの最大加速度 {a:.2f} m/s²"
      f" → 必要な摩擦力 m(g+a) = {need:.3f} N\n")
z_ref = z[-4:].mean()                                          # 最も強く握った場合の箱の高さ
print(f"{'把持力F(1本)':>12s} {'2μF':>8s} {'理論':>6s} {'箱の高さ（4環境）':>26s} {'平均の滑り':>8s} {'成功':>5s}")
for j, F in enumerate(GRIP_FORCES):
    zs = z[4 * j: 4 * j + 4]
    theory = "持てる" if 2 * mu * F >= need else "滑る"
    slip = (z_ref - zs[zs > 0.05]).mean() * 100 if (zs > 0.05).any() else float("nan")
    slip_s = f"{slip:5.1f} cm" if not np.isnan(slip) else "  落下"
    print(f"{F:10.2f} N {2 * mu * F:6.2f} N {theory:>6s}   {np.array2string(zs, precision=3):>24s}"
          f" {slip_s:>8s} {int((zs > 0.15).sum())}/4")
