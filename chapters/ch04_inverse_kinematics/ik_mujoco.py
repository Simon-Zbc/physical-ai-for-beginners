"""第4章 (3): 差分 IK で、手先に円を描かせる（MuJoCo の位置制御アクチュエータで実行）

VLA などの学習済み方策が「手先をこれだけ動かせ」という指令（Δx）を出し、
差分 IK が関節角の指令に変換する ― という現在よく使われる構成の最小版。
"""
import mujoco
import numpy as np

from fk import MJCF, forward_kinematics
from ik_numeric import numeric_jacobian

# 第3章のアームに、位置制御アクチュエータ（関節角の目標値を受け取るモーター）を追加
MJCF_ACT = MJCF.replace("<mujoco model=\"arm4dof\">",
                        "<mujoco model=\"arm4dof\">\n  <option gravity=\"0 0 0\"/>"
                        "\n  <default><joint damping=\"2\"/></default>")
MJCF_ACT = MJCF_ACT.replace("</mujoco>", """  <actuator>
    <position joint="yaw" kp="200"/> <position joint="shoulder" kp="200"/>
    <position joint="elbow" kp="200"/> <position joint="wrist_roll" kp="50"/>
  </actuator>
</mujoco>""")


def dls(J, e, lam=0.05):
    return J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(J.shape[0]), e)


model = mujoco.MjModel.from_xml_string(MJCF_ACT)
data = mujoco.MjData(model)
sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "hand")
data.qpos[:] = np.deg2rad([0, -30, 60, 0])
data.ctrl[:] = data.qpos
mujoco.mj_forward(model, data)

center, radius, period = np.array([0.40, 0.0, 0.25]), 0.10, 4.0   # 円の中心・半径・周期
q_cmd = data.qpos.copy()                       # 関節角の指令値
errors = []

for step in range(int(8.0 / model.opt.timestep)):          # 8 秒（円を 2 周）
    t = data.time
    # 1) 方策（ここでは決め打ちの軌道）が手先の目標位置を出す
    target = center + radius * np.array([0, np.cos(2 * np.pi * t / period),
                                         np.sin(2 * np.pi * t / period)])
    # 2) 差分 IK：指令値 q_cmd の手先と目標の差 Δx を、関節角の変化 Δq に変換
    #    （実機の現在値ではなく指令値で計算すると、追従遅れで指令が暴走しない）
    e_cmd = target - forward_kinematics(q_cmd)[:3, 3]
    q_cmd = q_cmd + dls(numeric_jacobian(q_cmd), e_cmd)
    e = target - data.site_xpos[sid]                         # 実際の追従誤差（評価用）
    # 3) 位置制御アクチュエータに関節角の指令を送る
    data.ctrl[:] = q_cmd
    mujoco.mj_step(model, data)
    if t > 1.0:                                              # 最初の 1 秒（立ち上がり）は除く
        errors.append(np.linalg.norm(e))
    if step % 500 == 0:
        print(f"t={t:4.1f}s  目標={target.round(3)}  手先={data.site_xpos[sid].round(3)}  "
              f"誤差={np.linalg.norm(e) * 1000:5.1f} mm")

print(f"\n追従誤差（1秒以降）: 平均 {np.mean(errors) * 1000:.1f} mm / 最大 {np.max(errors) * 1000:.1f} mm")
