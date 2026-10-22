"""第3章 (2): 4自由度アームの順運動学を NumPy で実装し、MuJoCo と答え合わせする"""
import mujoco
import numpy as np

from transforms import make_T, quat_to_mat, rot_x, rot_y, rot_z, trans

np.set_printoptions(precision=4, suppress=True)

# ---- ロボットの定義（MJCF）: 旋回 → 肩 → 肘 → 手首ロール ----
MJCF = """
<mujoco model="arm4dof">
  <worldbody>
    <body name="base" pos="0 0 0.1">
      <joint name="yaw" type="hinge" axis="0 0 1"/>
      <geom type="cylinder" size="0.04 0.02" rgba="0.3 0.3 0.3 1"/>
      <body name="upper_arm">
        <joint name="shoulder" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" rgba="0.2 0.4 0.8 1"/>
        <body name="forearm" pos="0.3 0 0">
          <joint name="elbow" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.018" rgba="0.8 0.4 0.2 1"/>
          <body name="wrist" pos="0.25 0 0">
            <joint name="wrist_roll" type="hinge" axis="1 0 0"/>
            <geom type="box" size="0.02 0.03 0.01" pos="0.025 0 0" rgba="0.2 0.7 0.3 1"/>
            <site name="hand" pos="0.05 0 0" size="0.01"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

L0, L1, L2, L3 = 0.10, 0.30, 0.25, 0.05   # 基部の高さ・上腕・前腕・手先までの長さ [m]


def forward_kinematics(q, return_all=False):
    """関節角 q = [yaw, shoulder, elbow, wrist_roll] から手先の姿勢 T_base_hand を計算する

    各関節について「親リンクからの固定オフセット」→「関節の回転」を順に掛けていく。
    """
    q1, q2, q3, q4 = q
    T_yaw      = trans(0, 0, L0) @ make_T(rot_z(q1))   # 基部の高さに上がって、z 軸まわりに旋回
    T_shoulder = make_T(rot_y(q2))                     # 肩：y 軸まわり
    T_elbow    = trans(L1, 0, 0) @ make_T(rot_y(q3))   # 上腕の先端で、肘：y 軸まわり
    T_wrist    = trans(L2, 0, 0) @ make_T(rot_x(q4))   # 前腕の先端で、手首：x 軸まわり
    T_hand     = trans(L3, 0, 0)                       # 手首から手先まで

    frames = [T_yaw]
    for T in [T_shoulder, T_elbow, T_wrist, T_hand]:
        frames.append(frames[-1] @ T)                  # T_0n = T_01 · T_12 · ... · T_(n-1)n
    return frames if return_all else frames[-1]


def mujoco_fk(model, data, q):
    """同じ関節角での手先姿勢を MuJoCo に計算させる（答え合わせ用）"""
    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)                  # 物理は進めず、運動学だけ計算
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "hand")
    return data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy()


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_string(MJCF)
    data = mujoco.MjData(model)

    # 1) 特定の姿勢で比較
    q = np.deg2rad([30, -45, 60, 90])
    T = forward_kinematics(q)
    p_mj, R_mj = mujoco_fk(model, data, q)
    print("関節角 [deg]:", np.rad2deg(q))
    print("手先位置  NumPy :", T[:3, 3])
    print("手先位置  MuJoCo:", p_mj)
    print("手先の姿勢（回転行列） NumPy:\n", T[:3, :3])

    # 2) ランダムな 1000 姿勢で誤差を確認
    rng = np.random.default_rng(0)
    pos_err, rot_err = [], []
    for _ in range(1000):
        q = rng.uniform(-np.pi, np.pi, size=4)
        T = forward_kinematics(q)
        p_mj, R_mj = mujoco_fk(model, data, q)
        pos_err.append(np.linalg.norm(T[:3, 3] - p_mj))
        rot_err.append(np.abs(T[:3, :3] - R_mj).max())
    print(f"\nランダム 1000 姿勢: 位置誤差の最大 {max(pos_err):.2e} m / 回転行列の誤差の最大 {max(rot_err):.2e}")

    # 3) MuJoCo のクォータニオン（[w, x, y, z]）とも一致するか
    q = np.deg2rad([30, -45, 60, 90])
    frames = forward_kinematics(q, return_all=True)
    mujoco_fk(model, data, q)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist")
    quat_wxyz = data.xquat[bid]
    print("\n手首のクォータニオン（MuJoCo, wxyz）:", quat_wxyz)
    print("wxyz として解釈 → FK と一致？", np.allclose(quat_to_mat(quat_wxyz), frames[3][:3, :3]))
    quat_xyzw = quat_wxyz[[1, 2, 3, 0]]               # SciPy / ROS の順番に並べ替えたもの
    print("xyzw を wxyz と誤解釈 → 一致？", np.allclose(quat_to_mat(quat_xyzw), frames[3][:3, :3]))

    # 4) 各関節の位置（どの座標系が、どこにあるか）
    print("\n各座標系の原点（基部座標）:")
    names = ["旋回(yaw)", "肩(shoulder)", "肘(elbow)", "手首(wrist)", "手先(hand)"]
    for name, F in zip(names, forward_kinematics(np.deg2rad([30, -45, 60, 90]), return_all=True)):
        print(f"  {name:14s} {F[:3, 3]}")
