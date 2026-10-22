"""第5章: 重力のある4自由度アーム（第3・4章と同じ形状に、質量・重力・モーターを追加）"""
import mujoco
import numpy as np

MJCF_DYN = """
<mujoco model="arm4dof_dynamics">
  <option timestep="0.001" gravity="0 0 -9.81"/>
  <default><joint damping="0.1" armature="0.01"/><geom density="1000"/></default>
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
  <actuator>
    <motor joint="yaw" ctrlrange="-30 30"/> <motor joint="shoulder" ctrlrange="-30 30"/>
    <motor joint="elbow" ctrlrange="-30 30"/> <motor joint="wrist_roll" ctrlrange="-5 5"/>
  </actuator>
</mujoco>
"""

KP = np.array([150.0, 150.0, 100.0, 10.0])    # 関節ごとの比例ゲイン
KD = np.array([10.0, 10.0, 6.0, 0.5])         # 関節ごとの微分ゲイン
LIMIT = np.array([30.0, 30.0, 30.0, 5.0])     # トルク上限 [Nm]


def make(q0):
    """モデルを作り、関節角 q0 から始める"""
    model = mujoco.MjModel.from_xml_string(MJCF_DYN)
    data = mujoco.MjData(model)
    data.qpos[:] = q0
    mujoco.mj_forward(model, data)
    return model, data


def pd_gravity(data, q_ref, qd_ref):
    """PD + 重力補償：τ = Kp (q_ref − q) + Kd (q̇_ref − q̇) + g(q)

    data.qfrc_bias には重力とコリオリ力・遠心力の項が入っている（MuJoCo が毎ステップ計算）。
    """
    tau = KP * (q_ref - data.qpos) + KD * (qd_ref - data.qvel) + data.qfrc_bias
    return np.clip(tau, -LIMIT, LIMIT)
