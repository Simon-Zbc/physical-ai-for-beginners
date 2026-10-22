"""第0章: Sense → Think → Act ループの最小実装（MuJoCo）"""
import mujoco
import numpy as np

# ---- 1. 世界（ロボット＋環境）を MJCF で定義 ----
MJCF = """
<mujoco model="two_link_arm">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="1 1 0.1" rgba="0.9 0.9 0.9 1"/>
    <body name="link1" pos="0 0 0.05">
      <joint name="shoulder" type="hinge" axis="0 0 1" damping="0.5"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" rgba="0.2 0.4 0.8 1"/>
      <body name="link2" pos="0.3 0 0">
        <joint name="elbow" type="hinge" axis="0 0 1" damping="0.5"/>
        <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.02" rgba="0.8 0.4 0.2 1"/>
        <site name="hand" pos="0.25 0 0" size="0.02"/>
      </body>
    </body>
    <site name="target" pos="0.2 0.35 0.05" size="0.025" rgba="0 1 0 0.6"/>
  </worldbody>
  <actuator>
    <motor joint="shoulder" gear="1" ctrlrange="-5 5"/>
    <motor joint="elbow"    gear="1" ctrlrange="-5 5"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(MJCF)
data = mujoco.MjData(model)
hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "hand")
target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target")

# 初期姿勢：腕が一直線（特異姿勢）だとヤコビ転置法ではうまく動かせないため、肘を少し曲げておく
data.qpos[:] = [0.0, 0.8]
mujoco.mj_forward(model, data)   # 位置・ヤコビアンなどを計算して状態を確定


def sense(model, data):
    """知覚: センサー値（ここではシミュレータの真値）を観測として取り出す"""
    return {
        "qpos": data.qpos.copy(),          # 関節角度 [rad]
        "qvel": data.qvel.copy(),          # 関節角速度 [rad/s]
        "hand": data.site_xpos[hand_id].copy(),
        "target": data.site_xpos[target_id].copy(),
    }


def think(model, data, obs, kp=300.0, kd=1.0):
    """判断: 手先を目標へ近づけるトルクを計算（ヤコビ転置法）"""
    error = obs["target"] - obs["hand"]            # 手先位置の誤差（ワールド座標）
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, hand_id)  # 手先の並進ヤコビアン
    tau = kp * jacp.T @ error - kd * obs["qvel"]   # τ = Kp・Jᵀ・e − Kd・q̇
    return tau


def act(model, data, tau):
    """行動: アクチュエータに指令を送り、物理を1ステップ進める"""
    data.ctrl[:] = np.clip(tau, -5, 5)
    mujoco.mj_step(model, data)


if __name__ == "__main__":
    # ---- 2. 制御ループ ----
    for step in range(1500):                      # 0.002s × 1500 = 3秒
        obs = sense(model, data)
        tau = think(model, data, obs)
        act(model, data, tau)
        if step % 100 == 0 and step <= 600:
            dist = np.linalg.norm(obs["target"] - obs["hand"])
            print(f"t={data.time:4.2f}s  手先={obs['hand'][:2].round(3)}  目標までの距離={dist:.4f} m")

    final = np.linalg.norm(data.site_xpos[target_id] - data.site_xpos[hand_id])
    print(f"最終誤差: {final * 1000:.1f} mm")
