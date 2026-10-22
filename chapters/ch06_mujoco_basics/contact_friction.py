"""第6章 (1): 斜面に置いた箱で、摩擦と接触力を確かめる"""
import mujoco
import numpy as np

XML = """
<mujoco>
  <option timestep="0.002" cone="elliptic"/>
  <worldbody>
    <!-- y 軸まわりに傾けた斜面 -->
    <geom name="slope" type="box" size="2 0.3 0.01" euler="0 @TILT@ 0" friction="@MU@ 0.005 0.0001"/>
    <body name="box" pos="0 0 0.05" euler="0 @TILT@ 0">
      <freejoint/>
      <geom name="box" type="box" size="0.03 0.03 0.03" mass="0.2" friction="@MU@ 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
"""


def simulate(tilt_deg, mu, T=0.5):
    xml = XML.replace("@TILT@", str(tilt_deg)).replace("@MU@", str(mu))   # euler は既定で「度」
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    start = data.qpos[:3].copy()
    for _ in range(int(0.2 / model.opt.timestep)):     # 着地して落ち着くまで
        mujoco.mj_step(model, data)
    settled = data.qpos[:3].copy()

    # 接触情報を読む：data.contact に接触点が入っている
    normal_total, tangent_total = 0.0, 0.0
    f6 = np.zeros(6)
    for i in range(data.ncon):
        mujoco.mj_contactForce(model, data, i, f6)   # 接触座標系での力 [法線, 接線1, 接線2, ...]
        normal_total += f6[0]
        tangent_total += np.linalg.norm(f6[1:3])
    ncon = data.ncon

    for _ in range(int(T / model.opt.timestep)):     # その後 T 秒でどれだけ滑るか
        mujoco.mj_step(model, data)
    slide = np.linalg.norm(data.qpos[:3] - settled)
    return slide, ncon, normal_total, tangent_total


if __name__ == "__main__":
    m, g = 0.2, 9.81
    print("理論：tanθ > μ なら滑る\n")
    print(f"{'傾き':>4s} {'μ':>4s} {'tanθ':>6s} {'0.5秒で滑った距離':>12s} {'接触点数':>6s} "
          f"{'法線力 N':>8s} {'摩擦力 F':>8s} {'F / N':>6s}")
    for tilt, mu in [(20, 0.5), (30, 0.5), (30, 0.7), (40, 0.7)]:
        slide, ncon, fn, ft = simulate(tilt, mu)
        th = np.deg2rad(tilt)
        print(f"{tilt:3d}° {mu:4.1f} {np.tan(th):6.3f} {slide * 100:10.2f} cm {ncon:6d} "
              f"{fn:7.3f} N {ft:7.3f} N {ft / fn:6.3f}")
    print(f"\n参考：止まっている場合の理論値  20°: N={m*g*np.cos(np.deg2rad(20)):.3f}, F={m*g*np.sin(np.deg2rad(20)):.3f}"
          f" / 30°: N={m*g*np.cos(np.deg2rad(30)):.3f}, F={m*g*np.sin(np.deg2rad(30)):.3f}")
