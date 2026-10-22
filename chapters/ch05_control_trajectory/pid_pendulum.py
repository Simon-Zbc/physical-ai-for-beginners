"""第5章 (1): 重力のある1関節アームで P / PD / PID / PD+重力補償 を比較する"""
import mujoco
import numpy as np

MJCF = """
<mujoco model="pendulum">
  <option timestep="0.001" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="link" pos="0 0 0.5">
      <!-- 水平な y 軸まわりに回る関節：角度 0 で腕が水平（x 方向）を向く -->
      <joint name="j" type="hinge" axis="0 -1 0" damping="0.05"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1.0"/>
    </body>
  </worldbody>
  <actuator><motor joint="j" ctrlrange="-@LIM@ @LIM@"/></actuator>
</mujoco>
"""


class PID:
    def __init__(self, kp, ki=0.0, kd=0.0, i_limit=None):
        self.kp, self.ki, self.kd, self.i_limit = kp, ki, kd, i_limit
        self.integral = 0.0

    def __call__(self, target, q, qdot, dt):
        e = target - q
        self.integral += e * dt
        if self.i_limit is not None:                       # アンチワインドアップ（積分値の上限）
            self.integral = np.clip(self.integral, -self.i_limit, self.i_limit)
        # D 項は誤差の微分ではなく「-速度」を使う（目標が急に変わっても D 項が跳ねない）
        return self.kp * e + self.ki * self.integral - self.kd * qdot


def run(ctrl, target, gravity_comp=False, torque_limit=20.0, T=3.0):
    model = mujoco.MjModel.from_xml_string(MJCF.replace("@LIM@", str(torque_limit)))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    log_q, log_tau = [], []
    for _ in range(int(T / model.opt.timestep)):
        q, qdot = data.qpos[0], data.qvel[0]
        tau = ctrl(target, q, qdot, model.opt.timestep)
        if gravity_comp:
            tau += data.qfrc_bias[0]                         # 重力（+コリオリ力）を打ち消すトルク
        data.ctrl[0] = np.clip(tau, -torque_limit, torque_limit)
        mujoco.mj_step(model, data)
        log_q.append(data.qpos[0]); log_tau.append(data.ctrl[0])
    return np.array(log_q), np.array(log_tau), model.opt.timestep


def metrics(q, target, dt, band=np.deg2rad(1.0)):
    """オーバーシュート・整定時間（±1°に収まって以後出ない時刻）・定常偏差"""
    overshoot = max(0.0, (q.max() - target) / target * 100)
    outside = np.where(np.abs(q - target) > band)[0]
    settle = (outside[-1] + 1) * dt if len(outside) else 0.0
    if settle > len(q) * dt - 0.2:
        return overshoot, float("nan"), float("nan")        # 時間内に整定しなかった
    sse = np.rad2deg(target - q[-200:].mean())
    return overshoot, settle, sse


if __name__ == "__main__":
    target = np.deg2rad(45)                                 # 水平から 45° 持ち上げる
    cases = [
        ("P   (Kp=20)",              PID(20),                     False),
        ("P   (Kp=100)",             PID(100),                    False),
        ("PD  (Kp=100, Kd=4)",       PID(100, kd=4),              False),
        ("PID (Kp=100, Ki=200, Kd=4)", PID(100, ki=200, kd=4),    False),
        ("PD + 重力補償",             PID(100, kd=4),              True),
    ]
    print(f"目標: {np.rad2deg(target):.0f}°（重力に逆らって持ち上げる）\n")
    print(f"{'制御則':28s} {'オーバーシュート':>10s} {'整定時間':>8s} {'定常偏差':>8s} {'最大トルク':>8s}")
    for name, ctrl, gc in cases:
        q, tau, dt = run(ctrl, target, gravity_comp=gc)
        os_, st, sse = metrics(q, target, dt)
        st_s = f"{st:.2f} s" if not np.isnan(st) else "未整定"
        sse_s = f"{sse:6.2f}°" if not np.isnan(sse) else "（振動）"
        print(f"{name:28s} {os_:9.1f} % {st_s:>8s} {sse_s:>8s} {np.abs(tau).max():7.1f} Nm")

    print("\n=== 積分ワインドアップ：トルク上限が厳しい（2.5 Nm）とき ===")
    target2 = np.deg2rad(60)
    for name, ctrl in [("PID（積分の上限なし）", PID(100, ki=200, kd=4)),
                       ("PID（積分の上限あり）", PID(100, ki=200, kd=4, i_limit=0.01))]:
        q, tau, dt = run(ctrl, target2, torque_limit=2.5, T=6.0)
        os_, st, sse = metrics(q, target2, dt)
        st_s = f"{st:.2f} s" if not np.isnan(st) else "未整定"
        print(f"{name:24s} オーバーシュート {os_:5.1f} %  整定時間 {st_s:>8s}  定常偏差 {sse:6.2f}°")
