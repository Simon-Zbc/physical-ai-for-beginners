"""第5章 (3): 学習済み方策（10Hz）の出力を、1kHz の制御ループでどう実行するか

方策の代わりに「10Hz で関節角の目標を返す関数」を使い、
(a) 届いた目標をそのまま保持する（ゼロ次ホールド）
(b) 前の目標から次の目標まで線形補間する
の2通りで実行して比べる。
"""
import mujoco
import numpy as np

from arm_model import make, pd_gravity

POLICY_HZ = 10                                   # 方策の出力周波数（VLA などを想定）


def policy(t):
    """方策の代役：時刻 t にいるべき関節角を返す（なめらかな往復運動）"""
    q = np.deg2rad([0, -40, 60, 0])
    return q + np.deg2rad([40, 15, -20, 45]) * np.sin(2 * np.pi * 0.5 * t)


def run(mode, total=4.0):
    model, data = make(policy(0.0))
    dt = model.opt.timestep
    per = int(1 / (POLICY_HZ * dt))              # 方策1回あたりの制御ステップ数（= 100）
    prev = cur = policy(0.0)
    errs, dtaus, last_tau = [], [], None
    for k in range(int(total / dt)):
        t = k * dt
        if k % per == 0:                         # 10Hz：方策が「次の周期の終わりの姿勢」を出す
            prev, cur = cur, policy(t + 1 / POLICY_HZ)
        if mode == "hold":                       # (a) 目標を階段状に切り替える
            q_ref, qd_ref = cur, np.zeros(4)
        else:                                    # (b) 周期内で等速に補間し、速度も指令する
            a = (k % per) / per
            q_ref, qd_ref = prev + a * (cur - prev), (cur - prev) * POLICY_HZ
        tau = pd_gravity(data, q_ref, qd_ref)    # 1kHz：PD + 重力補償
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        if t > 1.0:                              # 立ち上がりの1秒を除いて評価
            errs.append(np.abs(policy(t + dt) - data.qpos).max())
            if last_tau is not None:
                dtaus.append(np.abs(tau - last_tau).max() / dt)
        last_tau = tau
    return np.rad2deg(np.mean(errs)), np.rad2deg(np.max(errs)), np.max(dtaus)


if __name__ == "__main__":
    for mode, name in [("hold", "ゼロ次ホールド"), ("interp", "線形補間")]:
        mean_e, max_e, dtau = run(mode)
        print(f"{name:8s}: 理想の軌道からの誤差 平均 {mean_e:.2f}° / 最大 {max_e:.2f}°   "
              f"トルク変化率の最大 {dtau:7.0f} Nm/s")
