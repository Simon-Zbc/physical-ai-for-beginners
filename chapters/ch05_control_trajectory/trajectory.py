"""第5章 (2): 同じ移動を「ステップ」「台形速度」「5次多項式」の3つの軌道で比べる"""
import mujoco
import numpy as np

from arm_model import LIMIT, make, pd_gravity

np.set_printoptions(precision=3, suppress=True)


def step_traj(q0, q1, T, t):
    """目標を一瞬で切り替える（軌道なし）"""
    return q1.copy(), np.zeros_like(q0)


def trapezoid_traj(q0, q1, T, t, ratio=0.25):
    """台形速度：加速 → 等速 → 減速。加速度は区間ごとに一定（境界で不連続）"""
    ta = ratio * T                                  # 加速にかける時間
    vmax = 1.0 / (T - ta)                           # 正規化した最高速度（全体の移動量を 1 とする）
    if t < ta:
        s, sd = 0.5 * vmax / ta * t**2, vmax / ta * t
    elif t < T - ta:
        s, sd = 0.5 * vmax * ta + vmax * (t - ta), vmax
    elif t < T:
        tr = T - t
        s, sd = 1.0 - 0.5 * vmax / ta * tr**2, vmax / ta * tr
    else:
        s, sd = 1.0, 0.0
    return q0 + s * (q1 - q0), sd * (q1 - q0)


def quintic_traj(q0, q1, T, t):
    """5次多項式：始点と終点で速度・加速度がともに 0。加速度まで連続になる"""
    tau = np.clip(t / T, 0.0, 1.0)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    sd = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T
    return q0 + s * (q1 - q0), sd * (q1 - q0)


def run(traj, q0, q1, T=1.0, total=2.0):
    model, data = make(q0)
    dt = model.opt.timestep
    errs, taus, vmax = [], [], 0.0
    for k in range(int(total / dt)):
        q_ref, qd_ref = traj(q0, q1, T, k * dt)
        tau = pd_gravity(data, q_ref, qd_ref)
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        errs.append(np.abs(q_ref - data.qpos).max())
        taus.append(tau.copy())
        vmax = max(vmax, np.abs(data.qvel).max())
    taus = np.array(taus)
    dtau = np.abs(np.diff(taus, axis=0)).max() / dt          # トルクの変化率の最大値
    final = np.rad2deg(np.abs(q1 - data.qpos).max())
    sat = int((np.abs(taus) >= LIMIT - 1e-9).any(axis=1).sum())  # トルクが上限に張り付いたステップ数
    return np.rad2deg(max(errs)), np.abs(taus).max(), dtau, np.rad2deg(vmax), final, sat


if __name__ == "__main__":
    q0 = np.deg2rad([0, -20, 60, 0])
    q1 = np.deg2rad([90, -60, 30, 90])                       # 旋回しながら腕を持ち上げる
    print("移動: ", np.rad2deg(q0), "→", np.rad2deg(q1), "deg（1.0 秒で）\n")
    print(f"{'軌道':10s} {'最大追従誤差':>8s} {'最大トルク':>8s} {'トルク変化率の最大':>10s} {'最大関節速度':>8s} {'最終誤差':>7s} {'飽和ステップ':>6s}")
    for name, traj in [("ステップ", step_traj), ("台形速度", trapezoid_traj), ("5次多項式", quintic_traj)]:
        e, tmax, dtau, vmax, fin, sat = run(traj, q0, q1)
        print(f"{name:10s} {e:9.2f}° {tmax:7.1f} Nm {dtau:9.0f} Nm/s {vmax:8.0f}°/s {fin:7.3f}° {sat:8d}")
