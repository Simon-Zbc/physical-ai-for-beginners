"""第6章 (2): スクリプトで「近づく → 下ろす → 掴む → 持ち上げる」を実行し、成否を判定する"""
import mujoco
import numpy as np

from scene import make


def waypoints(box_xy):
    """(x, y, z, 指の開き, 所要時間[s])。z はステージの関節値（0 = 初期の高さ）"""
    x, y = box_xy
    return [
        (x, y,  0.000, 0.04, 1.0),   # 1. 箱の真上へ移動（指は開く）
        (x, y, -0.225, 0.04, 1.0),   # 2. 下ろす（指が箱の側面の高さに来る）
        (x, y, -0.225, 0.00, 1.0),   # 3. 指を閉じる（目標 0 = 箱より内側 → 把持力の上限で押し付ける）
        (x, y,  0.000, 0.00, 1.5),   # 4. 持ち上げる
    ]


def quintic(s):
    """第5章の5次多項式（0→1 をなめらかにつなぐ）"""
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def run(mu=1.0, grip_force=5.0, box_mass=0.1, smooth=True, verbose=False):
    model, data = make(mu, grip_force, box_mass)
    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    dt = model.opt.timestep
    prev = data.ctrl.copy()
    max_box_h = 0.0
    for i, (x, y, z, w, T) in enumerate(waypoints(data.xpos[box_id][:2])):
        goal = np.array([x, y, z, w, w])
        n = int(T / dt)
        for k in range(n):
            if smooth:   # 所要時間の 8 割で目標まで補間し、残りは保持
                data.ctrl[:] = prev + quintic(k / (0.8 * n)) * (goal - prev)
            else:        # ステップ指令：目標をいきなり渡す
                data.ctrl[:] = goal
            mujoco.mj_step(model, data)
            max_box_h = max(max_box_h, data.xpos[box_id][2])
        prev = goal
        if verbose:
            print(f"  step{i + 1}: 手先 z={data.qpos[2]:+.3f}  指={data.qpos[3]:.4f}/{data.qpos[4]:.4f} m"
                  f"  箱の高さ={data.xpos[box_id][2]:.3f} m  指の力={data.actuator_force[3]:+.2f} N")
    lifted = data.xpos[box_id][2]
    return lifted, lifted > 0.15, max_box_h          # 15cm 以上持ち上がっていれば成功


if __name__ == "__main__":
    print("=== 1. ステップ指令 vs なめらかな軌道（μ=1.0, 把持力 5 N, 箱 100 g）===")
    for smooth, name in [(False, "ステップ指令"), (True, "5次多項式")]:
        print(f"[{name}]")
        h, ok, hmax = run(smooth=smooth, verbose=True)
        print(f"  → 結果: {'成功' if ok else '失敗'}（箱の最高到達点 {hmax:.3f} m）\n")

    print("=== 2. 摩擦係数と把持力を変える（5次多項式の軌道）===")
    m, g = 0.1, 9.81
    a_max = 10 / np.sqrt(3) * 0.225 / (0.8 * 1.5) ** 2      # 5次多項式で持ち上げるときの最大加速度
    need = m * (g + a_max)
    print(f"箱の重さ mg = {m * g:.2f} N、持ち上げの最大加速度 {a_max:.2f} m/s² → 必要な摩擦力 m(g+a) = {need:.2f} N")
    print("理論：2 μ F ≥ m(g+a) なら滑らずに持ち上がる\n")
    print(f"{'μ':>5s} {'把持力F':>7s} {'2μF':>7s} {'理論':>6s} {'箱の最終高さ':>10s} {'結果':>4s}")
    for mu, F in [(1.0, 5.0), (0.3, 5.0), (0.1, 6.0), (0.1, 5.0), (0.1, 3.0), (0.05, 12.0), (0.05, 5.0)]:
        h, ok, _ = run(mu, F, m)
        theory = "持てる" if 2 * mu * F >= need else "滑る"
        print(f"{mu:5.2f} {F:5.1f} N {2 * mu * F:5.2f} N {theory:>6s} {h:9.3f} m  {'成功' if ok else '失敗'}")
