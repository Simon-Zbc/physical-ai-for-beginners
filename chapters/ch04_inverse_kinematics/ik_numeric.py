"""第4章 (2): 4自由度アームの数値 IK（ヤコビ転置・擬似逆行列・DLS）を比較する

第3章の transforms.py と fk.py を同じディレクトリに置いて実行する。
"""
import mujoco
import numpy as np

from fk import MJCF, forward_kinematics

np.set_printoptions(precision=4, suppress=True)
Q_MIN = np.deg2rad([-170, -120, -150, -180])       # 関節の可動範囲（関節リミット）
Q_MAX = np.deg2rad([170, 120, 150, 180])


def hand_pos(q):
    return forward_kinematics(q)[:3, 3]


def numeric_jacobian(q, eps=1e-6):
    """位置ヤコビアン（3×4）を数値微分で求める"""
    x0 = hand_pos(q)
    J = np.zeros((3, len(q)))
    for j in range(len(q)):
        dq = np.zeros(len(q)); dq[j] = eps
        J[:, j] = (hand_pos(q + dq) - x0) / eps
    return J


def ik_step(J, e, method, lam=0.05):
    """誤差 e を減らす関節角の更新量 Δq を計算する"""
    if method == "transpose":                       # Δq = α Jᵀ e
        g = J @ J.T @ e                             # α は Buss の方法で毎回自動調整する
        alpha = (e @ g) / (g @ g + 1e-12)
        return alpha * J.T @ e
    if method == "pinv":                            # Δq = J⁺ e
        return np.linalg.pinv(J) @ e
    if method == "dls":                             # Δq = Jᵀ (J Jᵀ + λ² I)⁻¹ e
        return J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(J.shape[0]), e)
    raise ValueError(method)


def solve_ik(target, q0, method, max_iter=200, tol=1e-4, max_step=0.2):
    """反復計算で IK を解く。戻り値: (関節角, 収束したか, 反復回数, 1回の最大更新量)"""
    q = q0.copy()
    biggest = 0.0
    for i in range(max_iter):
        e = target - hand_pos(q)
        if np.linalg.norm(e) < tol:
            return q, True, i, biggest
        dq = ik_step(numeric_jacobian(q), e, method)
        biggest = max(biggest, np.abs(dq).max())
        dq = np.clip(dq, -max_step, max_step)      # 1回の更新量を制限（暴れ防止）
        q = np.clip(q + dq, Q_MIN, Q_MAX)          # 関節リミットを守る
    return q, False, max_iter, biggest


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_string(MJCF)
    data = mujoco.MjData(model)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "hand")

    print("=== 1. 数値ヤコビアンと MuJoCo の mj_jacSite を比較 ===")
    q = np.deg2rad([30, -45, 60, 90])
    data.qpos[:] = q; mujoco.mj_forward(model, data)
    jacp = np.zeros((3, model.nv)); mujoco.mj_jacSite(model, data, jacp, None, sid)
    print("数値微分:\n", numeric_jacobian(q))
    print("MuJoCo  :\n", jacp)
    print("→ 4列目（手首ロール）が 0：手首をひねっても手先の『位置』は変わらない")

    print("\n=== 2. 届く目標 300 個で 3 手法を比較 ===")
    rng = np.random.default_rng(0)
    targets = [hand_pos(rng.uniform(Q_MIN * 0.8, Q_MAX * 0.8)) for _ in range(300)]
    q0 = np.deg2rad([0, -30, 60, 0])               # 毎回同じ初期姿勢から解く
    for method in ["transpose", "pinv", "dls"]:
        res = [solve_ik(t, q0, method) for t in targets]
        ok = [r for r in res if r[1]]
        print(f"{method:9s}: 成功率 {len(ok) / len(res) * 100:5.1f}%  "
              f"平均反復 {np.mean([r[2] for r in ok]):5.1f} 回  "
              f"更新量の最大 {max(r[3] for r in res):8.2f} rad")

    print("\n=== 3. 届かない目標（基部から 1m 先）を与えたとき ===")
    far = np.array([1.0, 0.0, 0.3])
    for method in ["transpose", "pinv", "dls"]:
        q, ok, it, big = solve_ik(far, q0, method)
        print(f"{method:9s}: 収束={ok}  最終誤差 {np.linalg.norm(far - hand_pos(q)):.3f} m  "
              f"更新量の最大 {big:10.2f} rad  最終姿勢 {np.rad2deg(q).round(1)} deg")
