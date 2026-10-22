"""第4章 (1): 2関節アームで「解析的 IK」「ヤコビアン」「特異姿勢」を確かめる"""
import mujoco
import numpy as np

np.set_printoptions(precision=4, suppress=True)
L1, L2 = 0.30, 0.25


def fk(q):
    """順運動学（第3章 6.2 節の式）"""
    return np.array([L1 * np.cos(q[0]) + L2 * np.cos(q[0] + q[1]),
                     L1 * np.sin(q[0]) + L2 * np.sin(q[0] + q[1])])


def analytic_ik(x, y):
    """解析的 IK：余弦定理で肘の角度を求める。解は最大2つ（肘を上げる / 下げる）"""
    c2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    if abs(c2) > 1:
        return []                                    # 届かない：解なし
    sols = []
    for s2 in [np.sqrt(1 - c2**2), -np.sqrt(1 - c2**2)]:
        q2 = np.arctan2(s2, c2)
        q1 = np.arctan2(y, x) - np.arctan2(L2 * s2, L1 + L2 * c2)
        sols.append(np.array([q1, q2]))
    return sols


def jacobian(q):
    """解析的ヤコビアン：J[i, j] = ∂x_i / ∂q_j"""
    s1, c1 = np.sin(q[0]), np.cos(q[0])
    s12, c12 = np.sin(q[0] + q[1]), np.cos(q[0] + q[1])
    return np.array([[-L1 * s1 - L2 * s12, -L2 * s12],
                     [ L1 * c1 + L2 * c12,  L2 * c12]])


def numeric_jacobian(f, q, eps=1e-6):
    """数値微分のヤコビアン：各関節を少しだけ動かして、手先の変化を測る"""
    x0 = f(q)
    J = np.zeros((len(x0), len(q)))
    for j in range(len(q)):
        dq = np.zeros(len(q)); dq[j] = eps
        J[:, j] = (f(q + dq) - x0) / eps
    return J


MJCF = """
<mujoco><worldbody>
  <body><joint type="hinge" axis="0 0 1"/>
    <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02"/>
    <body pos="0.3 0 0"><joint type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.02"/>
      <site name="hand" pos="0.25 0 0"/>
    </body></body>
</worldbody></mujoco>"""

if __name__ == "__main__":
    print("=== 1. 解析的 IK：解は 2つ・0個のこともある ===")
    for target in [(0.2, 0.35), (0.6, 0.3)]:
        sols = analytic_ik(*target)
        print(f"目標 {target}: 解の数 = {len(sols)}")
        for q in sols:
            print(f"   q = {np.rad2deg(q).round(1)} deg  → FK で確認: {fk(q)}")

    print("\n=== 2. ヤコビアン：解析解・数値微分・MuJoCo の比較 ===")
    q = np.array([0.3, 0.8])
    model = mujoco.MjModel.from_xml_string(MJCF); data = mujoco.MjData(model)
    data.qpos[:] = q; mujoco.mj_kinematics(model, data); mujoco.mj_comPos(model, data)
    jacp = np.zeros((3, 2)); mujoco.mj_jacSite(model, data, jacp, None, 0)
    print("解析的   :\n", jacobian(q))
    print("数値微分 :\n", numeric_jacobian(fk, q))
    print("MuJoCo   :\n", jacp[:2])

    print("\n=== 3. 特異姿勢に近づくと何が起きるか ===")
    print(" 肘の角度 | 可操作度 w | 条件数    | x方向へ 1cm/s 動かす関節速度 (pinv) | 同 (DLS, λ=0.01)")
    v = np.array([0.01, 0.0])                       # 手先を x 方向に 1cm/s で動かしたい
    for q2_deg in [90, 30, 10, 3, 1, 0.1]:
        q = np.array([0.0, np.deg2rad(q2_deg)])
        J = jacobian(q)
        w = np.sqrt(np.linalg.det(J @ J.T))         # 可操作度（Yoshikawa）
        cond = np.linalg.cond(J)
        dq_pinv = np.linalg.pinv(J) @ v
        lam = 0.01
        dq_dls = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(2), v)
        print(f" {q2_deg:7.1f}° | {w:.5f}   | {cond:9.1f} | {np.linalg.norm(dq_pinv):10.3f} rad/s"
              f"                | {np.linalg.norm(dq_dls):.3f} rad/s")
