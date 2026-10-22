"""第3章 (1): 座標変換の基本を動かして確かめる"""
import numpy as np

from transforms import (inv_T, make_T, quat_from_axis_angle, quat_mul, quat_rotate,
                        quat_to_mat, rot_x, rot_y, rot_z, slerp, transform_point, trans)

np.set_printoptions(precision=3, suppress=True)
deg = np.deg2rad

print("=== 1. 回転行列の性質 ===")
R = rot_z(deg(30)) @ rot_y(deg(45)) @ rot_x(deg(60))
print("R^T R = I ?", np.allclose(R.T @ R, np.eye(3)), "/ det(R) =", round(np.linalg.det(R), 6))

print("\n=== 2. 回転の順番を入れ替えると結果が変わる ===")
v = np.array([1.0, 0.0, 0.0])
print("Rz(90)·Ry(90)·v =", rot_z(deg(90)) @ rot_y(deg(90)) @ v)
print("Ry(90)·Rz(90)·v =", rot_y(deg(90)) @ rot_z(deg(90)) @ v)

print("\n=== 3. カメラ座標 → ロボット基部座標 ===")
# カメラはロボット基部から見て (0.5, 0, 0.6) の位置にあり、真下（-z 方向）を向いている
# カメラ座標系：z = 視線方向、x = 画像の右、y = 画像の下（OpenCV の慣習）
R_base_cam = np.array([[0, -1, 0],
                       [-1, 0, 0],
                       [0, 0, -1]], dtype=float)
T_base_cam = make_T(R_base_cam, [0.5, 0.0, 0.6])
p_cam = np.array([0.10, -0.05, 0.55])          # カメラが検出した物体の位置（カメラ座標）
p_base = transform_point(T_base_cam, p_cam)
print("物体（カメラ座標）:", p_cam, "→ 物体（基部座標）:", p_base)
print("逆変換で戻す      :", transform_point(inv_T(T_base_cam), p_base))

print("\n=== 4. 変換の連鎖：基部 → 手先 → 掴んだ物体 ===")
T_base_hand = make_T(rot_z(deg(90)), [0.3, 0.2, 0.1])   # 手先の姿勢
T_hand_obj = trans(0.0, 0.0, 0.05)                      # 物体は手先から z 方向に 5cm
T_base_obj = T_base_hand @ T_hand_obj
print("物体の位置（基部座標）:", T_base_obj[:3, 3])

print("\n=== 5. クォータニオン ===")
q1 = quat_from_axis_angle([0, 0, 1], deg(90))           # z 軸まわり 90°
q2 = quat_from_axis_angle([1, 0, 0], deg(90))           # x 軸まわり 90°
print("q1 =", q1)
print("回転行列と一致する？", np.allclose(quat_to_mat(q1), rot_z(deg(90))))
print("q1 で [1,0,0] を回転:", quat_rotate(q1, v))
print("合成 q1*q2 と Rz·Rx が一致？", np.allclose(quat_to_mat(quat_mul(q1, q2)),
                                              rot_z(deg(90)) @ rot_x(deg(90))))
print("q と -q は同じ回転？", np.allclose(quat_to_mat(q1), quat_to_mat(-q1)))

print("\n=== 6. 姿勢の補間（slerp） ===")
q_start = np.array([1.0, 0, 0, 0])                      # 回転なし
q_goal = quat_from_axis_angle([0, 0, 1], deg(120))
for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
    q = slerp(q_start, q_goal, t)
    angle = np.rad2deg(2 * np.arccos(np.clip(q[0], -1, 1)))
    print(f"t={t:.2f}  回転角 = {angle:6.1f}°")

print("\n=== 7. ジンバルロック ===")
# ZYX オイラー角（yaw, pitch, roll）で pitch = 90° のとき
R_a = rot_z(deg(30)) @ rot_y(deg(90)) @ rot_x(deg(0))
R_b = rot_z(deg(0)) @ rot_y(deg(90)) @ rot_x(deg(-30))
R_c = rot_z(deg(50)) @ rot_y(deg(90)) @ rot_x(deg(20))
print("(yaw,pitch,roll)=(30,90,0) と (0,90,-30) は同じ姿勢？", np.allclose(R_a, R_b))
print("(yaw,pitch,roll)=(30,90,0) と (50,90,20) は同じ姿勢？", np.allclose(R_a, R_c))
