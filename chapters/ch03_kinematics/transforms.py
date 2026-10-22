"""第3章: 座標変換の基本ツール（NumPy だけで実装）

- 回転行列 / 同次変換行列
- クォータニオン（MuJoCo と同じ [w, x, y, z] の順）
"""
import numpy as np


# ================= 回転行列 =================
def rot_x(theta):
    """x 軸まわりに theta [rad] 回転する回転行列"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(theta):
    """y 軸まわりに theta [rad] 回転する回転行列"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(theta):
    """z 軸まわりに theta [rad] 回転する回転行列"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# ================= 同次変換行列 =================
def make_T(R=np.eye(3), p=np.zeros(3)):
    """回転 R と並進 p から 4x4 の同次変換行列を作る"""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def trans(x, y, z):
    """並進だけの同次変換行列"""
    return make_T(p=np.array([x, y, z], dtype=float))


def inv_T(T):
    """同次変換の逆行列（np.linalg.inv より速く、数値的にも安定）"""
    R, p = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ p)


def transform_point(T, p):
    """点 p（3次元）を同次変換 T で変換する"""
    return (T @ np.append(p, 1.0))[:3]


# ================= クォータニオン [w, x, y, z] =================
def quat_from_axis_angle(axis, theta):
    """回転軸 axis まわりに theta [rad] 回転するクォータニオン"""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([[np.cos(theta / 2)], np.sin(theta / 2) * axis])


def quat_mul(q1, q2):
    """クォータニオンの積 q1 * q2（「q2 の回転をしてから q1 の回転」に相当）"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conj(q):
    """共役（単位クォータニオンでは逆回転）"""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate(q, v):
    """ベクトル v を q で回転する: q * (0, v) * q^-1"""
    return quat_mul(quat_mul(q, np.concatenate([[0.0], v])), quat_conj(q))[1:]


def quat_to_mat(q):
    """クォータニオン → 回転行列"""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def slerp(q0, q1, t):
    """球面線形補間：2つの姿勢の間を「一定の角速度」で補間する"""
    q0, q1 = q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1)
    dot = np.dot(q0, q1)
    if dot < 0:              # q と -q は同じ回転。近い方の経路を選ぶ
        q1, dot = -q1, -dot
    if dot > 0.9995:         # ほぼ同じ姿勢なら線形補間で十分
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    omega = np.arccos(dot)
    return (np.sin((1 - t) * omega) * q0 + np.sin(t * omega) * q1) / np.sin(omega)
