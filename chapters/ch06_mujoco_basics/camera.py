"""第6章 (3): カメラ画像（RGB・深度・セグメンテーション）を取得し、深度から箱の3次元位置を求める

画面のない環境では、import より前に環境変数 MUJOCO_GL を設定する:
  Linux（NVIDIA GPU）: MUJOCO_GL=egl   / CPU のみ: MUJOCO_GL=osmesa
  Windows / macOS（画面あり）: 設定不要
OpenGL が使えない環境では、レイキャスト（mj_ray）で深度とセグメンテーションを計算する代替経路で動く。
"""
import mujoco
import numpy as np

from scene import make

W, H = 320, 240


def intrinsics(model, cam_id, width, height):
    """MuJoCo の fovy（縦の画角）から、ピンホールカメラの内部パラメータを求める"""
    fovy = np.deg2rad(model.cam_fovy[cam_id])
    f = 0.5 * height / np.tan(fovy / 2)          # 焦点距離 [pixel]
    return f, f, (width - 1) / 2, (height - 1) / 2   # fx, fy, cx, cy


def render_gl(model, data, cam):
    """OpenGL で RGB・深度・セグメンテーションを描画する"""
    with mujoco.Renderer(model, H, W) as r:
        r.update_scene(data, camera=cam)
        rgb = r.render()
        r.enable_depth_rendering(); r.update_scene(data, camera=cam)
        depth = r.render()                          # カメラの光軸方向の距離 [m]
        r.disable_depth_rendering()
        r.enable_segmentation_rendering(); r.update_scene(data, camera=cam)
        seg = r.render()[..., 0]                    # 各画素に写っている geom の ID
    return rgb, depth, seg


def render_raycast(model, data, cam):
    """OpenGL なしの代替：画素ごとにレイを飛ばして深度と geom ID を求める（遅いが確実）"""
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    fx, fy, cx, cy = intrinsics(model, cid, W, H)
    pos, R = data.cam_xpos[cid], data.cam_xmat[cid].reshape(3, 3)
    depth = np.zeros((H, W)); seg = -np.ones((H, W), dtype=int)
    gid = np.zeros(1, dtype=np.int32)
    for v in range(H):
        for u in range(W):
            # MuJoCo のカメラ座標系：x = 右、y = 上、-z = 視線方向（OpenGL の慣習）
            d_cam = np.array([(u - cx) / fx, -(v - cy) / fy, -1.0])
            d_world = R @ (d_cam / np.linalg.norm(d_cam))
            dist = mujoco.mj_ray(model, data, pos, d_world, None, 1, -1, gid)
            if dist >= 0:
                depth[v, u] = dist * (-d_cam[2] / np.linalg.norm(d_cam))   # 光軸方向の距離に直す
                seg[v, u] = gid[0]
    return None, depth, seg


def depth_to_points(model, data, cam, depth, mask):
    """深度画像の画素 → カメラ座標 → ワールド座標の3次元点（第3章の同次変換と同じ考え方）"""
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    fx, fy, cx, cy = intrinsics(model, cid, W, H)
    v, u = np.nonzero(mask)
    z = depth[v, u]
    p_cam = np.stack([(u - cx) / fx * z, -(v - cy) / fy * z, -z], axis=1)   # OpenGL 慣習
    R, t = data.cam_xmat[cid].reshape(3, 3), data.cam_xpos[cid]
    return p_cam @ R.T + t                                                   # p_world = R p_cam + t


if __name__ == "__main__":
    model, data = make()
    box_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box")
    cam = "top"
    try:
        rgb, depth, seg = render_gl(model, data, cam)
        mode = "OpenGL"
    except Exception as e:                           # OpenGL が使えない環境
        print(f"OpenGL が使えないため、レイキャストで代替します（{type(e).__name__}）")
        rgb, depth, seg = render_raycast(model, data, cam)
        mode = "レイキャスト"

    mask = seg == box_geom
    pts = depth_to_points(model, data, cam, depth, mask)
    print(f"[{mode}] 画像 {W}x{H} / 深度の範囲 {depth[depth > 0].min():.3f}〜{depth.max():.3f} m "
          f"/ 箱が写っている画素数 {mask.sum()}")
    print("深度から推定した箱の上面の中心:", np.median(pts, axis=0).round(4), "（外れ値に強い中央値）")
    print("シミュレータの真値（箱の中心）  :", data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'box')].round(4),
          "→ 上面は z = 0.04")

    # 画像を保存（Pillow があれば）
    try:
        from PIL import Image
        d8 = (255 * (1 - (depth - depth.min()) / (np.ptp(depth) + 1e-9))).astype(np.uint8)
        Image.fromarray(d8).save("depth_top.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save("mask_top.png")
        if rgb is not None:
            Image.fromarray(rgb).save("rgb_top.png")
        print("depth_top.png / mask_top.png" + (" / rgb_top.png" if rgb is not None else "") + " を保存しました")
    except ImportError:
        pass
