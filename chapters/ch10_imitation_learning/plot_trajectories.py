"""第10章: 手先の軌跡を俯瞰図の PNG に描く（Pillow だけを使う）

使い方:
  python plot_trajectories.py --demos data/demos_mix.npz          # 示範 50 本 -> plots/demos_mix.png
  python plot_trajectories.py models/bc_demos_mix_s0.pt models/diffusion_demos_mix_s0.pt
                                                                  # 方策を 50 環境で1エピソードずつ動かす
  python plot_trajectories.py models/*_demos_mix_s0.pt --same-start
                                                                  # 50 本すべて同じ起点・目標から（実験 E1b）
  （Windows の cmd / PowerShell は * を展開しないので、ファイル名を並べて書く）

1つの入力につき1枚（plots/<名前>.png）。入力が2つ以上なら、横に並べた plots/<--out>.png も作る。
座標はアームの根元が原点、上が +y（目標側）、右が +x。縮尺はどの図も同じ。
Diffusion のサンプルの乱数は、ファイル名の _s<数字>（experiments.py が付ける学習 seed）を使う。なければ --seed。
"""
import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from env import (EPISODE_STEPS, HAND_RADIUS, L1, OBS_CENTER, OBS_RADIUS, SPAWN_HALF, START_CENTER,
                 TARGET_CENTER, VecObstacleEnv, ik)

SCALE = 700                                  # 1m = 700 ピクセル
X_RANGE, Y_RANGE = (-0.05, 0.60), (-0.32, 0.32)
TOP = 64                                     # 上の見出し用の余白 [px]
COLORS = {-1: (31, 119, 180), 1: (255, 127, 14), 0: (130, 130, 130)}   # 左 = 青、右 = 橙、未通過 = 灰
try:
    FONT = ImageFont.load_default(size=15)  # Pillow 10.1 以降は大きさを指定できる
except TypeError:
    FONT = ImageFont.load_default()


def px(xy):
    """机の座標 [m] -> 画像のピクセル座標（y は上向きが正なので反転する）"""
    return ((xy[0] - X_RANGE[0]) * SCALE, TOP + (Y_RANGE[1] - xy[1]) * SCALE)


def rollout(path, n, spawn_half, seed):
    """方策を n 環境で1エピソードずつ動かし、(軌跡のリスト, 通った側, 結果) を返す"""
    from policies import load_runner                    # torch は方策を描くときだけ必要（遅延 import）
    runner = load_runner(path, seed)
    env = VecObstacleEnv(n, seed=1, spawn_half=spawn_half)    # evaluate() と同じ seed=1 の起点・目標
    obs = env.reset()
    hands = [obs[:, 6:8] - obs[:, 8:10]]                # 観測の「目標 − (目標 − 手先)」= 手先
    side = np.zeros(n, dtype=int)                       # 通った側
    result = np.full(n, "", dtype=object)               # "success" / "collision" / "timeout"
    length = np.zeros(n, dtype=int)                     # 軌跡の点の数
    alive = np.ones(n, dtype=bool)
    for t in range(EPISODE_STEPS):
        obs, info = env.step(runner(obs))
        final = info["final_obs"]                       # 自動リセット前の観測（終わった瞬間の手先が取れる）
        hands.append(final[:, 6:8] - final[:, 8:10])
        end = alive & info["done"]
        side[end] = info["final_side"][end]
        length[end] = t + 2
        for key in ["success", "collision", "timeout"]:
            result[end & info[key]] = key
        alive &= ~info["done"]
        if not alive.any():
            break
    hands = np.stack(hands, axis=1)
    return [hands[i, :length[i]] for i in range(n)], side, result


def demo_paths(path, n):
    """示範データの最初の n エピソードの手先の軌跡"""
    d = np.load(path)
    hand = d["obs"][:, 6:8] - d["obs"][:, 8:10]
    eps = np.arange(n)
    result = np.where(d["success"][eps], "success", np.where(d["collision"][eps], "collision", "timeout"))
    return [hand[d["episode_index"] == e] for e in eps], d["side"][eps], result


def draw_scene(img, spawn_half):
    """アーム（起点の姿勢）・障害物・起点と目標の範囲を描く"""
    d = ImageDraw.Draw(img)
    q1 = ik(START_CENTER[None])[0, 0]
    elbow = L1 * np.array([np.cos(q1), np.sin(q1)])
    d.line([px((0, 0)), px(elbow), px(START_CENTER)], fill=(215, 215, 215), width=7)   # 起点の姿勢のアーム
    d.ellipse([px((-0.02, 0.02)), px((0.02, -0.02))], fill=(90, 90, 90))               # 根元（原点）
    o, r, rr = OBS_CENTER, OBS_RADIUS, OBS_RADIUS + HAND_RADIUS
    d.ellipse([px(o + [-rr, rr]), px(o + [rr, -rr])], outline=(230, 150, 150), width=1)   # 手先の中心がこの内側に入ると衝突
    d.ellipse([px(o + [-r, r]), px(o + [r, -r])], fill=(200, 60, 60))                    # 障害物
    h = spawn_half
    for c in (START_CENTER, TARGET_CENTER):            # 起点・目標がランダムに置かれる範囲と、その中心
        d.rectangle([px(c + [-h, h]), px(c + [h, -h])], outline=(40, 160, 70), width=2)
        d.ellipse([px(c + [-0.006, 0.006]), px(c + [0.006, -0.006])], fill=(40, 160, 70))


def draw_paths(img, paths, side, result):
    """軌跡を半透明で重ねて描き（たくさん通った所ほど濃い）、衝突した所に × 印を付ける"""
    for p, s in zip(paths, side):
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(layer).line([px(q) for q in p], fill=COLORS[int(s)] + (110,), width=2)
        img.alpha_composite(layer)
    d = ImageDraw.Draw(img)
    for p, res in zip(paths, result):
        if res == "collision":
            x, y = px(p[-1])
            d.line([(x - 5, y - 5), (x + 5, y + 5)], fill=(0, 0, 0), width=2)
            d.line([(x - 5, y + 5), (x + 5, y - 5)], fill=(0, 0, 0), width=2)


def draw_labels(img, title, side, result):
    """上に見出しと集計、左上に凡例を書く"""
    d = ImageDraw.Draw(img)
    black = (0, 0, 0)
    left, right, none = [int((side == v).sum()) for v in (-1, 1, 0)]
    success, collision, timeout = [int((result == v).sum()) for v in ("success", "collision", "timeout")]
    d.text((8, 6), f"{title}  ({len(side)} episodes)", fill=black, font=FONT)
    d.text((8, 26), f"left {left} / right {right} / not passed {none}", fill=black, font=FONT)
    d.text((8, 44), f"success {success} / collision {collision} / timeout {timeout}", fill=black, font=FONT)
    for i, (label, color) in enumerate([("left", COLORS[-1]), ("right", COLORS[1]), ("not passed", COLORS[0])]):
        y = TOP + 10 + 18 * i                           # 凡例は軌跡の通らない左上に
        d.line([(10, y + 8), (34, y + 8)], fill=color, width=3)
        d.text((40, y), label, fill=black, font=FONT)
    d.text((10, TOP + 64), "x = collision", fill=black, font=FONT)
    d.text((10, TOP + 82), "green = start / target", fill=black, font=FONT)


def draw_panel(title, paths, side, result, spawn_half):
    """1枚ぶんの俯瞰図を作る。軌跡の色は通った側、× は衝突した場所"""
    w = int((X_RANGE[1] - X_RANGE[0]) * SCALE)
    h = TOP + int((Y_RANGE[1] - Y_RANGE[0]) * SCALE)
    img = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    draw_scene(img, spawn_half)
    draw_paths(img, paths, side, result)
    draw_labels(img, title, np.asarray(side), np.asarray(result))
    return img.convert("RGB")


def sample_seed(model_path, default):
    """ファイル名が ..._s<数字>.pt なら、その数字（学習 seed）を Diffusion のサンプルの乱数に使う"""
    m = re.search(r"_s(\d+)$", Path(model_path).stem)
    return int(m.group(1)) if m else default


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="*")
    p.add_argument("--demos", default=None)
    p.add_argument("--same-start", action="store_true")    # 全エピソードを同じ起点・目標から（spawn_half=0）
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)          # ファイル名に _s<数字> がないときの、サンプルの乱数
    p.add_argument("--out", default="compare")
    args = p.parse_args()
    if args.models:                                        # 方策を動かすときだけ torch を使う
        import torch
        torch.set_num_threads(1)                           # 学習・評価と同じ1スレッドにそろえる
    spawn_half = 0.0 if args.same_start else SPAWN_HALF
    suffix = "_same" if args.same_start else ""

    panels = []                                            # [(名前, 画像), ...]
    if args.demos:
        name = Path(args.demos).stem
        panels.append((name, draw_panel(name, *demo_paths(args.demos, args.n), SPAWN_HALF)))
    for m in args.models:
        name = Path(m).stem + suffix
        trajectories = rollout(m, args.n, spawn_half, sample_seed(m, args.seed))
        panels.append((name, draw_panel(name, *trajectories, spawn_half)))
    Path("plots").mkdir(exist_ok=True)
    for name, img in panels:
        img.save(f"plots/{name}.png")
        print(f"plots/{name}.png")
    if len(panels) > 1:                                    # 横に並べた1枚
        w, h = panels[0][1].size
        row = Image.new("RGB", (w * len(panels), h), (255, 255, 255))
        for i, (_, img) in enumerate(panels):
            row.paste(img, (w * i, 0))
            ImageDraw.Draw(row).line([(w * i, 0), (w * i, h)], fill=(180, 180, 180), width=2)   # 区切り線
        row.save(f"plots/{args.out}{suffix}.png")
        print(f"plots/{args.out}{suffix}.png")
