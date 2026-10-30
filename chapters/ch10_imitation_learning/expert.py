"""第10章: 脚本示範者（scripted expert）

起点 → 経由点 → 目標 を滑らかな曲線でつなぎ、各ステップの目標関節角 q_des を作る。
経由点は障害物の左 (side = -1) か右 (side = +1) に置く。どちらを通っても正解なので、
示範データには「同じ場所から左へ行く例」と「右へ行く例」が混ざる（第10章の主題）。
"""
import numpy as np

from env import EPISODE_STEPS, OBS_CENTER, ik

DETOUR = 0.12       # 経由点 = 障害物の中心から横に 12cm（障害物の半径 5cm + 手先 2cm に余裕をもたせた値）
DEMO_STEPS = 100    # 動く部分のステップ数（2秒）。その後は目標で止まって待つ


def _min_jerk(s):
    """min-jerk の時間配分 s∈[0,1] -> [0,1]。両端で速度・加速度が 0"""
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def plan(start_xy, target_xy, side, pause_steps=0):
    """起点 -> 経由点 -> 目標 の目標関節角列 (EPISODE_STEPS, 2) を作る

    経路は経由点をちょうど通る2次ベジェ曲線（経由点で速度が連続）で、時間配分は min-jerk。
    pause_steps > 0 のときは、経由点でいったん止まって pause_steps ステップ待ち、そのあと動き出す。
    """
    via = OBS_CENTER + np.array([side * DETOUR, 0.0])
    ctrl = 2 * via - (start_xy + target_xy) / 2          # u = 0.5 で経由点を通る制御点
    t = np.arange(EPISODE_STEPS)
    if pause_steps == 0:
        u = _min_jerk(np.clip(t / (DEMO_STEPS - 1), 0, 1))
    else:                                                # 前半（起点→経由点）、停止、後半（経由点→目標）
        half = DEMO_STEPS // 2
        first = 0.5 * _min_jerk(np.clip(t / (half - 1), 0, 1))
        second = 0.5 + 0.5 * _min_jerk(np.clip((t - half - pause_steps) / (half - 1), 0, 1))
        u = np.where(t < half + pause_steps, first, second)
    u = u[:, None]
    xy = (1 - u)**2 * start_xy + 2 * u * (1 - u) * ctrl + u**2 * target_xy
    return ik(xy)
