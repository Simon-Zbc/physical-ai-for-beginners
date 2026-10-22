"""第0章 7.6節: ビューアで Sense → Think → Act の動きを見る

ローカルPC専用（画面表示のない環境では実行できません）。
macOS では `python` の代わりに `mjpython` コマンドで実行してください。
"""
import time

import mujoco.viewer

from sense_think_act import act, data, model, sense, think

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and data.time < 5.0:
        obs = sense(model, data)
        act(model, data, think(model, data, obs))
        viewer.sync()
        time.sleep(model.opt.timestep)   # 実時間に合わせる
