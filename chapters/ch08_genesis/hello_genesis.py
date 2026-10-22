"""第8章 (1): Genesis の最小プログラム ― 初期化 → シーン → エンティティ → build → step"""
import time

import genesis as gs

gs.init(backend=gs.cpu, logging_level="warning")   # GPU があれば gs.gpu

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01),
    show_viewer=False,                              # 画面に表示する場合は True
)
plane = scene.add_entity(gs.morphs.Plane())
box = scene.add_entity(gs.morphs.Box(pos=(0.0, -0.6, 0.3), size=(0.04, 0.04, 0.04)))
franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))

t0 = time.perf_counter()
scene.build()                                       # ここでカーネルがコンパイルされる
print(f"build（コンパイル）: {time.perf_counter() - t0:.1f} 秒")

t0 = time.perf_counter()
for i in range(100):
    scene.step()
print(f"100 ステップ: {time.perf_counter() - t0:.2f} 秒")
print("箱の位置（落下後）:", box.get_pos())
print("Franka の自由度:", franka.n_dofs, "/ 関節角（重力で垂れ下がる）:", franka.get_dofs_position()[:7])
