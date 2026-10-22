"""第8章 (3): 環境数を変えて、Genesis のスループット（環境ステップ/秒）を測る

  python benchmark.py 64          # 64 環境
"""
import sys
import time

import torch

import genesis as gs

n_envs = int(sys.argv[1]) if len(sys.argv) > 1 else 16
backend = gs.gpu if torch.cuda.is_available() else gs.cpu
gs.init(backend=backend, logging_level="warning", seed=0)

scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01), show_viewer=False)
scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
scene.build(n_envs=n_envs)

home = torch.tensor([0, -0.3, 0, -2.2, 0, 2.0, 0.79, 0.04, 0.04])
franka.set_dofs_kp(torch.tensor([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
franka.set_dofs_kv(torch.tensor([450, 450, 350, 350, 200, 200, 200, 10, 10]))
franka.set_qpos(home.repeat(n_envs, 1))


def run(steps):
    for k in range(steps):
        # 全環境に、少しずつ異なる関節目標を与える（N, 9）
        target = home + 0.3 * torch.sin(torch.arange(n_envs)[:, None] * 0.1 + k * 0.05)
        franka.control_dofs_position(target)
        scene.step()


run(20)                                    # ウォームアップ（初回のカーネル実行を除く）
t0 = time.perf_counter()
STEPS = 200
run(STEPS)
sec = time.perf_counter() - t0
print(f"backend={backend.name if hasattr(backend, 'name') else backend}  n_envs={n_envs:5d}  "
      f"{STEPS * n_envs / sec:10,.0f} env-step/s  （実時間の {STEPS * n_envs * 0.01 / sec:,.1f} 倍）")
