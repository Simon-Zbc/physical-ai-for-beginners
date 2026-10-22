"""第1章: LLM から呼び出す「ロボットのスキル」層（MuJoCo 2関節アーム）"""
import json
import mujoco
import numpy as np

MJCF = """
<mujoco model="tabletop_arm">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="1 1 0.1" rgba="0.9 0.9 0.9 1"/>
    <body name="link1" pos="0 0 0.05">
      <joint name="shoulder" type="hinge" axis="0 0 1" damping="0.5"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" rgba="0.2 0.4 0.8 1"/>
      <body name="link2" pos="0.3 0 0">
        <joint name="elbow" type="hinge" axis="0 0 1" damping="0.5"/>
        <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.02" rgba="0.8 0.4 0.2 1"/>
        <site name="hand" pos="0.25 0 0" size="0.02"/>
      </body>
    </body>
    <!-- 机の上の物体（見た目だけの目印。衝突判定なし） -->
    <site name="red_cup"    pos="0.20  0.35 0.05" size="0.03" rgba="1 0 0 1"/>
    <site name="blue_box"   pos="0.35 -0.20 0.05" size="0.03" type="box" rgba="0 0 1 1"/>
    <site name="green_ball" pos="0.60  0.30 0.05" size="0.03" rgba="0 1 0 1"/>
  </worldbody>
  <actuator>
    <motor joint="shoulder" gear="1" ctrlrange="-5 5"/>
    <motor joint="elbow"    gear="1" ctrlrange="-5 5"/>
  </actuator>
</mujoco>
"""

OBJECTS = ["red_cup", "blue_box", "green_ball"]
LINK1, LINK2 = 0.30, 0.25
R_MAX, R_MIN = LINK1 + LINK2, abs(LINK1 - LINK2)   # 届く範囲: 0.05 m 〜 0.55 m


class ArmEnv:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(MJCF)
        self.data = mujoco.MjData(self.model)
        self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "hand")
        self.data.qpos[:] = [0.0, 0.8]          # 特異姿勢を避けた初期姿勢（第0章参照）
        mujoco.mj_forward(self.model, self.data)

    # ---------- 内部ユーティリティ ----------
    def _site_xy(self, name):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        return self.data.site_xpos[sid][:2].copy()

    def _hand_xy(self):
        return self.data.site_xpos[self.hand_id][:2].copy()

    @staticmethod
    def _affordance(x, y):
        """実行可能性スコア（0〜1）。SayCan の価値関数を、幾何計算で簡易的に置き換えたもの"""
        r = float(np.hypot(x, y))
        outer = np.clip((R_MAX - r) / 0.05, 0.0, 1.0)   # 外側の限界に近いほど低い
        inner = np.clip((r - R_MIN) / 0.05, 0.0, 1.0)   # 根元に近すぎても低い
        return round(float(min(outer, inner)), 2), r

    # ---------- LLM に公開するスキル（ツール） ----------
    def get_scene(self):
        """シーン内の物体と手先の位置を返す"""
        return {
            "hand": [round(float(v), 3) for v in self._hand_xy()],
            "objects": {n: [round(float(v), 3) for v in self._site_xy(n)] for n in OBJECTS},
            "unit": "meter",
        }

    def check_affordance(self, x, y):
        """指定座標に手先を動かせるかを評価する"""
        score, r = self._affordance(x, y)
        return {"x": float(x), "y": float(y), "distance_from_base": round(r, 3), "affordance": score,
                "feasible": score >= 0.5}

    def move_hand_to(self, x, y, timeout=3.0):
        """手先を (x, y) へ動かす。内部で 500Hz の制御ループを回す"""
        # 安全層：LLM の出力をそのままモーターに送らず、必ず実行可能性を検証する
        check = self.check_affordance(x, y)
        if not check["feasible"]:
            return {"success": False,
                    "reason": f"到達不可能（基部からの距離 {check['distance_from_base']} m、"
                              f"届く範囲は {R_MIN:.2f}〜{R_MAX:.2f} m）"}

        m, d = self.model, self.data
        target = np.array([x, y])
        jacp = np.zeros((3, m.nv))
        t0, steps = d.time, 0
        while d.time - t0 < timeout:
            err = target - self._hand_xy()
            mujoco.mj_jacSite(m, d, jacp, None, self.hand_id)
            tau = 300.0 * jacp[:2].T @ err - 1.0 * d.qvel      # 第0章のヤコビ転置法
            d.ctrl[:] = np.clip(tau, -5, 5)
            mujoco.mj_step(m, d)
            steps += 1
            if np.linalg.norm(err) < 0.002 and np.linalg.norm(d.qvel) < 0.05:
                break
        final_err = float(np.linalg.norm(target - self._hand_xy()))
        return {"success": final_err < 0.005,
                "final_error_mm": round(final_err * 1000, 1),
                "sim_time_s": round(d.time - t0, 3),
                "control_steps": steps}

    # ---------- ツール呼び出しのディスパッチ ----------
    def call(self, name, args):
        fn = {"get_scene": self.get_scene,
              "check_affordance": self.check_affordance,
              "move_hand_to": self.move_hand_to}.get(name)
        if fn is None:
            return {"error": f"unknown tool: {name}"}
        try:
            return fn(**args)
        except TypeError as e:              # 引数の間違いも LLM に返して自己修正させる
            return {"error": str(e)}


if __name__ == "__main__":
    # LLM を使わずにスキル層だけを確認する（固定の計画を実行）
    env = ArmEnv()
    print("scene:", json.dumps(env.get_scene(), ensure_ascii=False))
    for name in OBJECTS:
        x, y = env.get_scene()["objects"][name]
        print(f"\n[{name}] check_affordance ->", env.check_affordance(x, y))
        print(f"[{name}] move_hand_to      ->", json.dumps(env.move_hand_to(x, y), ensure_ascii=False))
