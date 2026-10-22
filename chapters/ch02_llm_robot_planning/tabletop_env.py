"""第2章: ブロックを掴んで置ける卓上環境（MuJoCo 2関節アーム）

把持は物理的なグリッパーではなく「手先に吸着させる」簡易モデルで表現する。
LLM（生成コード）から呼べるのは ROBOT_API に登録した関数だけ。
"""
import mujoco
import numpy as np

MJCF = """
<mujoco model="tabletop">
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
    <site name="red_block"   type="box" size="0.02 0.02 0.02" pos="0.40  0.10 0.05" rgba="1 0 0 1"/>
    <site name="blue_block"  type="box" size="0.02 0.02 0.02" pos="0.30 -0.25 0.05" rgba="0 0 1 1"/>
    <site name="green_block" type="box" size="0.02 0.02 0.02" pos="0.15  0.40 0.05" rgba="0 1 0 1"/>
    <site name="yellow_block" type="box" size="0.02 0.02 0.02" pos="0.62 -0.05 0.05" rgba="1 1 0 1"/>
    <site name="tray" type="cylinder" size="0.06 0.002" pos="-0.25 0.30 0.05" rgba="0.6 0.6 0.6 0.5"/>
  </worldbody>
  <actuator>
    <motor joint="shoulder" gear="1" ctrlrange="-5 5"/>
    <motor joint="elbow"    gear="1" ctrlrange="-5 5"/>
  </actuator>
</mujoco>
"""

BLOCKS = ["red_block", "blue_block", "green_block", "yellow_block"]
PLACES = ["tray"]
R_MAX, R_MIN = 0.55, 0.05


class SkillError(Exception):
    """スキルの実行に失敗したときの例外（エラーメッセージは LLM にも返す）"""


class TabletopEnv:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(MJCF)
        self.data = mujoco.MjData(self.model)
        self.hand_id = self._sid("hand")
        self.held = None                      # 現在掴んでいるブロック名
        self.log = []                         # 実行したスキルの記録
        self.data.qpos[:] = [0.0, 0.8]        # 特異姿勢を避ける（第0章）
        mujoco.mj_forward(self.model, self.data)

    def _sid(self, name):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise SkillError(f"unknown object: {name}")
        return sid

    def _hand(self):
        return self.data.site_xpos[self.hand_id][:2].copy()

    def _move(self, target, timeout=3.0):
        """第0・1章と同じヤコビ転置法で手先を動かす（掴んでいる物体は手先に追従）"""
        m, d = self.model, self.data
        jacp = np.zeros((3, m.nv))
        t0 = d.time
        while d.time - t0 < timeout:
            err = target - self._hand()
            mujoco.mj_jacSite(m, d, jacp, None, self.hand_id)
            d.ctrl[:] = np.clip(300.0 * jacp[:2].T @ err - 1.0 * d.qvel, -5, 5)
            mujoco.mj_step(m, d)
            if self.held:                                   # 吸着モデル
                self.model.site_pos[self._sid(self.held)][:2] = self._hand()
            if np.linalg.norm(err) < 0.002 and np.linalg.norm(d.qvel) < 0.05:
                break
        mujoco.mj_forward(m, d)
        return float(np.linalg.norm(target - self._hand()))

    # ================= LLM に公開するロボット API =================
    def get_object_names(self):
        """シーン内の物体名のリストを返す"""
        return BLOCKS + PLACES

    def get_obj_pos(self, name):
        """物体の位置 [x, y]（メートル）を返す"""
        return [round(float(v), 3) for v in self.model.site_pos[self._sid(name)][:2]]

    def is_reachable(self, x, y):
        """手先が (x, y) に届くかを返す"""
        r = float(np.hypot(x, y))
        return R_MIN + 0.02 <= r <= R_MAX - 0.02

    def holding(self):
        """現在掴んでいる物体名（何も持っていなければ None）"""
        return self.held

    def pick(self, name):
        """物体を掴む"""
        if name not in BLOCKS:
            raise SkillError(f"{name} は掴める物体ではありません")
        if self.held:
            raise SkillError(f"すでに {self.held} を持っています")
        x, y = self.get_obj_pos(name)
        if not self.is_reachable(x, y):
            raise SkillError(f"{name} ({x}, {y}) は届く範囲外です")
        err = self._move(np.array([x, y]))
        if err > 0.005:
            raise SkillError(f"{name} に到達できませんでした（誤差 {err*1000:.1f} mm）")
        self.held = name
        self.log.append(f"pick({name})")

    def place(self, x, y):
        """掴んでいる物体を (x, y) に置く"""
        if not self.held:
            raise SkillError("何も持っていません")
        if not self.is_reachable(x, y):
            raise SkillError(f"({x:.3f}, {y:.3f}) は届く範囲外です")
        err = self._move(np.array([x, y]))
        if err > 0.005:
            raise SkillError(f"({x:.3f}, {y:.3f}) に到達できませんでした")
        self.log.append(f"place({self.held} -> {x:.3f}, {y:.3f})")
        self.held = None

    def api(self):
        """生成コードから呼び出してよい関数の一覧（ホワイトリスト）"""
        return {f.__name__: f for f in [self.get_object_names, self.get_obj_pos,
                                        self.is_reachable, self.holding,
                                        self.pick, self.place]}
