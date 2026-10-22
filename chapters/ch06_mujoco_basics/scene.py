"""第6章: XYZ ステージ + 2本指グリッパー + 箱 の把持シーン（MJCF を文字列で組み立てる）"""
import mujoco

TEMPLATE = """
<mujoco model="gripper_scene">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" cone="elliptic" integrator="implicitfast"/>

  <visual><global offwidth="640" offheight="480"/></visual>

  <!-- CSS のように既定値をまとめて指定する -->
  <default>
    <geom condim="4" friction="1 0.005 0.0001"/>
    <default class="stage">
      <joint type="slide" damping="20" armature="0.1"/>
      <position kp="2000" dampratio="1"/>
    </default>
    <default class="finger">
      <joint type="slide" damping="5" armature="0.01" range="0 0.04"/>
      <!-- 指の把持力 = min(kp × 位置の誤差, forcerange)。kp を大きくして forcerange で力を決める -->
      <position kp="3000" dampratio="1" ctrlrange="0 0.04" forcerange="-@GRIP@ @GRIP@"/>
      <geom type="box" size="0.008 0.006 0.03" rgba="0.3 0.3 0.3 1" friction="@MU@ 0.005 0.0001"/>
    </default>
  </default>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.85 0.85 0.85" rgb2="0.7 0.7 0.7" width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="8 8"/>
  </asset>

  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="1 1 0.1" material="grid"/>

    <!-- 固定カメラ：真上と斜め前 -->
    <camera name="top"   pos="0 0 1.0" xyaxes="1 0 0 0 1 0" fovy="60"/>
    <camera name="front" pos="0.6 -0.6 0.5" xyaxes="0.707 0.707 0 -0.3 0.3 0.905" fovy="50"/>

    <!-- XYZ ステージ（ガントリー）：重力補償（gravcomp）付き -->
    <body name="stage_x" pos="0 0 0.3" gravcomp="1">
      <joint name="x" class="stage" axis="1 0 0" range="-0.3 0.3"/>
      <geom type="box" size="0.02 0.02 0.02" mass="1" rgba="0.6 0.6 0.6 0.3" contype="0" conaffinity="0"/>
      <body name="stage_y" gravcomp="1">
        <joint name="y" class="stage" axis="0 1 0" range="-0.3 0.3"/>
        <geom type="box" size="0.02 0.02 0.02" mass="1" rgba="0.6 0.6 0.6 0.3" contype="0" conaffinity="0"/>
        <body name="hand" gravcomp="1">
          <joint name="z" class="stage" axis="0 0 1" range="-0.28 0.2"/>
          <geom name="palm" type="box" size="0.02 0.06 0.01" mass="0.5" rgba="0.2 0.4 0.8 1"/>
          <site name="tcp" pos="0 0 -0.07" size="0.005"/>
          <!-- 2本の指：y 方向に開閉。qpos = 指の内側の面と中心線の距離（0 で閉じきる） -->
          <body name="left_finger" pos="0 0.008 -0.04" gravcomp="1">
            <joint name="left" class="finger" axis="0 1 0"/>
            <geom name="left_pad" class="finger" mass="0.05"/>
          </body>
          <body name="right_finger" pos="0 -0.008 -0.04" gravcomp="1">
            <joint name="right" class="finger" axis="0 -1 0"/>
            <geom name="right_pad" class="finger" mass="0.05"/>
          </body>
        </body>
      </body>
    </body>

    <!-- 掴む対象：自由に動く箱（freejoint） -->
    <body name="box" pos="0.1 0.05 0.02">
      <freejoint name="box"/>
      <geom name="box" type="box" size="0.02 0.02 0.02" mass="@MASS@" rgba="0.9 0.2 0.2 1"
            friction="@MU@ 0.005 0.0001"/>
    </body>
  </worldbody>

  <actuator>
    <position name="x" joint="x" class="stage" ctrlrange="-0.3 0.3"/>
    <position name="y" joint="y" class="stage" ctrlrange="-0.3 0.3"/>
    <position name="z" joint="z" class="stage" ctrlrange="-0.28 0.2"/>
    <position name="left"  joint="left"  class="finger"/>
    <position name="right" joint="right" class="finger"/>
  </actuator>
</mujoco>
"""


def make(mu=1.0, grip_force=5.0, box_mass=0.1):
    """摩擦係数・把持力の上限・箱の質量を変えたシーンを作る"""
    xml = (TEMPLATE.replace("@MU@", str(mu))
                   .replace("@GRIP@", str(grip_force))
                   .replace("@MASS@", str(box_mass)))
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[3:5] = 0.04                       # 指は開いた状態から始める
    data.ctrl[3:5] = 0.04
    mujoco.mj_forward(model, data)
    return model, data


if __name__ == "__main__":
    model, data = make()
    print(f"body 数 {model.nbody} / joint 数 {model.njnt} / 自由度 nq={model.nq}, nv={model.nv} "
          f"/ actuator 数 {model.nu} / camera 数 {model.ncam}")
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        jtype = mujoco.mjtJoint(model.jnt_type[j]).name
        print(f"  joint {j}: {name:6s} {jtype:12s} qpos の位置 {model.jnt_qposadr[j]}")
    print("箱の初期位置 qpos[5:8] =", data.qpos[5:8], " 姿勢（wxyz）qpos[8:12] =", data.qpos[8:12])
