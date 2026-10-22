---
title: 【フィジカルAI入門 #06】MuJoCo 入門 ― 接触・摩擦・グリッパー・カメラで「掴める」シミュレーション環境を作る
tags: PhysicalAI ロボティクス MuJoCo シミュレーション 物理エンジン
---

# はじめに

Part 3「シミュレーション」に入ります。

ここまでの章でも MuJoCo を使ってきましたが、扱ってきたのは **何にも触れないアーム** だけでした。第2章の「掴む」も、手先に物体を貼り付けるだけの簡易モデルでした。しかし、フィジカルAIの本題であるロボットの操作（マニピュレーション）は、**物体に触れて、押して、掴む** ことの連続です。

本章では、MuJoCo を本格的に使い、次の4つを身につけます。

- MuJoCo の基本構造（**MjModel と MjData**）と、**MJCF** の書き方
- **接触と摩擦** のモデル化と、接触力の読み出し方
- **2本指グリッパー** で箱を掴んで持ち上げ、成否が物理法則どおりに決まることを確かめる
- **カメラ画像（RGB・深度・セグメンテーション）** を取得し、深度から物体の3次元位置を求める

この章で作る「掴める環境」と「カメラ画像」は、Part 4 の強化学習・模倣学習でそのまま土台になります。

:::note info
**前提**
- 必要なライブラリは `mujoco`、`numpy`、（画像を保存する場合）`pillow` です
- 本章のコードは、ディレクトリ内で完結しています
:::

---

# 1. なぜ MuJoCo なのか

MuJoCo（Multi-Joint dynamics with Contact）は、Google DeepMind が開発・保守しているオープンソースの物理エンジンです。名前のとおり、**多関節の物体と接触** を高速かつ安定に計算することに重点を置いています。

本連載で MuJoCo をメインに使う理由は次のとおりです。

| 理由 | 内容 |
|---|---|
| 軽い | CPU だけで動き、`pip install mujoco` だけで使える |
| 研究で広く使われている | 強化学習のベンチマークや模倣学習の研究で、標準的な環境として採用されることが多い |
| ロボットモデルが豊富 | **MuJoCo Menagerie** に、品質が確認されたロボットモデルが多数公開されている[^menagerie] |

Menagerie には、Franka Emika Panda のような研究用アーム、Unitree G1 のような人型ロボット、ALOHA の双腕システム、第20章で扱う予定の **SO-101** などのモデルが含まれています[^menagerie-panda]。物理シミュレーションの結果はモデルの質に大きく左右されるため、「動作を確認済みのモデルから始められる」ことは大きな利点です[^menagerie-doc]。

---

# 2. MuJoCo の基本構造：MjModel と MjData

MuJoCo のプログラムは、2つのオブジェクトを中心に動きます。

| オブジェクト | 中身 | 変わるか | ソフトウェアでたとえると |
|---|---|---|---|
| **MjModel** | ロボットや環境の構造（リンクの長さ、質量、関節の種類、摩擦係数など） | シミュレーション中は基本的に変わらない | クラス定義・設定ファイル |
| **MjData** | 現在の状態（関節角 `qpos`、速度 `qvel`、接触点、センサー値など） | 毎ステップ更新される | インスタンス・実行時の状態 |

1つの MjModel から複数の MjData を作れば、**同じロボットを、異なる状態で並列に動かす** ことができます。強化学習で多数の環境を並べるときの基本的な考え方です。

よく使う関数は次の3つです。

| 関数 | やること |
|---|---|
| `mj_step(model, data)` | 物理を1ステップ進める（力の計算 → 加速度 → 速度・位置の更新） |
| `mj_forward(model, data)` | 時間は進めずに、現在の状態から位置・接触・力などを計算し直す |
| `mj_kinematics(model, data)` | 運動学（各部の位置・姿勢）だけを計算する（第3章で使用） |

関節角を直接書き換えたあとは、`mj_forward` を呼んで **派生した量（手先の位置など）を計算し直す** 必要があります。第0章から何度も書いてきた `mujoco.mj_forward(model, data)` は、このためのものでした。

---

# 3. MJCF の書き方

## 3.1 木構造と主な要素

MJCF は、`<worldbody>` を根とする **body の入れ子構造** でロボットと環境を表します。子の body は、文字どおり XML の子要素として書きます[^mujoco-modeling]。

| 要素 | 役割 |
|---|---|
| `<body>` | 剛体（1つのかたまりとして動く部品）。`pos` で親からの位置を指定 |
| `<joint>` | body と親の間に **動ける自由度** を作る。joint がない body は親に固定される |
| `<geom>` | 形状。**衝突判定と見た目** の両方に使われ、質量もここから計算される |
| `<site>` | 位置を取得するための目印。衝突しない |
| `<camera>` | カメラ。body に付ければ一緒に動く |
| `<actuator>` | モーター。どの joint をどう動かすかを指定 |

第3章で説明したとおり、MuJoCo は body の木構造をたどって、同次変換を順に掛けて各部の位置を計算しています。

## 3.2 関節の種類と qpos のサイズ

| 関節 | 自由度 | `qpos` の数 | `qvel` の数 | 用途 |
|---|---|---|---|---|
| `hinge` | 回転 1 | 1 | 1 | アームの関節 |
| `slide` | 直動 1 | 1 | 1 | 直動ステージ、グリッパーの指 |
| `ball` | 回転 3 | 4（クォータニオン） | 3 | 肩関節など |
| `free` | 並進 3 + 回転 3 | **7**（位置 3 + クォータニオン 4） | **6** | 自由に動く物体（掴む対象など） |

`free` 関節は、**位置3つとクォータニオン4つ** で物体の位置と姿勢を表します。クォータニオンは第3章で学んだ **$(w, x, y, z)$ の順** です。回転の自由度は3なので、速度 `qvel` は 6 になり、**`qpos` と `qvel` の長さが一致しなくなります**（実践では `nq=12`、`nv=11`）。`qpos` の何番目がどの関節かは、`model.jnt_qposadr` で確認できます。

## 3.3 default クラス：CSS のような既定値

MJCF では、`<default>` に既定値をまとめて書けます。HTML に対する CSS のように、クラスを定義して、要素ごとに `class="..."` で適用します[^mujoco-modeling]。

```xml
<default>
  <geom condim="4" friction="1 0.005 0.0001"/>   <!-- すべての geom の既定値 -->
  <default class="finger">                       <!-- 指だけの既定値 -->
    <joint type="slide" range="0 0.04"/>
    <geom type="box" size="0.008 0.006 0.03"/>
  </default>
</default>
```

同じ部品が何度も出てくるロボットでは、設定の重複を大きく減らせます。

## 3.4 アクチュエータの種類

| アクチュエータ | 入力（`ctrl`） | 中身 |
|---|---|---|
| `motor` | トルク・力 | 指令をそのまま力として加える（第0・5章で使用） |
| `position` | 目標位置 | 物理エンジン内で PD 制御を行う（第4章で使用）。`kp` と、減衰の `kv` または減衰比 `dampratio` を指定 |
| `velocity` | 目標速度 | 速度の比例制御 |

`position` アクチュエータの `forcerange` を使うと、**出せる力の上限** を設定できます。グリッパーでは、これが **把持力の上限** になります。実践パートで、この設定が結果を大きく左右することを確認します。

また、`<body gravcomp="1">` と書くと、その body にかかる重力を打ち消す力が自動的に加わります。第5章で自分で計算した **重力補償** を、MuJoCo に任せる機能です。

---

# 4. 接触と摩擦

## 4.1 MuJoCo の接触モデル

物体どうしの接触は、物理シミュレーションで最も難しい部分の1つです。完全に硬い物体どうしの接触は、ぶつかった瞬間に速度が不連続に変わるため、数値的に扱いにくいからです。

MuJoCo は、**ほんのわずかなめり込みを許す「柔らかい接触」** のモデルを採用しています。接触を凸最適化問題として解くことで、安定性と計算速度を両立しています[^mujoco-comp]。このため、実践の結果でも、物体がサブミリメートル程度めり込んだり、止まっているはずの物体がごくわずかに動いたりします。

接触の硬さは `solref`（ばねの時定数と減衰比）と `solimp`（めり込みに対する硬さの変化）で調整できますが、まずは既定値のままで問題ありません。

## 4.2 摩擦のパラメータ

geom の `friction` 属性には、3つの摩擦係数を指定します[^mujoco-friction]。

| 値 | 摩擦の種類 | 抵抗する動き | 典型値 |
|---|---|---|---|
| 1つ目 | **すべり摩擦** | 面に沿ってすべる動き | 0.5〜1 程度 |
| 2つ目 | ねじり摩擦 | 接触面の法線まわりに回る動き | 0.005 程度 |
| 3つ目 | 転がり摩擦 | 転がる動き | 0.0001 程度 |

どの摩擦を計算に含めるかは、`condim`（接触の次元）で決まります。

| `condim` | 含まれる力 | 用途 |
|---|---|---|
| 1 | 法線方向（めり込まない）だけ | 摩擦なし |
| 3 | + すべり摩擦（接線2方向） | 一般的な接触（既定値） |
| 4 | + ねじり摩擦 | **把持**（掴んだ物体が指の間で回るのを防ぐ） |
| 6 | + 転がり摩擦 | ボールや車輪 |

2つの geom が接触したときの摩擦係数は、既定では **2つの geom の値の大きいほう** が使われます。

## 4.3 摩擦円錐

すべり摩擦の大きさには上限があり、法線力（押し付ける力）を $N$、摩擦係数を $\mu$ とすると、**摩擦力 $F \le \mu N$** です。

- 物体が止まっているとき（静止摩擦）：摩擦力はちょうど外力を打ち消す大きさで、$F \le \mu N$
- 物体がすべっているとき（動摩擦）：摩擦力は上限の $F = \mu N$

接線方向の2成分を合わせた摩擦力が $\mu N$ 以下という条件は、力の空間で **円錐** の形になるため、**摩擦円錐** と呼ばれます。MuJoCo では、これを角錐で近似する `cone="pyramidal"`（既定値）と、円錐のまま扱う `cone="elliptic"` を選べます。本章では、物理的に忠実な `elliptic` を使います。

## 4.4 接触情報の読み出し

シミュレーション中の接触は、`data.contact` に入っています。

```python
for i in range(data.ncon):                  # 現在の接触点の数
    c = data.contact[i]                     # c.geom1, c.geom2, c.pos（位置）, c.frame（向き）
    f6 = np.zeros(6)
    mujoco.mj_contactForce(model, data, i, f6)   # [法線力, 接線力1, 接線力2, ねじり, 転がり...]
```

力は **接触座標系**（1軸目が接触面の法線方向）で返ってきます。接触力は、把持の成否の判定、触覚センサーのシミュレーション、強化学習の報酬設計などに使います。

---

# 5. 把持の物理

## 5.1 持ち上げられる条件

2本の指で箱を左右から挟んで持ち上げる場合を考えます。指が箱を押し付ける力（**把持力**）を $F$ とすると、左右の指それぞれで摩擦力の上限は $\mu F$ です。箱を落とさずに持ち上げるには、摩擦力の合計が、箱の重さと加速に必要な力を上回る必要があります。

$$
2 \mu F \;\ge\; m (g + a)
$$

- $m$：箱の質量、$g$：重力加速度、$a$：持ち上げるときの上向きの加速度

ここから、把持の設計についていくつかのことがわかります。

- **滑りやすい物体ほど、強く握る必要がある**（$\mu$ が小さいと $F$ を大きく）
- **速く持ち上げるほど、強く握る必要がある**（$a$ が大きいと必要な力が増える）
- 壊れやすい物体は強く握れないので、**ゆっくり持ち上げる** 必要がある

## 5.2 position アクチュエータで把持力を決める

実践では、指を `position` アクチュエータで動かし、**目標位置を箱の内側（閉じきった位置）** に設定します。指は箱に当たって止まり、目標位置とのずれに比例した力で箱を押し付けます。

$$
F = \min\left(K_p \times (\text{目標位置と現在位置のずれ}),\; \text{forcerange の上限}\right)
$$

第5章で説明した「目標位置と実際の位置のずれが、PD 制御を通じて力になる」の具体例です。実践パートでは、ここで1つ落とし穴にはまります。

---

# 6. カメラとレンダリング

## 6.1 オフスクリーンレンダリング

フィジカルAIの方策の多くは、**カメラ画像** を入力にします。MuJoCo では、`mujoco.Renderer` を使って、ウィンドウを開かずに画像を取得（**オフスクリーンレンダリング**）できます[^mujoco-python]。

```python
with mujoco.Renderer(model, height, width) as r:
    r.update_scene(data, camera="top")
    rgb = r.render()                           # (H, W, 3) の uint8 配列
```

同じ Renderer で、次の3種類の画像が取れます。

| 画像 | 切り替え | 中身 |
|---|---|---|
| RGB | 既定 | カラー画像 |
| 深度 | `enable_depth_rendering()` | 各画素の、カメラの光軸方向の距離 [m] |
| セグメンテーション | `enable_segmentation_rendering()` | 各画素に写っている geom の ID |

深度とセグメンテーションは、**シミュレータだから追加コストなしで手に入る正解ラベル** です。実機では深度カメラや人手のアノテーションが必要な情報が、自動で得られます。第14章の合成データでも、この性質を活用します。

## 6.2 画面表示のない環境での設定

レンダリングには OpenGL が必要です。MuJoCo では、環境変数 `MUJOCO_GL` で描画のバックエンドを選びます。この変数は、**`import mujoco` より前** に設定する必要があります[^torchrl-mujoco]。

| `MUJOCO_GL` | 使う場面 |
|---|---|
| `glfw` | 画面のある PC（Windows / macOS / Linux デスクトップ）。既定値 |
| `egl` | 画面のない Linux サーバーで、GPU を使って描画する（クラウド GPU、Colab など） |
| `osmesa` | 画面も GPU もない環境で、CPU で描画する（遅い） |

本章の実践コードは、OpenGL が使えない環境でも動くように、**レイキャスト**（画素ごとに光線を飛ばして、最初に当たる物体と距離を求める `mj_ray`）で深度とセグメンテーションを計算する代替経路を用意しています。

## 6.3 カメラの内部パラメータ

深度画像から3次元の点を求めるには、カメラの **内部パラメータ**（焦点距離 $f_x, f_y$ と画像中心 $c_x, c_y$）が必要です。MuJoCo のカメラは縦の画角 `fovy` で定義されているので、画像の高さ $H$ から焦点距離を次のように求めます。

$$
f = \frac{H / 2}{\tan(\text{fovy} / 2)}
$$

画素 $(u, v)$ の深度が $z$ のとき、カメラ座標系での3次元の点は次のようになります。

$$
x = \frac{u - c_x}{f_x}\, z, \qquad y = \frac{v - c_y}{f_y}\, z
$$

## 6.4 カメラ座標系の約束事に注意

ここで、第3章の「座標系の約束事」がまた登場します。カメラ座標系には、2つの代表的な流儀があります。

| 流儀 | x | y | 視線方向 | 使われる場所 |
|---|---|---|---|---|
| **OpenCV** | 右 | **下** | **+z** | OpenCV、ROS、多くのコンピュータビジョンのライブラリ |
| **OpenGL** | 右 | **上** | **−z** | MuJoCo、3D グラフィックス |

第3章のカメラの例は OpenCV の流儀でしたが、**MuJoCo のカメラは OpenGL の流儀** です。そのため、上の式で求めた点は、y の符号を反転し、z を $-z$ にしてから、カメラの姿勢 `data.cam_xmat` と位置 `data.cam_xpos` でワールド座標に変換します。ここを間違えると、物体の位置が上下反転したり、カメラの後ろ側に計算されたりします。

---

# 7. 実践：掴めるシミュレーション環境を作る

## 7.1 作るもの

```text
ch06_mujoco_basics/
├── scene.py              # XYZ ステージ + 2本指グリッパー + 箱 のシーン
├── contact_friction.py   # 斜面の箱で、摩擦と接触力を確かめる
├── grasp.py              # 掴んで持ち上げる。軌道・摩擦・把持力を変えて比較
└── camera.py             # RGB・深度・セグメンテーション → 箱の3次元位置
```

ロボットには、第3〜5章の回転関節のアームではなく、**XYZ 方向に直動するステージ（ガントリー）の先に2本指グリッパーを付けたもの** を使います。IK を解かずに手先の位置を直接指定できるので、本章のテーマである **接触と把持** に集中できます。実際の工場でも、この構成の「直交ロボット」はよく使われています。

## 7.2 シーンの定義：`scene.py`

```python:scene.py
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
```

```bash
python scene.py
```

```text
body 数 7 / joint 数 6 / 自由度 nq=12, nv=11 / actuator 数 5 / camera 数 2
  joint 0: x      mjJNT_SLIDE  qpos の位置 0
  joint 1: y      mjJNT_SLIDE  qpos の位置 1
  joint 2: z      mjJNT_SLIDE  qpos の位置 2
  joint 3: left   mjJNT_SLIDE  qpos の位置 3
  joint 4: right  mjJNT_SLIDE  qpos の位置 4
  joint 5: box    mjJNT_FREE   qpos の位置 5
箱の初期位置 qpos[5:8] = [0.1  0.05 0.02]  姿勢（wxyz）qpos[8:12] = [1. 0. 0. 0.]
```

3.2 節で説明したとおり、自由に動く箱（`free` 関節）の `qpos` は 7 個（位置3 + クォータニオン4）あり、`nq=12` に対して `nv=11` になっています。箱の姿勢の初期値は `[1, 0, 0, 0]`、つまり $(w, x, y, z)$ の順の「回転なし」です。

主な設計のポイントは次のとおりです。

- **`gravcomp="1"`**：ステージとグリッパーの重さを MuJoCo が打ち消すので、position アクチュエータは位置合わせだけに集中できます（第5章の重力補償）
- **`armature`**：第5章で学んだとおり、軽い部品の数値的な不安定さを防ぐために、モーター側の慣性を入れています
- **指の `forcerange`**：把持力の上限です。`make()` の引数で変えられます
- **`condim="4"`**：把持した箱が指の間で回らないように、ねじり摩擦まで含めています

## 7.3 摩擦と接触力：`contact_friction.py`

斜面に箱を置いて、すべるかどうかと、そのときの接触力を調べます。斜面の傾きを $\theta$ とすると、箱がすべり出す条件は $\tan\theta > \mu$ です。

```python:contact_friction.py
"""第6章 (1): 斜面に置いた箱で、摩擦と接触力を確かめる"""
import mujoco
import numpy as np

XML = """
<mujoco>
  <option timestep="0.002" cone="elliptic"/>
  <worldbody>
    <!-- y 軸まわりに傾けた斜面 -->
    <geom name="slope" type="box" size="2 0.3 0.01" euler="0 @TILT@ 0" friction="@MU@ 0.005 0.0001"/>
    <body name="box" pos="0 0 0.05" euler="0 @TILT@ 0">
      <freejoint/>
      <geom name="box" type="box" size="0.03 0.03 0.03" mass="0.2" friction="@MU@ 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
"""


def simulate(tilt_deg, mu, T=0.5):
    xml = XML.replace("@TILT@", str(tilt_deg)).replace("@MU@", str(mu))   # euler は既定で「度」
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    start = data.qpos[:3].copy()
    for _ in range(int(0.2 / model.opt.timestep)):     # 着地して落ち着くまで
        mujoco.mj_step(model, data)
    settled = data.qpos[:3].copy()

    # 接触情報を読む：data.contact に接触点が入っている
    normal_total, tangent_total = 0.0, 0.0
    f6 = np.zeros(6)
    for i in range(data.ncon):
        mujoco.mj_contactForce(model, data, i, f6)   # 接触座標系での力 [法線, 接線1, 接線2, ...]
        normal_total += f6[0]
        tangent_total += np.linalg.norm(f6[1:3])
    ncon = data.ncon

    for _ in range(int(T / model.opt.timestep)):     # その後 T 秒でどれだけ滑るか
        mujoco.mj_step(model, data)
    slide = np.linalg.norm(data.qpos[:3] - settled)
    return slide, ncon, normal_total, tangent_total


if __name__ == "__main__":
    m, g = 0.2, 9.81
    print("理論：tanθ > μ なら滑る\n")
    print(f"{'傾き':>4s} {'μ':>4s} {'tanθ':>6s} {'0.5秒で滑った距離':>12s} {'接触点数':>6s} "
          f"{'法線力 N':>8s} {'摩擦力 F':>8s} {'F / N':>6s}")
    for tilt, mu in [(20, 0.5), (30, 0.5), (30, 0.7), (40, 0.7)]:
        slide, ncon, fn, ft = simulate(tilt, mu)
        th = np.deg2rad(tilt)
        print(f"{tilt:3d}° {mu:4.1f} {np.tan(th):6.3f} {slide * 100:10.2f} cm {ncon:6d} "
              f"{fn:7.3f} N {ft:7.3f} N {ft / fn:6.3f}")
    print(f"\n参考：止まっている場合の理論値  20°: N={m*g*np.cos(np.deg2rad(20)):.3f}, F={m*g*np.sin(np.deg2rad(20)):.3f}"
          f" / 30°: N={m*g*np.cos(np.deg2rad(30)):.3f}, F={m*g*np.sin(np.deg2rad(30)):.3f}")
```

```bash
python contact_friction.py
```

```text
理論：tanθ > μ なら滑る

  傾き    μ   tanθ   0.5秒で滑った距離   接触点数    法線力 N    摩擦力 F  F / N
 20°  0.5  0.364       0.05 cm      4   1.839 N   0.671 N  0.365
 30°  0.5  0.577      14.82 cm      2   0.748 N   0.374 N  0.500
 30°  0.7  0.577       0.09 cm      4   1.697 N   0.982 N  0.579
 40°  0.7  0.839      23.98 cm      2   3.384 N   2.369 N  0.700

参考：止まっている場合の理論値  20°: N=1.844, F=0.671 / 30°: N=1.699, F=0.981
```

すべてのケースで、理論どおりの結果になりました。

- **止まっているケース**（20°・μ=0.5、30°・μ=0.7）：法線力と摩擦力が理論値（$mg\cos\theta$、$mg\sin\theta$）とほぼ一致し、摩擦力と法線力の比 $F/N$ は $\tan\theta$ になっています。静止摩擦は「すべらないのに必要な分だけ」働き、上限 $\mu N$ には達していません
- **すべっているケース**（30°・μ=0.5、40°・μ=0.7）：$F/N$ が **ちょうど $\mu$**（0.500、0.700）になっています。すべっているときは、摩擦力が摩擦円錐の縁（上限 $\mu N$）に張り付くことが確認できます

止まっているケースでも、0.5 秒で 0.05〜0.09 cm だけ動いています。これは 4.1 節で説明した、MuJoCo の **柔らかい接触** による、ごくわずかなすべりです。

:::note warn
**角度の単位に注意**
MJCF の `euler` などの角度は、既定では **度** です（`<compiler angle="radian"/>` でラジアンに切り替えられます）。最初の実装では、このファイルでラジアンの値を渡してしまい、斜面がほとんど傾かずに「どの条件でもすべらない」という結果になりました。エラーは出ないので、気づきにくいバグです。`scene.py` ではラジアンを明示的に指定しています。
:::

## 7.4 掴んで持ち上げる：`grasp.py`

「箱の真上へ移動 → 下ろす → 指を閉じる → 持ち上げる」の4段階を、スクリプトで実行します。

```python:grasp.py
"""第6章 (2): スクリプトで「近づく → 下ろす → 掴む → 持ち上げる」を実行し、成否を判定する"""
import mujoco
import numpy as np

from scene import make


def waypoints(box_xy):
    """(x, y, z, 指の開き, 所要時間[s])。z はステージの関節値（0 = 初期の高さ）"""
    x, y = box_xy
    return [
        (x, y,  0.000, 0.04, 1.0),   # 1. 箱の真上へ移動（指は開く）
        (x, y, -0.225, 0.04, 1.0),   # 2. 下ろす（指が箱の側面の高さに来る）
        (x, y, -0.225, 0.00, 1.0),   # 3. 指を閉じる（目標 0 = 箱より内側 → 把持力の上限で押し付ける）
        (x, y,  0.000, 0.00, 1.5),   # 4. 持ち上げる
    ]


def quintic(s):
    """第5章の5次多項式（0→1 をなめらかにつなぐ）"""
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def run(mu=1.0, grip_force=5.0, box_mass=0.1, smooth=True, verbose=False):
    model, data = make(mu, grip_force, box_mass)
    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    dt = model.opt.timestep
    prev = data.ctrl.copy()
    max_box_h = 0.0
    for i, (x, y, z, w, T) in enumerate(waypoints(data.xpos[box_id][:2])):
        goal = np.array([x, y, z, w, w])
        n = int(T / dt)
        for k in range(n):
            if smooth:   # 所要時間の 8 割で目標まで補間し、残りは保持
                data.ctrl[:] = prev + quintic(k / (0.8 * n)) * (goal - prev)
            else:        # ステップ指令：目標をいきなり渡す
                data.ctrl[:] = goal
            mujoco.mj_step(model, data)
            max_box_h = max(max_box_h, data.xpos[box_id][2])
        prev = goal
        if verbose:
            print(f"  step{i + 1}: 手先 z={data.qpos[2]:+.3f}  指={data.qpos[3]:.4f}/{data.qpos[4]:.4f} m"
                  f"  箱の高さ={data.xpos[box_id][2]:.3f} m  指の力={data.actuator_force[3]:+.2f} N")
    lifted = data.xpos[box_id][2]
    return lifted, lifted > 0.15, max_box_h          # 15cm 以上持ち上がっていれば成功


if __name__ == "__main__":
    print("=== 1. ステップ指令 vs なめらかな軌道（μ=1.0, 把持力 5 N, 箱 100 g）===")
    for smooth, name in [(False, "ステップ指令"), (True, "5次多項式")]:
        print(f"[{name}]")
        h, ok, hmax = run(smooth=smooth, verbose=True)
        print(f"  → 結果: {'成功' if ok else '失敗'}（箱の最高到達点 {hmax:.3f} m）\n")

    print("=== 2. 摩擦係数と把持力を変える（5次多項式の軌道）===")
    m, g = 0.1, 9.81
    a_max = 10 / np.sqrt(3) * 0.225 / (0.8 * 1.5) ** 2      # 5次多項式で持ち上げるときの最大加速度
    need = m * (g + a_max)
    print(f"箱の重さ mg = {m * g:.2f} N、持ち上げの最大加速度 {a_max:.2f} m/s² → 必要な摩擦力 m(g+a) = {need:.2f} N")
    print("理論：2 μ F ≥ m(g+a) なら滑らずに持ち上がる\n")
    print(f"{'μ':>5s} {'把持力F':>7s} {'2μF':>7s} {'理論':>6s} {'箱の最終高さ':>10s} {'結果':>4s}")
    for mu, F in [(1.0, 5.0), (0.3, 5.0), (0.1, 6.0), (0.1, 5.0), (0.1, 3.0), (0.05, 12.0), (0.05, 5.0)]:
        h, ok, _ = run(mu, F, m)
        theory = "持てる" if 2 * mu * F >= need else "滑る"
        print(f"{mu:5.2f} {F:5.1f} N {2 * mu * F:5.2f} N {theory:>6s} {h:9.3f} m  {'成功' if ok else '失敗'}")
```

```bash
python grasp.py
```

```text
=== 1. ステップ指令 vs なめらかな軌道（μ=1.0, 把持力 5 N, 箱 100 g）===
[ステップ指令]
  step1: 手先 z=+0.000  指=0.0400/0.0400 m  箱の高さ=0.020 m  指の力=-0.00 N
  step2: 手先 z=-0.225  指=0.0400/0.0400 m  箱の高さ=0.020 m  指の力=+0.00 N
  step3: 手先 z=-0.225  指=0.0174/0.0174 m  箱の高さ=0.020 m  指の力=-5.00 N
  step4: 手先 z=-0.000  指=0.0000/-0.0000 m  箱の高さ=0.020 m  指の力=+0.00 N
  → 結果: 失敗（箱の最高到達点 0.206 m）

[5次多項式]
  step1: 手先 z=+0.000  指=0.0400/0.0400 m  箱の高さ=0.020 m  指の力=-0.00 N
  step2: 手先 z=-0.225  指=0.0400/0.0400 m  箱の高さ=0.020 m  指の力=-0.00 N
  step3: 手先 z=-0.225  指=0.0174/0.0174 m  箱の高さ=0.020 m  指の力=-5.00 N
  step4: 手先 z=-0.000  指=0.0174/0.0174 m  箱の高さ=0.242 m  指の力=-5.00 N
  → 結果: 成功（箱の最高到達点 0.243 m）

=== 2. 摩擦係数と把持力を変える（5次多項式の軌道）===
箱の重さ mg = 0.98 N、持ち上げの最大加速度 0.90 m/s² → 必要な摩擦力 m(g+a) = 1.07 N
理論：2 μ F ≥ m(g+a) なら滑らずに持ち上がる

    μ    把持力F     2μF     理論     箱の最終高さ   結果
 1.00   5.0 N 10.00 N    持てる     0.242 m  成功
 0.30   5.0 N  3.00 N    持てる     0.242 m  成功
 0.10   6.0 N  1.20 N    持てる     0.243 m  成功
 0.10   5.0 N  1.00 N     滑る     0.020 m  失敗
 0.10   3.0 N  0.60 N     滑る     0.020 m  失敗
 0.05  12.0 N  1.20 N    持てる     0.243 m  成功
 0.05   5.0 N  0.50 N     滑る     0.020 m  失敗
```

### ① ステップ指令だと、箱が飛んでいく

最初に作ったバージョンでは、各段階の目標位置を **いきなり** アクチュエータに渡していました（ステップ指令）。結果を見ると、指を閉じた時点までは同じなのに、持ち上げると **箱が 0.206 m の高さまで飛び上がり、指から離れて床に落ちて** います。

持ち上げの瞬間、ステージは目標との大きなずれに対して大きな力を出し、手先に **数百 m/s² という加速度** がかかります。5.1 節の式でいえば $a$ が巨大になり、摩擦力では箱を支えきれません。箱は指からすっぽ抜け、勢いで上に飛び出したのです。

第5章の **5次多項式の軌道** で目標をなめらかに動かすと、同じ設定で箱を 0.242 m まで持ち上げることに成功しました。**軌道生成は、把持の成否そのものを左右する** ことがわかります。

### ② 成否は 2μF ≥ m(g+a) で決まる

5次多項式の軌道で、摩擦係数と把持力を変えた結果です。持ち上げの最大加速度は約 0.90 m/s² なので、必要な摩擦力は $m(g + a) \approx 1.07$ N です。

| 条件 | 2μF | 理論 | 結果 |
|---|---|---|---|
| μ=1.0、F=5 N | 10.0 N | 持てる | 成功 |
| μ=0.1、F=6 N | 1.2 N | 持てる | 成功 |
| **μ=0.1、F=5 N** | **1.0 N** | **滑る** | **失敗** |
| μ=0.05、F=12 N | 1.2 N | 持てる | 成功 |
| μ=0.05、F=5 N | 0.5 N | 滑る | 失敗 |

7つの条件すべてで、理論と結果が一致しました。特に μ=0.1・F=5 N のケースは、箱の重さ（0.98 N）だけなら持てるはずの力（1.0 N）がありますが、**持ち上げる加速度の分（0.09 N）が足りずに失敗** しています。$a$ の項が実際に効いていることがわかります。また、μ=0.05 でも、把持力を 12 N に上げれば持ち上げられます。

### ③ つまずいたポイント：把持力は forcerange だけでは決まらない

最初のバージョンでは、指のアクチュエータの `kp` を 500 にしていました。この設定では、μ=0.05 で把持力の上限を 12 N や 20 N に上げても、**なぜか必ず失敗** しました。

原因は 5.2 節の式にありました。指は目標位置（0）から約 0.017 m 手前で箱に当たって止まるため、実際の把持力は $K_p \times 0.017 = 500 \times 0.017 \approx 8.6$ N にしかなりません。`forcerange` をいくら大きくしても、**8.6 N を超えることはなかった** のです。$2 \times 0.05 \times 8.6 = 0.86$ N は必要な 1.07 N に届かないので、失敗は理論どおりでした。

`kp` を 3000 に上げて、$K_p \times$ ずれ が常に `forcerange` を上回るようにすると、把持力が `forcerange` で決まるようになり、結果が理論と一致しました。**「設定したつもりの力」と「実際に出ている力」が違う** というのは、シミュレーションでも実機でもよくある落とし穴です。`data.actuator_force` で実際の力を確認する習慣をつけましょう。

## 7.5 カメラ画像から箱の位置を求める：`camera.py`

真上のカメラ（高さ 1 m、`fovy=60°`）の画像から、**深度とセグメンテーションだけを使って** 箱の3次元位置を推定します。

```python:camera.py
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
```

```bash
# 画面のある PC
python camera.py
# 画面のない Linux サーバー（GPU あり）
MUJOCO_GL=egl python camera.py
```

画面のある PC（筆者の Windows 環境）で実行すると、OpenGL による描画で次の結果になり、RGB・深度・セグメンテーションの画像が保存されます。

```text
[OpenGL] 画像 320x240 / 深度の範囲 0.680〜1.000 m / 箱が写っている画素数 81
深度から推定した箱の上面の中心: [0.0993 0.0485 0.04  ] （外れ値に強い中央値）
シミュレータの真値（箱の中心）  : [0.1  0.05 0.02] → 上面は z = 0.04
depth_top.png / mask_top.png / rgb_top.png を保存しました
```

OpenGL が使えないサーバーでは、レイキャストの代替経路で実行され、次のように表示されます。推定された位置は OpenGL 経路と同じでした。

```text
OpenGL が使えないため、レイキャストで代替します（FatalError）
[レイキャスト] 画像 320x240 / 深度の範囲 0.680〜1.000 m / 箱が写っている画素数 81
深度から推定した箱の上面の中心: [0.0993 0.0485 0.04  ] （外れ値に強い中央値）
シミュレータの真値（箱の中心）  : [0.1  0.05 0.02] → 上面は z = 0.04
depth_top.png / mask_top.png を保存しました
```

セグメンテーションで「箱が写っている画素」だけを取り出し、その深度から計算した3次元の点の中央値は (0.099, 0.049, 0.040) でした。真値は箱の中心が (0.1, 0.05, 0.02) で、上から見えるのは上面（z = 0.04）なので、**ほぼ正確に箱の上面の中心を求められています**。

この処理は、実機での「カメラで物体を見つけて、そこに手を伸ばす」の基本形です。実機では、セグメンテーションの部分を物体検出やセグメンテーションのモデルに置き換えます。そして、こうして得た位置に手を伸ばすのが、第4章の IK の役割です。

:::note warn
**OpenGL 経路の動作確認について**
`render_gl` の経路（`mujoco.Renderer` による描画）は、Windows の既定のバックエンド（`glfw`）で実行して確認しました。箱が写っている画素数（81）と推定位置は、レイキャストの結果と一致しています。ただし、OpenGL の深度画像は、光の届く範囲（near / far）の設定などにより、レイキャストとわずかに値が異なる場合があります。EGL を使うサーバーなど、ほかの環境では確認していません。
:::

## 7.6 演習

1. `contact_friction.py` で、`cone="elliptic"` を `cone="pyramidal"` に変えて、すべり始める角度や $F/N$ の値が変わるか確かめてみましょう。
2. `grasp.py` で、箱の質量を 0.3 kg に変えると、必要な把持力はいくつになるでしょうか。計算してから、シミュレーションで確かめてみましょう。
3. `grasp.py` の持ち上げ時間を 1.5 秒から 0.5 秒に縮めると、5.1 節の $a$ はどう変わり、成否の境目はどう動くでしょうか。
4. `scene.py` の箱の `condim` を 3 に変え、箱を x 方向にずらした位置（指の中心から外れた位置）で掴んで持ち上げてみましょう。箱が指の間で回転するか観察してください。
5. `camera.py` の推定位置を使って、`grasp.py` の `waypoints` の目標位置を決めるように書き換えてみましょう。**カメラで見て、掴みに行く** ロボットになります。
6. （発展）`pip install robot_descriptions` などで MuJoCo Menagerie のモデルを取得し、Franka Emika Panda のモデルを読み込んで、本章と同じ箱を掴ませてみましょう[^menagerie-panda]。第4章の差分 IK が必要になります。

---

# 8. まとめ

- MuJoCo は、構造を表す **MjModel** と、状態を表す **MjData** に分かれている。関節角を書き換えたら `mj_forward` で派生量を計算し直す
- **MJCF** は body の入れ子構造で書き、joint で自由度を、geom で形状・衝突・質量を、actuator でモーターを定義する。**default クラス** で既定値をまとめられる。`free` 関節の `qpos` は位置3 + クォータニオン4
- MuJoCo の接触は、わずかなめり込みを許す **柔らかい接触**。摩擦は `friction`（すべり・ねじり・転がり）と `condim` で決まり、すべっているときは摩擦力が **摩擦円錐の縁 $\mu N$** に張り付くことを確かめた
- 把持の成否は **$2\mu F \ge m(g + a)$** でほぼ決まる。**ステップ指令で持ち上げると加速度が大きすぎて箱が飛ぶ** ため、軌道生成が把持の成否を左右する。把持力は `kp × ずれ` と `forcerange` の小さいほうで決まることにも注意する
- `mujoco.Renderer` で **RGB・深度・セグメンテーション** を取得できる。画面のない環境では `MUJOCO_GL` を設定する。MuJoCo のカメラは **OpenGL の流儀（y が上、視線が −z）** であることに注意して、深度から3次元の点を求められる

## 次回予告

MuJoCo は CPU で手軽に動かせる一方、強化学習で **数千の環境を並列に動かして大量の試行錯誤をする** には、GPU を使ったシミュレーションが有利です。

**第7章「Isaac Sim / Isaac Lab」** では、NVIDIA の GPU シミュレーション環境を扱います。なぜロボット学習で GPU 並列シミュレーションが重要なのか、Isaac Sim と Isaac Lab の役割の違い、そしてローカルに RTX GPU がない場合にクラウド GPU で動かす方法を紹介します。

---

# 参考文献

[^menagerie]: Google DeepMind, "MuJoCo Menagerie". https://github.com/google-deepmind/mujoco_menagerie
[^menagerie-panda]: MuJoCo Menagerie, "franka_emika_panda". https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda
[^menagerie-doc]: MuJoCo Documentation, "Model Gallery". https://mujoco.readthedocs.io/en/stable/models.html
[^mujoco-modeling]: MuJoCo Documentation, "Modeling". https://mujoco.readthedocs.io/en/stable/modeling.html
[^mujoco-comp]: MuJoCo Documentation, "Computation". https://mujoco.readthedocs.io/en/latest/computation/index.html
[^mujoco-friction]: MuJoCo Documentation, "XML Reference"（`geom` の `friction` / `condim` 属性、`option` の `cone` 属性）. https://mujoco.readthedocs.io/en/stable/XMLreference.html
[^mujoco-python]: MuJoCo Documentation, "Python". https://mujoco.readthedocs.io/en/stable/python.html
[^torchrl-mujoco]: TorchRL Documentation, "Working with MuJoCo-based environments". https://docs.pytorch.org/rl/stable/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html
