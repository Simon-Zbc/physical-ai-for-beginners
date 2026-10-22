# 第6章: MuJoCo 入門

接触・摩擦・2本指グリッパー・カメラを含む「掴めるシミュレーション環境」を構築します。LLM は不要で、すべてローカルで実行できます。このディレクトリ内で完結しています。

## ファイル

| ファイル | 内容 |
|---|---|
| `scene.py` | XYZ ステージ + 2本指グリッパー + 箱のシーン。摩擦係数・把持力の上限・箱の質量を `make()` の引数で変更可能 |
| `contact_friction.py` | 斜面の箱で摩擦と接触力を確認。すべっているとき F/N がちょうど μ になることを確かめる |
| `grasp.py` | 「近づく → 下ろす → 掴む → 持ち上げる」を実行。ステップ指令と5次多項式の比較、`2μF ≥ m(g+a)` の検証 |
| `camera.py` | RGB・深度・セグメンテーションを取得し、深度から箱の3次元位置を推定。OpenGL が使えない場合はレイキャストで代替 |

## 実行方法

```bash
uv sync
cd chapters/ch06_mujoco_basics
uv run python scene.py
uv run python contact_friction.py
uv run python grasp.py
uv run python camera.py      # depth_top.png / mask_top.png / rgb_top.png を保存（git 管理外）
```

画面のない Linux サーバーでは `MUJOCO_GL=egl`（GPU あり）または `MUJOCO_GL=osmesa`（CPU のみ）を指定してください。

## 演習（元記事より）

1. `contact_friction.py` で `cone="elliptic"` を `"pyramidal"` に変えて、結果の変化を確認する
2. `grasp.py` で箱の質量を 0.3 kg にし、必要な把持力を計算してから確かめる
3. 持ち上げ時間を 1.5 秒から 0.5 秒に縮め、成否の境目がどう動くか確認する
4. 箱の `condim` を 3 にして、指の中心から外れた位置で掴み、箱が回転するか観察する
5. `camera.py` の推定位置を `grasp.py` の `waypoints` に使い、「カメラで見て掴みに行く」ようにする
6. （発展）MuJoCo Menagerie の Franka Emika Panda で同じ箱を掴ませる
