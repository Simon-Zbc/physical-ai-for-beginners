# 第8章: Genesis

Python で書かれた GPU ネイティブな物理シミュレータ Genesis World で、Franka Emika Panda に箱を掴ませます。16 環境を並列に動かし、箱の位置と把持力を環境ごとに変えて、第6章の把持の物理（`2μF ≥ m(g+a)`）を確かめます。LLM は不要で、CPU でも動作します。

## 動作環境

`genesis-world` は **Python 3.10〜3.13** が必要です（`<3.14`）。リポジトリ本体の依存関係（`pyproject.toml`）には含めず、この章専用の環境を作ってください。PyTorch も必要です。

```bash
# uv の場合（Python 3.12 を自動で用意する）
uv venv --python 3.12 .venv-genesis
uv pip install --python .venv-genesis genesis-world torch
```

検証は Genesis 1.4.4 / PyTorch 2.14（CPU）/ Python 3.12 で行いました。初回の `scene.build()` はカーネルのコンパイルで 1 分前後かかります。GPU がある場合は、スクリプト中の `gs.cpu` を `gs.gpu` に変えてください（`benchmark.py` は自動で選択します）。

## ファイル

| ファイル | 内容 |
|---|---|
| `hello_genesis.py` | 最小のプログラム（初期化 → シーン → エンティティ → build → step）。Franka と箱を置いて重力で落とす |
| `grasp_parallel.py` | 16 環境で、箱の位置と把持力を変えて掴む。IK を全環境分一括で解き、5次多項式で補間して移動 |
| `benchmark.py` | 環境数を変えてスループット（env-step/s）を測る |

## 実行方法

```bash
cd chapters/ch08_genesis
../../.venv-genesis/Scripts/python hello_genesis.py       # Windows。Linux/macOS は ../../.venv-genesis/bin/python
../../.venv-genesis/Scripts/python grasp_parallel.py
for n in 1 16 64 256; do ../../.venv-genesis/Scripts/python benchmark.py $n; done
```

## 結果についての注意

- `grasp_parallel.py` の把持力 0.1 N / 0.4 N / 2.0 N の結果は理論どおりで再現性があります。**境界付近の 0.3 N** は、環境・バージョンによって成功数が変わります（記事では 3/4、筆者の再検証では 2/4）。成否の境界付近ではシミュレータの接触モデルの違いが結果に現れる、というのが本章の論点です
- `benchmark.py` のスループットはハードウェア依存です。CPU では 16 環境前後でピークになり、それ以上では低下する傾向になります。GPU では環境数を増やしても伸びやすくなります

## 演習（元記事より）

1. GPU のある環境で `benchmark.py` を 1 / 64 / 1024 / 4096 環境で実行し、CPU と比較する
2. `grasp_parallel.py` の持ち上げ時間を 1.5 秒から 0.5 秒に短縮し、成否の境界の動きを見る
3. 箱を `gs.morphs.Sphere(radius=0.02)` に変えて、球を掴む
4. 第6章の `grasp.py` と同じ条件（ステップ指令）を Genesis で再現し、箱が飛ぶか確かめる
5. （発展）FEM や MPM のソルバーで、柔らかい物体や砂を Franka で扱う
