# フィジカルAI入門 ― ソフトウェアエンジニアのための理論と実践

ソフトウェアエンジニア（クラウド、AI、Web・業務システム開発など）が、**フィジカルAI（Physical AI）** を理論と実践の両面から体系的に学ぶための学習リポジトリです。Qiita で公開した記事の原稿を [`articles/`](articles/) に、記事の実践コードを [`chapters/`](chapters/) に、章ごとに収録しています。

## 学習の方針

- **Agent から Physical AI へ**：LLM Agent の考え方を出発点に、ロボティクスの基礎、ロボット学習、基盤モデル（VLA / World Model）へと進みます
- **理論 → 最小実装 → まとめ**：各章は、原理の解説と動かせるコードをセットにしています
- **実機がなくても学べる**：第19章までは原則としてシミュレーションで完結します。実機は低価格アーム（SO-101 など）を想定しています
- **数式は必要最小限**：厳密な導出は省略し、直感的な理解と実装を重視します

## リポジトリ構成

```text
physical-ai-for-beginners/
├── articles/                      # 記事の原稿（NN_タイトル.md）。記事間のリンク ./NN_xxx.md がそのまま使える
├── chapters/                      # 章ごとの実践コード（chNN_xxx/）。各ディレクトリの README に実行方法
│   ├── ch00_intro/
│   ├── ch01_tool_calling_to_action/
│   └── ...
├── pyproject.toml                 # 共通の依存関係（uv）
└── README.md
```

各章のコードは、原則としてそのディレクトリ内で完結します。章をまたいで再利用するファイル（第3章の `transforms.py` / `fk.py` など）は、章ごとにコピーを置いています。

## 目次

### Part 0：導入

| # | 記事 | 主なテーマ | 実践 | コード | 状態 |
|---|---|---|---|---|---|
| 00 | [フィジカルAIとは何か](articles/00_フィジカルAIとは何か.md) | 定義、生成AIとの違い、技術スタック、2026年の動向 | MuJoCo で Sense → Think → Act ループ | [`ch00_intro/`](chapters/ch00_intro/) | ✅ |

### Part 1：Agent から Embodied Agent へ

| # | 記事 | 主なテーマ | 実践 | コード | 状態 |
|---|---|---|---|---|---|
| 01 | [Tool Calling から Action へ](articles/01_ToolCallingからActionへ.md) | SayCan / RT-1 / RT-2 の系譜 | LLM の Function Calling でシミュレーション上のアームを操作 | [`ch01_tool_calling_to_action/`](chapters/ch01_tool_calling_to_action/) | ✅ |
| 02 | [LLM でロボットを動かす](articles/02_LLMでロボットを動かす.md) | Code as Policies、階層型プランニング | 自然言語指示 → タスク分解 → スキル実行 | [`ch02_llm_robot_planning/`](chapters/ch02_llm_robot_planning/) | ✅ |

### Part 2：ロボット工学の基礎（エンジニア版）

| # | 記事 | 主なテーマ | 実践 | コード | 状態 |
|---|---|---|---|---|---|
| 03 | [座標変換と運動学](articles/03_座標変換と運動学.md) | 座標系、同次変換、クォータニオン、順運動学 | NumPy で順運動学を計算 | [`ch03_kinematics/`](chapters/ch03_kinematics/) | ✅ |
| 04 | [逆運動学（IK）](articles/04_逆運動学.md) | ヤコビアン、数値 IK、特異姿勢 | MuJoCo でアームを指定位置へ移動 | [`ch04_inverse_kinematics/`](chapters/ch04_inverse_kinematics/) | ✅ |
| 05 | [制御と軌道生成](articles/05_制御と軌道生成.md) | PID 制御、軌道計画 | PID でアームを滑らかに制御 | [`ch05_control_trajectory/`](chapters/ch05_control_trajectory/) | ✅ |

### Part 3：シミュレーション

| # | 記事 | 主なテーマ | 実践 | コード | 状態 |
|---|---|---|---|---|---|
| 06 | [MuJoCo 入門](articles/06_MuJoCo入門.md) | MJCF、物理シミュレーション、レンダリング | 物体把持環境の構築 | [`ch06_mujoco_basics/`](chapters/ch06_mujoco_basics/) | ✅ |
| 07 | [Isaac Sim / Isaac Lab](articles/07_IsaacSim_IsaacLab.md) | GPU 並列シミュレーション、Omniverse | クラウド GPU で Isaac Lab を実行 | [`ch07_isaac_lab/`](chapters/ch07_isaac_lab/)（NumPy による並列環境の体験。Isaac Lab 本体は未実行） | ✅ |
| 08 | [Genesis](articles/08_Genesis_World.md) | GPU ネイティブ、生成的シミュレーション | 把持タスクの実行 | [`ch08_genesis/`](chapters/ch08_genesis/)（Python 3.10〜3.13 の別環境が必要） | ✅ |

### Part 4：ロボット学習

| # | 記事 | 主なテーマ | 実践 | コード | 状態 |
|---|---|---|---|---|---|
| 09 | [強化学習（RL）](articles/09_強化学習.md) | MDP、PPO / SAC、報酬設計 | Reach タスクの学習 | [`ch09_reinforcement_learning/`](chapters/ch09_reinforcement_learning/) | ✅ |
| 10 | [模倣学習（IL）](articles/10_模倣学習.md) | Behavior Cloning、ACT、Diffusion Policy | デモデータから方策を学習 | [`ch10_imitation_learning/`](chapters/ch10_imitation_learning/) | ✅ |
| 11 | LeRobot 実践 | LeRobotDataset、学習・評価 | 自作データセットで方策を学習 | - | ⬜ |

### Part 5：基盤モデルの時代

| # | 記事 | 主なテーマ | 実践 | コード | 状態 |
|---|---|---|---|---|---|
| 12 | Vision-Language-Action（VLA） | RT-2、OpenVLA、π0 系、SmolVLA、GR00T、行動トークン化 | VLA の推論とファインチューニング | - | ⬜ |
| 13 | World Model | Dreamer、Cosmos、「想像」による学習と評価 | World Model による未来予測 | - | ⬜ |
| 14 | 合成データ | シミュレーションデータ、動画生成 → 行動データ | 合成データを使った学習 | - | ⬜ |

### Part 6：Agent × フィジカルAI

| # | 記事 | 主なテーマ | 実践 | コード | 状態 |
|---|---|---|---|---|---|
| 15 | RAG for Robots | 手順書・安全規程・設備マニュアルの参照 | 知識を参照してから動くロボット Agent | - | ⬜ |
| 16 | マルチエージェント・ロボティクス | Planner / Executor / Safety / Vision Agent | 役割分担型のロボット Agent | - | ⬜ |
| 17 | MCP とロボット | Robot MCP Server の設計 | ロボットのスキルを MCP ツールとして公開 | - | ⬜ |

### Part 7：実世界へ

| # | 記事 | 主なテーマ | 実践 | コード | 状態 |
|---|---|---|---|---|---|
| 18 | Sim2Real | Sim2Real Gap、Domain Randomization | ランダム化による頑健性の向上 | - | ⬜ |
| 19 | エッジ推論と Jetson | 量子化、リアルタイム推論 | 方策モデルの軽量化と推論 | - | ⬜ |
| 20 | 実機実践 | SO-101 などの低価格アーム | 実機でのデータ収集と方策の実行 | - | ⬜ |

凡例：✅ 公開済み ／ 🚧 執筆中 ／ ⬜ 未着手

## 主な技術スタック

| 分類 | ツール |
|---|---|
| 言語・基盤 | Python 3.10+、PyTorch、NumPy、uv |
| シミュレータ | MuJoCo（メイン）、Genesis、Isaac Sim / Isaac Lab |
| ロボット学習 | Gymnasium、Stable-Baselines3、LeRobot |
| 基盤モデル | OpenVLA、SmolVLA、π0 系、GR00T、Cosmos |
| ミドルウェア・実機 | ROS 2、Jetson、SO-101 |
| Agent 連携 | LLM API（Function Calling）、RAG、MCP |

## 動作環境

| 対象 | 必要環境 |
|---|---|
| 第0〜6章 | CPU のみで実行可能（Windows / macOS / Linux） |
| 第7章（Isaac Sim / Lab） | RT コアを持つ RTX GPU が必須。ローカルにない場合はクラウド GPU を使用 |
| 第8章（Genesis） | Python 3.10〜3.13 の別環境と PyTorch が必要（CPU でも動作） |
| 第9章（強化学習） | CPU のみで実行可能（PyTorch は CPU 版で可） |
| 第10章（模倣学習） | CPU のみで実行可能（PyTorch は CPU 版で可） |
| 第11〜14章（学習） | NVIDIA GPU 推奨。Google Colab やクラウド GPU でも可 |
| 第20章 | 実機（SO-101 など）。未所持の場合はシミュレーションで代替 |

> **Note**：GTX 10xx（Pascal 世代）の GPU を使う場合、PyTorch は CUDA 12.6 ビルド（`cu126`）を選んでください。CUDA 12.8 以降のビルドは Pascal 世代に対応していません。

## セットアップ

Python 3.10 以上を想定しています。

```bash
git clone https://github.com/Simon-Zbc/physical-ai-for-beginners.git
cd physical-ai-for-beginners

# uv を使う場合（推奨）
uv sync

# pip を使う場合
python -m venv .venv
# Windows: .venv\Scripts\activate / macOS・Linux: source .venv/bin/activate
pip install mujoco numpy openai python-dotenv pillow
```

各章の実行方法は、それぞれのディレクトリ内の README を参照してください（例：[`chapters/ch00_intro/README.md`](chapters/ch00_intro/README.md)）。LLM を使う章（第1・2章）は、各章の `.env.example` を `.env` にコピーして API キーを設定します。強化学習の章（第9章）は、追加の依存関係（Stable-Baselines3・Gymnasium・PyTorch）が必要です（`uv sync --extra rl`。pip の場合は [`chapters/ch09_reinforcement_learning/README.md`](chapters/ch09_reinforcement_learning/README.md) を参照）。模倣学習の章（第10章）は、追加の依存関係（PyTorch）が必要です（`uv sync --extra il`。pip の場合は [`chapters/ch10_imitation_learning/README.md`](chapters/ch10_imitation_learning/README.md) を参照）。

## 注意事項

- 業界動向に関する記述は、各章の執筆時点の公開情報にもとづいています
- 各章の参考文献は、それぞれの記事末尾に記載しています
