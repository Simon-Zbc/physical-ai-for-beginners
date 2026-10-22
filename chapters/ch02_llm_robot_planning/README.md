# 第2章: LLM でロボットを動かす

LLM にツールを選ばせるのではなく、**ロボットを動かすコードそのもの**を書かせる Code as Policies と、計画・実行・再計画を層に分ける階層型プランニングを実装します。

## ファイル

| ファイル | 内容 | LLM |
|---|---|---|
| `tabletop_env.py` | スキル層。ブロックを掴んで置ける卓上環境（`pick` / `place` など6つのAPI）。把持は手先への吸着モデル | 不要 |
| `code_as_policies.py` | コード生成 → 静的検査（AST）→ サンドボックス実行 → 失敗時の書き直し | 必要 |
| `hierarchical_agent.py` | Planner（サブタスク分解）→ Code as Policies → 失敗時の再計画 | 必要（JSONモード対応モデル） |
| `demo_without_llm.py` | LLM なしで検査・サンドボックス・スキルの動作を確認するデモ | 不要 |

## 実行方法

### LLM なしで実行基盤を確認

```bash
uv sync
cd chapters/ch02_llm_robot_planning
uv run python demo_without_llm.py
```

届く3つのブロックをトレイへ運び、届かない黄色のブロックはスキップして報告します。また、`import` / `open` / `while` を含むコードが実行前に拒否されることも確認できます。

### LLM を使う（要APIキー）

`code_as_policies.py` と `hierarchical_agent.py` は OpenAI または Azure OpenAI の API キーが必要なため、**本リポジトリではまだ実行していません**。第1章と同様に、このディレクトリに `.env` を作成してください。

```bash
cp .env.example .env     # 編集して OPENAI_API_KEY / LLM_MODEL を設定（Azure OpenAI の場合は OPENAI_BASE_URL も）
```

OpenAI を使う場合、`OPENAI_BASE_URL` は書かないでください（値を空にして書くと接続に失敗します）。

```bash
# Code as Policies 単体
uv run python code_as_policies.py "トレイから一番遠いブロックから順に、届くものをトレイに入れて"

# 階層型 Agent
uv run python hierarchical_agent.py "テーブルを片付けて。ブロックは全部トレイに入れて"
```

## 注意

`code_as_policies.py` のサンドボックスは**学習用**です。`exec()` を AST 検査と名前空間の制限で守る方法は、セキュリティ境界としては不十分です。実運用では別プロセス・コンテナ等で隔離し、ロボット側にも物理的な安全機構を設けてください。

## 演習（元記事より）

1. `FEW_SHOT` に例を追加せずに「ブロックをトレイの周りに円形に並べて」と指示し、`math.cos` / `math.sin` を使えるか観察する
2. `pick` に20%の確率で失敗する処理を加え、Executor の書き直しと Planner の再計画の動きを観察する
3. 階層的なコード生成を実装する（未定義の関数呼び出しを `ast` で検出し、その実装を LLM に別途生成させる）
4. Planner の出力に ProgPrompt 風の前提条件を含めさせ、実行前に検証する仕組みを追加する
