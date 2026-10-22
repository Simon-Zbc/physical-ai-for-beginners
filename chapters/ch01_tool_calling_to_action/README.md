# 第1章: Tool Calling から Action へ

LLM の Function Calling（Tool Calling）でロボットアームを動かす Agent を実装します。SayCan の「Say（有用性）× Can（実行可能性）」の考え方を、スキル層の安全チェックとして組み込んでいます。

## ファイル

| ファイル | 内容 | LLM |
|---|---|---|
| `arm_env.py` | スキル層。`get_scene` / `check_affordance` / `move_hand_to` の3つのツールと、内部の500Hz制御ループ（第0章のヤコビ転置法） | 不要（単体で実行可能） |
| `llm_agent.py` | Function Calling で `arm_env.py` のスキルを呼び出す Agent ループ | 必要（OpenAI / Azure OpenAI の API キー） |

## 実行方法

### スキル層のみ（LLM不要）

```bash
uv sync
cd chapters/ch01_tool_calling_to_action
uv run python arm_env.py
```

赤いカップ・青い箱には到達でき、アームの可動範囲（0.05〜0.55m）の外にある緑のボールは `feasible: false` として拒否されることを確認できます。

### LLM Agent（要APIキー）

`llm_agent.py` は OpenAI または Azure OpenAI（v1 API）の API キーが必要なため、**本リポジトリではまだ実行していません**。実行する場合は、このディレクトリに `.env` ファイルを作成してください（`.env` は `.gitignore` で除外されているため、誤ってコミットされる心配はありません）。

```bash
cd chapters/ch01_tool_calling_to_action
cp .env.example .env
```

`.env` を編集して値を設定します。

```env
OPENAI_API_KEY=...
LLM_MODEL=<モデル名 または デプロイ名>
# Azure OpenAI の場合だけ追加する
OPENAI_BASE_URL=https://<リソース名>.openai.azure.com/openai/v1/
```

OpenAI を使う場合、`OPENAI_BASE_URL` は書かないでください。値を空にして書くと空文字として扱われ、接続に失敗します。

```bash
uv run python llm_agent.py "赤いカップ、青い箱、緑のボールの順に触ってください。"
```

## 演習（元記事より）

1. `OBJECTS` と MJCF に新しい物体（例: `yellow_block`）を追加し、「一番遠くにある、届く物体に触って」のような推論が必要な指示を試す
2. `touch(object_name)` というツールを追加し、座標ではなく物体名で指定できるようにする
3. システムプロンプトから安全確認の指示を消し、スキル層の安全チェックが働くことを確認する
4. `move_hand_to` にランダムな失敗（例: 10%の確率で `success: false`）を加え、LLMがリトライするか観察する
