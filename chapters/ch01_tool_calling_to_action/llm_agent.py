"""第1章: LLM の Function Calling でロボットのスキルを呼び出す Agent"""
import json
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

from arm_env import ArmEnv

load_dotenv()   # このディレクトリの .env から OPENAI_API_KEY / OPENAI_BASE_URL / LLM_MODEL を読み込む

# ---- ツール定義（LLM に見せる「ロボットにできること」のカタログ） ----
TOOLS = [
    {"type": "function", "function": {
        "name": "get_scene",
        "description": "机の上の物体と、ロボットの手先の現在位置（x, y [m]）を取得する。",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "check_affordance",
        "description": "座標 (x, y) に手先を動かせるかを評価する。affordance は 0〜1 の実行可能性スコア。",
        "parameters": {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"}},
            "required": ["x", "y"]}}},
    {"type": "function", "function": {
        "name": "move_hand_to",
        "description": "手先を座標 (x, y) [m] へ移動する。物体に「触れる」ときはその物体の座標を指定する。",
        "parameters": {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"}},
            "required": ["x", "y"]}}},
]

SYSTEM_PROMPT = """あなたは机の上で作業する2関節ロボットアームの制御エージェントです。
- 物体の位置は推測せず、必ず get_scene で確認してください。
- 動かす前に check_affordance で実行可能性を確認し、実行できない指示は理由を説明してください。
- 作業が終わったら、何を実行し、何ができなかったかを日本語で簡潔に報告してください。"""


def run_agent(instruction: str, max_turns: int = 10):
    # OPENAI_API_KEY / OPENAI_BASE_URL は環境変数から自動で読み込まれる
    client = OpenAI()
    model = os.environ["LLM_MODEL"]           # OpenAI: モデル名 / Azure OpenAI: デプロイ名
    env = ArmEnv()

    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instruction}]

    for turn in range(max_turns):
        t0 = time.perf_counter()
        res = client.chat.completions.create(model=model, messages=messages, tools=TOOLS)
        llm_sec = time.perf_counter() - t0
        msg = res.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:                     # ツール呼び出しがなければ最終回答
            print(f"\n[Agent] {msg.content}")
            return msg.content

        for tc in msg.tool_calls:                  # LLM が選んだスキルを実行（Act）
            args = json.loads(tc.function.arguments or "{}")
            result = env.call(tc.function.name, args)
            print(f"[turn {turn}] (LLM {llm_sec:.1f}s) {tc.function.name}({args}) -> "
                  f"{json.dumps(result, ensure_ascii=False)}")
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(result, ensure_ascii=False)})  # 結果を観測として返す（Sense）
    print("[Agent] 最大ターン数に達しました")


if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "赤いカップ、青い箱、緑のボールの順に触ってください。"
    run_agent(text)
