"""第2章 (2): 階層型プランニング ― Planner が分解し、Code as Policies が実行し、結果で再計画する"""
import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

import code_as_policies as cap
from tabletop_env import TabletopEnv

load_dotenv()   # このディレクトリの .env から OPENAI_API_KEY / OPENAI_BASE_URL / LLM_MODEL を読み込む

PLANNER_PROMPT = """あなたはロボットのタスクプランナーです。
ユーザーの指示を、ロボットが1つずつ実行できる短いサブタスク（日本語）に分解してください。
- 1つのサブタスクは「〇〇を掴んで△△に置く」程度の粒度にする
- 物理的に不可能なサブタスク（届かない物体の操作など）は計画に含めず、skipped に理由を書く
- 次の JSON だけを出力する: {"steps": ["...", "..."], "skipped": ["..."]}
"""


def plan(messages) -> dict:
    client = OpenAI()
    res = client.chat.completions.create(model=os.environ["LLM_MODEL"], messages=messages,
                                         response_format={"type": "json_object"})
    return json.loads(res.choices[0].message.content)


def run(instruction: str, max_replans: int = 2):
    env = TabletopEnv()
    history = []                                             # Inner Monologue：実行結果の記録
    messages = [{"role": "system", "content": PLANNER_PROMPT},
                {"role": "user", "content":
                 f"指示: {instruction}\n現在の状態: {cap.describe_state(env)}\n"
                 f"届く範囲: 基部から 0.07〜0.53 m"}]

    for replan in range(max_replans + 1):
        p = plan(messages)
        print(f"===== 計画 (replan {replan}) =====\n{json.dumps(p, ensure_ascii=False, indent=2)}\n")
        failed = None
        for step in p["steps"]:
            print(f">>> サブタスク: {step}")
            result = cap.run(step, env=env, max_retries=1)   # 下位層：コード生成と実行
            history.append({"step": step, "success": result["success"],
                            "error": result.get("error")})
            if not result["success"]:
                failed = step
                break
        if failed is None:
            print(f"===== 完了 =====\n実行ログ: {env.log}\nスキップ: {p.get('skipped')}")
            return env.log
        # 失敗したら、実行履歴と現在の状態を渡して残りを再計画する
        messages += [{"role": "assistant", "content": json.dumps(p, ensure_ascii=False)},
                     {"role": "user", "content":
                      f"サブタスク「{failed}」が失敗しました。\n"
                      f"これまでの実行結果: {json.dumps(history, ensure_ascii=False)}\n"
                      f"現在の状態: {cap.describe_state(env)}\n"
                      "残りの作業を再計画してください。"}]
    print("===== 再計画の上限に達しました =====")
    return env.log


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "テーブルを片付けて。ブロックは全部トレイに入れて")
