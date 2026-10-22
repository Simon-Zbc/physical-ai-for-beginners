"""第2章 (1): Code as Policies ― LLM にロボット制御コードを書かせて安全に実行する"""
import ast
import math
import os
import re
import sys

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from tabletop_env import SkillError, TabletopEnv

load_dotenv()   # このディレクトリの .env から OPENAI_API_KEY / OPENAI_BASE_URL / LLM_MODEL を読み込む

# ---------- 1. プロンプト：API 仕様 + Few-shot 例 ----------
API_DOC = '''
# 使える関数（これ以外の関数・モジュールは使用禁止。import も禁止）
get_object_names() -> list[str]        # シーン内の物体名
get_obj_pos(name: str) -> [x, y]       # 物体の位置 [m]。ロボットの基部が原点
is_reachable(x: float, y: float) -> bool  # 手先が届く位置か
holding() -> str | None                # 掴んでいる物体名
pick(name: str) -> None                # 物体を掴む（失敗すると例外）
place(x: float, y: float) -> None      # 掴んでいる物体を置く（失敗すると例外）
say(text: str) -> None                 # ユーザーへの報告
# 数値計算には np（NumPy）と math を使ってよい
'''

FEW_SHOT = '''
# 指示: 赤いブロックをトレイに置いて
pick("red_block")
place(*get_obj_pos("tray"))

# 指示: 青いブロックを赤いブロックの右に 10cm ずらして置いて
x, y = get_obj_pos("red_block")
pick("blue_block")
place(x, y - 0.10)   # ロボットから見て右 = y のマイナス方向

# 指示: 届くブロックを全部トレイに入れて
tx, ty = get_obj_pos("tray")
for name in get_object_names():
    if not name.endswith("_block"):
        continue
    if not is_reachable(*get_obj_pos(name)):
        say(f"{name} は届かないのでスキップします")
        continue
    pick(name)
    place(tx, ty)
'''

SYSTEM_PROMPT = f"""あなたは2関節ロボットアームを制御する Python コードを書くアシスタントです。
ユーザーの指示を実行するコードだけを ```python ``` ブロックで1つ出力してください。
{API_DOC}
# 例
{FEW_SHOT}"""


# ---------- 2. 生成コードの静的検査（AST） ----------
FORBIDDEN_NAMES = {"eval", "exec", "compile", "open", "__import__", "getattr",
                   "setattr", "delattr", "globals", "locals", "vars", "input"}


def validate(code: str) -> None:
    """危険な構文を含むコードを実行前に拒否する"""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("import は禁止されています")
        if isinstance(node, (ast.While, ast.AsyncFunctionDef, ast.Await)):
            raise ValueError(f"{type(node).__name__} は禁止されています（無限ループ防止）")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ValueError(f"{node.id} は使用禁止です")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("アンダースコアで始まる属性へのアクセスは禁止です")


# ---------- 3. サンドボックス実行 ----------
SAFE_BUILTINS = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
                 for k in ["abs", "min", "max", "sum", "len", "range", "enumerate", "zip",
                           "sorted", "round", "float", "int", "str", "list", "dict",
                           "print", "isinstance", "any", "all"]}


def execute(code: str, env: TabletopEnv, max_skill_calls: int = 20) -> dict:
    """ホワイトリストの関数だけを渡して実行する。ロボットを動かすスキルの呼び出し回数にも上限を設ける"""
    validate(code)
    reports, calls = [], {"n": 0}

    def limited(fn):
        def wrapper(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] > max_skill_calls:
                raise SkillError(f"スキル呼び出しが上限 {max_skill_calls} 回を超えました")
            return fn(*args, **kwargs)
        return wrapper

    api = env.api()
    for name in ["pick", "place"]:                    # 物理的に動くスキルだけ回数を制限
        api[name] = limited(api[name])
    api["say"] = reports.append
    sandbox = {"__builtins__": SAFE_BUILTINS, "np": np, "math": math, **api}
    try:
        exec(compile(code, "<llm_policy>", "exec"), sandbox)
        return {"success": True, "reports": reports, "log": list(env.log)}
    except Exception as e:                            # 失敗理由は LLM へのフィードバックに使う
        return {"success": False, "error": f"{type(e).__name__}: {e}",
                "reports": reports, "log": list(env.log)}


# ---------- 4. LLM 呼び出し ----------
def generate_code(messages) -> str:
    client = OpenAI()   # OPENAI_API_KEY / OPENAI_BASE_URL を環境変数から読む
    res = client.chat.completions.create(model=os.environ["LLM_MODEL"], messages=messages)
    text = res.choices[0].message.content
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def describe_state(env: TabletopEnv) -> str:
    pos = {n: env.get_obj_pos(n) for n in env.get_object_names()}
    return f"物体の位置: {pos} / 掴んでいる物体: {env.holding()}"


def run(instruction: str, env: TabletopEnv | None = None, max_retries: int = 2) -> dict:
    """コードを生成して実行。失敗したらエラーと現在の状態を返して書き直させる"""
    env = env or TabletopEnv()
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"# 指示: {instruction}"}]
    for attempt in range(max_retries + 1):
        code = generate_code(messages)
        print(f"----- 生成コード (attempt {attempt}) -----\n{code}\n")
        try:
            result = execute(code, env)
        except (ValueError, SyntaxError) as e:        # 静的検査で拒否された
            result = {"success": False, "error": f"検査で拒否: {e}", "log": list(env.log)}
        print(f"----- 実行結果 -----\n{result}\n")
        if result["success"]:
            return result
        messages += [{"role": "assistant", "content": f"```python\n{code}\n```"},
                     {"role": "user", "content":
                      f"実行に失敗しました: {result['error']}\n"
                      f"現在の状態: {describe_state(env)}\n"
                      "すでに実行済みの動作は繰り返さず、残りを実行するコードを書き直してください。"}]
    return result


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "トレイから一番遠いブロックから順に、届くものをトレイに入れて")
