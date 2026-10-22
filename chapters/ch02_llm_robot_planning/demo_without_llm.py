"""LLM を使わずに、実行基盤（検査 + サンドボックス + スキル）だけを確認する"""
import code_as_policies as cap
from tabletop_env import TabletopEnv

# LLM が生成した「つもり」のコード（Few-shot の3つ目の例と同じ）
POLICY = '''
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

env = TabletopEnv()
result = cap.execute(POLICY, env)
print("success:", result["success"])
print("reports:", result["reports"])
for line in result["log"]:
    print("  ", line)

# 危険なコードは実行前に拒否される
for bad in ['import os', 'open("secret.txt")', 'while True:\n    pick("red_block")']:
    try:
        cap.validate(bad)
    except ValueError as e:
        print(f"拒否: {bad!r:40} -> {e}")
