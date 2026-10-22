# 第7章: Isaac Sim / Isaac Lab

GPU 並列シミュレーションの考え方（状態を `(N, …)` のテンソルにまとめ、一括で計算する）を、NumPy だけで体験します。CPU で動作し、GPU や Isaac Sim は不要です。

Isaac Sim / Isaac Lab 本体（RT コア付き RTX GPU が必須）は本リポジトリでは実行していません。記事中の Isaac Lab のコマンドは Early Access 版の記載であり、正式版で変わる可能性があります。実行する場合は公式ドキュメントを確認してください。

## ファイル

| ファイル | 内容 |
|---|---|
| `vec_env.py` | N 環境を一括で扱う2関節アームの Reach 環境。メソッド構成は Isaac Lab の `DirectRLEnv`（`_apply_action` / `_get_observations` / `_get_rewards` / `_get_dones` / `_reset_idx`）に対応 |
| `benchmark.py` | 「ループで N 環境」と「一括で N 環境」のスループット比較。4096 環境で10秒分のエピソード統計も取得 |

## 実行方法

```bash
uv sync
cd chapters/ch07_isaac_lab
uv run python benchmark.py
```

スループットの絶対値は実行環境によって変わります。確認すべきは、ループ版は環境数を増やしてもほぼ一定で、一括版は環境数に応じて伸びるという傾向です。終了エピソード数・成功数はシード固定のため、環境によらず一致します（19,186 回 / 16,790 回 / 成功率 87.5%）。

## 演習（元記事より）

1. 環境数を 16384 / 65536 に増やし、スループットが頭打ちになる理由を考える
2. `scripted_policy` の Kp / Kd を変えて成功率の変化を調べる
3. `_reset_idx` で `DAMP` やリンク質量を環境ごとにランダム化する（ドメインランダマイゼーションの最小版）
4. `np.` を `torch.` に置き換え、`device="cuda"` で動かして比較する
5. （クラウドが使える場合）Isaac Lab の Cartpole を学習し、Direct ワークフローのコードと `vec_env.py` の対応を確かめる
