# 第5章: 制御と軌道生成

PID 制御・重力補償・軌道生成を MuJoCo で実装し、「関節角の目標を、どうなめらかに正確に実行するか」を数値で確認します。LLM は不要で、すべてローカルで実行できます。このディレクトリ内で完結しています（第3・4章のファイルは不要）。

## ファイル

| ファイル | 内容 |
|---|---|
| `pid_pendulum.py` | 重力のある1関節アームで P / PD / PID / PD+重力補償を比較。積分ワインドアップとその対策も確認 |
| `arm_model.py` | 重力・質量・トルク上限付きの4自由度アームと、PD + 重力補償の制御則（`armature` でモーター側の慣性を追加） |
| `trajectory.py` | ステップ・台形速度・5次多項式の3つの軌道を、追従誤差・トルク・トルク変化率で比較 |
| `chunk_execution.py` | 10Hz の方策出力を 1kHz の制御ループで実行する（ゼロ次ホールド vs 線形補間） |

## 実行方法

```bash
uv sync
cd chapters/ch05_control_trajectory
uv run python pid_pendulum.py
uv run python trajectory.py
uv run python chunk_execution.py
```

## 演習（元記事より）

1. 減衰比 ζ = 0.5 / 1.0 / 2.0 になるように Kd を決め、応答を比べる
2. 重力補償に使う質量をわざと 20% 小さく見積もり、定常偏差と小さな I 項の効果を確かめる
3. `trajectory.py` に、関節ごとの速度上限から移動時間 T を自動で決める機能を追加する
4. `chunk_execution.py` に、簡易版の Temporal Ensembling（過去の予測の指数加重平均）を追加する
5. `arm_model.py` の `armature` や `timestep` を小さくして、シミュレーションが不安定になる境界を調べる
