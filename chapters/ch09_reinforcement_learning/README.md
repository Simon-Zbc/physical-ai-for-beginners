# 第9章: 強化学習（RL）

第7章の2関節アームの Reach 並列環境を、Stable-Baselines3 の PPO で学習します。第0章のヤコビ転置法（手書きの制御器）と成功率・動きのなめらかさを比べ、報酬の書き方を変えたときに何が起きるかを実験します。CPU だけで動作し、GPU は不要です。

## 動作環境

リポジトリ本体の依存関係に加えて、Stable-Baselines3・Gymnasium・PyTorch が必要です。PyTorch は CPU 版で十分です。

```bash
# uv の場合
uv sync --extra rl

# pip の場合：先に CPU 版の PyTorch を入れる
# （順番が逆だと、Linux では依存関係として GPU 版の大きなパッケージが入ってしまう）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy stable-baselines3 gymnasium
```

検証は Stable-Baselines3 2.9.0 / Gymnasium 1.4.0 / PyTorch 2.14.1（CPU）/ NumPy 2.5.4 / Python 3.14 / Windows 11 で行いました。

## ファイル

| ファイル | 内容 |
|---|---|
| `vec_env.py` | 第7章の Reach 並列環境のコピー。報酬を `reward` 引数で切り替えられるようにし、`step()` が `terminated`（成功）/ `truncated`（時間切れ）/ リセット前の観測を返すように変更。動力学・観測・成功判定は第7章と同じ |
| `sb3_env.py` | `VecReachEnv` を Stable-Baselines3 の `VecEnv` として使うためのアダプタ |
| `evaluate.py` | 方策の評価（成功率・平均ステップ数・トルク変化率）。学習とは別のシードで、1024 環境 × 10 秒 |
| `train_ppo.py` | PPO で学習し、`models/` に保存する |
| `reward_experiments.py` | 5種類の報酬（`dense` / `no_action_penalty` / `sparse` / `progress` / `potential`）で同じ条件の学習・評価を行う |

## 実行方法

```bash
cd chapters/ch09_reinforcement_learning
python evaluate.py                          # ヤコビ転置法だけを評価（数秒）
python train_ppo.py                         # PPO を 300 万ステップ学習（筆者の環境で約 5 分）
python train_ppo.py --no-normalize          # 観測の正規化なしで学習（失敗例）
python evaluate.py models/ppo_dense.zip models/ppo_dense_nonorm.zip
python reward_experiments.py                # 報酬 5 種類 × 300 万ステップ（筆者の環境で 10〜20 分）
```

時間がかかる場合は `--steps 1000000` のように学習ステップ数を減らせます（成功率は下がります）。

Windows のターミナルで日本語の出力が文字化けする、または `UnicodeEncodeError` になる場合は、環境変数 `PYTHONIOENCODING=utf-8` を設定してください。

## 結果についての注意

- 学習は `torch.set_num_threads(1)` とシードの固定により、同じ環境では同じ結果になります（筆者の環境で、2回の実行の出力が完全に一致）。ただし、PyTorch・NumPy のバージョンや CPU が変わると、浮動小数点の計算順序の違いで結果が変わることがあります
- 特に **観測の正規化なし**（`--no-normalize`）の結果はシードによるばらつきが大きく、筆者の環境ではシード 0 で 45.7%、シード 1 で 28.7% でした。正規化ありでは、どちらのシードでも 94〜96% でした
- 学習時間はハードウェア依存です。小さなネットワークでは PyTorch のスレッドを増やしても速くならず、筆者の環境では、既定のスレッド数（8）より 1 スレッドのほうが約 2.5 倍速くなりました（20 万ステップで 21.2 秒 → 8.6 秒）

## 演習（元記事より）

1. `vec_env.py` の `dense` 報酬で、行動のペナルティの係数 `0.01` を `0.1`、`1.0` と大きくすると、成功率とトルク変化率はどう変わるか。また、ペナルティを前のステップとの差 Σ(a_t − a_{t−1})² に変えて比べる
2. `train_ppo.py` で、`PPO_KWARGS` の `clip_range` を 0.2 から 1.0 に広げたり、`n_epochs` を 30 に増やしたりして、学習が不安定になるか確かめる
3. Stable-Baselines3 の `SAC` で同じタスクを学習し、同じ環境ステップ数での成功率を PPO と比べる（ヒント：SAC は環境数を少なく、例えば 4 にする）
4. `potential` 報酬の `SUCCESS_BONUS` を 0 にすると、何が起きるか
5. （発展）`_reset_idx` でリンクの質量や関節の粘性を環境ごとにランダムにして学習し、ランダム化しない方策と、質量を変えた環境での成功率を比べる（第18章 Sim2Real の予習）
