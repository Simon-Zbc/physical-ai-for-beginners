# 第10章: 模倣学習（IL）

机の上に障害物を1つ置いた2関節アームの並列環境で、脚本示範者（手書きの経路計画）から示範を集め、BC・チャンク（Temporal Ensembling）・Diffusion Policy の3つの方策を学習します。障害物は左からも右からも避けられるので、示範には「同じ場所から左へ行く例」と「右へ行く例」が混ざります。この状況で3つの方策がどう振る舞うかを、同じ条件で比べます。CPU だけで動作し、GPU は不要です。

## 動作環境

リポジトリ本体の依存関係（NumPy・Pillow など）に加えて、PyTorch が必要です。PyTorch は CPU 版で十分です。この章では MuJoCo・Stable-Baselines3・Gymnasium は使いません。

```bash
# uv の場合
uv sync --extra il

# pip の場合：先に CPU 版の PyTorch を入れる
# （順番が逆だと、Linux では依存関係として GPU 版の大きなパッケージが入ってしまう）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pillow
```

Pillow は軌跡の図（`plot_trajectories.py`）にだけ使います。

検証は PyTorch 2.14.1（CPU）/ NumPy 2.5.4 / Pillow 13.0.0 / Python 3.14.6 / Windows 11 で行いました。

## ファイル

| ファイル | 内容 |
|---|---|
| `env.py` | 障害物つき 2関節アームの並列環境。第9章の `vec_env.py` の動力学をコピーし、行動を「目標関節角 `q_des`」（内部の PD 制御がトルクに変換）に、報酬をなしに変更。障害物は固定で、手先が触れると衝突として終了。結果（成功・衝突・時間切れ）と、障害物を通った側を `info` で返す |
| `expert.py` | 脚本示範者。起点 → 経由点 → 目標 をなめらかな曲線でつなぐ。経由点は障害物の左か右。経由点で一時停止する版もある |
| `collect_demos.py` | 示範者を並列に走らせて、観測と行動の組を `data/` に保存する |
| `policies.py` | 3つの方策のネットワーク・実行器・保存と読み込み。BC は次の1ステップ、チャンクは未来 16 ステップを予測して Temporal Ensembling で平均、Diffusion はノイズからチャンクを DDPM で生成する |
| `train.py` | 示範から方策を学習し、`models/` に保存して、その場で評価する |
| `evaluate.py` | 方策の評価（成功率・衝突率・時間切れ率・障害物を通った側・推論時間）。学習とは別のシードで、1024 環境 × 600 ステップ |
| `experiments.py` | 3つの方策を同じ示範で学習・評価して並べる実験（E1: 左右が混ざった示範、E1b: 毎回同じ起点、E2: 経由点で止まる示範とその対照）を、複数のシードで行う |
| `plot_trajectories.py` | 手先の軌跡を俯瞰図の PNG に描く（Pillow だけを使う） |

生成物は `data/`（示範）、`models/`（モデル）、`plots/`（図）に出力されます。この3つのディレクトリは `.gitignore` に入っています。

## 実行方法

```bash
cd chapters/ch10_imitation_learning

# 1. 示範を集める（それぞれ数秒）
python collect_demos.py                        # 左右 50/50 の示範 200 本 -> data/demos_mix.npz
python collect_demos.py --sides left           # 左だけ通る示範 -> data/demos_left.npz
python collect_demos.py --sides left --pause   # 左だけ + 経由点で止まる示範 -> data/demos_left_pause.npz

# 2. 3つの方策を学習する（学習のあと、その場で評価も表示される）
python train.py bc                             # 筆者の環境で約 1 分
python train.py chunk                          # 筆者の環境で約 1 分
python train.py diffusion                      # 学習は約 30 秒、評価は約 40 秒（サンプルを毎回生成するため）

# 3. 学習済みの方策をまとめて評価する（ファイル名を並べる。Windows のシェルは `*` を展開しない）
python evaluate.py models/bc_demos_mix.pt models/chunk_demos_mix.pt models/diffusion_demos_mix.pt

# 4. 実験（シード 0, 1, 2 で E1 / E1b / E2 / E2 対照）
python experiments.py --seeds 0 1 2            # 筆者の環境で約 12 分（CPU が混んでいたときは約 26 分。下の注意を参照）

# 5. 軌跡を図にする（plots/ に PNG が出る）
python plot_trajectories.py --demos data/demos_mix.npz
python plot_trajectories.py models/bc_demos_mix_s0.pt models/chunk_demos_mix_s0.pt models/diffusion_demos_mix_s0.pt --same-start
```

`experiments.py` は、モデルを `models/<方策>_<示範>_s<シード>.pt` という名前で保存します（手順 5 の2つ目のコマンドは、これを読み込みます）。

時間がかかる場合は、`python train.py diffusion --epochs 50` や `python experiments.py --seeds 0 --epochs 5` のようにエポック数やシードを減らせます（動作確認用で、結果は本来のものとは変わります）。

Windows のターミナルで日本語の出力が文字化けする、または出力をファイルやパイプに流したときに `UnicodeEncodeError` になる場合は、環境変数 `PYTHONIOENCODING=utf-8` を設定してください。

## 結果についての注意

- 学習は `torch.set_num_threads(1)` とシードの固定により、同じ環境では同じ結果になります（筆者の環境で、`train.py` を2回実行した出力の成功率・左右の件数が一致）。ただし、PyTorch・NumPy のバージョンや CPU が変わると、浮動小数点の計算順序の違いで結果が変わることがあります
- **結果はシードによって変わります**。左右が混ざった示範（`demos_mix`）で学習した BC は、筆者の環境では、どのシードでも「ほぼ片側だけ」を選びましたが、どちら側になるかはシード次第でした（障害物を通った左／右の件数：シード 0 は 2343 / 52、シード 1 は 6 / 7984、シード 2 は 6119 / 24）。1回の学習結果だけで「BC はこうなる」と言い切らず、複数のシードで見てください。`experiments.py` が複数のシードの平均と最小〜最大を出力するのはこのためです
- 同じく筆者の環境で、チャンクと Diffusion は E1 で 99.7〜100.0% の成功率でした。Diffusion はどのシードでも左右の両方を使いました（左の割合は 33.7〜81.7%）。チャンクも左右の両方を使うシードがありますが、これは確率的に選んでいるのではなく、起点・目標が数 cm ずれると決定的に別の側へ行く、という振る舞いです。本当に「同じ場所から両方に行ける」のは Diffusion だけです（E1b：全エピソードを同じ起点から走らせたとき、Diffusion は3シードとも左右の両方を使い、BC とチャンクは毎回同じ動きしかしませんでした）
- BC はときどき、途中で止まったまま時間切れになります（筆者の環境で、E1 のシード 0 では 40.5%）。左だけの示範で学習した場合でも同様のことが起きる（E2 対照：シード 1 で 36.8%、シード 2 で 26.4%）ので、「左右の平均化」とは別の弱点です。また、左右が混ざった示範（E1）で学習した BC の衝突率は、筆者の環境では3シードとも 0.0% でした。「平均化して障害物にぶつかる」という単純な図式にはなりませんでした
- 経由点で止まる示範（E2）では、現在の状態だけを見る BC は、その地点で「止まれ」を学んでしまい、動き出せなくなりました（時間切れ 42.2% / 100.0% / 96.7%）。未来 16 ステップをまとめて出すチャンクは、3シードとも 100.0% 成功でした（Temporal Ensembling あり。補足として Temporal Ensembling を切って評価すると、チャンクの方策も大半が時間切れになりました。記事 5.6 節を参照）
- 学習時間・推論時間はハードウェアと負荷に依存します。筆者の環境で CPU が空いていたときは、`experiments.py`（E2 対照を含む）が約 12 分でした。1回の推論は、CPU が空いていた別の実行（E2 対照を加える前のコード）で、BC が 2.6〜3.3 ms、Diffusion が 54.6〜61.7 ms（1024 環境ぶんの1回の呼び出し）でした。同じ環境でほかの処理が動いていたときは、E2 対照を含めて約 26 分になり、Diffusion は 100 ms を超えました（成功率などの数値は、どの実行でも一致しました）。Diffusion は DDPM の 100 ステップを毎回回すので、BC やチャンクの約 20 倍の時間がかかります
- 示範者自身の成績は、E1・E2・E2対照では成功率 100.0%・衝突 0 本でした（E1 の `示範者` 行）

## 演習（元記事より）

1. `policies.py` の `CHUNK`（チャンクの長さ、既定は 16）を 4 や 32 に変えて `python train.py chunk` を実行し、成功率と軌跡のなめらかさがどう変わるか確かめる（`CHUNK = 1` は Temporal Ensembling なしの BC に近い）。注意：`CHUNK` は Diffusion のネットワークの形も変え、`CHUNK` は `N_ACTION`（8）以上である必要があります（4 を試すときは Diffusion を使わないか、`N_ACTION` も下げる）。保存済みのチャンク・Diffusion の `.pt` は読み込めなくなるので学習し直し、試したあとは 16 に戻してください
2. `collect_demos.py` で左右の比率を変えた示範を作り（例：左 90% / 右 10%。`rng.choice([-1, 1], episodes)` に `p=[0.9, 0.1]` を渡す）、BC・チャンク・Diffusion が、右に行く割合をどこまで再現するか比べる
3. DAgger を試す。学習済みの BC を走らせ、その途中の状態（手先の位置）から目標まで、脚本示範者で `plan` を作り直してラベルとして付け、示範に追加して再学習する。BC の「止まったまま」は減るか
4. Diffusion の拡散ステップ数 `T_DIFF`（既定は 100）を 50、20、10 に減らして学習し、成功率と推論時間（`evaluate.py` の推論 [ms] の列）の関係を調べる
5. （発展）第9章の PPO を、この障害物つきの環境（`env.py`）で学習し、報酬を自分で設計して、模倣学習と成功率・動きのなめらかさ・必要な試行回数を比べる
