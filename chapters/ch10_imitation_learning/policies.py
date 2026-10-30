"""第10章: 模倣学習の3つの方策（ネットワーク・実行器・保存と読み込み）

  - BC        : 観測 → 次の1ステップの行動。MSE で学習する、いちばん素直な模倣学習
  - チャンク  : 観測 → 未来 CHUNK ステップぶんの行動。毎ステップ予測し、重なった予測を
                Temporal Ensembling で加重平均して実行する（ACT の考え方）
  - Diffusion : ノイズからチャンクを少しずつ「削り出す」生成モデル（DDPM）。N_ACTION ステップ
                実行するごとに、新しいチャンクをサンプルし直す（Diffusion Policy の考え方）

実行器は evaluate.py の約束に従う: runner(obs (N,10)) -> q_des (N,2)、runner.reset(ids)。
"""
import math

import numpy as np
import torch
from torch import nn

CHUNK = 16        # チャンクの長さ（何ステップ先まで一度に予測するか）
N_ACTION = 8      # Diffusion で、1回サンプルしたチャンクのうち実際に実行するステップ数
T_DIFF = 100      # 拡散のステップ数
TE_M = 0.01       # Temporal Ensembling の重み w = exp(-TE_M · s) の係数
CLIP_X0 = 3.0     # サンプル中の x̂0 を ±3 に切る。平均0・標準偏差1 の空間なので ±3σ
                  # （Diffusion Policy は min-max で [-1, 1] に揃えるので ±1 で切る）
OBS_DIM, ACT_DIM, HIDDEN = 10, 2, 256


def cosine_betas(n, s=0.008):
    """cosine スケジュール（Diffusion Policy と同じ）の β_t (n,) を作る

    f(t) = cos²((t/n + s)/(1 + s) · π/2) として ᾱ(t) = f(t) / f(0) になるよう β_t = 1 − f(t+1)/f(t)。
    ただし最後のほうで 1 に近づきすぎないよう β は 0.999 で頭打ちにする。
    """
    f = torch.cos((torch.arange(n + 1, dtype=torch.float64) / n + s) / (1 + s) * math.pi / 2) ** 2
    return torch.clamp(1 - f[1:] / f[:-1], max=0.999).float()


# サンプルは純粋なノイズ N(0, I) から始めるので、学習でも最後のステップでは信号がほぼ消えている（ᾱ ≈ 0）必要がある
BETAS = cosine_betas(T_DIFF)                      # 各ステップで足すノイズの分散 β_t
ALPHAS = 1.0 - BETAS
ALPHA_BARS = torch.cumprod(ALPHAS, dim=0)         # ᾱ_t = α_1 · … · α_t（元の信号がどれだけ残っているか）


class Normalizer:
    """各次元を「平均 0・標準偏差 1」に揃える。統計は学習データで1回だけ計算し、評価でも同じ値を使う"""

    def __init__(self, x):
        x = np.asarray(x, dtype=np.float64)                  # 統計は倍精度で計算する
        self.mean = x.mean(axis=0).astype(np.float32)
        self.std = np.maximum(x.std(axis=0), 1e-6).astype(np.float32)   # 一定の次元でも 0 で割らない

    def norm(self, x):
        return (x - self.mean) / self.std

    def denorm(self, x):
        return x * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}   # list にすると torch.load で安全に読める

    @classmethod
    def from_state_dict(cls, d):
        """保存した統計から作り直す（学習データは使わないので __init__ は通さない）"""
        n = cls.__new__(cls)
        n.mean = np.array(d["mean"], dtype=np.float32)
        n.std = np.array(d["std"], dtype=np.float32)
        return n


def make_chunk_labels(actions, episode_index, k):
    """各ステップのラベル = 同じエピソード内の未来 k ステップの行動 (M, k, 2)

    エピソードの終わりを越える部分は、そのエピソードの最後の行動で埋める
    （次の示範の行動が混ざらないように）。各エピソードの行は時間順に並んでいる前提。
    """
    labels = np.empty((len(actions), k, actions.shape[1]), dtype=actions.dtype)
    for e in np.unique(episode_index):
        ids = np.nonzero(episode_index == e)[0]
        a = actions[ids]
        padded = np.concatenate([a, np.repeat(a[-1:], k - 1, axis=0)])   # 最後の行動を k-1 回足す
        labels[ids] = padded[np.arange(len(a))[:, None] + np.arange(k)]  # i 行目 = padded[i : i+k]
    return labels


def mlp(n_in, n_out):
    """隠れ層 256 次元 × 3 層、ReLU の多層パーセプトロン"""
    return nn.Sequential(nn.Linear(n_in, HIDDEN), nn.ReLU(),
                         nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                         nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                         nn.Linear(HIDDEN, n_out))


class BCNet(nn.Module):
    """観測 → 次の行動 (B, 2)"""

    def __init__(self):
        super().__init__()
        self.mlp = mlp(OBS_DIM, ACT_DIM)

    def forward(self, obs):
        return self.mlp(obs)


class ChunkNet(nn.Module):
    """観測 → 未来 CHUNK ステップの行動 (B, CHUNK, 2)"""

    def __init__(self):
        super().__init__()
        self.mlp = mlp(OBS_DIM, CHUNK * ACT_DIM)

    def forward(self, obs):
        return self.mlp(obs).view(-1, CHUNK, ACT_DIM)


def time_embedding(t, dim=32):
    """拡散のステップ番号 t (B,) を、周波数の違う sin / cos の並び (B, dim) に変える"""
    freqs = torch.exp(-math.log(10000) * torch.arange(dim // 2) / (dim // 2))
    angles = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class DiffusionNet(nn.Module):
    """(観測, ノイズの乗ったチャンク, ステップ t) → 乗っているノイズの推定 (B, CHUNK, 2)"""

    def __init__(self):
        super().__init__()
        self.mlp = mlp(OBS_DIM + CHUNK * ACT_DIM + 32, CHUNK * ACT_DIM)

    def forward(self, obs, noisy_chunk, t):
        x = torch.cat([obs, noisy_chunk.flatten(1), time_embedding(t)], dim=1)
        return self.mlp(x).view(-1, CHUNK, ACT_DIM)


def predict_x0(x, eps, t):
    """x_t とノイズの推定 ε̂ から「ノイズを全部取り除いたチャンク」x̂0 を推定し、±CLIP_X0 で切る"""
    ab = ALPHA_BARS[t]
    return ((x - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)).clamp(-CLIP_X0, CLIP_X0)


@torch.no_grad()
def sample_chunks(net, obs_n, rng_seed):
    """DDPM の逆過程: 純粋なノイズから T_DIFF ステップかけてチャンクを作る（正規化した空間で）

    同じ rng_seed なら同じ結果になる。obs_n は正規化済みの観測 (B, 10) の tensor。
    """
    g = torch.Generator().manual_seed(rng_seed)
    x = torch.randn(len(obs_n), CHUNK, ACT_DIM, generator=g)
    for t in reversed(range(T_DIFF)):
        eps = net(obs_n, x, torch.full((len(obs_n),), t))
        x0 = predict_x0(x, eps, t)                                   # ① 元のチャンクを推定（切る）
        ab, ab_prev = ALPHA_BARS[t], (ALPHA_BARS[t - 1] if t > 0 else torch.tensor(1.0))
        # ② x̂0 と今の x_t を混ぜて、1ステップ前 x_{t-1} の平均にする（DDPM の事後分布の平均）
        x = (torch.sqrt(ab_prev) * BETAS[t] / (1 - ab)) * x0 + (torch.sqrt(ALPHAS[t]) * (1 - ab_prev) / (1 - ab)) * x
        if t > 0:                                                    # ③ 最後のステップ以外はノイズを足す
            var = BETAS[t] * (1 - ab_prev) / (1 - ab)                # 事後分布の分散 β̃_t
            x = x + torch.sqrt(var) * torch.randn(x.shape, generator=g)
    return x


def _predict(net, obs_norm, obs):
    with torch.no_grad():
        return net(torch.as_tensor(obs_norm.norm(obs), dtype=torch.float32)).numpy()


class BCRunner:
    """毎ステップ、BCNet の出力をそのまま実行する（内部状態なし）"""

    def __init__(self, net, obs_norm, act_norm):
        self.net, self.obs_norm, self.act_norm = net, obs_norm, act_norm

    def __call__(self, obs):
        return self.act_norm.denorm(_predict(self.net, self.obs_norm, obs))

    def reset(self, ids):
        pass


class ChunkRunner:
    """毎ステップ チャンクを予測し、同じ時刻に向けた過去の予測と加重平均して実行する

    num[i, j] / den[i, j] は env i の「j ステップ後」に向けた予測の重みつき和と重みの和。
    ステップごとに1つ前へずらすので、num[:, 0] / den[:, 0] が今のステップの行動になる。
    """

    def __init__(self, net, obs_norm, act_norm, ensemble=True):
        self.net, self.obs_norm, self.act_norm, self.ensemble = net, obs_norm, act_norm, ensemble
        self.num = None                                         # 最初の呼び出しで env の数に合わせて作る

    def __call__(self, obs):
        if self.num is None:
            self.num = np.zeros((len(obs), CHUNK, ACT_DIM)); self.den = np.zeros((len(obs), CHUNK))
            self.step = np.zeros(len(obs), dtype=int)           # リセットから数えたステップ数 s
        chunk = self.act_norm.denorm(_predict(self.net, self.obs_norm, obs))   # (N, CHUNK, 2)
        if not self.ensemble:
            return chunk[:, 0]
        # 早い時刻の予測ほど重い（ACT と同じ）。w の絶対値は s とともに小さくなるが、
        # num / den の比で打ち消し合うので、効くのは予測どうしの相対的な重みだけ
        w = np.exp(-TE_M * self.step)
        self.num += w[:, None, None] * chunk
        self.den += w[:, None]
        action = self.num[:, 0] / self.den[:, 0, None]
        self.num = np.roll(self.num, -1, axis=1); self.num[:, -1] = 0   # 1ステップ進める
        self.den = np.roll(self.den, -1, axis=1); self.den[:, -1] = 0
        self.step += 1
        return action

    def reset(self, ids):
        if self.num is not None:                                # 終わった env のぶんだけ捨てる
            self.num[ids] = 0; self.den[ids] = 0; self.step[ids] = 0


class DiffusionRunner:
    """env ごとに行動のキューを持ち、空になった env だけチャンクをサンプルして先頭 N_ACTION 個を入れる"""

    def __init__(self, net, obs_norm, act_norm, seed=0):
        self.net, self.obs_norm, self.act_norm = net, obs_norm, act_norm
        self.rng = np.random.default_rng(seed)                  # サンプルごとの乱数シードを決める
        self.queue = None

    def __call__(self, obs):
        if self.queue is None:
            self.queue = np.zeros((len(obs), N_ACTION, ACT_DIM), dtype=np.float32)
            self.left = np.zeros(len(obs), dtype=int)           # キューに残っている行動の数
        empty = np.nonzero(self.left == 0)[0]
        if len(empty):
            obs_n = torch.as_tensor(self.obs_norm.norm(obs[empty]), dtype=torch.float32)
            chunk = sample_chunks(self.net, obs_n, int(self.rng.integers(2**31))).numpy()
            self.queue[empty] = self.act_norm.denorm(chunk[:, :N_ACTION])
            self.left[empty] = N_ACTION
        action = self.queue[np.arange(len(obs)), N_ACTION - self.left]
        self.left -= 1
        return action

    def reset(self, ids):
        if self.queue is not None:
            self.left[ids] = 0                                  # 次の呼び出しで、この env だけ新しくサンプルする


NETS = {"bc": BCNet, "chunk": ChunkNet, "diffusion": DiffusionNet}
RUNNERS = {"bc": BCRunner, "chunk": ChunkRunner, "diffusion": DiffusionRunner}


def save_checkpoint(path, kind, net, obs_norm, act_norm, meta):
    """重みと、学習データの正規化の統計をいっしょに保存する（評価で同じ統計を使うため。第9章の教訓）"""
    torch.save({"kind": kind, "net": net.state_dict(), "obs_norm": obs_norm.state_dict(),
                "act_norm": act_norm.state_dict(), "meta": meta}, path)


def load_runner(path, seed=0):
    """保存した方策を読み込み、kind に合った実行器を返す（seed は Diffusion のサンプルの乱数にだけ使う）"""
    ck = torch.load(path)
    net = NETS[ck["kind"]]()
    net.load_state_dict(ck["net"])
    net.eval()
    args = (net, Normalizer.from_state_dict(ck["obs_norm"]), Normalizer.from_state_dict(ck["act_norm"]))
    if ck["kind"] == "diffusion":
        return DiffusionRunner(*args, seed=seed)
    return RUNNERS[ck["kind"]](*args)
