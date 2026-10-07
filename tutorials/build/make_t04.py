"""Builds T04_transformers.ipynb (Tutorial 4, Deep Learning 89-6877, Fall 2026)."""
from nbkit import Builder
from recap import make_recap

STEM = "T04_transformers"
B = Builder()
md, code = B.md, B.code

md(rf"""
# Tutorial 4: attention and the Transformer, built by hand
**Deep Learning 89-6877, Fall 2026.** TA: Tal Fiskus.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idansc/dl-course-2026/blob/main/tutorials/{STEM}.ipynb)
· [Recap slides (PDF)](https://github.com/idansc/dl-course-2026/blob/main/tutorials/recap/{STEM}_recap.pdf)

Plan for today (≈ 85 min):
1. Lecture recap: attention as gating, scaled dot-product, multi-head, the block, the 2026 block, ViT (10 min)
2. Scaled dot-product attention, causal mask, multi-head attention (10 min)
3. The modern block: RMSNorm, SwiGLU, RoPE → a char-level mini-GPT on Tiny Shakespeare (15 min)
4. Grouped-query attention and a top-k mixture-of-experts FFN (10 min)
5. Symmetry: attention is permutation-equivariant; positional encoding breaks it (5 min)
6. Where does position enter? Absolute, relative, rotary, none: five schemes trained side by side, RoPE in depth, context extension (15 min)
7. Patchify + a tiny ViT on CIFAR-10 (10 min)
8. Inside the trained mini-GPT: logit lens and attention sinks (10 min)
9. If time: train the MoE mini-GPT with and without the balancing loss

Runs on CPU (Colab or laptop, ≈ 10 min end to end). Cells marked ✏️ are for you to try.
<<STUDENT>>
**Before the tutorial:** fill in every `# TODO` (replace the `...`). Each one is followed by a check cell that prints the PyTorch answer next to yours. The solutions are presented in the tutorial.
<</STUDENT>>
""")

code(r"""
import math, os, time, urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
torch.manual_seed(0); np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device, "| torch", torch.__version__)
""")

md(r"""
## 1. Lecture recap

**Attention as gating.** Attentive pooling scores each of $n$ elements and returns a weighted sum, $z = \sum_i [g(x)]_i\, x_i$ with $g(x)=\mathrm{softmax}(\cdot)\in\Delta^{n}$. The weights are computed from the data, so the "gate" changes with every input.

**Scaled dot-product attention.** Queries $Q\in\mathbb{R}^{n_q\times d}$, keys $K\in\mathbb{R}^{n_k\times d}$, values $V\in\mathbb{R}^{n_k\times d_v}$:
$$\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\Big(\frac{QK^\top}{\sqrt d}\Big)V .$$
The $\sqrt d$ keeps the logits at unit scale: for $q,k$ with i.i.d. unit-variance entries, $\mathrm{Var}(q^\top k)=d$.
**Self-attention:** $Q=XW_Q,\ K=XW_K,\ V=XW_V$ from the same sequence $X$. **Causal (masked)** self-attention sets $E_{ij}=-\infty$ for $j>i$, so token $i$ cannot look ahead (language modeling).

**Multi-head.** $H$ heads of size $d/H$ in parallel, each with its own $W_Q^h,W_K^h,W_V^h$; concatenate and project with $W_O$. Same FLOPs as one head of size $d$.

**Transformer block (2017).** $x \leftarrow \mathrm{LN}(x+\mathrm{MHA}(x))$, $x\leftarrow\mathrm{LN}(x+\mathrm{MLP}(x))$. Attention is the only operation that mixes tokens; norm and MLP act per token.

**The 2026 block** (Llama 3 / Qwen 3 style): same skeleton, with
- **pre-norm**: $x \leftarrow x + \mathrm{Attn}(\mathrm{Norm}(x))$, $x \leftarrow x + \mathrm{FFN}(\mathrm{Norm}(x))$, so the residual stream is never normalized;
- **RMSNorm**: $\mathrm{RMSNorm}(x)=\dfrac{x}{\sqrt{\tfrac1d\sum_i x_i^2+\epsilon}}\odot g$ (no mean-centering, no bias);
- **SwiGLU**: $\mathrm{FFN}(x)=W_2\big(\mathrm{SiLU}(W_1x)\odot W_3x\big)$, hidden size $\approx\tfrac83 d$ to keep the parameter count of a $4d$ MLP;
- **RoPE**: rotate each pair of coordinates of $q$ and $k$ by an angle $m\theta_j$ at position $m$; then $\langle R_m q, R_n k\rangle$ depends only on $n-m$;
- **GQA**: $H$ query heads share $H_{kv}<H$ key/value heads (KV cache $H/H_{kv}$ times smaller);
- **MoE**: replace the FFN by $E$ expert FFNs and a router that sends each token to its top-$k$; an auxiliary loss keeps the experts evenly used.

**Symmetry.** Without positions, self-attention is **permutation-equivariant**: $f(PX)=P\,f(X)$. A sequence is then a set. Positional encoding (learned, sinusoidal or RoPE) breaks this on purpose.

**ViT.** Cut the image into $16\times16$ patches, flatten, project linearly → tokens; add a [CLS] token and learned position embeddings; run a Transformer encoder; classify from [CLS].
""")

md(r"""
## 2. Scaled dot-product attention and multi-head attention

Shapes: `q (B, H, Tq, d)`, `k, v (B, H, Tk, d)`. A boolean `mask` is `True` where attention is **allowed**, as in PyTorch.
""")

code(r"""
def attention(q, k, v, mask=None):
    # returns (output, attention weights)
    #>> scores = q k^T / sqrt(d); set disallowed positions to -inf; softmax over keys; weights @ v
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    w = scores.softmax(dim=-1)
    #<<
    return w @ v, w

def causal_mask(T, device=None):
    return torch.ones(T, T, dtype=torch.bool, device=device).tril()   # True on and below the diagonal
""")

code(r"""
torch.manual_seed(0)
q, k, v = (torch.randn(2, 4, 10, 16, dtype=torch.float64) for _ in range(3))
out, w = attention(q, k, v)
print("no mask: max |mine − torch| =", (out - F.scaled_dot_product_attention(q, k, v)).abs().max().item())
out, w = attention(q, k, v, causal_mask(10))
print("causal : max |mine − torch| =", (out - F.scaled_dot_product_attention(q, k, v, is_causal=True)).abs().max().item())
print("rows sum to 1:", torch.allclose(w.sum(-1), torch.ones(1, dtype=torch.float64)), "| weight above diagonal:", w.triu(1).abs().max().item())
""")

md(r"""
**Multi-head attention** with one fused `qkv` projection. Split the last dimension into heads, attend per head, merge back:
`(B, T, D) → (B, H, T, D/H) → attention → (B, T, D)`.
""")

code(r"""
class MHA(nn.Module):
    def __init__(self, d, n_heads, bias=False):
        super().__init__()
        self.h = n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=bias)
        self.proj = nn.Linear(d, d, bias=bias)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)                       # each (B, T, D)
        #>> reshape q, k, v to (B, H, T, D/H); y = attention(...); merge heads back to (B, T, D)
        q, k, v = (t.view(B, T, self.h, D // self.h).transpose(1, 2) for t in (q, k, v))
        y, _ = attention(q, k, v, mask)
        y = y.transpose(1, 2).reshape(B, T, D)
        #<<
        return self.proj(y)
""")

md(r"""
**Check against `nn.MultiheadAttention`.** Its `in_proj_weight` stacks $W_Q, W_K, W_V$ in the same order as our `qkv`, so copying the weights must give the same output. Note PyTorch's `attn_mask` convention for this module is the opposite: `True` = **blocked**.
""")

code(r"""
torch.manual_seed(0)
D, H, T = 32, 4, 12
mine = MHA(D, H, bias=True).double()
ref = nn.MultiheadAttention(D, H, batch_first=True, dtype=torch.float64)
with torch.no_grad():
    ref.in_proj_weight.copy_(mine.qkv.weight); ref.in_proj_bias.copy_(mine.qkv.bias)
    ref.out_proj.weight.copy_(mine.proj.weight); ref.out_proj.bias.copy_(mine.proj.bias)
x = torch.randn(3, T, D, dtype=torch.float64)
y_ref, _ = ref(x, x, x, attn_mask=~causal_mask(T), need_weights=False)
print("causal MHA: max |mine − nn.MultiheadAttention| =", (mine(x, causal_mask(T)) - y_ref).abs().max().item())
""")

md(r"""
✏️ Remove the `/ math.sqrt(...)` and look at `w.max(-1).values.mean()` for `d = 16` vs `d = 1024`. Without the scaling, the softmax saturates to one-hot as $d$ grows and its gradient vanishes.
""")

md(r"""
**Past exam question (Moed B, 2026)**

A single Transformer encoder block, INPUT → MHA → FFN:
- INPUT: a sequence of length $T=20$ with embedding dimension $d_{\text{model}}=64$.
- Multi-head self-attention with $h=4$ heads; assume $d_k = d_v = d_{\text{model}}/h$ in every head.
- FFN: a linear layer to $d_{ff}=128$, then a linear layer back to 64.
- Ignore LayerNorm and biases.

Fill in the table (activation dimensions and number of learned parameters for INPUT $20\times64$ / 0, MHA, FFN). Show the MHA parameter computation explicitly.

<<STUDENT>>
✏️ Your answer:
<</STUDENT>>
<<SOLUTION>>
$d_k=d_v=64/4=16$. Per head $W_Q^i, W_K^i, W_V^i\in\mathbb{R}^{64\times16}$; over the 4 heads these stack into three $64\times64$ matrices, plus the output projection $W_O\in\mathbb{R}^{64\times64}$: MHA $=4\cdot64\cdot64 = \mathbf{16{,}384}$, output $20\times64$. The number of heads does not change the count. FFN $= 64\cdot128+128\cdot64 = \mathbf{16{,}384}$, output $20\times64$.
<</SOLUTION>>
""")

code(r"""
d_m, n_h, d_ff, T_q = 64, 4, 128, 20
#>> MHA and FFN parameter counts by formula (no biases)
p_mha = 3 * d_m * d_m + d_m * d_m
p_ffn = d_m * d_ff + d_ff * d_m
#<<
mha_q = MHA(d_m, n_h, bias=False)
ffn_q = nn.Sequential(nn.Linear(d_m, d_ff, bias=False), nn.ReLU(), nn.Linear(d_ff, d_m, bias=False))
x_q = torch.randn(1, T_q, d_m)
print(f"MHA: yours {p_mha}, module {sum(p.numel() for p in mha_q.parameters())}, output {tuple(mha_q(x_q).shape[1:])}")
print(f"FFN: yours {p_ffn}, module {sum(p.numel() for p in ffn_q.parameters())}, output {tuple(ffn_q(mha_q(x_q)).shape[1:])}")
""")

md(r"""
**Past exam question (Moed C, 2026)**

True or false: in a Transformer with self-attention layers, changing the input sequence length $T$ (keeping the same embedding dimension) does not change the number of learned parameters of the model.

<<STUDENT>>
✏️ Your answer:
<</STUDENT>>
<<SOLUTION>>
**True.** $W_Q,W_K,W_V,W_O$ and the FFN act on each token's $d$-dimensional vector and are shared over positions; $T$ only sets the size of the activations and of the $T\times T$ attention matrix (the block above runs unchanged on $T=2000$). One caveat the exam does not raise: a **learned absolute position table** (GPT-2, the ViT in section 7; section 6 compares it with the alternatives) has $T_{\max}\cdot d$ parameters, so changing the maximum length changes that table. RoPE and sinusoidal positions have no parameters.
<</SOLUTION>>
""")

code(r"""
#>> run the same MHA + FFN on T = 20 and T = 2000; print the output shapes and the parameter count
for T_len in [20, 2000]:
    out_q = ffn_q(mha_q(torch.randn(1, T_len, d_m)))
    print(f"T = {T_len:4d}: output {tuple(out_q.shape[1:])}, parameters {sum(p.numel() for p in [*mha_q.parameters(), *ffn_q.parameters()])}")
#<<
""")

md(r"""
## 3. The modern block → a mini-GPT

Three components, each checked on its own, then assembled.
""")

code(r"""
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight  # @student: return None  # TODO: x / sqrt(mean(x^2) + eps) * weight

class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.w1 = nn.Linear(d, hidden, bias=False)   # gate branch
        self.w3 = nn.Linear(d, hidden, bias=False)   # linear branch
        self.w2 = nn.Linear(hidden, d, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))  # @student: return None  # TODO: W2 (SiLU(W1 x) ⊙ W3 x)

x = torch.randn(4, 7, 64, dtype=torch.float64)
mine, ref = RMSNorm(64).double(), nn.RMSNorm(64, eps=1e-6).double()
with torch.no_grad(): mine.weight.uniform_(0.5, 1.5); ref.weight.copy_(mine.weight)
print("RMSNorm: max |mine − nn.RMSNorm| =", (mine(x) - ref(x)).abs().max().item())
ffn = SwiGLU(64, 171).double()
h = x @ ffn.w1.weight.T
print("SwiGLU : max |mine − h·sigmoid(h)·(x W3) W2| =",
      (ffn(x) - ((h * torch.sigmoid(h)) * (x @ ffn.w3.weight.T)) @ ffn.w2.weight.T).abs().max().item())
""")

md(r"""
**RoPE.** Split the head dimension into $d/2$ pairs $(x_{2j}, x_{2j+1})$. At position $m$, rotate pair $j$ by angle $m\theta_j$, $\theta_j = 10000^{-2j/d}$:
$$\begin{pmatrix}x'_{2j}\\x'_{2j+1}\end{pmatrix}=\begin{pmatrix}\cos m\theta_j & -\sin m\theta_j\\ \sin m\theta_j & \cos m\theta_j\end{pmatrix}\begin{pmatrix}x_{2j}\\x_{2j+1}\end{pmatrix}.$$
Applied to $q$ and $k$ only (not $v$). Since $R_m^\top R_n = R_{n-m}$, the score $\langle R_m q, R_n k\rangle$ depends on the offset $n-m$, not on $m$ and $n$ separately. That is the check.
""")

code(r"""
def rope_cache(T, head_dim, base=10000.0):
    theta = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)   # (head_dim/2,)
    angles = torch.outer(torch.arange(T, dtype=torch.float64), theta)                  # (T, head_dim/2)
    return angles.cos(), angles.sin()

def apply_rope(x, cos, sin, pos=None):
    # x (..., T, head_dim); pos: optional (T,) positions, default 0..T-1
    T = x.shape[-2]
    c, s = (cos[:T], sin[:T]) if pos is None else (cos[pos], sin[pos])
    c, s = c.to(x.dtype), s.to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    #>> rotate each pair (x1, x2) by its angle, then interleave back to (..., T, head_dim)
    out = torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    return out.flatten(-2)
    #<<
""")

code(r"""
cos, sin = rope_cache(512, 16)
q1, k1 = torch.randn(16, dtype=torch.float64), torch.randn(16, dtype=torch.float64)
def score(m, n):  # <R_m q, R_n k>
    return (apply_rope(q1[None], cos, sin, torch.tensor([m])) @ apply_rope(k1[None], cos, sin, torch.tensor([n])).T).item()
print("offset 3 at (5, 8), (100, 103), (400, 403):", [round(score(m, m + 3), 6) for m in (5, 100, 400)])
print("offset 7 at (5, 12):", round(score(5, 12), 6), "  (different offset → different score)")
print("norm preserved:", torch.allclose(apply_rope(q1[None], cos, sin, torch.tensor([77])).norm(), q1.norm()))
# reference: the same rotation written with complex numbers (as in the Llama code)
zq = torch.view_as_complex(q1.view(-1, 2).contiguous()) * torch.polar(torch.ones(8, dtype=torch.float64), 77 * (10000.0 ** (-torch.arange(0, 16, 2, dtype=torch.float64) / 16)))
print("max |mine − complex form| =", (apply_rope(q1[None], cos, sin, torch.tensor([77]))[0] - torch.view_as_real(zq).flatten()).abs().max().item())
""")

md(r"""
**The block.** Pre-norm, RMSNorm, RoPE inside attention, SwiGLU. Flags `causal` and `rope` let us switch both off in section 5.
For training we call `F.scaled_dot_product_attention` (fused, faster); `return_attn=True` routes through our `attention` so we can look at the weights in section 8.
""")

code(r"""
class Attention(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads=None):
        super().__init__()
        self.h, self.h_kv = n_heads, n_kv_heads or n_heads
        self.hd = d // n_heads
        self.wq = nn.Linear(d, d, bias=False)
        self.wkv = nn.Linear(d, 2 * self.h_kv * self.hd, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x, rope=None, causal=True, return_attn=False):
        B, T, D = x.shape
        q = self.wq(x).view(B, T, self.h, self.hd).transpose(1, 2)                  # (B, H, T, hd)
        k, v = self.wkv(x).view(B, T, 2, self.h_kv, self.hd).permute(2, 0, 3, 1, 4)  # (B, H_kv, T, hd) each
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        if self.h_kv != self.h:   # GQA, section 4
            k, v = repeat_kv(k, self.h // self.h_kv), repeat_kv(v, self.h // self.h_kv)
        if return_attn:
            y, w = attention(q, k, v, causal_mask(T, x.device) if causal else None)
        else:
            y, w = F.scaled_dot_product_attention(q, k, v, is_causal=causal), None
        return self.proj(y.transpose(1, 2).reshape(B, T, D)), w

def repeat_kv(t, n_rep):   # filled in section 4; identity when n_rep == 1
    return t if n_rep == 1 else t.repeat_interleave(n_rep, dim=1)  # @student: return t if n_rep == 1 else None  # TODO (section 4)

class Block(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads=None, ffn=None):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
        self.attn = Attention(d, n_heads, n_kv_heads)
        self.ffn = ffn if ffn is not None else SwiGLU(d, 64 * math.ceil(8 * d / 3 / 64))

    def forward(self, x, rope=None, causal=True, return_attn=False):
        #>> pre-norm residual block: x + attn(norm1(x)), then x + ffn(norm2(x)); return (x, attention weights)
        a, w = self.attn(self.norm1(x), rope, causal, return_attn)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        #<<
        return x, w

class MiniGPT(nn.Module):
    def __init__(self, vocab, d=128, n_layers=4, n_heads=4, n_kv_heads=None, ctx=128, ffn_fn=None):
        super().__init__()
        self.ctx = ctx
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, n_heads, n_kv_heads, ffn_fn() if ffn_fn else None) for _ in range(n_layers)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        cos, sin = rope_cache(ctx, d // n_heads)
        self.register_buffer("cos", cos.float(), persistent=False)
        self.register_buffer("sin", sin.float(), persistent=False)

    def forward(self, idx, return_internals=False):
        x = self.emb(idx)
        resid, attns = [x], []
        for blk in self.blocks:
            x, w = blk(x, (self.cos, self.sin), causal=True, return_attn=return_internals)
            resid.append(x); attns.append(w)
        logits = self.head(self.norm(x))
        return (logits, resid, attns) if return_internals else logits
""")

md(r"""
**Data: Tiny Shakespeare**, 1.1M characters, 65-symbol vocabulary. Each training example is a random window of `ctx` characters; the target is the same window shifted by one.
""")

code(r"""
os.makedirs("data", exist_ok=True)
path = "data/tinyshakespeare.txt"
if not os.path.exists(path):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", path)
text = open(path).read()
chars = sorted(set(text)); vocab = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
encode = lambda s: torch.tensor([stoi[c] for c in s])
decode = lambda t: "".join(chars[i] for i in t.tolist())
data = encode(text); n_train = int(0.9 * len(data))
train_data, val_data = data[:n_train], data[n_train:]
print(f"{len(text):,} chars, vocab {vocab}")

def get_batch(src, B, T):
    ix = torch.randint(len(src) - T - 1, (B,))
    x = torch.stack([src[i:i + T] for i in ix]); y = torch.stack([src[i + 1:i + T + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def val_loss(model, n=10, B=32):
    model.eval()
    l = sum(F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item() for x, y in (get_batch(val_data, B, model.ctx) for _ in range(n))) / n
    model.train(); return l
""")

md(r"""
**Train briefly.** 4 layers, $d=128$, 4 heads, context 128, ≈ 0.8M parameters, AdamW with a one-cycle schedule. A uniform guess over 65 symbols has loss $\ln 65 = 4.17$.
On a Colab GPU, scale to `d=384, n_layers=6, n_heads=6, ctx=256, steps=5000` for loss ≈ 1.5 and much better text.
""")

code(r"""
def train_lm(model, steps, B=32, lr=3e-3, aux_coef=0.0, log_every=100):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    hist, t0 = [], time.time()
    for step in range(steps):
        x, y = get_batch(train_data, B, model.ctx)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        aux = sum(getattr(b.ffn, "aux_loss", 0.0) for b in model.blocks)
        opt.zero_grad(); (loss + aux_coef * aux).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        hist.append(loss.item())
        if step % log_every == 0 or step == steps - 1:
            print(f"step {step:4d}  train {loss.item():.3f}  val {val_loss(model):.3f}  ({time.time() - t0:.0f}s)")
    return hist

torch.manual_seed(0)
gpt = MiniGPT(vocab).to(device)
print(f"{sum(p.numel() for p in gpt.parameters()):,} parameters")
hist = train_lm(gpt, steps=600)
plt.figure(figsize=(6, 3.2)); plt.plot(hist, lw=0.8)
plt.axhline(math.log(vocab), ls="--", c="gray", label="uniform guess, ln 65")
plt.title("mini-GPT on Tiny Shakespeare"); plt.xlabel("step"); plt.ylabel("train loss (nats/char)"); plt.legend(); plt.show()
""")

code(r"""
@torch.no_grad()
def generate(model, prompt, n=300, temperature=0.8):
    model.eval()
    idx = encode(prompt)[None].to(device)
    for _ in range(n):
        logits = model(idx[:, -model.ctx:])[:, -1] / temperature
        idx = torch.cat([idx, torch.multinomial(logits.softmax(-1), 1)], dim=1)
    model.train()
    return decode(idx[0].cpu())

torch.manual_seed(1)
print(generate(gpt, "ROMEO:\n"))
""")

md(r"""
After 600 steps (validation loss ≈ 1.63 nats/char, from 4.17) the model has the format of the play: speaker names in capitals followed by a colon, line breaks, verse-length lines, mostly real English words. The words do not form sentences yet. The scaled-up GPU setting reaches loss ≈ 1.5 with the same code.
""")

md(r"""
## 4. Grouped-query attention and mixture-of-experts

**GQA.** $H$ query heads, $H_{kv}$ key/value heads; query head $h$ uses KV head $\lfloor h / (H/H_{kv}) \rfloor$. Implementation: compute $H_{kv}$ KV heads and **repeat each one** $H/H_{kv}$ times along the head axis (`repeat_kv` in the `Attention` class above, fill it in now).
$H_{kv}=H$ is MHA; $H_{kv}=1$ is multi-query attention (MQA). The KV cache shrinks by $H/H_{kv}$.
Checks: (a) with $H_{kv}=H$ our `Attention` equals the `MHA` of section 2 with the same weights; (b) with $H_{kv}<H$ it equals PyTorch's `enable_gqa=True`.
""")

code(r"""
torch.manual_seed(0)
D, H, T = 64, 8, 10
x = torch.randn(2, T, D, dtype=torch.float64)

gqa_full = Attention(D, H, n_kv_heads=H).double()    # groups = heads
mha = MHA(D, H).double()
with torch.no_grad():
    mha.qkv.weight.copy_(torch.cat([gqa_full.wq.weight, gqa_full.wkv.weight]))   # [W_Q; W_K; W_V]
    mha.proj.weight.copy_(gqa_full.proj.weight)
print("(a) H_kv = H: max |GQA − MHA| =", (gqa_full(x)[0] - mha(x, causal_mask(T))).abs().max().item())

gqa = Attention(D, H, n_kv_heads=2).double()
q = gqa.wq(x).view(2, T, H, D // H).transpose(1, 2)
k, v = gqa.wkv(x).view(2, T, 2, 2, D // H).permute(2, 0, 3, 1, 4)
ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
ref = gqa.proj(ref.transpose(1, 2).reshape(2, T, D))
print("(b) H_kv = 2: max |mine − SDPA(enable_gqa)| =", (gqa(x)[0] - ref).abs().max().item())
print(f"KV parameters: MHA {gqa_full.wkv.weight.numel():,}  vs  GQA(H_kv=2) {gqa.wkv.weight.numel():,}")
""")

md(r"""
**Top-$k$ MoE.** Router logits $s = W_r x \in \mathbb{R}^E$, $p=\mathrm{softmax}(s)$. Keep the top-$k$ probabilities, renormalize them to sum to 1 (Mixtral), and output
$$y = \sum_{i \in \mathrm{TopK}(p)} g_i\, \mathrm{FFN}_i(x),\qquad g_i = \frac{p_i}{\sum_{j\in\mathrm{TopK}} p_j}.$$
Only $k$ of the $E$ experts run per token: $E/k$ times more FFN parameters at the same FLOPs.

**Load-balancing loss (Switch Transformer).** Over the $N$ tokens of a batch, $f_i$ = fraction of routing slots assigned to expert $i$, $P_i$ = mean router probability of expert $i$:
$$\mathcal{L}_{\text{aux}} = E \sum_{i=1}^{E} f_i\,P_i .$$
$f$ is not differentiable, $P$ is; the gradient pushes probability away from over-used experts. With perfectly uniform routing $f_i = P_i = 1/E$ and $\mathcal{L}_{\text{aux}}=1$.
""")

code(r"""
class MoE(nn.Module):
    def __init__(self, d, hidden, n_experts=4, k=2):
        super().__init__()
        self.E, self.k = n_experts, k
        self.router = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(d, hidden) for _ in range(n_experts)])
        self.aux_loss, self.load = torch.tensor(0.0), None

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])                                   # (N, d) tokens
        probs = self.router(x).softmax(-1)                             # (N, E)
        #>> top-k of probs → (gates, idx), each (N, k); renormalize gates to sum to 1 per token
        gates, idx = probs.topk(self.k, dim=-1)
        gates = gates / gates.sum(-1, keepdim=True)
        #<<
        y = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            tok, slot = (idx == e).nonzero(as_tuple=True)              # tokens routed to expert e
            if len(tok):
                y.index_add_(0, tok, gates[tok, slot, None] * expert(x[tok]))
        #>> f = fraction of the N·k routing slots that went to each expert (E,); P = mean router prob (E,); aux = E Σ f P
        f = F.one_hot(idx, self.E).float().sum((0, 1)) / idx.numel()
        P = probs.mean(0)
        self.aux_loss = self.E * (f * P).sum()
        #<<
        self.load = f.detach()
        return y.reshape(shape)
""")

code(r"""
torch.manual_seed(0)
moe = MoE(16, 32, n_experts=4, k=2).double()
x = torch.randn(50, 16, dtype=torch.float64)
y = moe(x)
# reference: one token at a time, written directly from the formula
ref = []
for xt in x:
    p = moe.router(xt).softmax(-1); top = p.topk(2).indices
    ref.append(sum(p[i] / p[top].sum() * moe.experts[i](xt) for i in top))
print("MoE vs per-token loop: max diff =", (y - torch.stack(ref)).abs().max().item())
print("load f =", moe.load.numpy().round(3), " aux =", round(moe.aux_loss.item(), 3))

with torch.no_grad(): moe.router.weight.zero_()
moe(x)
print("uniform router (all logits 0) → aux =", round(moe.aux_loss.item(), 3))
with torch.no_grad(): moe.router.weight[0, 0], moe.router.weight[1, 0] = 1.0, 0.5   # logits (10, 5, 0, 0) when x_0 = 10
x_biased = x.clone(); x_biased[:, 0] = 10.0
moe(x_biased)
print("router collapsed onto experts 0, 1 → load f =", moe.load.numpy().round(3), " aux =", round(moe.aux_loss.item(), 3))
""")

md(r"""
Three values of $\mathcal{L}_{\text{aux}}$: a random router gives an uneven but spread load and $\mathcal{L}_{\text{aux}}\approx 1.01$; an all-zero router gives exactly 1 (top-$k$ ties break by index, so $f$ is not uniform, but $P_i=1/E$ makes $E\sum_i f_i P_i = \sum_i f_i = 1$); a router that sends every token to experts 0 and 1 with almost all its probability gives $f=(0.5,0.5,0,0)$ and $\mathcal{L}_{\text{aux}}=2=E/k$, its maximum for top-2. Experts 2 and 3 then receive no gradient at all and never recover: that is the collapse the loss prevents.
""")

md(r"""
## 5. Symmetry: permutation equivariance, and how positions break it

Take one block **without the causal mask**. Permute the input tokens with a permutation $\pi$. If the block has no positional information, the output is the same set of vectors, permuted the same way: $f(x_\pi) = f(x)_\pi$. RoPE ties each token to its index, so the equality breaks. A causal mask also breaks it (token $i$ sees a different set of tokens after permuting).
""")

code(r"""
torch.manual_seed(0)
T, D = 16, 64
blk = Block(D, 4).double()
x = torch.randn(1, T, D, dtype=torch.float64)
perm = torch.randperm(T)
rope = rope_cache(T, D // 4)

def equiv_error(**kw):
    return (blk(x[:, perm], **kw)[0] - blk(x, **kw)[0][:, perm]).abs().max().item()

print(f"no positions, no mask : max |f(x_π) − f(x)_π| = {equiv_error(rope=None, causal=False):.1e}")
print(f"RoPE,         no mask : max |f(x_π) − f(x)_π| = {equiv_error(rope=rope, causal=False):.1e}")
print(f"no positions, causal  : max |f(x_π) − f(x)_π| = {equiv_error(rope=None, causal=True):.1e}")
print(f"output scale          : mean |f(x)| = {blk(x, causal=False)[0].abs().mean().item():.2f}")
pooled = lambda z: blk(z, causal=False)[0].mean(1)
print(f"mean-pooled output, no positions: max |pool f(x_π) − pool f(x)| = {(pooled(x[:, perm]) - pooled(x)).abs().max().item():.1e}  (invariant)")
""")

md(r"""
- **No positions, no mask:** the error is ~1e-16, float64 round-off. The block is exactly permutation-equivariant, and after mean pooling exactly invariant: the model sees a **set**.
- **RoPE:** the error is ~0.07 on outputs of size ~0.8. Same tokens, different positions → different output.
- **Causal mask:** the error is ~1, larger still: after permuting, token $i$ sees a different prefix.

✏️ The trained mini-GPT has a causal mask *and* RoPE. Which of the two would already be enough for it to know word order? (Hint: run the third line above on a causal model without RoPE: "NoPE" decoders do learn order from the mask alone; section 6 trains one.)
""")

md(r"""
**Past exam question (Moed C, 2026)**

True or false: in standard self-attention, reordering the input tokens (the same permutation applied to the queries, keys and values) changes the output matrix only by the same permutation of its rows.

<<STUDENT>>
✏️ Your answer:
<</STUDENT>>
<<SOLUTION>>
**True** (no mask, no positional encoding). For a permutation matrix $P$: $\mathrm{softmax}\big(PQK^\top P^\top/\sqrt{d_k}\big)PV = P\,\mathrm{softmax}\big(QK^\top/\sqrt{d_k}\big)P^\top P V = P\,\mathrm{Attn}(X)$, because a row-wise softmax commutes with permuting rows and columns, and $P^\top P=I$. Self-attention is permutation-**equivariant**, the first line of the check above; a causal mask or RoPE breaks it.
<</SOLUTION>>
""")

code(r"""
torch.manual_seed(0)
Xp = torch.randn(1, 1, 6, 8, dtype=torch.float64)       # (B, heads, T, d)
Wq, Wk, Wv = (torch.randn(8, 8, dtype=torch.float64) for _ in range(3))
P = torch.randperm(6)
#>> single-head self-attention of X and of X[P] with the attention() function above; compare Attn(X[P]) with Attn(X)[P]
attn_of = lambda X: attention(X @ Wq, X @ Wk, X @ Wv)[0]
err = (attn_of(Xp[:, :, P]) - attn_of(Xp)[:, :, P]).abs().max().item()
#<<
print("max |Attn(PX) − P·Attn(X)| =", err)
""")

md(r"""
## 6. Where does position enter? Absolute, relative, rotary, none

Self-attention sees a set (section 5). A positional scheme injects the index at one of three places: the input $x_m$, the attention logit $E_{mn}$, or the vectors $q_m, k_n$. Notation: query position $m$, key position $n \le m$, offset $r = m-n$, model width $d$, head size $d_h$, $H$ heads.

**(a) Absolute, added to the input:** $x_m \leftarrow x_m + p_m$.
- *Learned table* $P\in\mathbb{R}^{T_{\max}\times d}$: GPT-2, BERT ([1810.04805](https://arxiv.org/abs/1810.04805)), ViT ([2010.11929](https://arxiv.org/abs/2010.11929)). $T_{\max}\cdot d$ parameters; there is no row for $m\ge T_{\max}$.
- *Sinusoidal* (Vaswani et al. 2017, [1706.03762](https://arxiv.org/abs/1706.03762)): $p_{m,2j}=\sin(m\,\omega_j)$, $p_{m,2j+1}=\cos(m\,\omega_j)$, $\omega_j=10000^{-2j/d}$. Per pair,
$\begin{pmatrix}\sin (m+k)\omega\\ \cos (m+k)\omega\end{pmatrix}=\begin{pmatrix}\cos k\omega & \sin k\omega\\ -\sin k\omega & \cos k\omega\end{pmatrix}\begin{pmatrix}\sin m\omega\\ \cos m\omega\end{pmatrix}$,
so $p_{m+k}=R_k\,p_m$ with $R_k$ block-diagonal, orthogonal and independent of $m$: an offset is a fixed linear map. The model can use it only through $W_Q, W_K$, and mixed with content: $\langle W_Q(x_m+p_m), W_K(x_n+p_n)\rangle$ has content–content, content–position and position–position terms.

**(b) Relative, a bias on the logits:** $E_{mn}=q_m^\top k_n/\sqrt{d_h}+b_h(m-n)$.
- Shaw et al. 2018 ([1803.02155](https://arxiv.org/abs/1803.02155)): learned vectors for the clipped offset $\mathrm{clip}(m-n,\pm K)$, added to the keys (and values).
- T5 (Raffel et al. 2020, [1910.10683](https://arxiv.org/abs/1910.10683)): one learned scalar per head per offset bucket, shared over layers; 32 buckets, exact for small offsets, log-spaced up to 128, one bucket for everything beyond.
- ALiBi (Press et al. 2022, [2108.12409](https://arxiv.org/abs/2108.12409)): no parameters, $b_h(m-n) = -s_h\,|m-n|$ with geometric slopes $s_h = 2^{-8h/H}$, $h=1,\dots,H$ ($\tfrac12,\tfrac14,\dots,\tfrac1{256}$ for $H=8$). Each head is a recency prior with its own scale $\approx 1/s_h$ tokens. Defined for every offset, so it runs at any length.

**(c) Rotary, on $q$ and $k$** (RoFormer, Su et al. 2021, [2104.09864](https://arxiv.org/abs/2104.09864)): $q_m\leftarrow R_m q_m$, $k_n\leftarrow R_n k_n$ (section 3). The logit depends on content and on $m-n$ only; no additive term, no parameters. Llama 2/3, Qwen 2/3, Mistral, Gemma, DeepSeek-V3 (on a separate 64-dim slice of each head, the "decoupled RoPE" of MLA). Llama 4 interleaves RoPE layers with NoPE layers.

**(d) None (NoPE).** In a causal decoder token $m$ attends over exactly $m+1$ tokens, so position leaks through the mask: e.g. a head with uniform weights outputs a mean of $m+1$ vectors, whose norm shrinks with $m$. Haviv et al. 2022 ([2203.16634](https://arxiv.org/abs/2203.16634)): NoPE LMs come close to the perplexity of LMs with positions. Kazemnejad et al. 2023 ([2305.19466](https://arxiv.org/abs/2305.19466)): on small synthetic tasks NoPE length-generalizes better than absolute, T5 bias, ALiBi and RoPE. Encoders (no mask) cannot use NoPE: they are permutation-equivariant.

**(e) Extending RoPE past the training length.** Pair $j$ rotates with frequency $\theta_j=b^{-2j/d_h}$, wavelength $\lambda_j=2\pi/\theta_j$. Pairs with $\lambda_j<T_{\text{train}}$ complete full turns in training and see every angle. Pairs with $\lambda_j>T_{\text{train}}$ only ever see angles in $[0, T_{\text{train}}\theta_j)$; at a longer offset they produce angles never seen in training. Fixes, all without new parameters, with $s = T/T_{\text{train}}$:
- *Position interpolation* (PI, Chen et al. 2023, [2306.15595](https://arxiv.org/abs/2306.15595)): $m \to m/s$. Every angle is back in the trained range, but neighbouring tokens are now $\theta_j/s$ apart, in the high-frequency pairs too. Used with a short fine-tune (1000 steps in the paper).
- *NTK-aware* scaling (bloc97, 2023): $b \to b\, s^{d_h/(d_h-2)}$. The highest frequency $\theta_0=1$ is unchanged, the lowest is divided by exactly $s$, geometric in between.
- *YaRN* (Peng et al. 2023, [2309.00071](https://arxiv.org/abs/2309.00071)), "NTK-by-parts": $\theta_j$ unchanged for $\lambda_j \ll T_{\text{train}}$, $\theta_j/s$ (PI) for $\lambda_j > T_{\text{train}}$, a linear ramp in between; plus a logit temperature $\sqrt{1/t}=0.1\ln s+1$. Used for the long-context versions of Qwen 2.5 and DeepSeek-V3; Llama 3.1's released code uses the same piecewise rule.
- Raise the base at training time: Llama 3 uses $b=500{,}000$; "ABF" (Xiong et al. 2023, [2309.16039](https://arxiv.org/abs/2309.16039)) raises it before long-context fine-tuning. Fewer pairs then complete a turn within the context.

**(f) Multimodal.** Images: 2D RoPE (Heo et al. 2024, [2403.13298](https://arxiv.org/abs/2403.13298)) rotates half of the pairs by the row index and half by the column index, so the logit depends on $(\Delta y, \Delta x)$. Qwen2-VL's M-RoPE ([2409.12191](https://arxiv.org/abs/2409.12191)) splits the pairs into three sections driven by (time, height, width) indices; a text token gets the same index in all three, which reduces to 1D RoPE. A learned table (the ViT in section 7) must be resampled (bicubic, as in ViT/DeiT fine-tuning at higher resolution) to change the number of patches.

In this section: implement the sinusoidal table and ALiBi, derive RoPE's relative property and look at what each rotary pair can encode, train the five schemes (none, learned, sinusoidal, ALiBi, RoPE) on the same task and test them at 2–8× the training length, then extend RoPE with PI, NTK-aware and by-parts scaling.
""")

md(r"""
**Sinusoidal table.** Even columns $\sin(m\omega_j)$, odd columns $\cos(m\omega_j)$. Check: the identity $p_{m+k} = R_k\,p_m$ for every $m$, with one matrix $R_k$ per offset.
""")

code(r"""
def sinusoidal_table(T, d, base=10000.0):
    m = torch.arange(T, dtype=torch.float64)[:, None]                    # (T, 1)
    omega = base ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)   # (d/2,)
    pe = torch.zeros(T, d, dtype=torch.float64)
    #>> even columns sin(m ω_j), odd columns cos(m ω_j)
    pe[:, 0::2] = torch.sin(m * omega)
    pe[:, 1::2] = torch.cos(m * omega)
    #<<
    return pe
""")

code(r"""
pe = sinusoidal_table(512, 64)
print("spot check |PE[37, 10] − sin(37 / 10000^(10/64))| =", abs(pe[37, 10].item() - math.sin(37 / 10000 ** (10 / 64))))
omega = 10000.0 ** (-torch.arange(0, 64, 2, dtype=torch.float64) / 64)

def R_offset(k):   # block-diagonal: maps (sin mω_j, cos mω_j) to (sin (m+k)ω_j, cos (m+k)ω_j) for every j
    Rk = torch.zeros(64, 64, dtype=torch.float64)
    c, s = torch.cos(k * omega), torch.sin(k * omega)
    Rk[0::2, 0::2], Rk[0::2, 1::2] = torch.diag(c), torch.diag(s)
    Rk[1::2, 0::2], Rk[1::2, 1::2] = torch.diag(-s), torch.diag(c)
    return Rk

for k in (1, 7, 100):
    print(f"k = {k:3d}: max over m < 400 of |PE(m+k) − R_k PE(m)| = {(pe[k:k + 400] - pe[:400] @ R_offset(k).T).abs().max().item():.1e}")
""")

md(r"""
**ALiBi.** A bias tensor `(H, T, T)` with entry $-s_h|m-n|$, added to the logits before the causal mask and the softmax. Check against a loop, and against the form in the ALiBi reference code, which adds $s_h\cdot n$ (key index only): for $n\le m$, $s_h n = -s_h(m-n) + s_h m$, a per-row constant, and softmax is invariant to adding a constant to a row.
""")

code(r"""
def alibi_slopes(H):
    return 2.0 ** (-8.0 * torch.arange(1, H + 1, dtype=torch.float64) / H)   # (H,): 2^(-8/H), 2^(-16/H), ..., 2^(-8)

def alibi_bias(T, H):
    # (H, T, T) additive logit bias, entry [h, m, n] = −s_h |m − n|; the causal mask is applied separately
    #>> |m − n| as a (T, T) matrix, times −s_h per head
    dist = (torch.arange(T)[:, None] - torch.arange(T)[None, :]).abs()
    return -alibi_slopes(H)[:, None, None] * dist
    #<<
""")

code(r"""
H, T = 8, 12
print("slopes (H = 8):", [f"1/{round(1 / s)}" for s in alibi_slopes(H).tolist()])
ref = torch.tensor([[[-(2.0 ** (-8 * (h + 1) / H)) * abs(m - n) for n in range(T)] for m in range(T)] for h in range(H)], dtype=torch.float64)
print("max |mine − loop| =", (alibi_bias(T, H) - ref).abs().max().item())
scores = torch.randn(H, T, T, dtype=torch.float64)
key_only = alibi_slopes(H)[:, None, None] * torch.arange(T, dtype=torch.float64)    # s_h · n, the reference-code form
probs = lambda b: (scores + b).masked_fill(~causal_mask(T), float("-inf")).softmax(-1)
print("softmax with −s|m−n| vs with s·n: max diff =", (probs(alibi_bias(T, H)) - probs(key_only)).abs().max().item())
""")

md(r"""
**RoPE in complex form.** Write pair $j$ of a vector as one complex number, $z_j = x_{2j} + i\,x_{2j+1}$. Rotating the pair by $m\theta_j$ is multiplying by $e^{im\theta_j}$, and for two real vectors $\langle a,b\rangle = \mathrm{Re}\sum_j a_j\bar b_j$. Hence
$$\langle R_m q,\, R_n k\rangle = \mathrm{Re}\sum_{j=0}^{d_h/2-1} q_j e^{im\theta_j}\,\overline{k_j e^{in\theta_j}} = \mathrm{Re}\sum_j q_j\bar k_j\, e^{i(m-n)\theta_j} = \sum_j |q_j|\,|k_j|\cos\!\big((m-n)\theta_j + \phi_j\big),\qquad \phi_j=\arg q_j-\arg k_j .$$
- The logit depends on the positions only through $r = m-n$. Content sets the amplitude $|q_j||k_j|$ and the phase $\phi_j$ of each term.
- As a function of the offset, the logit is a **trigonometric polynomial with fixed frequencies** $\theta_0 > \theta_1 > \dots$: each query–key pair picks amplitudes and phases for $d_h/2$ given sinusoids of $r$. Pair 0 ($\theta_0 = 1$, wavelength $2\pi \approx 6.3$ tokens) separates offsets 1, 2, 3: local position. The last pairs turn by a small angle over the whole context: they hardly depend on $r$ and act as a content channel with a slow trend (Barbero et al. 2024, [2410.06205](https://arxiv.org/abs/2410.06205), find that Gemma uses its lowest frequencies this way).
- **Long-range decay** (RoFormer §3.4.3). With $h_j = q_j\bar k_j$ and $S_j = \sum_{l<j} e^{ir\theta_l}$, Abel summation gives $\big|\sum_j h_j e^{ir\theta_j}\big| \le \max_j|h_{j+1}-h_j| \sum_{j=1}^{d_h/2}|S_j|$. The factor $\frac{2}{d_h}\sum_j |S_j|$ decreases (with oscillations) as $r$ grows: the attainable logit for distant tokens is smaller.

Check: the complex formula against `apply_rope` from section 3, at pairs $(m,n)$ with the same and with different offsets, up to $m = 4000$.
""")

code(r"""
torch.manual_seed(0)
hd = 16
cos_c, sin_c = rope_cache(4096, hd)
theta16 = 10000.0 ** (-torch.arange(0, hd, 2, dtype=torch.float64) / hd)
qv, kv = torch.randn(hd, dtype=torch.float64), torch.randn(hd, dtype=torch.float64)
zq, zk = torch.view_as_complex(qv.view(-1, 2)), torch.view_as_complex(kv.view(-1, 2))   # d_h/2 complex numbers each

def score_rotated(m, n):   # <R_m q, R_n k> with section 3's apply_rope
    return (apply_rope(qv[None], cos_c, sin_c, torch.tensor([m])) @ apply_rope(kv[None], cos_c, sin_c, torch.tensor([n])).T).item()

def score_complex(r):      # Re Σ_j q_j conj(k_j) e^{i r θ_j}
    return (zq * zk.conj() * torch.exp(1j * r * theta16)).real.sum().item()

for m, n in [(5, 2), (300, 297), (4000, 3997), (10, 900), (1000, 10)]:
    print(f"(m, n) = ({m:4d}, {n:4d}), r = {m - n:5d}: rotated {score_rotated(m, n):+.6f}  complex {score_complex(m - n):+.6f}  |diff| {abs(score_rotated(m, n) - score_complex(m - n)):.1e}")
""")

md(r"""
**What each pair can encode.** Left: $\cos(r\theta_j)$ for the 8 pairs of our head ($d_h = 16$, $b=10^4$) over offsets $0$–$255$; the model below is trained on 64 tokens (dashed line). Middle: wavelength per pair for our head and for a Llama-size head ($d_h=128$) with $b = 10^4$ (Llama 2) and $b = 5\cdot10^5$ (Llama 3); horizontal lines at the training lengths 64 (ours) and 8192 (Llama 3 pretraining). Right: the RoFormer decay factor $\frac{2}{d_h}\sum_j|S_j|$ for $d_h = 128$.
""")

code(r"""
r = torch.arange(256, dtype=torch.float64)
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
im = axes[0].imshow(torch.cos(torch.outer(theta16, r)), aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
axes[0].axvline(64, c="k", ls="--", lw=1)
axes[0].set_yticks(range(8), [f"j={j}, λ={2 * math.pi / t:.0f}" for j, t in enumerate(theta16.tolist())])
axes[0].set(title="cos(r θ_j), d_h = 16, b = 1e4", xlabel="offset r (tokens)")
fig.colorbar(im, ax=axes[0], shrink=0.8)

for hd_, b, lab in [(16, 1e4, "d_h=16, b=1e4 (ours)"), (128, 1e4, "d_h=128, b=1e4"), (128, 5e5, "d_h=128, b=5e5")]:
    th = b ** (-torch.arange(0, hd_, 2, dtype=torch.float64) / hd_)
    axes[1].semilogy(torch.arange(hd_ // 2) / (hd_ // 2), 2 * math.pi / th, marker="o" if hd_ == 16 else None, ms=4, label=lab)
for T_tr, lab in [(64, "T_train = 64"), (8192, "T_train = 8192")]:
    axes[1].axhline(T_tr, c="gray", ls="--", lw=1); axes[1].text(0.02, T_tr * 1.3, lab, fontsize=8)
axes[1].set(title="wavelength 2π/θ_j", xlabel="pair index j / (d_h/2)", ylabel="tokens"); axes[1].legend(fontsize=8)

rr = torch.logspace(0, 5, 600, dtype=torch.float64)
for b in (1e4, 5e5):
    th = b ** (-torch.arange(0, 128, 2, dtype=torch.float64) / 128)
    S = torch.exp(1j * torch.outer(rr, th)).cumsum(-1).abs()              # |S_j| for j = 1..64
    axes[2].semilogx(rr, S.mean(-1), lw=1, label=f"b = {b:.0e}")
axes[2].set(title="RoFormer decay factor, d_h = 128", xlabel="relative distance r", ylabel="(2/d_h) Σ_j |S_j|"); axes[2].legend()
plt.tight_layout(); plt.show()

for hd_, b, T_tr in [(16, 1e4, 64), (128, 1e4, 4096), (128, 5e5, 8192)]:
    lam = 2 * math.pi * b ** (torch.arange(0, hd_, 2, dtype=torch.float64) / hd_)
    print(f"d_h = {hd_:3d}, b = {b:.0e}, T_train = {T_tr:4d}: {(lam > T_tr).sum().item():2d} of {hd_ // 2} pairs never complete a turn in training")
""")

md(r"""
- Left: pairs 0–2 cycle within 64 tokens (pair 2's wavelength is 63); pairs 3–7 turn by less than a full cycle over the training window (pair 3 by 0.32 turns, pair 7 by 0.02 rad). Beyond the dashed line, pairs 3–7 reach angles the trained model has never seen.
- Middle: a larger base stretches all wavelengths except pair 0's. With $b = 5\cdot10^5$, 29 of Llama 3's 64 pairs never complete a turn within 8192 tokens (14 of 64 for $b=10^4$ at Llama 2's 4096).
- Right: the decay factor falls with distance; with the larger base it stays high to larger $r$ (it is a bound, not the actual logit of a trained model).

**The experiment.** Character-level Tiny Shakespeare (section 3's data). One model, five schemes: `none`, `learned` (table of 64 rows; positions $\ge 64$ reuse row 63, since the table has nothing else), `sinusoidal`, `alibi`, `rope`. 2 layers, $d=64$, 4 heads ($d_h=16$), causal, 600 steps of batch 32 at length 64, 3 seeds each. Evaluation: 64 validation windows of 512 characters; length $L$ uses their first $L$ characters, and we report the mean loss over all $L$ positions (as in the ALiBi paper). The attention below reuses section 3's `Block`, `RMSNorm`, `SwiGLU` and `apply_rope`; `Block.forward` hands its `rope` argument to the attention unchanged, so we use it to carry `(rope tables, ALiBi bias)`.
""")

code(r"""
def rope_theta(head_dim, base=10000.0):
    return base ** (-torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)

def rope_tables(pos, theta):   # cos, sin of the angles pos_m · θ_j, (T, d_h/2); positions may be fractional
    ang = torch.outer(pos.double(), theta)
    return ang.cos(), ang.sin()

class PosAttention(nn.Module):
    # causal attention; position from RoPE on q, k (rope = (cos, sin)) and/or an additive logit bias (H, T, T)
    def __init__(self, d, n_heads):
        super().__init__()
        self.h, self.hd = n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x, pos_ctx, causal=True, return_attn=False):
        rope, bias = pos_ctx
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)    # (B, H, T, d_h) each
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        if bias is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.masked_fill(~causal_mask(T, x.device), float("-inf")))
        return self.proj(y.transpose(1, 2).reshape(B, T, D)), None

class PosGPT(nn.Module):
    def __init__(self, vocab, scheme, d=64, n_layers=2, n_heads=4, train_len=64):
        super().__init__()
        self.scheme, self.h, self.hd, self.train_len = scheme, n_heads, d // n_heads, train_len
        self.rope_mode, self.rope_scale = "plain", None      # context extension, used further below
        self.emb = nn.Embedding(vocab, d)
        if scheme == "learned":
            self.pos = nn.Parameter(torch.randn(train_len, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, n_heads) for _ in range(n_layers)])
        for blk in self.blocks:
            blk.attn = PosAttention(d, n_heads)
        self.norm, self.head = RMSNorm(d), nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        T, dev = idx.shape[1], idx.device
        x, rope, bias = self.emb(idx), None, None
        if self.scheme == "learned":
            x = x + self.pos[torch.arange(T, device=dev).clamp(max=self.train_len - 1)]
        elif self.scheme == "sinusoidal":
            x = x + sinusoidal_table(T, x.shape[-1]).float().to(dev)
        elif self.scheme == "alibi":
            bias = alibi_bias(T, self.h).float().to(dev)
        elif self.scheme == "rope":
            tabs = (rope_tables(torch.arange(T), rope_theta(self.hd)) if self.rope_mode == "plain"
                    else extended_rope(T, self.hd, self.train_len, self.rope_mode, self.rope_scale))
            rope = tuple(t.float().to(dev) for t in tabs)
        for blk in self.blocks:
            x, _ = blk(x, (rope, bias))
        return self.head(self.norm(x))
""")

code(r"""
def train_pos(model, steps=600, T=64, B=32, lr=3e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    for _ in range(steps):
        x, y = get_batch(train_data, B, T)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    return model

torch.manual_seed(1234)
x_eval, y_eval = get_batch(val_data, 64, 512)          # length L uses the first L characters of these windows
EVAL_LENS = (64, 128, 256, 512)

@torch.no_grad()
def eval_pos(model):
    # mean loss over all positions at each eval length, and the per-position loss at length 512
    model.eval()
    out = {L: F.cross_entropy(model(x_eval[:, :L]).flatten(0, 1), y_eval[:, :L].flatten()).item() for L in EVAL_LENS}
    per_pos = F.cross_entropy(model(x_eval).transpose(1, 2), y_eval, reduction="none").mean(0).cpu()
    model.train()
    return out, per_pos

SCHEMES, SEEDS = ("none", "learned", "sinusoidal", "alibi", "rope"), (0, 1, 2)
results, per_pos, pos_models = {}, {}, {}
t0 = time.time()
for scheme in SCHEMES:
    for seed in SEEDS:
        torch.manual_seed(seed)
        m = train_pos(PosGPT(vocab, scheme).to(device))
        results[scheme, seed], per_pos[scheme, seed] = eval_pos(m)
        pos_models[scheme, seed] = m
    print(f"{scheme:10s} done ({time.time() - t0:.0f}s)")

def table(rows, title):
    print(f"{title:22s}" + "".join(f"   L = {L:<10d}" for L in EVAL_LENS))
    for name, runs in rows.items():
        cells = [np.array([r[L] for r in runs]) for L in EVAL_LENS]
        print(f"{name:22s}" + "".join(f"   {c.mean():.3f} ± {c.std():.3f}" for c in cells))

table({s: [results[s, sd] for sd in SEEDS] for s in SCHEMES}, "loss (nats/char)")
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for s in SCHEMES:
    mean = np.array([[results[s, sd][L] for L in EVAL_LENS] for sd in SEEDS])
    axes[0].errorbar(EVAL_LENS, mean.mean(0), yerr=mean.std(0), marker="o", capsize=3, label=s)
    curve = torch.stack([per_pos[s, sd] for sd in SEEDS]).mean(0)
    axes[1].plot(np.convolve(curve.numpy(), np.ones(16) / 16, mode="valid"), lw=1, label=s)
axes[0].set_xscale("log", base=2); axes[0].set_xticks(EVAL_LENS, [str(L) for L in EVAL_LENS])
axes[0].set(title="mean loss vs eval length (train length 64)", xlabel="eval length L", ylabel="loss (nats/char)")
axes[1].axvline(64, c="gray", ls="--", lw=1)
axes[1].set(title="loss per position at L = 512 (moving average of 16)", xlabel="position", ylabel="loss (nats/char)")
for ax in axes: ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

md(r"""
Mean ± std over 3 seeds (loss in nats/char; the first 64 positions of every length are in-distribution, so the mean at $L$ dilutes the damage; the right panel shows it per position).
- **In distribution ($L=64$):** RoPE 1.93 < ALiBi 2.02 < sinusoidal 2.08 < learned 2.23 < none 2.29. Every scheme beats NoPE, and RoPE is best by 0.09 nats.
- **ALiBi is flat:** 2.02 at every length up to 8× the training length, the per-position loss stays at its in-distribution level. This is the result of the ALiBi paper, and it reproduces here. With $H=4$ the slopes are $\tfrac14,\tfrac1{16},\tfrac1{64},\tfrac1{256}$; beyond offset 64 the extra keys only receive larger penalties (weights near zero in the three fast heads), so the attention pattern over the recent tokens is nearly unchanged.
- **Sinusoidal breaks at once:** the per-position loss jumps at position 64 to ≈ 3.1, worse than NoPE. The added vectors $p_m$, $m\ge 64$, are new inputs to $W_Q, W_K$ and the MLPs.
- **Learned (clamped) degrades gradually:** all positions $\ge 64$ share one row, so the model sees many tokens "at position 63"; loss 2.80 at 512.
- **RoPE holds to ≈ 100 and then breaks:** the per-position loss stays at 1.9 up to position ≈ 100 and rises to 3.4–3.7 beyond position 200, the worst at 512 (3.10). The break is not at 64. Next cell: the pair the model uses most has wavelength 199; its trained angles are $r\theta_3 \in [0, 2.0]$ rad, and $\cos(r\theta_3)$ decreases monotonically in $r$ only until $r\theta_3=\pi$, i.e. $r\approx 100$. Past that, $\cos$ rises again, so a token 150 back looks like a token 50 back.
- **NoPE degrades slowly** (2.29 → 2.50): its position signal (the $1/(m+1)$ averaging) shifts smoothly with length. It is worse than ALiBi at every length; the claim that NoPE length-generalizes better than ALiBi and RoPE (Kazemnejad et al.) comes from small synthetic tasks and does not show on character-level text at this scale.
""")

md(r"""
**Which rotary pairs does the trained model use?** For each pair $j$, the mean of $|q_j|\,|k_j|$ (the amplitude of the $j$-th sinusoid in the logit) over tokens, heads and both layers of the RoPE models, normalized to sum to 1.
""")

code(r"""
@torch.no_grad()
def pair_amplitudes(model, x):
    qk = []
    hooks = [blk.attn.qkv.register_forward_hook(lambda mod, inp, out: qk.append(out)) for blk in model.blocks]
    model.eval(); model(x); model.train()
    for h in hooks: h.remove()
    amps = []
    for out in qk:                                                      # (B, T, 3D) per layer
        q, k, _ = out.view(*out.shape[:2], 3, model.h, model.hd // 2, 2).unbind(2)
        amps.append((q.norm(dim=-1) * k.norm(dim=-1)).mean((0, 1, 2)))  # (d_h/2,)
    a = torch.stack(amps).mean(0).cpu()
    return a / a.sum()

amp = torch.stack([pair_amplitudes(pos_models["rope", sd], x_eval[:, :64]) for sd in SEEDS])
lam16 = 2 * math.pi / theta16
for j in range(8):
    print(f"pair {j}: wavelength {lam16[j].item():7.0f} tokens   amplitude share {amp[:, j].mean():.3f} ± {amp[:, j].std():.3f}")
""")

md(r"""
Pair 3 (wavelength 199, the slowest pair that still turns by a visible angle within 64 tokens, 0.32 turns) carries the largest share, 0.27; pairs 0–2 (local offsets) 0.16–0.19 each; pairs 5–7, whose angle hardly changes over 64 tokens, have the smallest amplitude, 0.046 each. In this model the slow pairs are almost unused, and pair 3 serves as a monotone "how far back" signal. That is the pair whose angles leave the trained range first, and it explains where the RoPE curve breaks.
""")

md(r"""
**Context extension for RoPE.** `extended_rope` returns the cos/sin tables for positions $0,\dots,T-1$ of a model trained on `train_len` positions, with scale $s$ (default $T/T_{\text{train}}$, recomputed per length, i.e. "dynamic" scaling):
- `pi`: positions $m \to m/s$ (your TODO);
- `ntk`: base $b \to b\,s^{d_h/(d_h-2)}$;
- `parts`: $\theta_j \to \theta_j/s$ only for pairs with $\lambda_j > T_{\text{train}}$ (YaRN's rule with a hard threshold instead of the ramp, and no temperature).
""")

code(r"""
def extended_rope(T, head_dim, train_len, mode, scale=None, base=10000.0):
    s = scale if scale is not None else max(1.0, T / train_len)
    pos, theta = torch.arange(T, dtype=torch.float64), rope_theta(head_dim, base)
    if mode == "pi":
        #>> position interpolation: positions m → m / s
        pos = pos / s
        #<<
    elif mode == "ntk":
        theta = rope_theta(head_dim, base * s ** (head_dim / (head_dim - 2)))
    elif mode == "parts":
        theta = torch.where(2 * math.pi / theta > train_len, theta / s, theta)
    return rope_tables(pos, theta)
""")

code(r"""
c_pi, s_pi = extended_rope(256, 16, 64, "pi")
c_tr, s_tr = rope_cache(64, 16)
print("PI at length 256: rows 0, 4, 8, ... equal the training rows 0, 1, 2, ...: max diff =",
      max((c_pi[::4] - c_tr).abs().max().item(), (s_pi[::4] - s_tr).abs().max().item()))
print("PI: row 1 is the fractional position 1/4: max |angle − θ/4| =", (torch.atan2(s_pi[1], c_pi[1]) - theta16 / 4).abs().max().item())
th_ntk = rope_theta(16, 1e4 * 4 ** (16 / 14))
print(f"NTK-aware, s = 4: θ_0 ratio {th_ntk[0] / theta16[0]:.3f} (unchanged), θ_7 ratio {th_ntk[-1] / theta16[-1]:.3f} (= 1/s)")
""")

md(r"""
First without any training: the three rules applied to the trained RoPE models from above. Then a short fine-tune at length 256 (100 steps, batch 8, peak learning rate $10^{-3}$, about 2% of the pretraining compute), once with plain RoPE and once with PI at a fixed $s=4$ (as in the PI paper, the fine-tuned model keeps $s=4$ at every length).
""")

code(r"""
import copy
ext = {}
for mode in ("plain", "pi", "ntk", "parts"):
    ext[f"{mode}, no fine-tune"] = []
    for sd in SEEDS:
        pos_models["rope", sd].rope_mode = mode
        ext[f"{mode}, no fine-tune"].append(eval_pos(pos_models["rope", sd])[0])
        pos_models["rope", sd].rope_mode = "plain"

t0 = time.time()
for mode, scale in (("plain", None), ("pi", 4.0)):
    ext[f"{mode}, fine-tuned 256"] = []
    for sd in SEEDS:
        torch.manual_seed(100 + sd)
        m = copy.deepcopy(pos_models["rope", sd])
        m.rope_mode, m.rope_scale = mode, scale
        ext[f"{mode}, fine-tuned 256"].append(eval_pos(train_pos(m, steps=100, T=256, B=8, lr=1e-3))[0])
print(f"fine-tuning: {time.time() - t0:.0f}s")
table({"alibi (reference)": [results["alibi", sd] for sd in SEEDS], **ext}, "RoPE extension")

fig, ax = plt.subplots(figsize=(7, 3.8))
for name, runs in ext.items():
    v = np.array([[r[L] for L in EVAL_LENS] for r in runs])
    ax.errorbar(EVAL_LENS, v.mean(0), yerr=v.std(0), marker="o", capsize=3, ls="-" if "no fine" in name else "--", label=name)
ax.set_xscale("log", base=2); ax.set_xticks(EVAL_LENS, [str(L) for L in EVAL_LENS])
ax.set(title="RoPE context extension (trained at 64)", xlabel="eval length L", ylabel="loss (nats/char)"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

md(r"""
- **PI without fine-tuning is worse than doing nothing** (2.64 vs 2.02 at $L=128$). PI divides all angles by $s$, including pairs 0–2, which resolve neighbouring characters; a character-level model depends on them (half of the amplitude above). The PI paper also reports that PI needs fine-tuning.
- **NTK-aware and by-parts scaling work without training up to 2×:** 1.93 / 1.96 at $L=128$ vs 1.93 in distribution. They leave the high frequencies alone and stretch only the slow pairs. At 4× and 8× they still lose 0.2–0.6 nats, less than plain RoPE (2.66, 3.10); by-parts is best at 512.
- **Short fine-tune at 256 (100 steps):** plain RoPE reaches 1.97 at $L=256$ and keeps 1.96 at 64; PI with $s=4$ reaches 2.04 at 256 and costs 0.11 nats at $L=64$ (positions there are compressed too). The PI paper's result, that PI + fine-tuning beats fine-tuning plain RoPE, **does not reproduce at this scale**: 100 steps are enough for a 2-layer model to learn the new angles of the 2–3 pairs that matter, while PI permanently coarsens the local pairs it needs. At Llama scale, Chen et al. found direct fine-tuning extended the effective context only slowly (from 2048 to ≈ 2560 after 10k steps), which is the regime PI was designed for. Both fine-tuned models still degrade at 512, beyond their new training length.

✏️ Train the RoPE model with `rope_theta(self.hd, 500.0)` and with `500000.0` instead of $10^4$ (change it in `PosGPT.forward`), and re-run the length table. Predict first from the wavelength plot: which base leaves fewer unseen angles at $L = 256$, and what does it cost at $L = 64$?
""")

md(r"""
**Exam-style question**

(a) A decoder LM with a learned absolute position table was trained on sequences of length 2048. Why can it not be run on 4096 tokens, while the same model with ALiBi or RoPE can at least be run? (b) What goes wrong with RoPE beyond the training length, in terms of the rotation angles of each pair? (c) What does position interpolation change, why does it need fine-tuning, and what does NTK-aware / by-parts scaling do differently?

<<STUDENT>>
✏️ Your answer:
<</STUDENT>>
<<SOLUTION>>
(a) The learned table has one parameter vector per index $0,\dots,T_{\max}-1$; position 2048 has no row, so the forward pass is undefined (clamping or extrapolating the table gives vectors the model never saw; loss 2.80 vs 2.23 above). ALiBi's bias $-s_h|m-n|$ and RoPE's rotation $R_m$ are closed-form functions of the index, defined for every $m$, and both enter only through the offset $m-n$, so the model runs at any length.

(b) Pair $j$ rotates by $r\theta_j$. Pairs with wavelength $2\pi/\theta_j < T_{\text{train}}$ have seen all angles; pairs with longer wavelength have seen only $[0, T_{\text{train}}\theta_j)$, and at longer offsets they produce angles outside this range. The logit $\sum_j |q_j||k_j|\cos(r\theta_j+\phi_j)$ then takes values the attention was not trained on; in particular $\cos$ is not monotone beyond $r\theta_j=\pi$, so distant keys can look near (our model: loss rises from 1.9 to ≈ 3.5 after offset ≈ 100, where pair 3 passes $\pi$).

(c) PI maps $m\to m\,T_{\text{train}}/T$: all angles are back in the trained range, but adjacent tokens are now $\theta_j/s$ apart in every pair, so the high-frequency pairs lose resolution; the model needs fine-tuning to adapt to the compressed spacing (here PI without fine-tuning was worse than plain RoPE). NTK-aware scaling raises the base, $b\to b\,s^{d_h/(d_h-2)}$: $\theta_0$ is unchanged and the lowest frequency is divided by $s$, so local resolution is kept and only slow pairs are stretched. By-parts (YaRN) interpolates only the pairs with wavelength $> T_{\text{train}}$ and leaves the others exactly as trained; here both work without training at 2× (1.93–1.96 vs 1.93).
<</SOLUTION>>
""")

md(r"""
**Practical notes**
- **Base.** Llama 2: $b = 10^4$, 4k context. Llama 3: $b = 5\cdot 10^5$ for 8k pretraining, then 128k by by-parts scaling (Llama 3.1). Code Llama: $10^6$. A larger base lengthens every wavelength except pair 0's: fewer pairs complete a turn within the context, less out-of-range angle at a longer length, slower decay of the RoFormer bound; the cost is coarser offset resolution in the middle pairs.
- **Partial rotary.** Rotate only some pairs and leave the rest position-free: GPT-NeoX/Pythia rotate 25% of each head ([2204.06745](https://arxiv.org/abs/2204.06745)), GPT-J 64 of 256 dims, Phi-2 40%. DeepSeek's MLA adds a separate 64-dim rotated part to $q$ and $k$, because rotated keys cannot be absorbed into the compressed latent KV.
- **KV cache.** The cache stores keys after rotation, $R_n k_n$; a new token rotates only its own $q, k$, at its absolute index. Changing the RoPE rule (PI/NTK scale) after the fact invalidates the cache. StreamingLLM ([2309.17453](https://arxiv.org/abs/2309.17453)) caches keys *before* rotation so it can re-index positions inside a sliding window.
- **Precision.** Compute positions and angles in fp32, not bf16: bf16 has an 8-bit significand, so integer positions above 256 are rounded: positions 0–8191 collapse to 897 distinct values (below). fp32 represents integers exactly up to $2^{24}$, but the angle $m\theta_j$ is also rounded: at $m=10^6$ the fp32 angle of pair 1 is off by 0.05 rad, at $m=8000$ by $4\cdot10^{-4}$. Most implementations build the cos/sin table in fp32 and cast afterwards.
""")

code(r"""
pos_all = torch.arange(8192)
print(f"distinct values of positions 0..8191: fp32 {pos_all.float().unique().numel()}, bf16 {pos_all.bfloat16().unique().numel()}, fp16 {pos_all.half().unique().numel()}")
th1 = rope_theta(128, 5e5)[1]                                   # second-fastest pair of a Llama 3 head
for m_big in (8_000, 1_000_000):
    ang64 = m_big * th1
    ang32 = (torch.tensor(float(m_big)) * th1.float()).double()
    print(f"m = {m_big:>9,}: angle m·θ_1 = {ang64.item():.5f} rad, fp32 error {abs(ang32 - ang64).item():.1e} rad")
""")

md(r"""
## 7. ViT: patchify, then a Transformer encoder

**Patchify.** An image `(B, C, H, W)` with patch size $p$ becomes `(B, N, C·p·p)` with $N = HW/p^2$, flattened in `(C, p, p)` order. A linear layer on these vectors **is** a convolution with `kernel_size = stride = p`, which is how ViT code implements it. That is the check.
""")

code(r"""
def patchify(img, p):
    B, C, H, W = img.shape
    #>> (B, C, H, W) → (B, C, H/p, p, W/p, p) → (B, H/p, W/p, C, p, p) → (B, N, C·p·p), patches in row-major order
    x = img.view(B, C, H // p, p, W // p, p).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(B, (H // p) * (W // p), C * p * p)
    #<<

img = torch.randn(2, 3, 32, 32, dtype=torch.float64)
conv = nn.Conv2d(3, 96, kernel_size=4, stride=4).double()
via_linear = patchify(img, 4) @ conv.weight.view(96, -1).T + conv.bias
via_conv = conv(img).flatten(2).transpose(1, 2)                 # (B, N, 96)
print("patchify shape:", tuple(patchify(img, 4).shape), "| max |linear(patchify) − conv| =", (via_linear - via_conv).abs().max().item())
""")

code(r"""
class TinyViT(nn.Module):
    def __init__(self, img=32, patch=4, d=128, n_layers=4, n_heads=4, n_classes=10):
        super().__init__()
        self.p, n = patch, (img // patch) ** 2
        self.embed = nn.Linear(3 * patch * patch, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, n + 1, d) * 0.02)     # learned absolute positions
        self.blocks = nn.ModuleList([Block(d, n_heads) for _ in range(n_layers)])
        self.norm, self.head = RMSNorm(d), nn.Linear(d, n_classes)

    def forward(self, img):
        x = self.embed(patchify(img, self.p))
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], dim=1) + self.pos
        for blk in self.blocks:
            x, _ = blk(x, rope=None, causal=False)                       # encoder: every patch sees every patch
        return self.head(self.norm(x[:, 0]))                             # classify from [CLS]
""")

md(r"""
**Train on a CIFAR-10 subset** (10k train / 2k test images, 32×32, patch 4 → 64 tokens + [CLS]), random flips and crops, 8 epochs.
ViTs need a lot of data or strong augmentation; with 10k images a small CNN does better (for scale: ResNet-18 on the full set reaches > 90%). On a Colab GPU, use the full 50k images, 100+ epochs and RandAugment/mixup.
""")

code(r"""
import torchvision, torchvision.transforms as Tr
mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
train_tf = Tr.Compose([Tr.RandomCrop(32, padding=4), Tr.RandomHorizontalFlip(), Tr.ToTensor(), Tr.Normalize(mean, std)])
test_tf = Tr.Compose([Tr.ToTensor(), Tr.Normalize(mean, std)])
g = torch.Generator().manual_seed(0)
train_set = torch.utils.data.Subset(torchvision.datasets.CIFAR10("data", train=True, download=True, transform=train_tf), torch.randperm(50000, generator=g)[:10000])
test_set = torch.utils.data.Subset(torchvision.datasets.CIFAR10("data", train=False, download=True, transform=test_tf), torch.randperm(10000, generator=g)[:2000])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=500)

@torch.no_grad()
def accuracy(model):
    model.eval()
    correct = sum((model(x.to(device)).argmax(-1).cpu() == y).sum().item() for x, y in test_loader)
    model.train(); return correct / len(test_set)

torch.manual_seed(0)
vit = TinyViT().to(device)
epochs = 8
opt = torch.optim.AdamW(vit.parameters(), lr=1e-3, weight_decay=0.05)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=epochs * len(train_loader), pct_start=0.15)
print(f"{sum(p.numel() for p in vit.parameters()):,} parameters")
t0, accs = time.time(), []
for ep in range(epochs):
    for xb, yb in train_loader:
        loss = F.cross_entropy(vit(xb.to(device)), yb.to(device))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    accs.append(accuracy(vit))
    print(f"epoch {ep + 1}  train loss {loss.item():.3f}  test acc {accs[-1]:.3f}  ({time.time() - t0:.0f}s)")
""")

md(r"""
What did the learned position embeddings learn? Cosine similarity between the embedding of one patch position and all 64 positions, shown on the 8×8 patch grid.
""")

code(r"""
pos = F.normalize(vit.pos[0, 1:].detach().cpu(), dim=-1)          # (64, d), skip [CLS]
sim = (pos @ pos.T).view(64, 8, 8)
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
for ax, (r, c) in zip(axes, [(0, 0), (3, 4), (7, 7), (0, 7)]):
    im = ax.imshow(sim[r * 8 + c], cmap="viridis", vmin=-1, vmax=1)
    ax.set_title(f"patch (row {r}, col {c})"); ax.set_xlabel("col"); ax.set_ylabel("row")
fig.colorbar(im, ax=axes, shrink=0.8, label="cosine similarity")
fig.suptitle(f"ViT learned position embeddings after {epochs} epochs on 10k images")
plt.show()
""")

md(r"""
Each position embedding is most similar to itself and to its spatial neighbours: the corner patches match their own row and column edge, the centre patch a blob around the centre. Nothing told the model the patches lie on a 2D grid (positions are just indices 0–63); it learned the 2D layout from 10k images in 8 epochs. Test accuracy is ≈ 55% on 2k images, far below a CNN on the same data: the lecture's point that ViTs lack the locality prior and must learn it from data.
""")

md(r"""
**Past exam question (Moed C, 2026)**

In a Vision Transformer (ViT) the image is split into fixed-size patches (e.g. $16\times16$); each patch is flattened into a vector and used as a token, and a learned positional embedding is added to every token.

Explain why ViT needs positional embeddings while a standard CNN does not. Refer explicitly to the nature of the self-attention operation and to the convolution operation along the spatial dimensions.

<<STUDENT>>
✏️ Your answer:
<</STUDENT>>
<<SOLUTION>>
Self-attention treats its input as a **set**: it is permutation-equivariant (section 5), and with [CLS] or mean pooling the prediction is permutation-invariant. Without position embeddings, shuffling the patches gives exactly the same output, so the model cannot tell where a patch was, nor which patches are neighbours. The embeddings attach each token's location to its content. A convolution is defined on the 2-D grid: each output combines a fixed $K\times K$ neighbourhood, and the weight of each neighbour is tied to its relative offset, so the spatial layout (locality and relative position) is built into the operation, and stacked layers plus pooling turn it into global position. Check below: with its position embeddings removed, our trained TinyViT gives identical logits (up to float32 round-off) for an image and its patch-shuffled version; with them, the logits change by up to ~3. Accuracy on shuffled images, however, stays at ≈ 0.50: after 8 epochs on 10k images this small ViT classifies mostly from the bag of patch contents and has learned little use of their layout, the same missing locality prior as above.
<</SOLUTION>>
""")

code(r"""
@torch.no_grad()
def vit_logits(model, img, perm=None, use_pos=True):
    x = model.embed(patchify(img, model.p))
    if perm is not None:
        x = x[:, perm]                                  # shuffle the patches
    x = torch.cat([model.cls.expand(len(x), -1, -1), x], dim=1)
    if use_pos:
        x = x + model.pos
    for blk in model.blocks:
        x, _ = blk(x, rope=None, causal=False)
    return model.head(model.norm(x[:, 0]))

vit.eval()
imgs, labels = next(iter(test_loader))
imgs = imgs.to(device)
perm = torch.randperm(64, generator=torch.Generator().manual_seed(0)).to(device)
#>> max |logits(shuffled) − logits(original)| without and with the position embeddings, and the accuracy on shuffled images
diff_nopos = (vit_logits(vit, imgs, perm, use_pos=False) - vit_logits(vit, imgs, use_pos=False)).abs().max().item()
diff_pos = (vit_logits(vit, imgs, perm) - vit_logits(vit, imgs)).abs().max().item()
acc_orig = (vit_logits(vit, imgs).argmax(-1).cpu() == labels).float().mean().item()
acc_shuf = (vit_logits(vit, imgs, perm).argmax(-1).cpu() == labels).float().mean().item()
#<<
vit.train()
print(f"no position embeddings: max |Δ logits| after shuffling = {diff_nopos:.1e}")
print(f"with position embeddings: max |Δ logits| after shuffling = {diff_pos:.2f}")
print(f"accuracy on {len(labels)} test images: original {acc_orig:.3f}, patch-shuffled {acc_shuf:.3f}")
""")

md(r"""
## 8. Inside the trained mini-GPT: logit lens and attention sinks

**Logit lens** (nostalgebraist, 2020). The residual stream after every block has the same shape as the final one. Apply the model's own final norm and unembedding to it, $\ \mathrm{softmax}(W_U\,\mathrm{RMSNorm}(x^{(\ell)}))$, and read it as a next-character prediction. Question: at which layer does the right next character appear?
""")

code(r"""
@torch.no_grad()
def logit_lens(model, x):
    _, resid, _ = model(x, return_internals=True)
    #>> for every residual state (embedding + each block), decode with the final norm and the head → list of (B, T, vocab) logits
    return [model.head(model.norm(r)) for r in resid]
    #<<

torch.manual_seed(0)
gpt.eval()
xv, yv = get_batch(val_data, 64, gpt.ctx)
lens = logit_lens(gpt, xv)
print("max |last lens layer − model logits| =", (lens[-1] - gpt(xv)).abs().max().item())
lens_loss = [F.cross_entropy(l.flatten(0, 1), yv.flatten()).item() for l in lens]
lens_acc = [(l.argmax(-1) == yv).float().mean().item() for l in lens]
names = ["emb"] + [f"block {i + 1}" for i in range(len(gpt.blocks))]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
ax[0].plot(names, lens_loss, marker="o"); ax[0].set(title="logit lens: loss per layer", xlabel="residual stream after", ylabel="cross-entropy (nats)")
ax[1].plot(names, lens_acc, marker="o"); ax[1].set(title="logit lens: top-1 next-char accuracy", xlabel="residual stream after", ylabel="accuracy")
plt.tight_layout(); plt.show()
for n, l, a in zip(names, lens_loss, lens_acc): print(f"{n:8s} loss {l:.2f}  acc {a:.3f}")
""")

md(r"""
The same view on one line of text: the top prediction at every position, decoded from each layer. Read each column top to bottom.
""")

code(r"""
prompt = "To be, or not to be, that is the question"
xs = encode(prompt)[None].to(device)
rows = [l[0].argmax(-1).cpu() for l in logit_lens(gpt, xs)]

show = lambda s: " ".join("_" if c == " " else ("⏎" if c == "\n" else c) for c in s)
print(f"{'input':>9}: " + show(prompt))
print(f"{'target':>9}: " + show(prompt[1:]) + " ?")
for n, r in zip(names, rows):
    print(f"{n:>9}: " + show(decode(r)))
""")

md(r"""
- The loss falls at every layer (3.94 → 2.91 → 2.47 → 2.20 → 1.64) and top-1 accuracy rises from 0.10 to 0.52. The prediction is built up gradually in the residual stream, and the last block contributes the largest single drop.
- In the example, bigram-like completions appear early: after `T` the `o` is there from block 1, after `t` the `h` from block 1. Completions that need the whole word are late: the space after `To` appears only at block 3, the `e` of `the` only at block 2.
- The final rows still miss characters that need context beyond the current word (the `?` after `question`): a 4-layer char model has little room for that.
""")

md(r"""
**Attention sinks** (Xiao et al., 2023, StreamingLLM). Trained LMs put a large share of attention on the **first token**, whatever it is: softmax weights must sum to 1, so a head with nothing useful to read parks its mass on a token every query can see. Measure it: for queries at positions $t \ge 32$, the mean weight on key 0, per layer and head. A head that attended uniformly over its causal window would give the average of $\frac{1}{t+1}$, ≈ 0.014 here.
Our training windows start at random positions, so position 0 is an arbitrary character, never a BOS token.
""")

code(r"""
@torch.no_grad()
def first_token_mass(attns, t_min=32):
    # attns: list over layers of (B, H, T, T) attention weights → (layers, heads)
    return torch.stack([w[:, :, t_min:, 0].float().mean((0, 2)) for w in attns]).cpu()  # @student: return None  # TODO: mean weight on key 0 over queries t ≥ t_min

def plot_sink(ax, sink, title, annotate=True):
    im = ax.imshow(sink, cmap="magma", vmin=0, vmax=1, aspect="auto")
    if annotate:
        for (l, h), val in np.ndenumerate(sink.numpy()): ax.text(h, l, f"{val:.2f}", ha="center", va="center", color="w" if val < 0.5 else "k")
    ax.set(title=title, xlabel="head", ylabel="layer")
    return im

uniform = (1 / torch.arange(33, gpt.ctx + 1).float()).mean().item()
with torch.no_grad():
    sink_gpt = first_token_mass(gpt(xv, return_internals=True)[2])
print(f"uniform baseline {uniform:.3f} | mini-GPT: mean {sink_gpt.mean():.3f}, max {sink_gpt.max():.3f}")
""")

md(r"""
For comparison, the same measurement on a real LM with the 2026 block: **SmolLM2-135M** (30 layers, 9 query heads, 3 KV heads (GQA), RMSNorm, SwiGLU, RoPE, trained on ~2T tokens). We feed it 128 BPE tokens of Tiny Shakespeare from the middle of a line, without a BOS token, and ask for the attention weights (`attn_implementation="eager"`; the fused kernels do not return them). ≈ 270 MB download.
""")

code(r"""
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
lm = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", attn_implementation="eager", dtype=torch.float32).eval()
ids = tok(text[5000:6000], return_tensors="pt").input_ids[:, :128]
print("first token:", repr(tok.decode(ids[0, :1])), "| config:", lm.config.num_hidden_layers, "layers,",
      lm.config.num_attention_heads, "heads,", lm.config.num_key_value_heads, "KV heads")
with torch.no_grad():
    sink_lm = first_token_mass(lm(ids, output_attentions=True).attentions)

fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), gridspec_kw={"width_ratios": [1, 1.6]})
plot_sink(axes[0], sink_gpt, "our mini-GPT (0.9M params, 600 steps)")
axes[0].set_yticks(range(len(gpt.blocks)))
im = plot_sink(axes[1], sink_lm, "SmolLM2-135M", annotate=False)
fig.colorbar(im, ax=axes, label="mean attention on token 0", shrink=0.8)
fig.suptitle(f"Attention on the first token, queries t ≥ 32 (uniform ≈ {uniform:.3f})")
plt.show()
for lo, hi in [(0, 3), (3, 12), (12, 30)]:
    print(f"SmolLM2 layers {lo}-{hi - 1}: mean attention on token 0 = {sink_lm[lo:hi].mean():.2f}")
""")

md(r"""
- **Our mini-GPT has no sink:** every head puts ≈ the uniform share or less on token 0.
- **SmolLM2 does, from the middle of the network on:** in layers 0–11 token 0 gets 1–8% of the attention on average; in layers 12–29 it gets 53% on average, close to 100% in one head of layer 18, although token 0 here is the arbitrary word piece `or`. A few upper-layer heads stay below 20%: those are the heads that still read the context. The rest spend most of their attention on a token that carries no information for the query.
- This is why StreamingLLM keeps the first few tokens in the KV cache when it slides a window: evict them and the softmax mass has nowhere to go, and the model breaks. Sinks are a property of large, long-trained models; a 0.9M-parameter model after 600 steps does not show it (the cheap test: the same measurement on our model after 5000 steps on a GPU).

✏️ Is the sink about **position** or **content**? Replace the first token of `ids` by another one (e.g. `ids2 = ids.clone(); ids2[0, 0] = tok(" the").input_ids[0]`), and also drop it (`ids[:, 1:]`), then re-measure. Then plot `attentions[l][0, h, -1]` for a strong sink head: what else does it attend to?
""")

md(r"""
## 9. If time: the MoE mini-GPT, with and without the balancing loss

Replace every SwiGLU in the mini-GPT by an `MoE` with 4 experts, top-2, each expert half the hidden size (same active FFN parameters per token as the dense model). Train two copies for 200 steps: $\alpha=0$ and $\alpha=0.01$ on $\sum_\ell \mathcal{L}_{\text{aux}}^{(\ell)}$. Compare the expert load.
""")

code(r"""
def moe_gpt():
    return MiniGPT(vocab, ffn_fn=lambda: MoE(128, 192, n_experts=4, k=2)).to(device)

loads = {}
for coef in (0.0, 0.01):
    torch.manual_seed(0)
    m = moe_gpt()
    print(f"--- aux coefficient {coef} ---")
    train_lm(m, steps=200, aux_coef=coef, log_every=199)
    with torch.no_grad(): m(xv)
    loads[coef] = torch.stack([b.ffn.load for b in m.blocks]).cpu()
    print("aux per layer:", [round(b.ffn.aux_loss.item(), 3) for b in m.blocks])

fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
for ax, (coef, ld) in zip(axes, loads.items()):
    for l in range(ld.shape[0]): ax.bar(np.arange(4) + 0.2 * (l - 1.5), ld[l], width=0.2, label=f"block {l + 1}")
    ax.axhline(0.25, c="gray", ls="--"); ax.set(title=f"expert load, aux coef = {coef}", xlabel="expert", ylabel="fraction of routing slots")
axes[0].legend(); plt.tight_layout(); plt.show()
""")

md(r"""
Without the balancing loss the routers are already uneven after 200 steps: loads range from 10% to 37% of the slots, and $\mathcal{L}_{\text{aux}}$ is 1.03–1.12. With $\alpha=0.01$ every load is within 21–28% and $\mathcal{L}_{\text{aux}}\le 1.02$, at the same validation loss (1.92 in both runs). At this scale nothing collapses; in large models an unbalanced router overloads some devices and starves others, which is why every MoE recipe uses either this loss or DeepSeek-V3's per-expert bias.
""")

md(r"""
## Summary
- Attention is a data-dependent weighted average: $\mathrm{softmax}(QK^\top/\sqrt d)V$. Our version matches `F.scaled_dot_product_attention` and `nn.MultiheadAttention` to ~1e-16 in float64.
- The 2026 block changes only parts of the 2017 one: pre-norm with RMSNorm, SwiGLU, RoPE on $q,k$, GQA. Each piece is a few lines; RoPE's check is that the score depends only on the offset.
- GQA is MHA with KV heads repeated; MoE is a router + top-$k$ + a balancing loss $E\sum_i f_iP_i$ that equals 1 when the experts are evenly used.
- Without positions a Transformer block is permutation-equivariant (error ~1e-16); RoPE or a causal mask breaks the symmetry, and that is the only way the model knows token order.
- Position enters at the input (learned, sinusoidal), on the logits (T5, ALiBi) or on $q,k$ (RoPE). RoPE's logit is $\mathrm{Re}\sum_j q_j\bar k_j e^{i(m-n)\theta_j}$: fixed frequencies, content sets amplitudes and phases. Trained at 64 characters: RoPE is best in distribution (1.93) and breaks at offset ≈ 100, where its most-used pair passes $\pi$; ALiBi stays at 2.02 to 8× the length; sinusoidal and learned tables fail past 64. NTK-aware / by-parts scaling recover RoPE at 2× without training; PI needs fine-tuning, and at this scale plain fine-tuning was better.
- ViT = patchify (a strided convolution) + the same encoder blocks. On 10k CIFAR images it is data-starved; the learned position embeddings are where the 2D structure has to come from.
- Inside the mini-GPT, the logit lens shows the next-character prediction forming layer by layer. Attention sinks (most of a head's attention on the first token) are absent in our small model and present in most upper layers of SmolLM2-135M.

**Further watching:** 3Blue1Brown, [*Transformers, the tech behind LLMs* (ch. 5)](https://www.youtube.com/watch?v=wjZofJX0v4M) and [*Attention in transformers, step-by-step* (ch. 6)](https://www.youtube.com/watch?v=eMlx5fFNoYc); Karpathy, [*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY) (section 3 follows it, with the 2026 block); Umar Jamil, [*Attention is all you need (Transformer): model explanation, math, inference and training*](https://www.youtube.com/watch?v=bCz4OMemCcA).

**Further reading:** Su et al., [RoFormer](https://arxiv.org/abs/2104.09864) (2104.09864); Press et al., [ALiBi](https://arxiv.org/abs/2108.12409) (2108.12409); Chen et al., [position interpolation](https://arxiv.org/abs/2306.15595) (2306.15595); Peng et al., [YaRN](https://arxiv.org/abs/2309.00071) (2309.00071); Kazemnejad et al., [NoPE and length generalization](https://arxiv.org/abs/2305.19466) (2305.19466); Barbero et al., [Round and round we go! What makes rotary positional encodings useful?](https://arxiv.org/abs/2410.06205) (2410.06205); EleutherAI blog, [Rotary embeddings: a relative revolution](https://blog.eleuther.ai/rotary-embeddings/). Stanford CME 295 (2025), [Lecture 1: the Transformer](https://cme295.stanford.edu/slides/fall25-cme295-lecture1.pdf) and [Lecture 2: MHA/MQA/GQA, RoPE](https://cme295.stanford.edu/slides/fall25-cme295-lecture2.pdf).
""")

for k, p in B.write(STEM).items():
    print(k, p)

print("recap", make_recap(STEM, [
    ("slides/lectures/p7.pdf", [17]),                              # seq2seq + attention: where it came from
    ("slides/lectures/p8.pdf", [2, 12, 20, 30, 34, 36, 48]),       # gating, attention layer, self-attn, equivariance, mask, MHA, Transformer
    ("slides/lectures/p9.pdf", [2]),                               # ViT
    ("slides/2026-updates/L8b_architectures_looped_2026.pdf", [2, 5, 15, 26, 47, 32]),  # 2017 vs 2026 block, why each change, RoPE, GQA, MoE, attention sink
]))
