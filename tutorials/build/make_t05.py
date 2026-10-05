"""Builds T05_compute_efficiency.ipynb (Tutorial 5, Deep Learning 89-6877, Fall 2026)."""
from nbkit import Builder
from recap import make_recap

STEM = "T05_compute_efficiency"
B = Builder()
md, code = B.md, B.code

md(r"""
# Tutorial 5: Compute and scaling laws
**Deep Learning 89-6877, Fall 2026.** TA: Tal Fiskus.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idansc/dl-course-2026/blob/main/tutorials/T05_compute_efficiency.ipynb)
Recap slides: [T05_compute_efficiency_recap.pdf](https://github.com/idansc/dl-course-2026/blob/main/tutorials/recap/T05_compute_efficiency_recap.pdf)

Plan for today (≈ 75 min):
1. Lecture recap: scaling laws, quantization, pruning, distillation, KV cache, FlashAttention, speculative decoding (10 min)
2. From earlier tutorials: a char mini-GPT and a small CNN, trained here (run while we talk)
3. Resource accounting: parameters, FLOPs, memory, MFU (10 min)
4. Scaling laws: an IsoFLOP study on CPU, a Chinchilla fit, and what $10^{23}$ FLOPs buys (15 min; the training runs take ~3–4 min, start them early)
5. KV cache: identical tokens, fewer FLOPs (8 min)
6. Quantization by hand: per-channel int8 and int4, loss vs bits (8 min)
7. Magnitude pruning, unstructured and channel-level (8 min)
8. Knowledge distillation (6 min)
9. bf16 autocast, `torch.compile`, profiling (4 min)
10. Why decoding is memory-bound: tokens/s vs batch size and a roofline (6 min)
11. If time: speculative decoding with a 4-bit draft

Runs on CPU (Colab or laptop) in a few minutes. Cells marked ✏️ are for you to try.
<<STUDENT>>
**Before the tutorial:** fill in every `# TODO` (replace the `...`). Each one is followed by a check cell that prints the PyTorch answer next to yours. The solutions are presented in the tutorial.
<</STUDENT>>
""")

code(r"""
import copy, math, time
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt
torch.manual_seed(0); np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
bench = "cpu"      # device for the KV-cache timing and the profiler (sections 5 and 9); section 10 uses `device`
print("device:", device, "| bench:", bench, "| threads:", torch.get_num_threads())

def sync(dev):
    if dev == "cuda": torch.cuda.synchronize()
    if dev == "mps": torch.mps.synchronize()

def time_fn(fn, dev, reps=10):              # ms per call (best of reps)
    fn(); fn(); sync(dev)                      # warm-up (and compilation, for torch.compile)
    best = float("inf")
    for _ in range(reps):                      # min over repeats: robust to other processes on the machine
        t0 = time.time(); fn(); sync(dev); best = min(best, time.time() - t0)
    return best * 1e3
""")

md(r"""
## 1. Lecture recap

**Compute.** A dense transformer with $N$ weights costs $\approx 2N$ FLOPs per token forward and $\approx 6N$ per token in training; training with Adam holds $\approx 16$ bytes/param of state. Achieved FLOP/s over peak FLOP/s is the MFU.

**Scaling laws.** Loss falls as a power law in parameters, data and compute (Kaplan et al., 2020). Chinchilla (Hoffmann et al., 2022): $L(N,D)=E+A/N^{\alpha}+B/D^{\beta}$; at fixed $C\approx6ND$ the optimum grows $N$ and $D$ together, $N_{\text{opt}},D_{\text{opt}}\propto C^{\approx0.5}$, $\approx20$ tokens/param. Production models train far past that point because inference costs $2N$ per token.

**Why efficiency.** A 70B model is 140 GB in bf16. Decoding one token re-reads every weight, so at small batch
$\text{tokens/s} \approx \text{memory bandwidth} / \text{model bytes}$. The levers: fewer bits, fewer weights, fewer recomputed FLOPs.

**Linear quantization.** $r = S(q - Z)$ with integer $q$. Symmetric ($Z=0$), $b$ bits:
$$S = \frac{\max|r|}{2^{b-1}-1},\qquad q = \mathrm{clamp}\big(\mathrm{round}(r/S),\,-(2^{b-1}-1),\,2^{b-1}-1\big),\qquad \hat r = S\,q .$$
**Per-channel**: one $S$ per output row of $W$, so a single large row does not set the step size for all rows. Rounding error per weight is at most $S/2$.
GPTQ rounds column by column and corrects the remaining columns with second-order (Hessian) information; AWQ scales up the input channels with large activations before rounding. Rule of thumb: 8-bit weights are free, 4-bit costs little, 3-bit hurts, 2-bit needs QAT. FP8/FP4 trade mantissa bits for range (FP8 training in DeepSeek-V3, NVFP4 with per-16-element scales).

**Pruning.** Magnitude criterion: drop the weights with the smallest $|w|$. **Unstructured** (any weight) keeps accuracy at high sparsity but needs sparse kernels to be faster; **structured** (whole channels/rows, or 2:4) gives dense smaller matrices and real speed-ups. Prune → fine-tune, iterated. **Lottery tickets**: a sparse subnetwork, rewound to its original init, trains to full accuracy. **SparseGPT** (OBS-style updates) and **Wanda** (score $|W_{ij}|\cdot\|x_j\|$) prune LLMs one-shot, without retraining.

**Knowledge distillation** (Hinton et al., 2015). Soften both outputs with temperature $T$, $p^T = \mathrm{softmax}(z/T)$:
$$L = \alpha\,T^2\,\mathrm{KL}\big(p_{\text{teacher}}^T \,\|\, p_{\text{student}}^T\big) + (1-\alpha)\,\mathrm{CE}(z_{\text{student}}, y).$$
$T^2$ keeps the soft-target gradient on the same scale as the CE gradient.

**Serving.** **KV cache**: in autoregressive decoding the keys and values of past tokens never change, so store them and compute $q,k,v$ only for the new token. Memory: $2\cdot n_{\text{layers}}\cdot n_{\text{tokens}}\cdot d\cdot$ bytes per value, per sequence (10 GB for Llama-2-70B with MHA at 4k tokens). **FlashAttention**: tile $QK^\top$ so the $N\times N$ matrix never goes to HBM; exact, not an approximation. **Speculative decoding**: a small draft proposes $k$ tokens, the large model checks them in one forward pass; output is identical to the large model's, 2–3× faster because one weight read now serves several tokens.
""")

md(r"""
## 2. From earlier tutorials: a char mini-GPT and a small CNN

A compact version of the decoder-only transformer (Tiny Shakespeare, characters as tokens) and a small CNN on a FashionMNIST subset. Both train in 1–3 minutes on a laptop. On a Colab GPU, increase `n_embd`, `n_layer`, `steps` and the data subsets.
""")

code(r"""
import os, urllib.request
os.makedirs("data", exist_ok=True)
path = "data/tinyshakespeare.txt"
if not os.path.exists(path):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", path)
text = open(path).read()
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
encode = lambda s: torch.tensor([stoi[c] for c in s]); decode = lambda t: "".join(chars[i] for i in t)
data = encode(text); n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
vocab, block_size = len(chars), 128

def get_batch(split, bs=16, gen=None):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size - 1, (bs,), generator=gen)
    x = torch.stack([d[i:i + block_size] for i in ix]); y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.n_head, self.qkv, self.proj = n_head, nn.Linear(d, 3 * d), nn.Linear(d, d)
    def split(self, x):                       # (B, T, d) -> q, k, v each (B, heads, T, d/heads)
        B, T, C = x.shape
        return [t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in self.qkv(x).split(C, dim=2)]
    def merge(self, y):                       # (B, heads, T, hd) -> (B, T, d)
        B, h, T, hd = y.shape
        return self.proj(y.transpose(1, 2).reshape(B, T, h * hd))
    def forward(self, x):
        q, k, v = self.split(x)
        return self.merge(F.scaled_dot_product_attention(q, k, v, is_causal=True))

class Block(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(d), CausalSelfAttention(d, n_head)
        self.ln2, self.mlp = nn.LayerNorm(d), nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class MiniGPT(nn.Module):
    def __init__(self, vocab, n_embd=96, n_layer=3, n_head=4):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, n_embd), nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f, self.head = nn.LayerNorm(n_embd), nn.Linear(n_embd, vocab)
    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))

@torch.no_grad()
def val_loss(model, n_batches=20):
    gen = torch.Generator().manual_seed(123)  # same validation batches every call
    model.eval()
    losses = [F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
              for x, y in (get_batch("val", 16, gen) for _ in range(n_batches))]
    return sum(losses) / len(losses)

torch.manual_seed(0)
gpt = MiniGPT(vocab).to(device)
opt = torch.optim.AdamW(gpt.parameters(), lr=1e-2)
steps, t0 = 1500, time.time()
for step in range(steps):
    for g in opt.param_groups: g["lr"] = 1e-2 * min(1, (step + 1) / 50) * 0.5 * (1 + math.cos(math.pi * step / steps))
    x, y = get_batch("train")
    loss = F.cross_entropy(gpt(x).flatten(0, 1), y.flatten())
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0 or step == steps - 1:
        print(f"step {step:4d}  train loss {loss.item():.3f}")
gpt.eval()
print(f"mini-GPT: {sum(p.numel() for p in gpt.parameters())/1e3:.0f}k params, val loss {val_loss(gpt):.3f}, trained in {time.time()-t0:.0f}s")
""")

code(r"""
from torchvision import datasets
def load_fmnist(train, n):
    ds = datasets.FashionMNIST("data", train=train, download=True)
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(ds), generator=g)[:n]
    X = (ds.data[idx].float() / 255 - 0.286) / 0.353
    return X.unsqueeze(1).to(device), ds.targets[idx].to(device)
Xtr, ytr = load_fmnist(True, 10000)
Xte, yte = load_fmnist(False, 2000)

class SmallCNN(nn.Module):
    def __init__(self, c1=32, c2=64, hidden=128):
        super().__init__()
        self.conv1, self.conv2 = nn.Conv2d(1, c1, 3, padding=1), nn.Conv2d(c1, c2, 3, padding=1)
        self.fc1, self.fc2 = nn.Linear(c2 * 7 * 7, hidden), nn.Linear(hidden, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))

def train(model, X, y, epochs, lr=2e-3, bs=128, loss_fn=None, after_step=None, seed=0):
    # loss_fn(logits, y, xb) -> loss; default CE. after_step() runs after every optimizer step (used to keep pruning masks).
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(X), generator=g).split(bs):
            xb, yb = X[idx], y[idx]
            logits = model(xb)
            loss = loss_fn(logits, yb, xb) if loss_fn else F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            if after_step: after_step()
    model.eval()
    return model

@torch.no_grad()
def accuracy(model, X=None, y=None):
    X, y = (Xte, yte) if X is None else (X, y)
    model.eval()
    return (model(X).argmax(1) == y).float().mean().item()

n_params = lambda m: sum(p.numel() for p in m.parameters())
torch.manual_seed(0); t0 = time.time()
cnn = train(SmallCNN().to(device), Xtr, ytr, epochs=4)
print(f"CNN: {n_params(cnn)/1e3:.0f}k params, test accuracy {accuracy(cnn):.3f}, trained in {time.time()-t0:.0f}s")
""")

md(r"""
## 3. Resource accounting: parameters, FLOPs, memory, MFU

Before making a model cheaper, count what it costs. For a decoder with $L$ layers, width $d$, vocabulary $V$, context $T$:

**Parameters.** Per block: $W_{qkv}$ $3d^2+3d$, $W_o$ $d^2+d$, MLP $8d^2+5d$, two LayerNorms $4d$, so $12d^2+13d$. Plus token and position embeddings $Vd + Td$, final LayerNorm $2d$, head $dV+V$.

**FLOPs per token.** A matmul with an $m\times n$ weight costs $2mn$ FLOPs per input row (one multiply, one add), so the weight matmuls cost $2N$ per token, with $N$ the number of matmul weights (embedding lookups are free). Attention adds $QK^\top$ and $AV$: $2Td$ each per layer, so $4LTd$. Backward costs twice the forward (gradient w.r.t. the input and w.r.t. the weights):
$$C_{\text{fwd}} \approx 2N + 4LTd,\qquad C_{\text{train}} \approx 3\,C_{\text{fwd}} \approx 6N \text{ per token}.$$

**Memory for training with Adam.** Weights, gradients and the two Adam moments: $4+4+8 = 16$ bytes/param in fp32. Mixed precision (bf16 weights and grads, fp32 master copy and moments): $2+2+4+8 = 16$ bytes/param again; bf16 saves *activation* memory and time, not optimizer memory. Activations scale with $B\cdot T\cdot L\cdot d$ and are freed after the backward pass.

**MFU** (model FLOPs utilization) $=\dfrac{C_{\text{train}}\cdot\text{tokens/s}}{\text{peak FLOP/s}}$: the fraction of the hardware doing useful model math. Good LLM training runs reach 40–60% on H100s.
""")

code(r"""
def count_params(V, T, d, L):
    #>> total parameters of MiniGPT from the formula above (embeddings, L blocks, final LayerNorm, head)
    return V * d + T * d + L * (12 * d * d + 13 * d) + 2 * d + d * V + V
    #<<

def flops_per_token(N_matmul, L, T, d, train=False):
    fwd = 2 * N_matmul + 4 * L * T * d  # @student: fwd = ...  # TODO: weight matmuls + attention (QK^T and AV), forward only
    return 3 * fwd if train else fwd

d, L = gpt.tok.embedding_dim, len(gpt.blocks)
N_matmul = sum(m.weight.numel() for m in gpt.modules() if isinstance(m, nn.Linear))
print(f"params: formula {count_params(vocab, block_size, d, L):,}   sum(p.numel()) {n_params(gpt):,}")
print(f"matmul weights N = {N_matmul:,}; attention term 4LTd = {4*L*block_size*d:,} ({4*L*block_size*d / (2*N_matmul):.0%} of 2N at T={block_size})")
""")

md(r"""
**Check the FLOP count** with `torch.utils.flop_counter.FlopCounterMode`, which counts the FLOPs of every matmul PyTorch actually runs. It does not see inside the fused `scaled_dot_product_attention` kernel on CPU/MPS, so for counting we temporarily swap in the textbook version of attention.
""")

code(r"""
from torch.utils.flop_counter import FlopCounterMode
def sdpa_math(q, k, v, attn_mask=None, is_causal=False):
    a = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if is_causal: attn_mask = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool, device=q.device).tril()
    if attn_mask is not None: a = a.masked_fill(~attn_mask, float("-inf"))
    return a.softmax(-1) @ v

xc, yc = get_batch("train")
gpt_cpu = copy.deepcopy(gpt).cpu()            # count on CPU: on GPU/MPS the backward runs in another thread the counter does not see
sdpa_fused, F.scaled_dot_product_attention = F.scaled_dot_product_attention, sdpa_math
try:
    with FlopCounterMode(display=False) as fc:
        gpt_cpu(xc.cpu())
    fwd_counted = fc.get_total_flops() / xc.numel()
    with FlopCounterMode(display=False) as fc:
        F.cross_entropy(gpt_cpu(xc.cpu()).flatten(0, 1), yc.cpu().flatten()).backward()
    train_counted = fc.get_total_flops() / xc.numel()
finally:
    F.scaled_dot_product_attention = sdpa_fused
print(f"forward  FLOPs/token: formula {flops_per_token(N_matmul, L, block_size, d):,}   counted {fwd_counted:,.0f}")
print(f"training FLOPs/token: formula {flops_per_token(N_matmul, L, block_size, d, train=True):,}   counted {train_counted:,.0f}")
""")

md(r"""
The formula matches the counter exactly, for the forward pass and for a full training step (both ignore LayerNorm, GELU, softmax and the optimizer update, which are a few percent of the total). Relative to $2N\approx 24Ld^2$, the attention term is $4LTd/24Ld^2 = T/6d$: 22% for this narrow model at $T=128$, 3% for GPT-3 ($d=12288$, $T=2048$). That is why "$6N$ FLOPs per training token" is the standard compute budget, e.g. $C\approx 6ND$ for $D$ training tokens.

**Memory.** The AdamW optimizer we trained with holds its state in `opt.state`; we check the 16 bytes/param rule against the actual tensors, and measure activation memory by recording every tensor autograd saves for the backward pass.
""")

code(r"""
def train_state_bytes(N, mixed=False):
    # weights + grads + Adam (m, v); mixed: bf16 weights and grads + fp32 master copy + fp32 moments
    return N * (2 + 2 + 4 + 8) if mixed else N * (4 + 4 + 8)  # @student: return ...  # TODO: bytes for weights, grads, Adam moments

N = sum(p.numel() for p in gpt.parameters())
F.cross_entropy(gpt(xc).flatten(0, 1), yc.flatten()).backward()           # populate .grad
measured = sum(p.nbytes + p.grad.nbytes for p in gpt.parameters()) + \
           sum(v.nbytes for st in opt.state.values() for k, v in st.items() if k in ("exp_avg", "exp_avg_sq"))
gpt.zero_grad(set_to_none=True)
print(f"weights+grads+Adam, fp32: formula {train_state_bytes(N)/1e6:.2f} MB   measured {measured/1e6:.2f} MB   "
      f"(mixed bf16: {train_state_bytes(N, mixed=True)/1e6:.2f} MB)")

def activation_bytes(model, x, dtype=None):
    saved = []
    ctx = torch.autocast(device_type=x.device.type, dtype=dtype) if dtype else nullcontext()
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t.nbytes) or t, lambda t: t), ctx:
        model(x)
    return sum(saved)                           # an upper bound: a tensor saved by two ops is counted twice
model_act = activation_bytes(gpt, xc)
print(f"activations saved for backward, batch {xc.shape[0]}×{xc.shape[1]}: {model_act/1e6:.1f} MB fp32 "
      f"= {model_act / (xc.numel() * L * d * 4):.0f} floats per token per layer per channel")
for name, (V_, T_, d_, L_) in {"GPT-2 small": (50257, 1024, 768, 12), "Llama-2-7B": (32000, 4096, 4096, 32)}.items():
    Nb = count_params(V_, T_, d_, L_)
    print(f"{name}: ~{Nb/1e9:.2f}B params (GPT-style blocks), Adam training state {train_state_bytes(Nb)/1e9:.0f} GB, "
          f"bf16 weights {2*Nb/1e9:.1f} GB, KV cache per 1 sequence of {T_} tokens in bf16 {2*L_*T_*d_*2/1e9:.2f} GB")
""")

md(r"""
The 16 bytes/param rule matches the actual tensors exactly (5.77 MB for 361k parameters). Mixed precision gives the same 16 bytes. Activations for one batch of 16×128 tokens take 66 MB, about 11× the whole training state: for small models and long sequences activations dominate, which is why activation checkpointing (recompute in the backward pass instead of storing) and smaller micro-batches are the first memory knobs. For a 7B model the training state alone is ~108 GB, more than one 80 GB GPU holds before a single activation is stored; this is what ZeRO/FSDP shard across GPUs.

**MFU on this device.** Time a training step, convert to model FLOP/s with the formula, and divide by the best FLOP/s this device reaches on one large matmul (on a laptop we have no reliable datasheet number; on a GPU use the datasheet peak, e.g. 65 TFLOP/s fp16 for a T4, 989 TFLOP/s bf16 for an H100).
""")

code(r"""
def step_fn():
    F.cross_entropy(gpt(xc).flatten(0, 1), yc.flatten()).backward()
    gpt.zero_grad(set_to_none=True)
ms_step = time_fn(step_fn, device, reps=10)
achieved = flops_per_token(N_matmul, L, block_size, d, train=True) * xc.numel() / (ms_step / 1e3)
A_, B_ = torch.randn(2048, 2048, device=device), torch.randn(2048, 2048, device=device)
ms_mm = time_fn(lambda: A_ @ B_, device, reps=30)
peak_dev = 2 * 2048 ** 3 / (ms_mm / 1e3)
print(f"training step: {ms_step:.1f} ms for {xc.numel()} tokens -> {achieved/1e9:.1f} GFLOP/s of model math")
print(f"2048x2048 fp32 matmul on {device}: {peak_dev/1e9:.0f} GFLOP/s   ->   MFU ≈ {achieved/peak_dev:.0%}")
""")

md(r"""
The training step reaches a few hundred GFLOP/s of model math, a small fraction (5–20% across our runs; timings on a shared laptop GPU vary) of what the same device does on one large matmul. A 361k-parameter model is a long chain of small ops: per-op launch cost and memory traffic of the elementwise ops (LayerNorm, GELU, residual adds), not arithmetic, set the step time.

✏️ Re-run the MFU cell with `n_embd=384` (and fewer steps), or with batch 32. Which change raises MFU more, and why? (Hint: per-op overhead vs work per op.)
""")

md(r"""
**Past exam question (Moed C, 2026).** A single-head self-attention block acts on an input sequence $X\in\mathbb R^{T\times d}$ ($T$ = sequence length, $d$ = embedding dimension):
$$\mathrm{Attn}(X)=\mathrm{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}}\Big)V,\qquad Q=XW_Q,\ K=XW_K,\ V=XW_V.$$
Explain why the memory complexity of computing the attention matrix $\mathrm{softmax}(QK^\top/\sqrt{d_k})$ is $\mathcal O(T^2)$, and why the compute complexity of the full attention operation is $\mathcal O(T^2 d)$. Break the cost down step by step: computing $QK^\top$, the softmax, and the multiplication by $V$.

<<STUDENT>>
✏️ Your answer:
<</STUDENT>>
<<SOLUTION>>
**Answer.** $Q,K,V\in\mathbb R^{T\times d}$.
- $QK^\top$: one dot product of length $d$ for each of the $T^2$ pairs of positions, $T^2 d$ multiply-adds ($2T^2d$ FLOPs). The result is a $T\times T$ matrix, so storing it takes $\mathcal O(T^2)$ memory.
- Softmax: exp, row sum and division per entry, $\mathcal O(T^2)$ time; its output is again $T\times T$, $\mathcal O(T^2)$ memory.
- $AV$: a $(T\times T)(T\times d)$ product, again $T^2 d$ multiply-adds.

Total: $\mathcal O(T^2 d)$ time, $\mathcal O(T^2)$ memory for the attention matrix. (The official solution stops here. The projections $XW_Q, XW_K, XW_V$ add $3Td^2$ multiply-adds, so the whole block is $\mathcal O(T^2d + Td^2)$; the $T^2 d$ term dominates once $T > d$.) The check below counts the FLOPs with PyTorch and measures the bytes of $A$.
<</SOLUTION>>
""")

code(r"""
def attn_cost(T, d):
    #>> flops_qk, flops_av = FLOPs (2 per multiply-add) of QK^T and of A·V; bytes_A = bytes of the T×T fp32 attention matrix A
    flops_qk = 2 * T * T * d
    flops_av = 2 * T * T * d
    bytes_A = 4 * T * T
    #<<
    return flops_qk, flops_av, bytes_A

d_h = 64
for T_a in [256, 512, 1024]:
    q_a, k_a, v_a = torch.randn(3, T_a, d_h).unbind(0)
    with FlopCounterMode(display=False) as fc:
        A_att = (q_a @ k_a.T / math.sqrt(d_h)).softmax(-1)
        o_att = A_att @ v_a
    f_qk, f_av, b_A = attn_cost(T_a, d_h)
    print(f"T={T_a:5d}: FLOPs formula {f_qk + f_av:>11,}  counted {fc.get_total_flops():>11,}   "
          f"A: formula {b_A/1e6:5.2f} MB  measured {A_att.nbytes/1e6:5.2f} MB")
""")

md(r"""
<<SOLUTION>>
Formula and counter agree exactly (the counter does not count the softmax, which is $\mathcal O(T^2)$, lower order). Doubling $T$ multiplies both the FLOPs and the bytes of $A$ by 4.

<</SOLUTION>>
**Past exam question (Moed C, 2026).** In a standard Transformer the memory complexity of the attention matrix is $\mathcal O(T^2)$ in the sequence length $T$. This becomes a critical bottleneck when training on **very long sequences** (long documents, video, or a long-context LLM with $T\sim10^4$ and more). **Propose one idea** (algorithmic or architectural) for reducing this complexity for long sequences, and **explain briefly** how it achieves it (in terms of the structure of the attention matrix, or in terms of the new complexity).

<<STUDENT>>
✏️ Your answer:
<</STUDENT>>
<<SOLUTION>>
**Answer (any one, as in the official solution).**
- **Sparse / sliding-window attention** (Longformer, BigBird): each token attends to a local window of $w$ tokens plus a few global tokens. The attention matrix is banded, $Tw$ entries instead of $T^2$: $\mathcal O(Twd)$ time, $\mathcal O(Tw)$ memory. For $T=16{,}384$, $w=512$: 32× fewer scored pairs.
- **Linear attention / Performer**: replace $\mathrm{softmax}(QK^\top)$ by a kernel feature map $\phi(Q)\phi(K)^\top$ and compute $\phi(Q)\big(\phi(K)^\top V\big)$ right to left: $\mathcal O(Td^2)$, no $T\times T$ matrix.
- **Linformer**: project $K,V$ along the sequence axis to $k\ll T$ rows: $\mathcal O(Tkd)$.
- **FlashAttention** (lecture): exact attention with the same $\mathcal O(T^2d)$ FLOPs, computed tile by tile in on-chip SRAM with an online softmax, so the $T\times T$ matrix is never written to HBM. Extra memory drops from $\mathcal O(T^2)$ to $\mathcal O(T)$; compute does not.
<</SOLUTION>>
""")

md(r"""
**Practical note: FlashAttention is exact, via the online softmax**

FlashAttention does not approximate the softmax; it reorders it. Each query row keeps a running max $m$, a running sum $\ell$ and an unnormalized output $o$, and visits $K,V$ one block at a time. For a new block of scores $s_j$:
$$m' = \max\big(m, \max_j s_j\big),\qquad \ell' = e^{m-m'}\,\ell + \sum_j e^{s_j-m'},\qquad o' = e^{m-m'}\,o + \sum_j e^{s_j-m'}\,v_j,$$
and the output is $o/\ell$ after the last block. The factor $e^{m-m'}$ rescales everything accumulated under the old max, so the result equals $\mathrm{softmax}(QK^\top/\sqrt d)V$ up to float rounding, while only one $B\times B$ tile of scores exists at a time. Subtracting the running max also keeps $e^{s}$ from overflowing.
""")

code(r"""
def flash_attention(q, k, v, block=128, causal=False):
    # q, k, v: (batch, heads, T, d). Query tiles of `block` rows, K/V tiles of `block` columns.
    T, d = q.shape[-2], q.shape[-1]
    out = torch.empty_like(q)
    for i in range(0, T, block):
        qi = q[..., i:i + block, :] / math.sqrt(d)
        m = torch.full(qi.shape[:-1] + (1,), float("-inf"))     # running max per row
        l = torch.zeros(qi.shape[:-1] + (1,))                   # running sum of exp
        o = torch.zeros_like(qi)                                # running unnormalized output
        for j in range(0, min(i + block, T) if causal else T, block):   # causal: skip tiles entirely in the future
            s = qi @ k[..., j:j + block, :].transpose(-2, -1)  # one (block × block) tile of scores
            if causal and j + block > i:
                rows = torch.arange(i, i + s.shape[-2])[:, None]; cols = torch.arange(j, j + s.shape[-1])[None]
                s = s.masked_fill(cols > rows, float("-inf"))
            #>> online-softmax update: new max m_new, rescale factor exp(m − m_new), update l and o with this tile, then m = m_new
            m_new = torch.maximum(m, s.amax(-1, keepdim=True))
            p = torch.exp(s - m_new)
            corr = torch.exp(m - m_new)
            l = l * corr + p.sum(-1, keepdim=True)
            o = o * corr + p @ v[..., j:j + block, :]
            m = m_new
            #<<
        out[..., i:i + block, :] = o / l
    return out

torch.manual_seed(0)
Bf, Hf, Tf, Df, blk = 2, 4, 1024, 64, 128
qf, kf, vf = torch.randn(3, Bf, Hf, Tf, Df).unbind(0)
for causal in (False, True):
    diff = (flash_attention(qf, kf, vf, blk, causal) - F.scaled_dot_product_attention(qf, kf, vf, is_causal=causal)).abs().max().item()
    print(f"causal={causal!s:5}: max |blocked − F.scaled_dot_product_attention| = {diff:.1e}")
    assert diff < 1e-5
q_big = 30 * qf                                    # logits up to ~10^3: a naive exp overflows
print(f"large logits: naive exp(max score) = {torch.exp((q_big[0, 0] @ kf[0, 0].T / 8).max()).item()},  "
      f"max |blocked − SDPA| = {(flash_attention(q_big, kf, vf, blk) - F.scaled_dot_product_attention(q_big, kf, vf)).abs().max().item():.1e}")
full_mb, tile_mb = Bf * Hf * Tf * Tf * 4 / 1e6, Bf * Hf * blk * blk * 4 / 1e6
state_mb = Bf * Hf * blk * (Df + 2) * 4 / 1e6
print(f"score matrix never materialized: {full_mb:.1f} MB (B·H·T²·4 bytes); one tile {tile_mb:.2f} MB ({full_mb / tile_mb:.0f}× less), "
      f"plus m, l, o for one query tile {state_mb:.2f} MB")
""")

md(r"""
The blocked version matches PyTorch's fused kernel to ~1e-6 with and without the causal mask, and stays exact when the logits are large enough that $e^{s}$ alone is `inf` in fp32. At $T=1024$ the $T\times T$ scores of 2×4 heads would take 34 MB; the loop never holds more than one 0.5 MB tile (64× less, and the ratio grows as $T^2/B^2$). The FLOPs are the same as standard attention; the saving is memory traffic: on a GPU the tile and $m,\ell,o$ live in SRAM, and the scores are never written to HBM. The backward pass recomputes the tiles instead of storing them. In Python this loop is slower than the fused kernel; the point is the algorithm, not the speed.
""")

md(r"""
## 4. Scaling laws: what a compute budget buys

Section 3 gave the cost of training: $C \approx 6ND$ FLOPs for $N$ parameters and $D$ tokens. Scaling laws answer the next question: for a given $C$, which $(N, D)$ gives the lowest loss, and what loss will it be?

**Kaplan et al. (2020).** Test loss of a language model is a power law in each resource when the other two are not the bottleneck:
$$L(N) = (N_c/N)^{\alpha_N},\quad L(D) = (D_c/D)^{\alpha_D},\quad L(C_{\min}) = (C_c/C_{\min})^{\alpha_C},\qquad \alpha_N\approx0.076,\ \alpha_D\approx0.095,\ \alpha_C\approx0.050,$$
with $N$ the non-embedding parameters. Their compute-optimal allocation grew the model much faster than the data: $N_{\text{opt}}\propto C^{0.73}$, $D_{\text{opt}}\propto C^{0.27}$. GPT-3 (175B parameters, 300B tokens, 1.7 tokens/param) was trained in that spirit.

**Hoffmann et al. (2022), Chinchilla.** One law for both resources, with an irreducible term $E$ (the entropy of text):
$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}},\qquad E=1.69,\ A=406.4,\ B=410.7,\ \alpha=0.34,\ \beta=0.28 \quad\text{(Approach 3 fit)}.$$
Three methods (fixed $N$ with varying training lengths; **IsoFLOP profiles**; the parametric fit above) all give $N_{\text{opt}}\propto C^{a}$, $D_{\text{opt}}\propto C^{b}$ with $a\approx b\approx 0.5$: double the compute, grow the model and the data by $\sqrt2$ each. In numbers, $\approx 20$ tokens per parameter. Chinchilla (70B, 1.4T tokens) beat Gopher (280B, 300B tokens) with the same compute.

**IsoFLOP profile.** Fix $C$; for each $N$ train on $D=C/(6N)$ tokens; plot final loss vs $N$. Too small a model is capacity-limited, too large a model sees too few tokens, so the curve is a valley. Its minimum is $N_{\text{opt}}(C)$; the minima over several $C$ lie on a line in log-log.

**Why Kaplan got a different exponent.** (i) All Kaplan runs used one long learning-rate schedule and were read off early: a model stopped before its cosine schedule has decayed looks worse than it is, which penalises the small models trained on many tokens, the ones that should win at large $C$. Chinchilla matched the cosine length to each run's token budget. (ii) Kaplan counted non-embedding parameters and studied small models, where the embeddings are a large share of the total; Pearce & Song (2024) show that this alone pulls the fitted exponent from ≈ 0.5 toward 0.73, and Porian et al. (2024) add the warm-up length and the output layer's FLOPs as further causes.

**Train-optimal is not inference-optimal.** Chinchilla minimises training compute. A deployed model also pays $2N$ FLOPs per generated token, so a smaller model trained on more tokens is cheaper over its lifetime. Llama 3 8B was trained on 15T tokens, $\approx 1{,}900$ tokens/param, ~90× the Chinchilla ratio. Section 4.3 measures what that costs in loss.

**Data-constrained scaling** (Muennighoff et al., 2023). When unique data runs out, repeating it for up to ~4 epochs is almost as good as fresh tokens; beyond ~16 epochs extra repetitions are worth almost nothing.

**Test-time compute** is a further axis: sampling many answers and verifying them, or a longer chain of thought, has its own scaling curve. Snell et al. (2024) find that on problems a small model can partly solve, optimally spent inference compute can beat a 14× larger model; on hard problems pre-training compute is still the better buy (lecture 12, reasoning models).

### 4.1 A mini IsoFLOP study on CPU

Setup: the mini-GPT from section 2 at depth 2 and 9 widths $d=8\dots128$, i.e. $N$ = 1.7k to 400k non-embedding parameters (Kaplan's count: $12d^2+13d$ per block), context 64 characters of Tiny Shakespeare. Four budgets $C\in\{5\cdot10^{10},\,1.5\cdot10^{11},\,5\cdot10^{11},\,1.5\cdot10^{12}\}$ FLOPs, 4 consecutive sizes per budget (a window around the expected optimum, as in Chinchilla), $D = C/(6N)$ tokens per run, AdamW with a cosine schedule whose length equals the run's token budget (warm-up 5%, decay to 10% of the peak). The peak learning rate is $\min(0.08,\,1.6/d)$, from a short sweep at four widths: with one fixed rate, the wide models are mistuned and look worse than they are, the same kind of failure as in the Kaplan story above. The batch grows with $D$ so every run has ≥ 250 steps; one seed per run.

Why so small: a compute-optimal character model at $10^{12}$ FLOPs has tens of thousands of parameters (measured below). By our fit, a $10^6$-parameter model is compute-optimal only at a few $\times10^{15}$ FLOPs, ~2000× this section's largest budget; a 1M-parameter model at our budgets sees too few tokens and sits on the right wall of the valley. On a Colab GPU, set `scale_dev = device`, multiply `Cs` by 30 and append wider models, e.g. `(192, 2, 2)`, `(256, 2, 2)`, to the ladder.
""")

code(r"""
from scipy.optimize import minimize
from scipy.special import logsumexp
scale_dev, T_s = "cpu", 64                      # tiny models: CPU is faster than a GPU's per-kernel launch cost
ladder = [(d, 2, 2) for d in (8, 12, 16, 24, 32, 48, 64, 96, 128)]   # (n_embd, n_layer, n_head): depth 2, width ×1.5 per step
n_nonemb = lambda cfg: cfg[1] * (12 * cfg[0] ** 2 + 13 * cfg[0])   # blocks only: no embeddings, final LN or head
Ns = [n_nonemb(c) for c in ladder]

def tokens_for_budget(C, N):
    #>> the number of training tokens D that a budget of C FLOPs buys for a model with N parameters (C ≈ 6ND)
    return C / (6 * N)
    #<<

gv = torch.Generator().manual_seed(123)
vix = torch.randint(len(val_data) - T_s - 1, (2048,), generator=gv)   # fixed 2048 × 64 validation characters
VX = torch.stack([val_data[i:i + T_s] for i in vix]); VY = torch.stack([val_data[i + 1:i + T_s + 1] for i in vix])

def train_run(cfg, D, seed=0):
    d, L_, h = cfg
    torch.manual_seed(seed); m = MiniGPT(vocab, d, L_, h).to(scale_dev)
    bs = int(min(128, max(8, round(D / (T_s * 250)))))       # ≥ 250 steps per run
    steps = max(1, round(D / (bs * T_s)))
    lr0 = min(0.08, 1.6 / d)                                   # peak LR ∝ 1/width, from a short sweep
    opt = torch.optim.AdamW(m.parameters(), lr=lr0, betas=(0.9, 0.95), weight_decay=0.1)
    g, warm = torch.Generator().manual_seed(seed), max(1, steps // 20)
    m.train()
    for s in range(steps):                                     # cosine over exactly this run's token budget
        for gr in opt.param_groups: gr["lr"] = lr0 * min(1, (s + 1) / warm) * (0.1 + 0.45 * (1 + math.cos(math.pi * s / steps)))
        ix = torch.randint(len(train_data) - T_s - 1, (bs,), generator=g)
        x = torch.stack([train_data[i:i + T_s] for i in ix]).to(scale_dev)
        y = torch.stack([train_data[i + 1:i + T_s + 1] for i in ix]).to(scale_dev)
        loss = F.cross_entropy(m(x).flatten(0, 1), y.flatten())
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        vl = np.mean([F.cross_entropy(m(VX[i:i + 256].to(scale_dev)).flatten(0, 1), VY[i:i + 256].to(scale_dev).flatten()).item()
                      for i in range(0, len(VX), 256)])
    return steps * bs * T_s, vl

for cfg, N_ in zip(ladder, Ns):                                # check the parameter count against the module
    assert N_ == sum(p.numel() for p in MiniGPT(vocab, *cfg).blocks.parameters())
print("non-embedding N:", Ns)
print(f"check: 6·N·D = {6 * Ns[3] * tokens_for_budget(1.5e11, Ns[3]):.3e} (should be 1.5e11)")
""")

code(r"""
Cs = [5e10, 1.5e11, 5e11, 1.5e12]
runs, t_all = [], time.time()
for C in Cs:
    guess = math.sqrt(C / (6 * 300))                           # window centre: a prior guess D/N ≈ 300 (from a pilot run)
    k = int(np.argmin([abs(math.log(N_ / guess)) for N_ in Ns])); k = min(max(k, 1), len(Ns) - 3)
    for j in range(k - 1, k + 3):                              # 4 sizes around the guess (small models on many tokens are the slow runs)
        t0 = time.time()
        D_, l_ = train_run(ladder[j], tokens_for_budget(C, Ns[j]))
        runs.append(dict(C=C, N=Ns[j], D=D_, loss=l_))
        print(f"C={C:.2g}  N={Ns[j]:>7,}  D={D_:>11,}  D/N={D_ / Ns[j]:7.1f}  val loss {l_:.3f}  ({time.time() - t0:.0f}s)")
print(f"{len(runs)} runs in {time.time() - t_all:.0f}s")
rN, rD, rL, rC = (np.array([r[k] for r in runs], float) for k in ("N", "D", "loss", "C"))
""")

md(r"""
### 4.2 Fitting $L(N, D) = E + A/N^\alpha + B/D^\beta$

As in Chinchilla, fit in log space with $a=\log A$, $b=\log B$, $e=\log E$:
$$\log \hat L = \mathrm{LSE}\big(a-\alpha\log N,\ b-\beta\log D,\ e\big),\qquad \min_{a,b,e,\alpha,\beta}\ \sum_i \mathrm{Huber}_\delta\big(\log\hat L_i-\log L_i\big),\ \delta=10^{-3},$$
with L-BFGS from a grid of starting points (the objective is not convex). Huber with a small $\delta$ is almost an $\ell_1$ loss: one bad run does not drag the fit.
""")

code(r"""
def huber(r, delta=1e-3):
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * r ** 2, delta * (a - 0.5 * delta))

def objective(theta, N, D, L):
    a, b, e, alpha, beta = theta
    #>> log L̂ = logsumexp of (a − α log N, b − β log D, e); return the summed Huber loss of log L̂ − log L
    log_pred = logsumexp(np.stack([a - alpha * np.log(N), b - beta * np.log(D), np.full_like(N, e)]), axis=0)
    return huber(log_pred - np.log(L)).sum()
    #<<

def fit_law(N, D, L):
    best = None
    for a0 in (0, 5, 10):
        for b0 in (0, 5, 10):
            for e0 in (-1, 0, 0.5):
                for al0, be0 in ((0.2, 0.2), (0.5, 0.5), (0.3, 0.8), (0.8, 0.3)):
                    r = minimize(objective, [a0, b0, e0, al0, be0], args=(N, D, L), method="L-BFGS-B")
                    if best is None or r.fun < best.fun: best = r
    a, b, e, al, be = best.x
    return dict(E=math.exp(e), A=math.exp(a), B=math.exp(b), alpha=al, beta=be)

def law(p, N, D):
    return p["E"] + p["A"] / N ** p["alpha"] + p["B"] / D ** p["beta"]

show = lambda p: "  ".join(f"{k}={v:.3f}" for k, v in p.items())
true = dict(E=1.3, A=30.0, B=60.0, alpha=0.4, beta=0.3)       # check: refit noiseless synthetic losses at our (N, D) grid
p_syn = fit_law(rN, rD, law(true, rN, rD))
print("true     ", show(true)); print("recovered", show(p_syn))
assert all(abs(p_syn[k] / true[k] - 1) < 1e-2 for k in true)
""")

code(r"""
fit = fit_law(rN, rD, rL)
res = law(fit, rN, rD) - rL
print("fit on our runs:", show(fit))
print(f"residuals: mean |L̂ − L| = {np.abs(res).mean():.3f}, max = {np.abs(res).max():.3f}  (loss range {rL.min():.2f}–{rL.max():.2f})")
print("Chinchilla     :  E=1.69  A=406.4  B=410.7  alpha=0.34  beta=0.28")
""")

md(r"""
**Compute-optimal allocation from the law.** Substitute $D = C/(6N)$ and set the derivative in $N$ to zero:
$$\frac{\partial}{\partial N}\Big[A N^{-\alpha} + B\big(\tfrac{C}{6}\big)^{-\beta} N^{\beta}\Big] = -\alpha A N^{-\alpha-1} + \beta B \big(\tfrac{C}{6}\big)^{-\beta} N^{\beta-1} = 0
\ \Rightarrow\ N^{\alpha+\beta} = \frac{\alpha A}{\beta B}\Big(\frac{C}{6}\Big)^{\beta}.$$
$$N_{\text{opt}}(C) = G\Big(\frac{C}{6}\Big)^{\frac{\beta}{\alpha+\beta}},\qquad D_{\text{opt}}(C) = G^{-1}\Big(\frac{C}{6}\Big)^{\frac{\alpha}{\alpha+\beta}},\qquad G = \Big(\frac{\alpha A}{\beta B}\Big)^{\frac{1}{\alpha+\beta}}.$$
So $a = \beta/(\alpha+\beta)$ and $b = \alpha/(\alpha+\beta)$, with $a+b=1$. $a=b=\tfrac12$ exactly when $\alpha=\beta$; the larger exponent's resource gets the smaller share of the growth.
""")

code(r"""
def n_opt(p, C):
    #>> closed-form compute-optimal N for budget C (formula above); D_opt = C / (6 N_opt)
    G = (p["alpha"] * p["A"] / (p["beta"] * p["B"])) ** (1 / (p["alpha"] + p["beta"]))
    return G * (C / 6) ** (p["beta"] / (p["alpha"] + p["beta"]))
    #<<

for C in (1.5e12, 1e15, 1e23):                                    # check: numeric argmin of L(N, C/6N) on a fine log grid
    N_grid = np.logspace(2, 14, 200001)
    N_num = N_grid[np.argmin(law(fit, N_grid, C / (6 * N_grid)))]
    print(f"C={C:.2g}: closed form N_opt={n_opt(fit, C):.4g}   numeric argmin {N_num:.4g}   rel. diff {abs(N_num / n_opt(fit, C) - 1):.1e}")
a_exp = fit["beta"] / (fit["alpha"] + fit["beta"])
print(f"our fit: N_opt ∝ C^{a_exp:.2f}, D_opt ∝ C^{1 - a_exp:.2f};  tokens/param at C=1.5e12: {tokens_for_budget(1.5e12, n_opt(fit, 1.5e12)) / n_opt(fit, 1.5e12):.0f}")
""")

md(r"""
**Plots.** Left: the IsoFLOP curves (measured losses, the law's prediction dashed, and the minimum of a parabola in $\log N$ through each curve, i.e. Chinchilla's Approach 2). Middle: every run against its compute (x on a log scale), with the frontier $L^*(C) = L(N_{\text{opt}}, D_{\text{opt}})$. Right: $N_{\text{opt}}(C)$ from the IsoFLOP minima and from the fitted law.
""")

code(r"""
iso_min = {}
for C in Cs:                                                   # Approach 2: parabola in log N through each IsoFLOP curve
    m_ = rC == C
    c2, c1, c0 = np.polyfit(np.log(rN[m_]), rL[m_], 2)
    x_v = -c1 / (2 * c2)
    if c2 > 0 and np.log(rN[m_]).min() <= x_v <= np.log(rN[m_]).max():
        iso_min[C] = (math.exp(x_v), c0 - c1 ** 2 / (4 * c2))
    else:                                                      # not convex / vertex outside the window: take the best run
        iso_min[C] = (rN[m_][rL[m_].argmin()], rL[m_].min())
slope2, icpt2 = np.polyfit(np.log(Cs), np.log([iso_min[C][0] for C in Cs]), 1)
print("IsoFLOP minima:  " + "   ".join(f"C={C:.2g}: N*={iso_min[C][0]:,.0f} (D/N={tokens_for_budget(C, iso_min[C][0]) / iso_min[C][0]:.0f}), L*={iso_min[C][1]:.3f}" for C in Cs))
print(f"Approach 2 (power law through the minima): N_opt ∝ C^{slope2:.2f};  Approach 3 (fitted law): N_opt ∝ C^{a_exp:.2f}")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
cols = plt.cm.viridis(np.linspace(0, 0.85, len(Cs)))
for C, col in zip(Cs, cols):
    m_ = rC == C
    ax[0].plot(rN[m_], rL[m_], "o", color=col, label=f"C = {C:.2g}")
    Ng = np.logspace(np.log10(rN[m_].min() / 1.5), np.log10(rN[m_].max() * 1.5), 100)
    ax[0].plot(Ng, law(fit, Ng, C / (6 * Ng)), "--", color=col, lw=1)
    ax[0].plot(*iso_min[C], "*", color=col, ms=15, mec="k")
ax[0].set(xscale="log", xlabel="non-embedding parameters N", ylabel="validation loss (nats/char)", title="IsoFLOP curves (★ = minimum)")
ax[0].legend(fontsize=8)
sc_ = ax[1].scatter(6 * rN * rD, rL, c=np.log10(rN), cmap="plasma", s=25)
Cg = np.logspace(np.log10(min(Cs) / 2), 14, 100)
ax[1].plot(Cg, law(fit, n_opt(fit, Cg), Cg / (6 * n_opt(fit, Cg))), "k-", lw=1.5, label="frontier L*(C) from the fit")
ax[1].axhline(fit["E"], color="gray", ls=":", label=f"fitted E = {fit['E']:.2f}")
ax[1].set(xscale="log", xlabel="compute C = 6ND (FLOPs, log scale)", ylabel="validation loss", title="Loss vs compute")
plt.colorbar(sc_, ax=ax[1], label="log10 N"); ax[1].legend(fontsize=8)
ax[2].loglog(Cs, [iso_min[C][0] for C in Cs], "*", ms=14, mec="k", label="IsoFLOP minima (Approach 2)")
ax[2].loglog(Cg, np.exp(icpt2) * Cg ** slope2, "C0:", label=f"power-law fit, slope {slope2:.2f}")
ax[2].loglog(Cg, n_opt(fit, Cg), "k-", label=f"fitted law (Approach 3), slope {a_exp:.2f}")
ax[2].loglog(Cg, np.sqrt(Cg / 120), "C3--", lw=1, label="Chinchilla rule D = 20N")
ax[2].set(xlabel="compute C (FLOPs)", ylabel="compute-optimal N", title="N_opt(C)"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

md(r"""
**Prediction test.** The point of a scaling law is to forecast a run before paying for it. Hold out the largest-budget IsoFLOP curve ($C=1.5\cdot10^{12}$, 4 runs, 3× more compute than anything left in the fit), refit on the other 12 runs, and predict.
""")

code(r"""
held = rC == max(Cs)
fit_h = fit_law(rN[~held], rD[~held], rL[~held])
pred = law(fit_h, rN[held], rD[held])
print("fit without the largest C:", show(fit_h))
for N_, p_, l_ in zip(rN[held], pred, rL[held]):
    print(f"N={N_:>8,.0f}: predicted {p_:.3f}   measured {l_:.3f}   error {p_ - l_:+.3f}")
best = np.argmin(rL[held])
print(f"best held-out run: predicted {pred[best]:.3f} vs measured {rL[held][best]:.3f} ({(pred[best] - rL[held][best]) / rL[held][best]:+.1%});  "
      f"mean |error| over the curve {np.abs(pred - rL[held]).mean():.3f}")
""")

md(r"""
**What came out** (single-seed runs; the numbers move by a few hundredths with the seed).
- **IsoFLOP valleys.** Every budget has an interior or near-edge minimum, and it moves to larger models as $C$ grows: $N^* \approx$ 3.8k, 6.6k, 16k, 36k for $C = 5\cdot10^{10} \dots 1.5\cdot10^{12}$ (30× compute, ~10× parameters). At the two smallest budgets the parabola through 4 points is not convex (e.g. 14k and 25k parameters tie at 2.19 for $C=1.5\cdot10^{11}$), so the star is the best run; at $5\cdot10^{10}$ the two smallest models tie (2.253 vs 2.254). That is the noise level of this study.
- **Fitted law on all 16 runs:** $E=1.10$, $A=5.5$, $B=29.8$, $\alpha=0.27$, $\beta=0.27$, so $N_{\text{opt}}\propto C^{0.50}$, $D_{\text{opt}}\propto C^{0.50}$: Chinchilla's exponents. Do not over-read this. Approach 2 (a line through the four minima) gives slope 0.67, and the fit without the largest budget below gives $\alpha=0.64$, $\beta=0.18$, i.e. $N_{\text{opt}}\propto C^{0.22}$. Sixteen noisy runs over 1.5 decades of compute do not pin down two exponents; Chinchilla used over 400 runs over ~4 decades, and Besiroglu et al. show that even their published constants were off.
- **Tokens per parameter.** Our optimum is ~200–600 characters per parameter, not 20. A character carries much less information than a BPE token (~4 characters per token), and the data and model are different: what transfers from Chinchilla is the method (IsoFLOP sweeps, fit, extrapolate), not the constant.
- **Fit quality.** Mean $|\hat L - L| = 0.027$, max 0.077. The fitted $E=1.10$ nats/char lies well below our best run (1.80), so it is an extrapolation, not a measurement. In the left plot the law (dashed) runs above the $1.5\cdot10^{12}$ points: the largest budget gains more than the 3-parameter power law gives it.
- **Prediction.** Fitted on the 12 runs below $C=1.5\cdot10^{12}$, the law predicts 1.865 for the best held-out run, measured 1.800 (+3.6%); mean $|$error$|$ over the held-out curve 0.078, and too pessimistic for 3 of 4 runs. Forecasting from noisy small runs is hard: real labs fit on many seeds and sizes, then predict the final run to ~0.01–0.02 nats (Llama 3 and GPT-4 report this).
""")

md(r"""
### 4.3 What $10^{23}$ FLOPs buys, with the real Chinchilla fit

Plug the published constants into the same `n_opt`. Besiroglu et al. (2024) re-extracted the data from the Chinchilla paper's figure and found the published Approach 3 fit inconsistent with the paper's own 20 tokens/param; we print their refit next to it.
""")

code(r"""
chin = dict(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28)          # Hoffmann et al. 2022, Approach 3
epoch = dict(E=1.8172, A=482.01, B=2085.43, alpha=0.3478, beta=0.3658)  # Besiroglu et al. 2024 refit
for name, p in (("Chinchilla (published)", chin), ("Besiroglu et al. refit", epoch)):
    for C in (1e23, 1e24):
        No = n_opt(p, C); Do = C / (6 * No)
        print(f"{name:23s} C={C:.2g}: N_opt={No / 1e9:6.1f}B  D_opt={Do / 1e12:5.2f}T  tokens/param={Do / No:5.1f}  predicted loss {law(p, No, Do):.3f}")

N8, D8 = 8e9, 15e12                                                  # Llama 3 8B
C8 = 6 * N8 * D8
print(f"\nLlama 3 8B on 15T tokens: C = 6ND = {C8:.2e} FLOPs, {D8 / N8:,.0f} tokens/param")
for name, p in (("Chinchilla (published)", chin), ("Besiroglu et al. refit", epoch)):
    No = n_opt(p, C8); Do = C8 / (6 * No)
    print(f"{name:23s}: optimal {No / 1e9:5.1f}B on {Do / 1e12:5.2f}T, loss {law(p, No, Do):.4f};  8B/15T loss {law(p, N8, D8):.4f}  "
          f"(+{law(p, N8, D8) - law(p, No, Do):.4f} nats)  ->  inference FLOPs/token 2N: {No / N8:.1f}× lower for the 8B")
""")

md(r"""
- **Published Chinchilla fit:** $10^{23}$ FLOPs → 14.6B parameters on 1.14T tokens (78 tokens/param), predicted loss 2.005; $10^{24}$ → 41B on 4.0T (98 tokens/param), 1.911. Since $\beta<\alpha$, $a = \beta/(\alpha+\beta) = 0.45$ and the tokens/param ratio grows with $C$. 78 is not 20: the published Approach-3 constants do not reproduce the paper's own Approaches 1 and 2.
- **Besiroglu et al. refit:** 29B on 0.57T tokens (19 tokens/param) at $10^{23}$, 96B on 1.74T (18) at $10^{24}$, consistent with the ≈ 20 rule.
- **Llama 3 8B on 15T tokens** ($C = 7.2\cdot10^{23}$, 1,875 tokens/param). The compute-optimal model for that budget has 36B (published fit) or 81B (refit) parameters. The 8B model's predicted loss is higher by only 0.026 / 0.054 nats (1.3% / 2.7%), while every generated token costs 4.5× / 10× fewer FLOPs, and its bf16 weights take 16 GB instead of 71 / 162 GB. For a model served to millions of users, inference dominates the lifetime compute, so the small, over-trained model is the cheaper one.

These laws were fit on Chinchilla's data, tokenizer and architecture; applied to Llama 3 they illustrate the trade-off, they do not predict its loss.

✏️ **Inference-aware optimum.** Suppose the model will generate $D_{\text{inf}} = 10^{13}$ tokens over its lifetime (2N FLOPs each). With the published Chinchilla law, find the $(N, D)$ that reaches the loss of the compute-optimal $10^{23}$-FLOP model at the lowest *total* cost $6ND + 2ND_{\text{inf}}$ (for each $N$ on a log grid, solve $L(N,D)=L^*$ for $D$, then take the cheapest). How many tokens/param is that? Repeat for $D_{\text{inf}}=10^{14}$.

**Exam-style question (new)**
A lab fits $L(N,D) = E + \dfrac{A}{N^{1/2}} + \dfrac{B}{D^{1/2}}$ with $A=100$, $B=400$, and plans a training run with $C = 6\cdot10^{20}$ FLOPs ($C=6ND$).
(a) Derive $N_{\text{opt}}$ and $D_{\text{opt}}$ for this budget. How many tokens per parameter is that?
(b) The budget grows to $2.4\cdot10^{21}$ FLOPs (×4). By what factor do $N_{\text{opt}}$ and $D_{\text{opt}}$ change?
(c) A second team spends the original budget on a $10^{10}$-parameter model. What is its loss minus the optimal loss? Which model is cheaper to serve, and by how much per token?

<<STUDENT>>
✏️ Your answer:
<</STUDENT>>
<<SOLUTION>>
**Answer.** (a) With $D=C/(6N)=10^{20}/N$: $L = E + 100N^{-1/2} + 400\cdot10^{-10}N^{1/2}$. Setting $\partial L/\partial N = -50N^{-3/2} + 200\cdot10^{-10}N^{-1/2} = 0$ gives $N = 50/(200\cdot10^{-10}) = 2.5\cdot10^{9}$, $D = 10^{20}/2.5\cdot10^{9} = 4\cdot10^{10}$: 16 tokens/param. (General formula with $\alpha=\beta$: $N_{\text{opt}} = (A/B)\sqrt{C/6}$.)
(b) $\alpha=\beta$, so $N_{\text{opt}}, D_{\text{opt}} \propto \sqrt C$: both grow 2×, to $5\cdot10^{9}$ and $8\cdot10^{10}$.
(c) $D = 10^{20}/10^{10} = 10^{10}$. Loss above $E$: $100/10^{5} + 400/10^{5} = 0.005$, vs $100/(5\cdot10^{4}) + 400/(2\cdot10^{5}) = 0.004$ for the optimum: $+0.001$ nats. The optimal $2.5\cdot10^9$ model is 4× cheaper to serve ($2N$ FLOPs per token: $5\cdot10^9$ vs $2\cdot10^{10}$) *and* has lower loss: over-sizing the model loses on both counts. (Over-*training* a smaller model is the direction that trades a little loss for cheaper inference.)
<</SOLUTION>>
""")

code(r"""
A_, B_, C_ = 100.0, 400.0, 6e20                                    # check (a)–(c) numerically
ex = dict(E=0.0, A=A_, B=B_, alpha=0.5, beta=0.5)
No = n_opt(ex, C_)
print(f"(a) N_opt = {No:.3g}, D_opt = {C_ / (6 * No):.3g}, tokens/param = {C_ / (6 * No) / No:.0f}")
print(f"(b) ×4 compute: N_opt ×{n_opt(ex, 4 * C_) / No:.2f}")
print(f"(c) 1e10 model: loss − optimal = {law(ex, 1e10, C_ / 6e10) - law(ex, No, C_ / (6 * No)):.4f}")
""")

md(r"""
## 5. KV cache

Without a cache, generating token $t+1$ runs the whole prefix of length $t$ through the model again: $O(t)$ work per token, $O(n^2)$ for $n$ tokens.
With a cache, each layer stores $K, V$ of all past positions; a step computes $q, k, v$ only for the new token, appends $k, v$, and attends over the cache.
The new queries at positions $t_0,\dots,t_0+T_{\text{new}}-1$ may see every cached position up to their own: a causal mask shifted by $t_0$.
""")

code(r"""
def new_cache(model):
    return [[] for _ in model.blocks]          # per layer: [] or [K, V], each (B, heads, T_past, hd)

def attn_cached(attn, x, layer_cache):
    q, k, v = attn.split(x)                    # projections of the NEW tokens only: (B, h, T_new, hd)
    #>> prepend the cached K, V (dim 2), store the result back in layer_cache, attend with a causal mask shifted by the cache length
    if layer_cache:
        k = torch.cat([layer_cache[0], k], dim=2)
        v = torch.cat([layer_cache[1], v], dim=2)
    layer_cache[:] = [k, v]
    T_new, T_all = q.shape[2], k.shape[2]
    mask = torch.ones(T_new, T_all, dtype=torch.bool, device=x.device).tril(diagonal=T_all - T_new)
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    #<<
    return attn.merge(y)

@torch.no_grad()
def forward_cached(model, idx_new, cache):
    past = cache[0][0].shape[2] if cache[0] else 0
    x = model.tok(idx_new) + model.pos(torch.arange(past, past + idx_new.shape[1], device=idx_new.device))
    for blk, c in zip(model.blocks, cache):
        x = x + attn_cached(blk.attn, blk.ln1(x), c)
        x = x + blk.mlp(blk.ln2(x))
    return model.head(model.ln_f(x))

@torch.no_grad()
def generate_nocache(model, idx, n_new):       # greedy; re-runs the whole prefix every step
    for _ in range(n_new):
        nxt = model(idx)[:, -1].argmax(-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
    return idx

@torch.no_grad()
def generate_cache(model, idx, n_new):         # greedy; prefill the prompt once, then one token per step
    cache = new_cache(model)
    logits = forward_cached(model, idx, cache)[:, -1]
    for _ in range(n_new):
        nxt = logits.argmax(-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
        logits = forward_cached(model, nxt, cache)[:, -1]  # @student: logits = ...  # TODO: feed only the new token through forward_cached
    return idx
""")

md(r"""
**Check.** Same prompt, same greedy decoding: the two must produce the same tokens, and the cached logits must equal the full forward pass up to float rounding.
""")

code(r"""
prompt = encode("ROMEO:\n").unsqueeze(0).to(device)
n_new = block_size - prompt.shape[1]           # fill the context exactly
out_full, out_kv = generate_nocache(gpt, prompt, n_new), generate_cache(gpt, prompt, n_new)
print("identical tokens:", torch.equal(out_full, out_kv))
cache = new_cache(gpt)
logits_kv = torch.cat([forward_cached(gpt, out_kv[:, :100], cache)] +
                      [forward_cached(gpt, out_kv[:, t:t + 1], cache) for t in range(100, block_size)], dim=1)
with torch.no_grad():
    print(f"max |logits_cached − logits_full| = {(logits_kv - gpt(out_kv)).abs().max().item():.2e}")
print(decode(out_kv[0].tolist())[:300])
""")

md(r"""
Now the speed. Generation of the 121 tokens that fill the context, batch 1, on `bench` (best of 3 runs):
""")

code(r"""
gpt_b = copy.deepcopy(gpt).to(bench)
p_b = prompt.to(bench)
for name, fn in [("no cache", generate_nocache), ("KV cache", generate_cache)]:
    dt = time_fn(lambda: fn(gpt_b, p_b, n_new), bench, reps=3) / 1e3
    print(f"{name:9s}: {n_new / dt:7.1f} tokens/s")
cache_bytes = sum(t.nbytes for c in cache for t in c)   # the cache filled in the check above: one sequence, block_size tokens
print(f"KV cache, one sequence of {block_size} tokens: measured {cache_bytes/1e3:.0f} kB, "
      f"formula 2·L·T·d·4 B = {2*L*block_size*d*4/1e3:.0f} kB (weights: {n_params(gpt)*4/1e3:.0f} kB)")
""")

md(r"""
<<SOLUTION>>
The cache gives the same tokens (the logit difference is float rounding, ~5e-6) at 3–6× the speed in our runs; the gap grows with context length, since the uncached cost per token grows linearly with the prefix and the cached one does not (only the attention over the cache does).
<</SOLUTION>>
<<STUDENT>>
Compare the two speeds. How should the gap change with the context length?
<</STUDENT>>

✏️ The cache costs memory instead: compute its size for Llama-2-7B (32 layers, $d=4096$, bf16) at 4096 tokens and batch 16. Grouped-query attention shares one $K,V$ head among 8 query heads; what does that do to the number?
""")

md(r"""
## 6. Quantization by hand: int8 and int4, per channel

Symmetric quantization of a weight matrix $W\in\mathbb{R}^{d_{out}\times d_{in}}$ to $b$ bits, one scale per output row (per channel) or one for the whole tensor:
$$s_i = \frac{\max_j |W_{ij}|}{2^{b-1}-1},\qquad q_{ij} = \mathrm{clamp}\big(\mathrm{round}(W_{ij}/s_i)\big),\qquad \hat W_{ij} = s_i\,q_{ij}.$$
""")

code(r"""
def quantize(W, bits, per_channel=True):
    qmax = 2 ** (bits - 1) - 1
    #>> scale s (shape (d_out, 1) if per_channel, else a scalar) = max|W| / qmax; q = clamp(round(W / s), −qmax, qmax)
    amax = W.abs().amax(dim=1, keepdim=True) if per_channel else W.abs().max()
    s = amax.clamp(min=1e-12) / qmax
    q = torch.clamp(torch.round(W / s), -qmax, qmax)
    #<<
    return q.to(torch.int8), s                 # int4 values are stored in int8 here; real kernels pack two per byte

def dequantize(q, s):
    return q.float() * s
""")

md(r"""
**Check** against PyTorch's fake-quantization op (quantize + dequantize in one call), for 8 and 4 bits, on the first MLP weight of the mini-GPT.
""")

code(r"""
W = gpt.blocks[0].mlp[0].weight.detach().cpu()
for bits in (8, 4):
    q, s = quantize(W, bits)
    qmax = 2 ** (bits - 1) - 1
    ref = torch.fake_quantize_per_channel_affine(W, s.flatten(), torch.zeros(W.shape[0], dtype=torch.int32), 0, -qmax, qmax)
    err = (dequantize(q, s) - W).abs()
    print(f"int{bits}: max |mine − torch| = {(dequantize(q, s) - ref).abs().max().item():.1e}, "
          f"max |Ŵ − W| = {err.max().item():.4f} ≤ max s/2 = {s.max().item() / 2:.4f}, distinct values per row ≤ {q[0].unique().numel()}")
""")

md(r"""
Now apply it to every `nn.Linear` in the mini-GPT (embeddings, LayerNorms and biases stay in float) and measure the validation loss as a function of bits. This is **round-to-nearest (RTN)** weight-only quantization, the baseline that GPTQ and AWQ improve on.
""")

code(r"""
def quantize_model(model, bits, per_channel=True):
    m = copy.deepcopy(model)
    for mod in m.modules():
        if isinstance(mod, nn.Linear):
            q, s = quantize(mod.weight.data, bits, per_channel)
            mod.weight.data = dequantize(q, s)
    return m

base = val_loss(gpt)
bit_list = [8, 6, 5, 4, 3, 2]
res = {pc: [val_loss(quantize_model(gpt, b, pc)) for b in bit_list] for pc in (True, False)}
lin_params = sum(m.weight.numel() for m in gpt.modules() if isinstance(m, nn.Linear))
print(f"fp32 val loss {base:.3f}   (Linear weights: {lin_params/1e3:.0f}k params = {lin_params*4/1e6:.2f} MB in fp32)")
for i, b in enumerate(bit_list):
    print(f"{b} bits: per-channel {res[True][i]:.3f}   per-tensor {res[False][i]:.3f}   size {lin_params*b/8/1e6:.3f} MB")

plt.figure(figsize=(6, 3.8))
plt.plot(bit_list, res[True], "o-", label="per-channel")
plt.plot(bit_list, res[False], "s--", label="per-tensor")
plt.axhline(base, color="k", lw=1, label="fp32")
plt.gca().invert_xaxis()
plt.xlabel("weight bits"); plt.ylabel("validation loss"); plt.title("Mini-GPT: RTN weight quantization")
plt.legend(); plt.tight_layout(); plt.show()
""")

md(r"""
<<SOLUTION>>
- 8 down to 5 bits per-channel: the loss does not move (≤ 0.005).
- 4 bits: +0.02 per-channel, +0.08 to +0.1 per-tensor. 3 bits: +0.15 per-channel, +0.7 to +0.9 per-tensor.
- 2 bits (3 levels per row): both break, loss ≈ 3.9–4.5, about uniform guessing over 65 characters ($\ln 65 = 4.17$).

(Exact values shift by ~0.01 between runs: the mini-GPT trains on the GPU, which is not bit-reproducible.)

Per-channel scales matter more as the bit-width shrinks: with one scale for the whole tensor, rows with small weights get only a few of the $2^b-1$ levels.
<</SOLUTION>>
<<STUDENT>>
Read the table: from which bit-width does the loss move, and where does per-channel start to matter?
<</STUDENT>>

✏️ **Group quantization** (one scale per group of 32 consecutive inputs in a row, as in GPTQ/AWQ checkpoints and NVFP4's per-16 scales): reshape `W` to `(d_out, d_in // 32, 32)`, quantize along the last dim, and redo the 3- and 2-bit rows. What does it cost in extra storage per weight (scale in fp16)?
""")

md(r"""
## 7. Magnitude pruning: unstructured vs channels

**Unstructured**: per layer, zero the fraction $p$ of weights with the smallest $|w|$.
**Channel-level** (structured): zero whole output channels (conv filters, rows of a Linear) with the smallest $\ell_1$ norm $\sum|w|$. A zeroed channel can be removed, giving a smaller dense layer.
We keep a mask $M$ and use $W\odot M$.
""")

code(r"""
def magnitude_mask(W, sparsity):
    k = int(round(sparsity * W.numel()))
    mask = torch.ones_like(W)
    #>> set mask to 0 at the k entries with the smallest |W| (hint: torch.topk(..., largest=False))
    if k > 0:
        idx = torch.topk(W.abs().flatten(), k, largest=False).indices
        mask.view(-1)[idx] = 0
    #<<
    return mask

def channel_mask(W, sparsity):                 # W: (out, ...) conv filters or Linear rows
    n_prune = int(round(sparsity * W.shape[0]))
    mask = torch.ones_like(W)
    #>> L1 norm of each output channel (sum |w| over all dims but 0); zero the n_prune channels with the smallest norm
    norms = W.abs().flatten(1).sum(1)
    mask[norms.argsort()[:n_prune]] = 0
    #<<
    return mask
""")

md(r"""
**Check** against `torch.nn.utils.prune` (`l1_unstructured` and `ln_structured` with $n=1$ along dim 0).
""")

code(r"""
W = cnn.conv2.weight.detach()
for name, mine, lib in [("unstructured", magnitude_mask, lambda m, a: prune.l1_unstructured(m, "weight", amount=a)),
                        ("channel", channel_mask, lambda m, a: prune.ln_structured(m, "weight", amount=a, n=1, dim=0))]:
    tmp = copy.deepcopy(cnn.conv2); lib(tmp, 0.6)
    M = mine(W, 0.6)
    print(f"{name:12s}: masks differ at {(M != tmp.weight_mask).sum().item()} of {M.numel()} entries, sparsity {1 - M.mean().item():.2f}")
""")

code(r"""
def pruned(model, sparsity, kind):
    m = copy.deepcopy(model)
    if kind == "unstructured":                 # global: one threshold over all weights, so each layer loses what it can spare
        layers = [m.conv1, m.conv2, m.fc1, m.fc2]
        flat = magnitude_mask(torch.cat([l.weight.data.flatten() for l in layers]), sparsity)
        masks = [M.view_as(l.weight) for M, l in zip(flat.split([l.weight.numel() for l in layers]), layers)]
    else:                                      # same fraction of channels in every hidden layer; never drop output classes
        layers = [m.conv1, m.conv2, m.fc1]
        masks = [channel_mask(l.weight.data, sparsity) for l in layers]
    def apply():
        with torch.no_grad():
            for l, M in zip(layers, masks):
                l.weight.mul_(M)
                if kind == "channel": l.bias.mul_(M.flatten(1)[:, 0])   # a removed channel has no bias either
    apply()
    return m, apply

sparsities = [0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98]
acc = {}
for kind in ("unstructured", "channel"):
    acc[kind] = [accuracy(pruned(cnn, s, kind)[0]) for s in sparsities]
    acc[kind + " + fine-tune"] = []
    for s in sparsities:                       # + 1 epoch of fine-tuning with the mask re-applied after every step
        m, apply = pruned(cnn, s, kind)
        acc[kind + " + fine-tune"].append(accuracy(train(m, Xtr, ytr, epochs=1, lr=1e-3, after_step=apply)))
print("sparsity  " + "  ".join(f"{k:>24s}" for k in acc))
for i, s in enumerate(sparsities):
    print(f"  {s:.2f}  " + "  ".join(f"{v[i]:24.3f}" for v in acc.values()))

plt.figure(figsize=(6.5, 3.8))
for (k, v), st in zip(acc.items(), ["o-", "o--", "s-", "s--"]):
    plt.plot(sparsities, v, st, label=k)
plt.xlabel("fraction of weights (unstructured) / channels (channel) removed"); plt.ylabel("test accuracy"); plt.ylim(0, 1)
plt.title("SmallCNN on FashionMNIST: magnitude pruning"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
""")

md(r"""
<<SOLUTION>>
- **Unstructured** (global threshold): no retraining needed up to 80% sparsity (0.869 vs 0.875 dense), 0.84 at 90%. With one epoch of fine-tuning, 95% of the weights can go at no cost (0.882; the fine-tuned nets are slightly above the dense one because they got an extra epoch), and 98% still gives 0.84.
- **Channel**: hurts sooner (0.84 at 50%, 0.30 at 90% without retraining). Fine-tuning recovers 0.87 at 50% and 0.85 at 70%.
- The trade-off: removing 50% of the channels gives a dense network with about a quarter of the conv FLOPs (`conv2` loses both input and output channels), on any hardware. 95% unstructured sparsity only pays off with sparse kernels or 2:4 hardware support.
<</SOLUTION>>
<<STUDENT>>
Up to what sparsity is unstructured pruning free without any retraining? How do channels compare, and what does one epoch of fine-tuning buy?
<</STUDENT>>

✏️ The masked model is not faster: it still multiplies by zeros. For 50% channel pruning, build a physically smaller `SmallCNN(c1=16, c2=32, hidden=64)` and copy the surviving filters into it (careful: `conv2` loses *input* channels too, and `fc1`'s inputs are `c2·7·7`). Check that its accuracy equals the masked model's.
""")

md(r"""
## 8. Knowledge distillation

Teacher: the CNN above. Student: `SmallCNN(c1=8, c2=16, hidden=64)`, 8× fewer parameters. To make the soft labels matter we give the student only 1,000 labelled images (the teacher saw 10,000), and train it twice from the same init: with CE alone and with the distillation loss.
""")

code(r"""
def kd_loss(s_logits, t_logits, y, T=4.0, alpha=0.9):
    #>> alpha · T² · KL(softmax(t/T) ‖ softmax(s/T)) averaged over the batch, + (1 − alpha) · CE(s, y). Use log_softmax for stability.
    log_p_t, log_p_s = F.log_softmax(t_logits / T, 1), F.log_softmax(s_logits / T, 1)
    kl = (log_p_t.exp() * (log_p_t - log_p_s)).sum(1).mean()
    return alpha * T * T * kl + (1 - alpha) * F.cross_entropy(s_logits, y)
    #<<

s_, t_, y_ = torch.randn(32, 10), torch.randn(32, 10), torch.randint(0, 10, (32,))
ref = 0.9 * 16 * F.kl_div(F.log_softmax(s_ / 4, 1), F.log_softmax(t_ / 4, 1), reduction="batchmean", log_target=True) \
      + 0.1 * F.cross_entropy(s_, y_)
print(f"mine {kd_loss(s_, t_, y_).item():.6f}   torch {ref.item():.6f}   diff {abs(kd_loss(s_, t_, y_) - ref).item():.1e}")
""")

code(r"""
Xs, ys = Xtr[:1000], ytr[:1000]
student_cfg = dict(c1=8, c2=16, hidden=64)
def kd_fn(logits, yb, xb, T=4.0):
    with torch.no_grad():
        t = cnn(xb)
    return kd_loss(logits, t, yb, T=T)

rows = []
for seed in range(3):
    torch.manual_seed(seed); init = SmallCNN(**student_cfg).to(device)
    alone = train(copy.deepcopy(init), Xs, ys, epochs=30, seed=seed)
    distilled = train(copy.deepcopy(init), Xs, ys, epochs=30, seed=seed, loss_fn=kd_fn)
    rows.append((accuracy(alone), accuracy(distilled)))
    print(f"seed {seed}: student alone {rows[-1][0]:.3f}   distilled {rows[-1][1]:.3f}")
r = np.array(rows)
print(f"teacher ({n_params(cnn)/1e3:.0f}k params) {accuracy(cnn):.3f} | student ({n_params(init)/1e3:.1f}k params): "
      f"alone {r[:,0].mean():.3f} ± {r[:,0].std():.3f}, distilled {r[:,1].mean():.3f} ± {r[:,1].std():.3f}")
""")

md(r"""
<<SOLUTION>>
Distillation adds 1.6 points (0.827 vs 0.811), several times the seed-to-seed spread, and wins in each of the three seeds; the student closes about a quarter of its gap to the teacher (0.875). The soft targets carry more information per image than the one-hot label (which classes look alike: shirt vs T-shirt vs pullover), which matters most when labelled images are few.
<</SOLUTION>>
<<STUDENT>>
Is the gap larger than the seed-to-seed spread?
<</STUDENT>>

✏️ Run the distilled student with `T=1` and `T=10` (pass `T` through `kd_fn`). Then set `alpha=1` (no labels at all): how close does the student get using only the teacher's soft outputs on 1,000 images?
""")

md(r"""
## 9. bf16 autocast, `torch.compile`, profiling

**bf16** keeps fp32's 8 exponent bits and cuts the mantissa to 7: same range, ~3 significant digits. Under `torch.autocast`, matmuls and convolutions run in bf16 while reductions (softmax, LayerNorm, losses) stay in fp32. On GPUs with tensor cores (A100, H100, and the T4's fp16) this is the default for training. **`torch.compile`** traces the model and fuses elementwise ops into larger kernels (fewer memory round-trips, fewer launches).
""")

code(r"""
xb, yb = get_batch("val", 16, torch.Generator().manual_seed(0))
dev_type = xb.device.type
try:
    with torch.autocast(device_type=dev_type, dtype=torch.bfloat16):  # @student: with nullcontext():  # TODO: autocast to bfloat16 on dev_type
        with torch.no_grad():
            logits_bf16 = gpt(xb)
    with torch.no_grad():
        logits_32 = gpt(xb)
    print("autocast output dtype:", logits_bf16.dtype)
    print(f"max |bf16 − fp32| logits = {(logits_bf16.float() - logits_32).abs().max().item():.3f}  "
          f"(logit scale {logits_32.abs().max().item():.1f});  loss fp32 {F.cross_entropy(logits_32.flatten(0,1), yb.flatten()).item():.4f}  "
          f"bf16 {F.cross_entropy(logits_bf16.float().flatten(0,1), yb.flatten()).item():.4f}")
    bf16_ok = True
except Exception as e:
    print("bf16 autocast not supported here:", type(e).__name__, str(e)[:100]); bf16_ok = False
""")

md(r"""
The logits move in the second or third digit, the loss in the third or fourth: that is bf16's precision, and it does not matter for training.

**Practical note: fp16 overflows, bf16 does not**

fp16 has 5 exponent bits: its largest value is 65,504 and its smallest normal number $6.1\cdot10^{-5}$. bf16 keeps fp32's 8 exponent bits (same range, ~$3.4\cdot10^{38}$) and pays with precision (7 mantissa bits, eps $=2^{-7}$). So in fp16 a sum of squares or an unshifted exp overflows to `inf`, and small gradients underflow to 0. **Loss scaling** (`torch.amp.GradScaler`) fixes the underflow: multiply the loss by $S$ (e.g. $2^{16}$) before backward, so every gradient is $S\times$ larger in fp16, and divide by $S$ in fp32 before the optimizer step; if any gradient became `inf`, skip the step and halve $S$. bf16 needs none of this, which is why it replaced fp16 for training on A100/H100.
""")

code(r"""
for dt in (torch.float32, torch.float16, torch.bfloat16):
    fi = torch.finfo(dt)
    print(f"{str(dt):15s} max {fi.max:9.3g}   smallest normal {fi.smallest_normal:9.3g}   eps {fi.eps:.3g}")

# overflow: a LayerNorm-style sum of squares over 768 channels of size 20, and exp of a logit of 12 (softmax without max-subtraction)
h = torch.full((768,), 20.0)
for dt in (torch.float16, torch.bfloat16):
    print(f"{str(dt):15s} sum(h²) = {(h.to(dt) ** 2).sum(dtype=dt).item():>9}   exp(12) = {torch.exp(torch.tensor(12.0, dtype=dt)).item():>9}"
          f"   (fp32: {(h ** 2).sum().item():.0f}, {math.exp(12):.0f})")
assert torch.isinf((h.half() ** 2).sum(dtype=torch.float16)) and torch.isfinite((h.bfloat16() ** 2).sum(dtype=torch.bfloat16))

# underflow: gradients spread over 10^-10 .. 10^-2 (log-uniform), cast to fp16 with and without loss scaling
torch.manual_seed(0)
g = 10 ** torch.empty(100_000).uniform_(-10, -2)
def to_fp16_grad(g, S):
    #>> scale by S in fp32, cast to fp16 (what backward would produce), cast back to fp32 and unscale
    return (g * S).half().float() / S
    #<<
for S in (1.0, 2.0 ** 16):
    g16 = to_fp16_grad(g, S)
    rel = ((g16 - g).abs() / g)
    print(f"scale S = {S:>7.0f}: zeroed {(g16 == 0).float().mean():.1%}   relative error > 1%: {(rel > 0.01).float().mean():.1%}   "
          f"max |g·S| = {(g * S).max():.3g} (fp16 max 65504)")
print(f"bf16, no scaling: zeroed {(g.bfloat16().float() == 0).float().mean():.1%}   "
      f"relative error > 1%: {(((g.bfloat16().float() - g).abs() / g) > 0.01).float().mean():.1%}")
assert (to_fp16_grad(g, 2.0 ** 16) == 0).sum() < (to_fp16_grad(g, 1.0) == 0).sum()

# what GradScaler does when S is too large: inf -> skip the step, halve S
S = 2.0 ** 24
while not torch.isfinite((g * S).half()).all():
    print(f"S = 2^{int(math.log2(S))}: overflow (max |g·S| = {(g * S).max():.2e}) -> skip step, halve S")
    S /= 2
print(f"S = 2^{int(math.log2(S))}: all gradients finite")

# real gradients of the mini-GPT: how many fall below fp16's normal range?
gpt.zero_grad(set_to_none=True)
F.cross_entropy(gpt(xb).flatten(0, 1), yb.flatten()).backward()
ga = torch.cat([p.grad.flatten().abs().cpu() for p in gpt.parameters()]); gpt.zero_grad(set_to_none=True)
ga = ga[ga > 0]
print(f"mini-GPT gradients: median |g| = {ga.median():.1e}; below fp16 smallest normal: {(ga < 6.1e-5).float().mean():.1%}; "
      f"flushed to 0 in fp16: {(ga.half() == 0).float().mean():.2%}")
""")

md(r"""
In fp16, 768 activations of size 20 already overflow the sum of squares ($307{,}200 > 65{,}504$), and so does $e^{12}$; bf16 returns both, rounded to 3 significant digits. Without scaling, 31% of gradients spread over $10^{-10}..10^{-2}$ become exactly 0 in fp16 and 51% lose more than 1% of their value (subnormals); with $S=2^{16}$ none are lost, and the largest scaled gradient (655) is still far from 65,504. Too large an $S$ overflows instead, which is what GradScaler's skip-and-halve loop detects.

On the mini-GPT's own gradients (one batch, fp32), the median $|g|$ is $1.4\cdot10^{-4}$: 26% of the entries lie below fp16's smallest normal number (kept only as subnormals, with fewer significant bits) and 0.1% flush to 0. Here the damage is mild; it grows as gradients shrink, which is why fp16 training always runs with a scaler.

Now timing, a forward pass over a batch of 16×128 tokens:
""")

code(r"""
timings = {}
with torch.no_grad():
    timings["fp32"] = time_fn(lambda: gpt(xb), device)
    if bf16_ok:
        def f_bf16():
            with torch.autocast(device_type=dev_type, dtype=torch.bfloat16): return gpt(xb)
        timings["bf16 autocast"] = time_fn(f_bf16, device)
    try:
        t0 = time.time(); gpt_c = torch.compile(gpt); gpt_c(xb); compile_s = time.time() - t0
        print(f"max |compiled − eager| = {(gpt_c(xb) - gpt(xb)).abs().max().item():.1e}  (compile took {compile_s:.0f}s)")
        timings["torch.compile"] = time_fn(lambda: gpt_c(xb), device)
    except Exception as e:
        print("torch.compile not available here:", type(e).__name__, str(e).split("\n")[0][:100])
for k, v in timings.items():
    print(f"{k:14s} {v:7.1f} ms / forward  ({device})")
""")

md(r"""
<<SOLUTION>>
On a laptop GPU (MPS) with this small model, the three timings are within noise of each other or bf16 is slower (extra casts, no bf16 matrix units); `torch.compile` was faster in some runs, slower in others. The gains come with size and hardware: on a CUDA GPU with tensor cores and a GPT-2-sized model, bf16 autocast typically gives ~2× and `torch.compile` another 1.2–1.5× (both are measured step by step in Karpathy's GPT-2 video). ✏️ On Colab, set `n_embd=384, n_layer=6` and re-time.
<</SOLUTION>>

**Profiling.** Before optimizing, find where the time goes. `torch.profiler` records every op of one training step (on a GPU add `ProfilerActivity.CUDA` and sort by `cuda_time_total`):
""")

code(r"""
from torch.profiler import profile, ProfilerActivity
gpt_p = copy.deepcopy(gpt).to(bench).train()
xp, yp = xb.to(bench), yb.to(bench)
step = lambda: F.cross_entropy(gpt_p(xp).flatten(0, 1), yp.flatten()).backward()
step()                                         # warm-up
with profile(activities=[ProfilerActivity.CPU]) as prof:
    step()
ka = prof.key_averages()
print(ka.table(sort_by="self_cpu_time_total", row_limit=10, max_name_column_width=40))
mm_ops = ("aten::addmm", "aten::mm", "aten::bmm", "aten::_scaled_dot_product_flash_attention_for_cpu",
          "aten::_scaled_dot_product_flash_attention_for_cpu_backward")
total = sum(e.self_cpu_time_total for e in ka)
print(f"matmul + attention kernels: {sum(e.self_cpu_time_total for e in ka if e.key in mm_ops) / total:.0%} of the step; "
      f"everything else (elementwise, LayerNorm, copies, reductions): the rest")
""")

md(r"""
The matmuls and attention are well under half of the step (about 25–40% across our runs); the rest is elementwise ops, LayerNorm, copies and reductions, each of which reads and writes whole activation tensors for very little arithmetic. These memory-bound ops are exactly what `torch.compile` fuses and what FlashAttention avoids materializing.
""")

md(r"""
## 10. Why decoding is memory-bound

**Arithmetic intensity** $I$ = FLOPs per byte moved from memory. A device with peak compute $P$ (FLOP/s) and bandwidth $\beta$ (B/s) reaches at most
$$\text{FLOP/s} \le \min(P,\; \beta\cdot I)\qquad\text{(the roofline)},$$
so below the **ridge point** $I^* = P/\beta$ it waits for memory, above it for arithmetic.

One decoding step multiplies a batch of $B$ token vectors by every weight matrix. For one $d\times d$ fp32 matrix: $2Bd^2$ FLOPs; bytes $= 4(d^2 + 2Bd)$ (weights + input + output). For $B \ll d$, $I \approx B/2$: **one token per sequence per weight read**. At $B=1$ almost all time is spent reading weights, and a larger batch is free until $I$ reaches $I^*$.

First the effect on the mini-GPT: cached greedy decoding of 32 tokens for a batch of prompts on `device` (best of 5 runs).
""")

code(r"""
gpt_b = copy.deepcopy(gpt).to(device)
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
tps = []
for bs in batch_sizes:
    prompts = get_batch("val", bs, torch.Generator().manual_seed(1))[0][:, :16].to(device)
    ms = time_fn(lambda: generate_cache(gpt_b, prompts, 32), device, reps=5)
    tps.append(bs * 32 / (ms / 1e3))
    print(f"batch {bs:4d}: {tps[-1]:9.0f} tokens/s   ({ms / 32:.2f} ms per step)")
""")

md(r"""
Going from batch 1 to batch 128 multiplies the tokens per step by 128 while the time per step stays at a few ms (flat within noise on the MPS run): throughput rises ~100×. A model this small is not limited by DRAM bandwidth (its 1.4 MB of weights fit in cache) but by the fixed per-step cost of launching ~50 kernels; the consequence is the same as for a large model limited by reading its weights: one step's fixed cost is shared by every sequence in the batch.

To see the roofline itself, strip the model down to its core operation: one $4096\times4096$ fp32 weight matrix (64 MB, larger than any on-chip cache) times a batch of $B$ vectors. This runs on `device`: a GPU shows the roofline cleanly, while on a CPU the BLAS library switches kernels at small $B$ and blurs the picture.
""")

code(r"""
def arithmetic_intensity(B, d, bytes_per=4):
    return 2 * B * d * d / (bytes_per * (d * d + 2 * B * d))  # @student: return ...  # TODO: FLOPs / bytes for (B×d) @ (d×d)

dm = 4096
Wbig = torch.randn(dm, dm, device=device)
Bs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
gflops, ints = [], []
for Bq in Bs:
    xq = torch.randn(Bq, dm, device=device)
    ms = time_fn(lambda: xq @ Wbig, device, reps=50)
    gflops.append(2 * Bq * dm * dm / (ms / 1e3) / 1e9); ints.append(arithmetic_intensity(Bq, dm))
    print(f"B={Bq:4d}: {ms:7.2f} ms   I = {ints[-1]:6.1f} FLOP/B   {gflops[-1]:7.1f} GFLOP/s")
print("check: I(B=1) =", round(arithmetic_intensity(1, dm), 3), "(≈ 0.5)")
""")

code(r"""
bw = max(g / i for g, i, Bq in zip(gflops, ints, Bs) if Bq <= 16)   # GB/s achieved while streaming weights (small B)
peak = max(gflops)
ridge = peak / bw
ms_list = [2 * Bq * dm * dm / (g * 1e9) * 1e3 for Bq, g in zip(Bs, gflops)]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
I_line = np.logspace(-1, 3, 200)
ax[0].loglog(I_line, np.minimum(peak, bw * I_line), "k--", lw=1, label=f"roofline: {bw:.0f} GB/s, {peak:.0f} GFLOP/s")
ax[0].loglog(ints, gflops, "o-", label="measured")
for Bq, i, g in zip(Bs, ints, gflops):
    if Bq in (1, 8, 64, 512): ax[0].annotate(f"B={Bq}", (i, g), textcoords="offset points", xytext=(5, -12))
ax[0].axvline(ridge, color="gray", lw=0.8)
ax[0].set(xlabel="arithmetic intensity (FLOP / byte)", ylabel="achieved GFLOP/s", title=f"Roofline on {device}: ridge at I* ≈ {ridge:.0f} FLOP/B")
ax[0].legend(loc="lower right")
B_line = np.logspace(0, np.log10(512), 100)
ax[1].loglog(Bs, ms_list, "o-", label="measured")
ax[1].loglog(B_line, [4 * dm * dm / (bw * 1e9) * 1e3] * len(B_line), "k--", lw=1, label="time to read W once (bandwidth)")
ax[1].loglog(B_line, 2 * B_line * dm * dm / (peak * 1e9) * 1e3, "k:", lw=1, label="time for the FLOPs (peak)")
ax[1].set(xlabel="batch size B", ylabel="ms per (B×4096) @ (4096×4096)", title="Time vs batch: flat while memory-bound")
ax[1].legend(); plt.tight_layout(); plt.show()
""")

md(r"""
- Left: at small $B$ the measured points follow the slanted (bandwidth) line, at large $B$ they reach the flat (compute) line. The ridge of this laptop GPU is ~10–20 FLOP/B; a data-center GPU's is much higher (below).
- Right: up to $B\approx 64$ the time stays near the time to read the 64 MB of weights once (dashed line), then grows linearly with $B$ once the FLOPs dominate (dotted line). In between, nothing is gained by computing less: a $B=16$ product costs about as much as a $B=1$ product.
- Individual points jump around (other processes share the GPU, and the library switches kernels with $B$); read the trend.

**The same estimate for a real model.** Llama-2-7B in bf16 is 13.5 GB. An H100 has $\beta\approx3.35$ TB/s and $P\approx990$ TFLOP/s (dense bf16), so $I^*\approx300$ FLOP/B; with 2-byte weights, $I\approx B$.
- At $B=1$: at most $3.35\,\text{TB/s} / 13.5\,\text{GB} \approx 250$ tokens/s, using about $0.3\%$ of the compute.
- The batch must reach ~300 sequences before compute is the limit (in practice the KV cache, which is read per sequence, often runs out of memory first).

This one inequality explains the serving tricks from the lecture: **weight-only int4** quarters the bytes, so ~4× tokens/s at $B=1$ (and why W4A16 beats W8A8 for single-user decoding); **speculative decoding** verifies $k$ tokens per weight read; **batching / PagedAttention** raises $B$; **FlashAttention** raises the intensity of attention itself by keeping tiles on-chip.

✏️ Redo the 7B estimate for int4 weights on a T4 ($\beta\approx 320$ GB/s). Then for a 70B model on 8×H100 with the weights split across GPUs.
""")

md(r"""
## 11. If time: speculative decoding with a 4-bit draft

Greedy speculative decoding: the draft proposes $k$ tokens one by one (cheap); the target runs all of them in **one** cached forward pass and keeps the longest prefix on which its own argmax agrees, plus its own next token. The output equals the target's greedy output exactly. Here the draft is the int4 copy of the mini-GPT (self-speculation with a quantized draft). It is as large as the target, so there is no wall-clock gain on this model; what we measure is how many target passes are saved.
""")

code(r"""
def crop(cache, n):
    for c in cache:
        if c: c[:] = [c[0][:, :, :n], c[1][:, :, :n]]

@torch.no_grad()
def speculative_greedy(target, draft, idx, n_new, k=4):
    seq = idx[0].tolist()
    tc, dc = new_cache(target), new_cache(draft)
    forward_cached(target, idx[:, :-1], tc); forward_cached(draft, idx[:, :-1], dc)   # caches hold all but the last token
    target_calls, accepted = 0, []
    total = idx.shape[1] + n_new
    while len(seq) < total:
        k_now = min(k, total - len(seq))       # do not run past the last position
        # 1) draft proposes k tokens greedily
        d_len = dc[0][0].shape[2]
        logits = forward_cached(draft, torch.tensor([seq[d_len:]], device=idx.device), dc)[:, -1]
        prop = []
        for i in range(k_now):
            prop.append(logits.argmax(-1).item())
            if i < k_now - 1: logits = forward_cached(draft, torch.tensor([[prop[-1]]], device=idx.device), dc)[:, -1]
        # 2) target scores [last token, proposals] in one pass
        L = tc[0][0].shape[2]
        t_pred = forward_cached(target, torch.tensor([[seq[-1]] + prop], device=idx.device), tc)[0].argmax(-1).tolist()
        target_calls += 1
        #>> n_ok = number of leading proposals equal to the target's prediction at that position; append prop[:n_ok] and the target's next token t_pred[n_ok] to seq
        n_ok = 0
        while n_ok < k_now and prop[n_ok] == t_pred[n_ok]: n_ok += 1
        seq += prop[:n_ok] + [t_pred[n_ok]]
        #<<
        accepted.append(n_ok)
        crop(tc, L + 1 + n_ok); crop(dc, min(dc[0][0].shape[2], len(seq) - 1))           # roll back rejected positions
    return torch.tensor([seq[:total]]), target_calls, accepted

n_spec = n_new
draft = quantize_model(gpt, 4)
ref_out = generate_cache(gpt, prompt, n_spec)
for name, dr in [("int4 draft", draft), ("int2 draft", quantize_model(gpt, 2))]:
    out, calls, acc = speculative_greedy(gpt, dr, prompt, n_spec, k=4)
    print(f"{name}: identical to target greedy: {torch.equal(out.to(device), ref_out)}   "
          f"target passes {calls} for {n_spec} tokens ({n_spec / calls:.2f} tokens/pass), mean accepted {np.mean(acc):.2f} of 4")
""")

md(r"""
Both drafts reproduce the target's greedy output exactly, as they must. The int4 draft agrees with the target on 2–3 of 4 proposals on average (3.1 in the run shown) (the exact number varies between runs, since GPU training is not bit-reproducible and the draft is a slightly different model each time), so the target runs 30–40 times for 121 tokens, 3–4 tokens per read of its weights. The int2 draft (loss ≈ 4 in section 6) is almost always rejected (≈ 0.1–0.2 of 4 accepted). A useful draft must be both much cheaper than the target and close to it.

✏️ Train a 1-layer `MiniGPT(vocab, n_embd=48, n_layer=1)` for 300 steps and use it as the draft. Its acceptance will be lower, but each draft step is now much cheaper: estimate the speed-up with cost(draft) ≈ (draft params / target params) × cost(target).
""")

md(r"""
## Summary
- **Accounting**: parameters $\approx 12Ld^2$ + embeddings; $2N$ FLOPs/token forward, $6N$ training, plus $4LTd$ for attention; 16 bytes/param for weights + grads + Adam in fp32 or mixed precision. Measured FLOP/s divided by peak (MFU) tells how much of the hardware is doing model math.
- **Scaling laws**: at fixed $C=6ND$, loss vs $N$ is a valley (IsoFLOP); $L=E+A/N^\alpha+B/D^\beta$ gives $N_{\text{opt}}\propto C^{\beta/(\alpha+\beta)}$ in closed form. Our CPU study shows the valley and its minimum moving to larger models with $C$, but 16 single-seed runs do not pin down the exponents (0.50 on all runs, 0.22 without the largest budget) and the held-out forecast is off by 3.6%. With the published Chinchilla fit, an 8B model on 15T tokens loses 0.03 nats against the compute-optimal 36B model and is 4.5× cheaper per generated token.
- **KV cache**: same tokens as recomputing the prefix, linear instead of quadratic work; the price is $2\,n_{\text{layers}}\,T\,d$ values of memory per sequence.
- **Quantization**: symmetric per-channel RTN is a few lines; 8-bit weights are free, 4-bit costs little, 2-bit breaks the model without GPTQ/AWQ-style error compensation or QAT. Per-channel scales matter most at low bit-widths.
- **Pruning**: unstructured magnitude pruning removes most weights of an over-parameterized net before accuracy moves; channel pruning hurts sooner but yields a genuinely smaller dense model, and a short fine-tune recovers much of the loss.
- **Distillation**: soft targets at temperature $T$, scaled by $T^2$, transfer the teacher's knowledge about class similarity; it helps most when labelled data is scarce.
- **Decoding is memory-bound**: at batch 1 each weight byte serves ~1 FLOP, so tokens/s is set by bandwidth / model bytes. Batching, int4 weights and speculative decoding all raise the work done per byte read.

**Further reading:** Stanford CS336 (Language Modeling from Scratch), lecture 2 *Resource accounting* (section 3 follows it), https://cs336.stanford.edu, lectures 9 and 11 *Scaling laws* (section 4); Kaplan et al., *Scaling Laws for Neural Language Models* (2020), https://arxiv.org/abs/2001.08361 ; Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla, 2022), https://arxiv.org/abs/2203.15556 ; Muennighoff et al., *Scaling Data-Constrained Language Models* (2023), https://arxiv.org/abs/2305.16264 ; Besiroglu et al., *Chinchilla Scaling: A replication attempt* (2024), https://arxiv.org/abs/2404.10102 ; Stanford CME 295 (Fall 2025), lecture 4 slides (quantization, hardware optimization), https://cme295.stanford.edu/slides/fall25-cme295-lecture4.pdf .

**Further watching:** Song Han, *Efficient Methods and Hardware for Deep Learning* (CS231n 2017, Lecture 15), https://www.youtube.com/watch?v=eZdOkDtYMoo ; MIT 6.5940 *TinyML and Efficient Deep Learning Computing* (the source of the lecture slides), https://efficientml.ai ; Karpathy, *Let's reproduce GPT-2 (124M)* (bf16, `torch.compile`, FlashAttention in practice), https://www.youtube.com/watch?v=l8pRSuU81PU .
""")

for k, p in B.write(STEM).items():
    print(k, p)

PDF = "slides/2026-updates/L11b_efficient_models_2026.pdf"
SCALING = "slides/2026-updates/L11_p11_with_scaling_laws_insert_2026.pdf"
# scaling pages: power laws (10), data scaling (9), joint model-data law (12), compute trade-off (13), IsoFLOP (14),
# Chinchilla-optimal table (6), train- vs inference-optimal (15), beyond Chinchilla (17)
print("recap", make_recap(STEM, [(PDF, [2]), (SCALING, [10, 9, 12, 13, 14, 6, 15, 17]),
                                 (PDF, [6, 7, 11, 13, 29, 33, 34, 38, 40, 41, 58, 60, 61, 24])]))
