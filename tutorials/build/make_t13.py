"""Builds T13_exam_practice.ipynb (Tutorial 13, Deep Learning 89-6877, Fall 2026): exam rehearsal."""
from nbkit import Builder
from recap import make_recap

STEM = "T13_exam_practice"
B = Builder()
md, code = B.md, B.code

md(rf"""
# Tutorial 13: exam rehearsal, by hand then by code
**Deep Learning 89-6877, Fall 2026.** TA: Tal Fiskus.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idansc/dl-course-2026/blob/main/tutorials/{STEM}.ipynb)
· [Recap slides (PDF)](https://github.com/idansc/dl-course-2026/blob/main/tutorials/recap/{STEM}_recap.pdf)

The exam is 75% of the grade and is pen-and-paper: shapes, parameter counts, FLOPs, memory, one or two backward passes, and the value of a loss on a tiny example. This session is 12 problems of that kind, one per course block. Each problem is solvable by hand in 5–10 minutes (a simple calculator is fine). Every problem has an **answer cell**, where the hand-derived numbers go, and a **check cell**, which computes the same quantity with PyTorch or an independent simulation.

Plan for today (≈ 60 min):
1. Course recap: the formula sheet (10 min)
2. Foundations: backprop, optimizers, CNNs, sequence models (P1–P4, 15 min)
3. Transformers and efficiency (P5–P6, 10 min)
4. Representations and generative models (P7–P9, 8 min)
5. Post-training and agents (P10–P12, 7 min)
6. Common exam mistakes: three wrong answers and the check that exposes each (10 min)
7. If time: three more problems

In class we solve about half the problems on the board; the rest are homework, in exam conditions.
<<STUDENT>>
**Before the tutorial:** solve every problem on paper first, without running anything. Then write your numbers in the answer cell (replace the `...`) and run the check cell. It prints `OK` or `WRONG` next to the reference value. The worked solutions are presented in the tutorial.
<</STUDENT>>
""")

code(r"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
torch.manual_seed(0); np.random.seed(0)

ANS, RESULTS = {}, {}          # your hand-derived answers go into ANS[...]

def _np(v):
    if torch.is_tensor(v): return v.detach().cpu().double().numpy()
    if isinstance(v, (list, tuple)): return np.array([_np(e) for e in v], dtype=float)
    return np.asarray(v, dtype=float)

def _fmt(v):
    a = _np(v)
    if a.size == 1 and float(a) == round(float(a)) and abs(float(a)) >= 10:
        return f"{int(round(float(a))):,}"
    return np.array2string(a, precision=4, separator=", ", suppress_small=True)

def check(key, ref, tol=2e-3):
    # integer-valued references (counts, shapes) must match exactly; real values to ~3 significant digits
    mine = ANS.get(key)
    if mine is None or mine is Ellipsis:
        print(f"{key:<26} not answered yet"); return
    a, r = _np(mine), _np(ref)
    if a.shape != r.shape:
        ok = False
    elif np.all(r == np.round(r)) and np.all(np.abs(r) >= 1):
        ok = bool(np.all(a == r))
    else:
        ok = bool(np.allclose(a, r, rtol=tol, atol=tol))
    RESULTS[key] = ok
    print(f"{key:<26} yours = {_fmt(a):<24} check = {_fmt(r):<24} {'OK' if ok else 'WRONG'}")
""")

md(r"""
## 1. Course recap: the formula sheet

The recap slides walk through the course arc (perceptron → CNNs/ResNet → SSL → attention/Transformers/ViT → diffusion → LLM training and RLVR). The exam tests the formulas below.

**MLP / backprop.** Downstream gradient = local × upstream. Linear $y = Wx+b$: $\;\partial L/\partial W = (\partial L/\partial y)\,x^\top$, $\;\partial L/\partial x = W^\top \partial L/\partial y$, $\;\partial L/\partial b = \partial L/\partial y$. ReLU masks the gradient with $\mathbb 1[y>0]$. Softmax + CE: $\partial L/\partial z = p - y$.

**Init.** $\mathrm{Var}(a) \approx n_{in}\sigma^2\mathrm{Var}(x)$ → Xavier $\sigma^2 = 1/n_{in}$, Kaiming $\sigma^2 = 2/n_{in}$; for a conv $n_{in} = C_{in}k^2$.

**Optimizers.** Momentum (PyTorch form) $v \leftarrow \beta v + g,\ \theta \leftarrow \theta - \eta v$. Adam: $m \leftarrow \beta_1 m + (1-\beta_1)g$, $v \leftarrow \beta_2 v + (1-\beta_2)g^2$, $\hat m = m/(1-\beta_1^t)$, $\hat v = v/(1-\beta_2^t)$, $\theta \leftarrow \theta - \eta\,\hat m/(\sqrt{\hat v}+\epsilon)$.

**Conv.** $H_{out} = \lfloor (H + 2p - d(k-1) - 1)/s \rfloor + 1$; params $C_{out}(C_{in}k^2 + 1)$; MACs $H_{out}W_{out}C_{out}C_{in}k^2$ (FLOPs ≈ 2 × MACs). Receptive field: $r \leftarrow r + (k_{\text{eff}}-1)\,j$, $j \leftarrow j\cdot s$, starting from $r=1, j=1$, with $k_{\text{eff}} = d(k-1)+1$.

**RNN / LSTM / SSM.** LSTM: 4 gates, each $W_{ih}\in\mathbb R^{h\times d}, W_{hh}\in\mathbb R^{h\times h}$, bias. Linear SSM $h_t = a h_{t-1} + b x_t,\ y_t = c h_t$ ⇔ causal convolution with kernel $K_k = c\,a^k b$.

**Transformer.** Block params ≈ $12d^2$ (attention $4d^2$, MLP $8d^2$ with $d_{ff}=4d$). Matmul FLOPs for $n$ tokens: $2n\cdot(\text{params in linears}) + 4n^2 d$ (scores and weighted sum). KV cache: $2 \cdot L \cdot n_{kv} \cdot d_{head} \cdot T \cdot B \cdot$ bytes.

**Efficiency.** Symmetric int8: $s = \max|w|/127$, $q = \mathrm{round}(w/s)$, $\hat w = sq$, $|w-\hat w| \le s/2$.

**SSL / multimodal.** InfoNCE row loss $-\log \frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ij}/\tau}}$; CLIP = mean of image→text (rows) and text→image (columns) CE.

**VAE / diffusion / flow.** $\mathrm{KL}(\mathcal N(\mu,\sigma^2)\|\mathcal N(0,1)) = \tfrac12(\mu^2+\sigma^2-1-\log\sigma^2)$ per dim. DDPM: $q(x_t|x_0) = \mathcal N(\sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)$, $\bar\alpha_t = \prod_{s\le t}(1-\beta_s)$. Flow matching (linear path): $x_t = (1-t)x_0 + t\epsilon$, target velocity $\epsilon - x_0$.

**Post-training.** LoRA on $W\in\mathbb R^{d_{out}\times d_{in}}$: $r(d_{in}+d_{out})$ params. DPO: $L = -\log\sigma\big(\beta[(\log\pi_\theta(y_w)-\log\pi_{ref}(y_w)) - (\log\pi_\theta(y_l)-\log\pi_{ref}(y_l))]\big)$. GRPO: $A_i = (r_i - \mathrm{mean}(r))/\mathrm{std}(r)$ within the group.

**Retrieval / agents.** MaxSim (ColBERT): $S(q,d) = \sum_i \max_j q_i^\top d_j$. pass@k from $n$ samples with $c$ correct: $1 - \binom{n-c}{k}/\binom{n}{k}$.
""")

# ---------------------------------------------------------------------------------------------------------
md(r"""
## 2. Foundations

### P1. Backprop through Linear → ReLU → squared error
$x = (1, 2)^\top$, $W = \begin{pmatrix}1 & -1\\ 2 & 0\end{pmatrix}$, $b = (0.5, -1)^\top$, target $t = (1, -1)^\top$.
$$y = Wx + b,\qquad h = \mathrm{ReLU}(y),\qquad L = \tfrac12\|h - t\|^2 .$$
Compute $L$, $\partial L/\partial W$, $\partial L/\partial b$, $\partial L/\partial x$.
""")

code(r"""
#>> P1: L, dL/dW (2x2), dL/db, dL/dx by hand
# y = (1-2+0.5, 2+0-1) = (-0.5, 1);  h = (0, 1);  h - t = (-1, 2);  L = (1 + 4)/2
# dL/dh = (-1, 2);  ReLU mask (0, 1) -> dL/dy = (0, 2)
ANS["P1 L"] = 2.5
ANS["P1 dL/dW"] = [[0, 0], [2, 4]]        # outer(dL/dy, x)
ANS["P1 dL/db"] = [0, 2]
ANS["P1 dL/dx"] = [4, 0]                  # W^T dL/dy = (1*0 + 2*2, -1*0 + 0*2)
#<<
""")

code(r"""
x = torch.tensor([1., 2.], requires_grad=True)
W = torch.tensor([[1., -1.], [2., 0.]], requires_grad=True)
b = torch.tensor([0.5, -1.], requires_grad=True)
t = torch.tensor([1., -1.])
L = 0.5 * ((torch.relu(W @ x + b) - t) ** 2).sum()
L.backward()
check("P1 L", L.item()); check("P1 dL/dW", W.grad); check("P1 dL/db", b.grad); check("P1 dL/dx", x.grad)
""")

md(r"""
The first unit has $y_1 = -0.5 < 0$, so its row of $\partial L/\partial W$ is zero even though its error $h_1 - t_1 = -1$ is not: a dead ReLU passes no gradient.

### P2. Momentum and Adam, two steps
A scalar parameter starts at $\theta_0 = 0$. The gradients at the two steps are $g_1 = 2$, $g_2 = -1$ (fixed, whatever $\theta$ is). Learning rate $\eta = 0.1$.
- (a) SGD with momentum $\beta = 0.9$ (PyTorch form, $v_0=0$). Give $\theta_1, \theta_2$.
- (b) Adam with $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon \to 0$. Give $\theta_1, \theta_2$.
""")

code(r"""
#>> P2: theta_1, theta_2 for momentum and for Adam
# momentum: v1 = 2 -> th1 = -0.2;  v2 = 0.9*2 - 1 = 0.8 -> th2 = -0.2 - 0.08
ANS["P2 momentum"] = [-0.2, -0.28]
# Adam step 1: m_hat = g, v_hat = g^2  ->  step = lr * sign(g) = 0.1
m2, v2 = 0.9 * 0.2 + 0.1 * (-1), 0.999 * 0.004 + 0.001 * 1
step2 = 0.1 * (m2 / (1 - 0.9**2)) / math.sqrt(v2 / (1 - 0.999**2))     # 0.1 * 0.4211 / 1.5809
ANS["P2 Adam"] = [-0.1, -0.1 - step2]
#<<
""")

code(r"""
def run(opt_cls, **kw):
    th = torch.zeros(1, requires_grad=True)
    opt = opt_cls([th], lr=0.1, **kw)
    out = []
    for g in (2.0, -1.0):
        opt.zero_grad(); (g * th).sum().backward(); opt.step()   # d(g*th)/dth = g
        out.append(th.item())
    return out
check("P2 momentum", run(torch.optim.SGD, momentum=0.9))
check("P2 Adam", run(torch.optim.Adam, eps=1e-12))
""")

md(r"""
Adam's first step is $\eta\cdot\mathrm{sign}(g_1)$ for any gradient size: bias correction makes $\hat m_1/\sqrt{\hat v_1} = g_1/|g_1|$. At step 2 the averaged direction is still positive ($\hat m_2 = 0.42$), so Adam keeps moving left although $g_2$ points right; momentum does the same ($v_2 = 0.8$).

### P3. A conv stack: shapes, parameters, MACs, receptive field
Input $3\times32\times32$. All convs have bias.

| layer | kernel | stride | padding | $C_{out}$ |
|---|---|---|---|---|
| conv1 | 3×3 | 1 | 1 | 16 |
| conv2 | 3×3 | 2 | 1 | 32 |
| conv3 | 3×3 | 2 | 1 | 64 |
| conv4 | 1×1 | 1 | 0 | 10 |

Give (a) the output shape $(C,H,W)$, (b) the total number of parameters, (c) the total multiply-accumulates (MACs, ignore bias adds), (d) the receptive field of one output pixel.
""")

code(r"""
#>> P3: output shape, #params, #MACs, receptive field
# shapes: 16x32x32 -> 32x16x16 -> 64x8x8 -> 10x8x8
ANS["P3 out shape"] = (10, 8, 8)
ANS["P3 params"] = (16*(3*9 + 1)) + (32*(16*9 + 1)) + (64*(32*9 + 1)) + (10*(64 + 1))    # 448 + 4640 + 18496 + 650
ANS["P3 MACs"] = 32*32*16*(3*9) + 16*16*32*(16*9) + 8*8*64*(32*9) + 8*8*10*64
# RF: r = 1, j = 1 -> conv1: r = 3 -> conv2: r = 5, j = 2 -> conv3: r = 5 + 2*2 = 9, j = 4 -> conv4 (1x1): 9
ANS["P3 RF"] = 9
#<<
""")

code(r"""
net = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1), nn.Conv2d(16, 32, 3, 2, 1),
                    nn.Conv2d(32, 64, 3, 2, 1), nn.Conv2d(64, 10, 1))
inp = torch.randn(1, 3, 32, 32, requires_grad=True)
fc = FlopCounterMode(display=False)
with fc:
    out = net(inp)
check("P3 out shape", tuple(out.shape[1:]))
check("P3 params", sum(p.numel() for p in net.parameters()))
check("P3 MACs", fc.get_total_flops() // 2)          # FlopCounter counts 2 FLOPs per MAC

# receptive field: all-positive weights (no cancellation), gradient of one central output pixel w.r.t. the input
with torch.no_grad():
    for m in net: m.weight.fill_(1.0)
inp.grad = None
net(inp)[0, 0, 4, 4].backward()
rows = inp.grad[0].abs().sum((0, 2)).nonzero().flatten()
check("P3 RF", rows.max().item() - rows.min().item() + 1)
""")

md(r"""
conv2 and conv3 cost the same 1.18M MACs each: halving $H$ and $W$ divides the positions by 4, doubling $C_{in}$ and $C_{out}$ multiplies the per-position cost by 4. conv1 has the fewest parameters (448) but 442k MACs, because it runs at full resolution. Parameters and compute are different budgets.

### P4. LSTM parameters and the SSM convolution view
- (a) How many parameters does `nn.LSTM(input_size=10, hidden_size=20)` have? (PyTorch keeps two bias vectors, $b_{ih}$ and $b_{hh}$.)
- (b) Linear SSM $h_t = a h_{t-1} + b x_t$, $y_t = c h_t$ with $a = 0.5, b = 1, c = 2$, $h_0 = 0$, input $x = (1, 0, 0, 1)$. Give the convolution kernel $K = (K_0..K_3)$ and the outputs $y_1..y_4$.
""")

code(r"""
#>> P4: LSTM #params; SSM kernel K and outputs y
ANS["P4 LSTM params"] = 4 * (20*10 + 20*20 + 20 + 20)     # 4 gates x (W_ih, W_hh, b_ih, b_hh)
ANS["P4 SSM kernel"] = [2, 1, 0.5, 0.25]                  # c a^k b
ANS["P4 SSM y"] = [2, 1, 0.5, 2 + 0.25]                   # y_4 = K_0 x_4 + K_3 x_1
#<<
""")

code(r"""
check("P4 LSTM params", sum(p.numel() for p in nn.LSTM(10, 20).parameters()))

a, b_, c = 0.5, 1.0, 2.0
xs = torch.tensor([1., 0., 0., 1.])
h, ys = 0.0, []
for xt in xs:                                   # the recurrence (what an RNN computes)
    h = a * h + b_ * xt; ys.append(c * h)
K = c * a ** torch.arange(4.) * b_
y_conv = F.conv1d(F.pad(xs, (3, 0)).view(1, 1, -1), K.flip(0).view(1, 1, -1)).flatten()   # causal conv
check("P4 SSM kernel", K); check("P4 SSM y", torch.stack(ys))
print("recurrence vs convolution, max diff:", (torch.stack(ys) - y_conv).abs().max().item())
""")

md(r"""
A linear time-invariant SSM is both an RNN (sequential, $O(1)$ state at inference) and a convolution (parallel over $t$ in training). Mamba makes $a, b, c$ depend on $x_t$, which keeps the recurrence but removes the fixed kernel.

## 3. Transformers and efficiency

### P5. One Transformer block: parameters, FLOPs, KV cache
Post-LN encoder block, $d = 512$, $h = 8$ heads, $d_{ff} = 2048$, biases in every linear layer, two LayerNorms (γ and β each).
- (a) Number of parameters. Compare with $12d^2$.
- (b) Forward matmul FLOPs for one sequence of $n = 128$ tokens (2 FLOPs per multiply-add; count only matmuls, not softmax/LN/bias).
- (c) KV cache in GiB for a 7B-style decoder: $L=32$ layers, 32 heads of $d_{head}=128$, fp16, context $T = 4096$, batch 1.
- (d) The same model with grouped-query attention (8 KV heads) at $T = 8192$.
""")

code(r"""
#>> P5: block #params, forward FLOPs at n=128, KV cache (GiB) for MHA and for GQA
d, dff, n = 512, 2048, 128
ANS["P5 params"] = 4*(d*d + d) + (d*dff + dff) + (dff*d + d) + 2*(2*d)   # QKVO + MLP + 2 LN = 3,152,384 (12d^2 = 3,145,728)
ANS["P5 FLOPs"] = 2*n*(4*d*d + 2*d*dff) + 2*(2*n*n*d)                    # linears + (QK^T and AV)
ANS["P5 KV MHA GiB"] = 2 * 32 * 32 * 128 * 4096 * 2 / 2**30               # K and V, layers, heads, d_head, T, bytes
ANS["P5 KV GQA GiB"] = 2 * 32 * 8 * 128 * 8192 * 2 / 2**30
#<<
""")

code(r"""
class Block(nn.Module):              # from earlier tutorials: post-LN encoder block, explicit attention
    def __init__(self, d, h, dff):
        super().__init__()
        self.h, self.qkv, self.o = h, nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, dff), nn.ReLU(), nn.Linear(dff, d))
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2) / math.sqrt(D // self.h)).softmax(-1)
        x = self.ln1(x + self.o((att @ v).transpose(1, 2).reshape(B, T, D)))
        return self.ln2(x + self.mlp(x))

blk = Block(512, 8, 2048)
ref_layer = nn.TransformerEncoderLayer(512, 8, 2048)
print("params, ours vs nn.TransformerEncoderLayer:",
      sum(p.numel() for p in blk.parameters()), sum(p.numel() for p in ref_layer.parameters()))
check("P5 params", sum(p.numel() for p in blk.parameters()))
fc = FlopCounterMode(display=False)
with fc:
    blk(torch.randn(1, 128, 512))
check("P5 FLOPs", fc.get_total_flops())

def kv_cache_gib(layers, kv_heads, d_head, T, batch=1, dtype=torch.float16):
    # build the actual cache tensors on the meta device (shapes only, no memory)
    cache = [torch.empty(2, batch, kv_heads, T, d_head, dtype=dtype, device="meta") for _ in range(layers)]
    return sum(c.numel() * c.element_size() for c in cache) / 2**30
check("P5 KV MHA GiB", kv_cache_gib(32, 32, 128, 4096))
check("P5 KV GQA GiB", kv_cache_gib(32, 8, 128, 8192))
""")

md(r"""
- The exact count (3,152,384) is within 0.2% of $12d^2$ (3,145,728): biases and LN are negligible at this width.
- At $n = 128$ the attention matmuls ($4n^2d$ = 33.6M) are 4% of the 839M FLOPs; the linears dominate. They break even at $n = 6d = 3072$ for $d_{ff}=4d$ (set $4n^2d = 24nd^2$).
- 2 GiB of KV cache per 4k-token sequence is why serving is memory-bound. GQA with 4× fewer KV heads at 2× the context costs half of it: 1 GiB.

### P6. Symmetric int8 quantization
$w = (0.52, -1.27, 0.034, 0.996)$, per-tensor symmetric int8 with range $[-127, 127]$.
Give the scale $s$, the integer codes $q$, the dequantized $\hat w$, and $\max_i |w_i - \hat w_i|$. What is the worst case over any $w$ with the same $\max|w|$?
""")

code(r"""
#>> P6: scale, codes, dequantized weights, max error
ANS["P6 scale"] = 1.27 / 127                      # 0.01
ANS["P6 q"] = [52, -127, 3, 100]                  # round(52.0), -127, round(3.4), round(99.6)
ANS["P6 w_hat"] = [0.52, -1.27, 0.03, 1.00]
ANS["P6 max err"] = 0.004                         # worst case s/2 = 0.005
#<<
""")

code(r"""
w = torch.tensor([0.52, -1.27, 0.034, 0.996])
s = w.abs().max() / 127
w_hat = torch.fake_quantize_per_tensor_affine(w, s.item(), 0, -127, 127)
check("P6 scale", s); check("P6 q", torch.round(w_hat / s)); check("P6 w_hat", w_hat)
check("P6 max err", (w - w_hat).abs().max())
""")

md(r"""
The error bound $s/2$ is set by the **largest** weight. One outlier of magnitude 12.7 would make $s = 0.1$ and round $0.034$ to 0 and $0.52$ to $0.5$. This is why LLM quantization uses per-channel or per-group scales and treats outlier features separately.

## 4. Representations and generative models

### P7. CLIP loss on a batch of 2
Similarity logits (already divided by $\tau$), rows = images, columns = texts, matched pairs on the diagonal:
$$S = \begin{pmatrix} 2 & 1 \\ 0 & 1 \end{pmatrix}.$$
Give the image→text loss (mean CE over rows), the text→image loss (mean CE over columns), and the CLIP loss (their mean).
""")

code(r"""
#>> P7: image->text, text->image, CLIP loss
# rows: row 1 -log(e^2/(e^2+e^1)) = log(1+e^-1); row 2 -log(e^1/(e^0+e^1)) = log(1+e^-1)
i2t = math.log(1 + math.exp(-1))
# columns: col 1 (2, 0) -> log(1+e^-2); col 2 (1, 1) -> log 2
t2i = (math.log(1 + math.exp(-2)) + math.log(2)) / 2
ANS["P7 i2t"], ANS["P7 t2i"], ANS["P7 CLIP"] = i2t, t2i, (i2t + t2i) / 2
#<<
""")

code(r"""
S = torch.tensor([[2., 1.], [0., 1.]])
labels = torch.arange(2)
l_i2t, l_t2i = F.cross_entropy(S, labels), F.cross_entropy(S.T, labels)
check("P7 i2t", l_i2t); check("P7 t2i", l_t2i); check("P7 CLIP", (l_i2t + l_t2i) / 2)
""")

md(r"""
Text 2 scores 1 against both images, so its column loss is $\log 2$: the model cannot tell which image it belongs to. The row view misses this, since image 2 does prefer text 2 ($1 > 0$). The symmetric loss catches errors in both directions.

✏️ SimCLR's NT-Xent on $N$ pairs builds a $2N\times 2N$ matrix over all views, masks the diagonal (self-similarity), and each view's positive is its partner. With $N=2$, how many negatives does each view have? (Answer: $2N-2 = 2$.)

### P8. VAE: KL to the standard normal
Encoder output for one input: $\mu = (1, 0)$, $\sigma = (0.5, 1)$. Give $\mathrm{KL}(q(z|x)\,\|\,\mathcal N(0, I))$.
""")

code(r"""
#>> P8: KL(N(mu, sigma^2) || N(0, I)), summed over the 2 dims
# dim 1: 0.5*(1 + 0.25 - 1 - ln 0.25) ; dim 2: 0.5*(0 + 1 - 1 - 0) = 0
ANS["P8 KL"] = 0.5 * (1 + 0.25 - 1 - math.log(0.25)) + 0.0
#<<
""")

code(r"""
from torch.distributions import Normal, kl_divergence
q = Normal(torch.tensor([1., 0.]), torch.tensor([0.5, 1.]))
check("P8 KL", kl_divergence(q, Normal(0., 1.)).sum())
""")

md(r"""
The second dimension already equals the prior and costs nothing. In the first, the shrunken variance ($-\tfrac12\log\sigma^2 = 0.69$) costs more than the shifted mean ($\tfrac12\mu^2 = 0.5$).

### P9. Diffusion forward marginal and the flow-matching target
- (a) DDPM with $\beta_1, \beta_2, \beta_3 = 0.1, 0.2, 0.5$ and $x_0 = 2$. Give the mean and standard deviation of $q(x_3 \mid x_0)$.
- (b) Flow matching with $x_t = (1-t)x_0 + t\epsilon$. For $x_0 = 2$, $\epsilon = -1$, $t = 0.25$, give $x_t$ and the regression target $v = dx_t/dt$.
""")

code(r"""
#>> P9: mean and std of q(x_3|x_0); x_t and target velocity for flow matching
alpha_bar = 0.9 * 0.8 * 0.5                       # 0.36
ANS["P9 DDPM mean, std"] = [math.sqrt(alpha_bar) * 2, math.sqrt(1 - alpha_bar)]    # (1.2, 0.8)
ANS["P9 FM x_t, v"] = [0.75 * 2 + 0.25 * (-1), -1 - 2]                             # (1.25, -3)
#<<
""")

code(r"""
betas = torch.tensor([0.1, 0.2, 0.5])
ab = torch.cumprod(1 - betas, 0)[-1]
check("P9 DDPM mean, std", [ab.sqrt() * 2, (1 - ab).sqrt()])

# independent check: run the 3-step noising chain x_t = sqrt(1-b) x_{t-1} + sqrt(b) eps on 200k samples
xt = torch.full((200_000,), 2.0)
for beta in betas:
    xt = (1 - beta).sqrt() * xt + beta.sqrt() * torch.randn_like(xt)
print(f"Monte Carlo: mean {xt.mean():.4f}, std {xt.std():.4f}")

tt = torch.tensor(0.25, requires_grad=True)
x_fm = (1 - tt) * 2.0 + tt * (-1.0)
x_fm.backward()
check("P9 FM x_t, v", [x_fm.item(), tt.grad.item()])
""")

md(r"""
The closed form and the simulated chain agree (mean 1.2, std 0.8 up to Monte Carlo noise): the per-step Gaussians compose into one Gaussian, so training can sample $x_t$ directly at any $t$ without running the chain. The flow-matching target does not depend on $t$ for a linear path: it is the constant velocity $\epsilon - x_0$.

## 5. Post-training and agents

### P10. LoRA trainable parameters
A 32-layer model with $d = 4096$. LoRA of rank $r = 8$ on $W_q$ and $W_v$ (both $4096\times4096$) in every layer. How many trainable parameters? What fraction of the four attention projections ($W_q, W_k, W_v, W_o$) is that?
""")

code(r"""
#>> P10: LoRA trainable params, and the fraction of the attention projections
ANS["P10 LoRA params"] = 32 * 2 * 8 * (4096 + 4096)     # layers x matrices x r(d_in + d_out)
ANS["P10 fraction"] = ANS["P10 LoRA params"] / (32 * 4 * 4096 * 4096)
#<<
""")

code(r"""
from peft import LoraConfig, get_peft_model

class Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = (nn.Linear(d, d, bias=False) for _ in range(4))
class Model(nn.Module):
    def __init__(self, L=32, d=4096):
        super().__init__(); self.layers = nn.ModuleList(Attn(d) for _ in range(L))

with torch.device("meta"):                # 2.1B parameters as shapes only: no memory is allocated
    model = get_peft_model(Model(), LoraConfig(r=8, target_modules=["q_proj", "v_proj"]))
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
check("P10 LoRA params", trainable); check("P10 fraction", trainable / frozen)
""")

md(r"""
4.2M trainable parameters, 0.2% of the attention weights and about 0.06% of a 6.7B model. The rank, not the width, controls the cost: per matrix it grows as $r(d_{in}+d_{out})$, against $d_{in}d_{out}$ for full fine-tuning.

### P11. DPO loss and GRPO advantages
- (a) DPO with $\beta = 0.1$. Policy log-probs: chosen $-10$, rejected $-12$. Reference log-probs: chosen $-11$, rejected $-11$. Give the loss and $\partial L/\partial \log\pi_\theta(y_w)$.
- (b) GRPO: a group of 8 sampled answers gets rewards $(1,0,0,1,1,0,0,0)$. Give the advantage of a correct and of a wrong answer (population std).
""")

code(r"""
#>> P11: DPO loss and its gradient w.r.t. the chosen log-prob; GRPO advantages
margin = 0.1 * ((-10 - -11) - (-12 - -11))        # beta * (1 - (-1)) = 0.2
sig = lambda z: 1 / (1 + math.exp(-z))
ANS["P11 DPO loss"] = -math.log(sig(margin))      # log(1 + e^-0.2)
ANS["P11 DPO grad"] = -0.1 * sig(-margin)         # dL/dmargin = -sigma(-margin), times beta
mean, std = 3 / 8, math.sqrt(3 / 8 * 5 / 8)
ANS["P11 GRPO A"] = [(1 - mean) / std, (0 - mean) / std]
#<<
""")

code(r"""
logp_w = torch.tensor(-10., requires_grad=True)
logp_l, ref_w, ref_l = torch.tensor(-12.), torch.tensor(-11.), torch.tensor(-11.)
loss = -F.logsigmoid(0.1 * ((logp_w - ref_w) - (logp_l - ref_l)))
loss.backward()
check("P11 DPO loss", loss); check("P11 DPO grad", logp_w.grad)

r = torch.tensor([1., 0., 0., 1., 1., 0., 0., 0.])
A = (r - r.mean()) / r.std(unbiased=False)
check("P11 GRPO A", [A[0], A[1]])
print("advantages sum to", A.sum().item(), "; an all-correct group gives std = 0 -> no learning signal")
""")

md(r"""
The DPO gradient is negative, so gradient descent **raises** $\log\pi_\theta(y_w)$. Its size, $\beta\,\sigma(-\text{margin})$, shrinks as the margin grows: pairs the policy already separates stop contributing. In GRPO the rarer outcome gets the larger advantage ($+1.29$ for the 3 correct vs $-0.77$ for the 5 wrong), and the advantages sum to zero. TRL uses the unbiased std ($N-1$), which changes the values (by $\sqrt{7/8}$) but not the signs.

### P12. MaxSim and pass@k
- (a) Query token embeddings $q_1 = (1, 0)$, $q_2 = (0.6, 0.8)$; document tokens $d_1 = (0.8, 0.6)$, $d_2 = (0, 1)$, $d_3 = (-1, 0)$. Give the ColBERT MaxSim score.
- (b) A model produced $n = 10$ samples for a problem; $c = 3$ pass the tests. Give the unbiased pass@1 and pass@5.
""")

code(r"""
#>> P12: MaxSim score; pass@1 and pass@5
# q1 . d = (0.8, 0, -1) -> 0.8 ;  q2 . d = (0.96, 0.8, -0.6) -> 0.96
ANS["P12 MaxSim"] = 0.8 + 0.96
ANS["P12 pass@k"] = [3 / 10, 1 - math.comb(7, 5) / math.comb(10, 5)]   # 1 - 21/252
#<<
""")

code(r"""
Q = torch.tensor([[1., 0.], [0.6, 0.8]])
D = torch.tensor([[0.8, 0.6], [0., 1.], [-1., 0.]])
check("P12 MaxSim", (Q @ D.T).max(dim=1).values.sum())

def pass_at_k(n, c, k):        # numerically stable form from the Codex paper (Chen et al., 2021)
    return 1.0 if n - c < k else 1 - np.prod(1 - k / np.arange(n - c + 1, n + 1))
check("P12 pass@k", [pass_at_k(10, 3, 1), pass_at_k(10, 3, 5)])

# Monte Carlo: draw k of the n samples without replacement, success if any is correct
rng = np.random.default_rng(0)
hits = [np.isin(rng.choice(10, 5, replace=False), [0, 1, 2]).any() for _ in range(100_000)]
print(f"Monte Carlo pass@5: {np.mean(hits):.4f}")
""")

md(r"""
pass@5 = 0.917 from a 30% per-sample success rate. This is why reported pass@k numbers must state $k$, and why RL that sharpens pass@1 can leave pass@k at large $k$ unchanged.
""")

code(r"""
print(f"score: {sum(RESULTS.values())} / {len(RESULTS)} checks OK")
for k, ok in RESULTS.items():
    if not ok: print("  wrong:", k)
""")

# ---------------------------------------------------------------------------------------------------------
md(r"""
## 6. Common exam mistakes

Three wrong answers that come up every year. Each is a plausible derivation with one slip; the check exposes it.

### 6.1 Parameter counts: the bias that is there, and the bias that is not
Question: parameters of a ResNet basic block, $64 \to 64$ channels (two 3×3 convs, each followed by BatchNorm).
- Wrong answer A: $2\cdot 64\cdot(64\cdot 9 + 1) = 73{,}856$. ResNet convs have **no bias** (`bias=False`): the BN that follows subtracts the mean, so a conv bias would be cancelled. This answer also forgets BN's own parameters.
- Wrong answer B: $2\cdot 64\cdot 64\cdot 9 = 73{,}728$. Bias correctly dropped, BN forgotten.
- Right: $2\cdot 64\cdot 64\cdot 9 + 2\cdot(2\cdot 64) = 73{,}984$: BN has a trainable $\gamma$ and $\beta$ per channel. The running mean and variance are buffers, not parameters.
""")

code(r"""
from torchvision.models.resnet import BasicBlock
blk = BasicBlock(64, 64)
n_params = sum(p.numel() for p in blk.parameters())
n_buffers = sum(b.numel() for name, b in blk.named_buffers() if "num_batches" not in name)
print(f"wrong A: {2*64*(64*9+1):,}   wrong B: {2*64*64*9:,}   right: {2*64*64*9 + 2*2*64:,}")
print(f"torchvision BasicBlock: {n_params:,} parameters (+ {n_buffers} BN running-stat buffers)")
print("conv biases:", [m.bias for m in blk.modules() if isinstance(m, nn.Conv2d)])
""")

md(r"""
### 6.2 Softmax cross-entropy: the sign of the gradient
Logits $z = (2, 1, 0)$, label $y = 0$. $p = \mathrm{softmax}(z) = (0.665, 0.245, 0.090)$.
- Wrong answer: $\partial L/\partial z = y_{\text{onehot}} - p = (0.335, -0.245, -0.090)$. This is the gradient of the **log-likelihood**, which we maximize; the loss is its negative.
- Right: $\partial L/\partial z = p - y_{\text{onehot}} = (-0.335, 0.245, 0.090)$. Sanity check without any algebra: increasing the correct logit must decrease the loss, so its component is negative.

A step of gradient descent with each answer shows the consequence.
""")

code(r"""
z = torch.tensor([2., 1., 0.], requires_grad=True)
y = torch.tensor(0)
loss = F.cross_entropy(z[None], y[None]); loss.backward()
p = z.detach().softmax(0); onehot = F.one_hot(y, 3).float()
print("p =", p.numpy().round(3))
print("autograd   :", z.grad.numpy().round(3))
print("p - y      :", (p - onehot).numpy().round(3))
print("y - p      :", (onehot - p).numpy().round(3), "<- wrong sign")
for name, g in [("p - y", p - onehot), ("y - p", onehot - p)]:
    new = F.cross_entropy((z.detach() - 1.0 * g)[None], y[None])
    print(f"one GD step (lr=1) with {name}: loss {loss.item():.3f} -> {new.item():.3f}")
""")

md(r"""
### 6.3 Kaiming init: where the 2 goes, and what fan-in means for a conv
Question: std of Kaiming-normal weights for `Conv2d(64, 128, 3)` (ReLU network).
- Wrong answer A: $\sigma = 2/n_{in}$ (the 2 put on the std instead of the variance).
- Wrong answer B: $\sigma = \sqrt{2/64}$ (fan-in taken as $C_{in}$; for a conv it is $C_{in}k^2 = 576$).
- Wrong answer C: $\sigma = \sqrt{1/576}$ (Xavier/LeCun, no factor 2 for ReLU).
- Right: $\sigma = \sqrt{2/576} = 0.0589$.

We check the std against `nn.init.kaiming_normal_` and then push a signal through 20 such conv+ReLU layers with each answer.
""")

code(r"""
conv = nn.Conv2d(64, 64, 3, padding=1, bias=False)
nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
fan_in = 64 * 3 * 3
answers = {"right  sqrt(2/576)": math.sqrt(2 / fan_in), "A      2/576": 2 / fan_in,
           "B      sqrt(2/64)": math.sqrt(2 / 64), "C      sqrt(1/576)": math.sqrt(1 / fan_in)}
print(f"nn.init.kaiming_normal_ empirical std: {conv.weight.std():.4f}")
x0 = torch.randn(4, 64, 16, 16)
with torch.no_grad():
    for name, std in answers.items():
        h = x0
        for _ in range(20):
            h = torch.relu(F.conv2d(h, torch.randn(64, 64, 3, 3) * std, padding=1))
        print(f"{name:<20} std = {std:.4f}  ->  activation RMS after 20 layers: {h.pow(2).mean().sqrt():.3g}")
""")

md(r"""
Only the right answer keeps the signal at order 1 after 20 layers (RMS 0.6; the zero padding at the borders of a 16×16 map loses a little each layer). Answer C shrinks it by $\sqrt{2}$ per layer ($2^{-10}\approx 10^{-3}$ predicted, $6\times10^{-4}$ measured); answer A vanishes to 0; answer B grows by $\sqrt{9}=3$ per layer, to $10^{9}$. The fan-in mistake (B) is the most common one on exams and the most destructive here.

**How to avoid all three:** write the shape of every tensor next to the formula, check one limiting case (a correct logit going up must lower the loss; a ReLU layer must preserve $\mathbb E[x^2]$), and count buffers separately from parameters.
""")

md(r"""
## 7. If time: three more problems

✏️ **ViT patchify.** A ViT-B/16 on a $224\times224$ RGB image with $d = 768$. How many tokens enter the encoder (with CLS)? How many parameters does the patch embedding (a `Conv2d` with kernel = stride = 16) have?

✏️ **BM25, one term.** $k_1 = 1.2$, $b = 0.75$, $N = 1000$ documents, the term appears in $n = 10$ of them and 3 times in a document of length 100 (average length 100). Use $\mathrm{IDF} = \ln\frac{N-n+0.5}{n+0.5} + 1$ (BM25+ style, as in Lucene) and score $= \mathrm{IDF}\cdot\frac{f(k_1+1)}{f + k_1(1 - b + b\,|D|/\mathrm{avgdl})}$.

✏️ **Receptive field with dilation.** Add a 3×3 conv with dilation 2 and stride 1 after conv3 in P3. What is the new receptive field?

Solve on paper, then run the cell.
""")

code(r"""
pe = nn.Conv2d(3, 768, 16, stride=16)
print("ViT tokens:", pe(torch.zeros(1, 3, 224, 224)).flatten(2).shape[-1] + 1,
      "| patch-embed params:", f"{sum(p.numel() for p in pe.parameters()):,}")

N_docs, n_t, f, k1, b_ = 1000, 10, 3, 1.2, 0.75
idf = math.log((N_docs - n_t + 0.5) / (n_t + 0.5)) + 1
print(f"BM25 term score: {idf * f * (k1 + 1) / (f + k1 * (1 - b_ + b_ * 1.0)):.3f}")

net_d = nn.Sequential(*list(net), nn.Conv2d(10, 10, 3, padding=2, dilation=2))
with torch.no_grad(): net_d[-1].weight.fill_(1.0)
inp = torch.randn(1, 3, 64, 64, requires_grad=True)
net_d(inp)[0, 0, 8, 8].backward()
rows = inp.grad[0].abs().sum((0, 2)).nonzero().flatten()
print("receptive field with the dilated conv:", rows.max().item() - rows.min().item() + 1)
""")

md(r"""
## Summary
- Every exam quantity here reduces to a shape: write the shape of each tensor, then count. Parameters, MACs, KV cache and LoRA cost are all products of the dimensions.
- Backward passes: local × upstream, masked by ReLU; softmax-CE gives $p - y$.
- Loss values on tiny batches (CLIP, KL, DPO) are a few logs and exponentials; the check is one library call.
- Compute and parameters are separate budgets (P3, P5); memory at inference is the KV cache (P5).
- The common mistakes are not conceptual: a missing or extra bias, a flipped sign, a misplaced factor 2 or a wrong fan-in. A limiting-case check catches each one.

**Further watching:** Karpathy, *Let's build GPT: from scratch, in code, spelled out* (block shapes and parameter counts); Karpathy, *Let's reproduce GPT-2 (124M)* (FLOPs, memory, and the 6N rule).
""")

make_recap(STEM, [("slides/lectures/p13.pdf", [4, 6, 8, 10, 18, 19, 21, 32, 36, 38, 41, 42, 49, 50])])
for k, p in B.write(STEM).items():
    print(k, p)
