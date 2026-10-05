"""Builds T06_self_supervised.ipynb (Tutorial 6, Deep Learning 89-6877, Fall 2026)."""
from nbkit import Builder
from recap import make_recap

STEM = "T06_self_supervised"
B = Builder()
md, code = B.md, B.code

md(r"""
# Tutorial 6: self-supervised learning and frozen features
**Deep Learning 89-6877, Fall 2026.** TA: Tal Fiskus.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idansc/dl-course-2026/blob/main/tutorials/T06_self_supervised.ipynb)
Recap slides: [T06_self_supervised_recap.pdf](https://github.com/idansc/dl-course-2026/blob/main/tutorials/recap/T06_self_supervised_recap.pdf)

Plan for today (≈ 60 min):
1. Lecture recap: pretext tasks, contrastive (SimCLR), masked modeling (MAE), self-distillation (DINO), JEPA, linear probes (10 min)
2. SimCLR: augmentations, NT-Xent, a short pretraining run, a linear probe (15 min)
3. MAE: patch masking and reconstruction on a tiny ViT (10 min)
4. Frozen DINOv2 as a dense backbone: a linear segmentation probe and a position probe (10 min)
5. One probe, three pretraining objectives (5 min)
6. Superposition and sparse autoencoders (10 min)
7. If time: why DINO needs centering and sharpening

Runs on a laptop CPU in under 10 minutes (CIFAR-10 and Oxford-IIIT Pets subsets; the Pets download is ≈ 800 MB). Cells marked ✏️ are for you to try.
On a Colab GPU, scale up by raising `N_PRE`, the epoch counts and the Pets subset size; the comments next to them say where.
<<STUDENT>>
**Before the tutorial:** fill in every `# TODO` (replace the `...`). Each one is followed by a check cell that prints the library answer next to yours. The solutions are presented in the tutorial.
<</STUDENT>>
""")

code(r"""
import math, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
torch.manual_seed(0); random.seed(0); np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)
T0 = time.time()
""")

md(r"""
## 1. Lecture recap

**Self-supervised learning.** Train a feature extractor $f$ on a pretext task whose labels come from the data itself; then freeze $f$ and evaluate it with a small head trained on few labels. The standard head is a **linear probe**: $\hat y = \mathrm{softmax}(W f(x) + b)$, only $W, b$ trained. It measures what the frozen features already make linearly accessible.

**Contrastive (SimCLR).** Two augmentations $\tilde x_i, \tilde x_j$ of the same image form a positive pair; the other $2N-2$ views in the batch are negatives. With $z = g(f(\tilde x))$ (projection head $g$) and cosine similarity $s_{ij} = z_i^\top z_j / \|z_i\|\|z_j\|$, the **NT-Xent / InfoNCE** loss for anchor $i$ is
$$\ell_i = -\log \frac{\exp(s_{i,p(i)}/\tau)}{\sum_{k\neq i}\exp(s_{ik}/\tau)} ,$$
a $(2N-1)$-way classification whose correct class is the other view. Larger batches give more negatives, and $I(f(x); f(x^+)) \ge \log N - \mathcal{L}$.

**Masked modeling (MAE).** Split the image into patches, drop 75% at random, encode only the visible 25%, then a light decoder fills in mask tokens and regresses the pixels; the loss is MSE **on the masked patches only**. After pretraining the decoder is thrown away.

**Self-distillation (DINO → DINOv2/v3).** A student matches a teacher on different crops, $\mathcal{L} = -\sum_k p_t^{(k)} \log p_s^{(k)}$, with $p = \mathrm{softmax}(z/\tau)$. The teacher is an EMA of the student, $\theta_t \leftarrow \lambda\theta_t + (1-\lambda)\theta_s$, and receives no gradient. Two tricks prevent collapse: **centering** (subtract a running mean of teacher logits, so no dimension dominates) and **sharpening** (small $\tau_t$, so the output is not uniform). DINOv3 is the current state of the art for frozen features on dense tasks (segmentation, depth, tracking).

**JEPA (I-JEPA → V-JEPA 2).** Predict the *latent* of masked target blocks from the visible context, $\mathcal{L} = \|\,\mathrm{pred}(s_x, \text{mask}) - \mathrm{sg}(\bar f(y))\|_2^2$, with an EMA target encoder $\bar f$. No pixels and no negatives. V-JEPA 2 applies this to video.

**Frozen features as the task backbone.** A linear layer on frozen DINOv2/v3 patch tokens gives semantic segmentation or depth; promptable segmenters (SAM 3) take points, boxes or text and return masks. Both replace a detector trained per task. CLIP (contrastive image-text) comes in the multimodal week.
""")

md(r"""
### From earlier tutorials
A small CNN (the SimCLR encoder, 64-d output) and a pre-norm transformer block (the MAE ViT).
""")

code(r"""
class SmallCNN(nn.Module):
    # 32x32 RGB in [0,1] -> 4w-d feature (64-d by default)
    def __init__(self, w=16):
        super().__init__()
        def blk(i, o, s): return [nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU()]
        self.net = nn.Sequential(*blk(3, w, 1), *blk(w, 2 * w, 2), *blk(2 * w, 4 * w, 2), *blk(4 * w, 4 * w, 2),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.register_buffer("mean", torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1))
        self.out_dim = 4 * w
    def forward(self, x):
        return self.net((x - self.mean) / self.std)

class Block(nn.Module):
    # pre-norm transformer block: x + MHA(LN(x)), then x + MLP(LN(x))
    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim))
    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))
""")

md(r"""
**Data.** `N_PRE = 5000` unlabeled CIFAR-10 training images for pretraining; the linear probes use the labels of the first 2000 of them and 1000 test images. (Colab GPU: `N_PRE = 50000` and more epochs below.)
""")

code(r"""
cifar_tr = torchvision.datasets.CIFAR10("./data", train=True, download=True)
cifar_te = torchvision.datasets.CIFAR10("./data", train=False, download=True)
CLASSES = cifar_tr.classes
def to_tensor(ds, idx):
    return torch.tensor(ds.data[idx]).permute(0, 3, 1, 2).float() / 255, torch.tensor(ds.targets)[idx]

N_PRE, N_PROBE, N_TEST = 5000, 2000, 1000          # Colab GPU: N_PRE = 50000
g = torch.Generator().manual_seed(0)
X_pre, y_pre = to_tensor(cifar_tr, torch.randperm(50000, generator=g)[:N_PRE])
X_te, y_te = to_tensor(cifar_te, torch.randperm(10000, generator=g)[:N_TEST])
X_probe, y_probe = X_pre[:N_PROBE], y_pre[:N_PROBE]
print(X_pre.shape, X_te.shape)
""")

md(r"""
## 2. SimCLR

### 2.1 Augmentations, batched
SimCLR's two crucial augmentations are **random resized crop** and **color distortion**. We write them as batched tensor ops so every image in the batch gets its own random parameters.
A crop of relative side $s$ centred at $(t_x, t_y)$ (in $[-1,1]$ coordinates) with an optional horizontal flip is one affine map from output to input coordinates,
$$\begin{pmatrix}u\\v\end{pmatrix} = \begin{pmatrix}\pm s & 0\\ 0 & s\end{pmatrix}\begin{pmatrix}x\\y\end{pmatrix} + \begin{pmatrix}t_x\\t_y\end{pmatrix},$$
which `F.affine_grid` turns into a sampling grid and `F.grid_sample` applies with bilinear interpolation.
""")

code(r"""
def crop_flip(x, side, tx, ty, flip):
    # x: (B,3,H,W); side, tx, ty: (B,); flip: (B,) of +1 / -1
    theta = torch.zeros(x.shape[0], 2, 3, device=x.device)
    #>> fill theta with the affine map above (row 0: ±s, 0, t_x; row 1: 0, s, t_y), then sample with affine_grid + grid_sample
    theta[:, 0, 0] = side * flip
    theta[:, 0, 2] = tx
    theta[:, 1, 1] = side
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    #<<

def augment(x, min_area=0.2):
    B, dev = x.shape[0], x.device
    side = torch.empty(B, device=dev).uniform_(min_area, 1.0).sqrt()       # crop area fraction in [0.2, 1]
    tx, ty = [(torch.rand(B, device=dev) * 2 - 1) * (1 - side) for _ in range(2)]
    flip = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
    x = crop_flip(x, side, tx, ty, flip)
    # color jitter with prob 0.8: brightness, contrast, saturation (strength 0.4)
    jit = (torch.rand(B, 1, 1, 1, device=dev) < 0.8).float()
    def r(): return 1 + jit * (torch.rand(B, 1, 1, 1, device=dev) * 0.8 - 0.4)
    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    x = x * r()
    x = (x - x.mean((1, 2, 3), keepdim=True)) * r() + x.mean((1, 2, 3), keepdim=True)
    x = gray + (x - gray) * r()
    # grayscale with prob 0.2
    to_gray = (torch.rand(B, 1, 1, 1, device=dev) < 0.2).float()
    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    x = to_gray * gray.expand_as(x) + (1 - to_gray) * x
    return x.clamp(0, 1)
""")

md(r"""
**Check.** With $s=1, t=0$ and a flip, the result must equal `torch.flip`. With $s=0.5$, $t=0$ it is the centre $16\times16$ crop resized to $32\times32$, i.e. `F.interpolate` on `x[..., 8:24, 8:24]`. The outermost pixel ring differs: `grid_sample` interpolates with the real neighbours outside the crop, `interpolate` clamps at the crop edge.
""")

code(r"""
xb = X_pre[:8]
ones, zeros = torch.ones(8), torch.zeros(8)
print("flip:        max |mine − torch.flip| =", (crop_flip(xb, ones, zeros, zeros, -ones) - torch.flip(xb, [-1])).abs().max().item())
ref = F.interpolate(xb[..., 8:24, 8:24], size=32, mode="bilinear", align_corners=False)
mine = crop_flip(xb, 0.5 * ones, zeros, zeros, ones)
print("centre crop: max |mine − interpolate| (interior) =", (mine - ref)[..., 1:-1, 1:-1].abs().max().item())

torch.manual_seed(0)
fig, axes = plt.subplots(3, 8, figsize=(12, 4.8))
for i in range(8):
    for row, img in enumerate([xb[i], augment(xb[i:i + 1])[0], augment(xb[i:i + 1])[0]]):
        axes[row, i].imshow(img.permute(1, 2, 0).numpy()); axes[row, i].axis("off")
for row, t in enumerate(["original", "view 1", "view 2"]):
    axes[row, 0].set_title(t, loc="left", fontsize=9)
fig.suptitle("Two random SimCLR views of each image"); plt.tight_layout(); plt.show()
""")

md(r"""
### 2.2 NT-Xent
Stack the two views as $z = [z^{(1)}; z^{(2)}] \in \mathbb{R}^{2N\times d}$, L2-normalize, and form the $2N\times 2N$ similarity matrix $S = zz^\top/\tau$. Mask the diagonal (a view is not its own negative); then row $i$ is a $(2N-1)$-way classification whose label is $i+N$ (for $i<N$) or $i-N$. So NT-Xent is `F.cross_entropy` on $S$.
""")

code(r"""
def nt_xent(z1, z2, tau=0.5):
    N = z1.shape[0]
    #>> normalize, similarity matrix / tau, mask the diagonal with -inf, targets = index of the other view, cross-entropy
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = (z @ z.T) / tau
    sim = sim.masked_fill(torch.eye(2 * N, dtype=torch.bool, device=z.device), float("-inf"))
    targets = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, targets)
    #<<

def nt_xent_reference(z1, z2, tau=0.5):
    # the formula, one anchor at a time
    z = torch.cat([z1, z2]); N2 = z.shape[0]; losses = []
    for i in range(N2):
        p = (i + N2 // 2) % N2
        sims = torch.stack([F.cosine_similarity(z[i], z[k], dim=0) / tau for k in range(N2)])
        denom = sum(torch.exp(sims[k]) for k in range(N2) if k != i)
        losses.append(-torch.log(torch.exp(sims[p]) / denom))
    return torch.stack(losses).mean()
""")

code(r"""
z1, z2 = torch.randn(6, 16), torch.randn(6, 16)
print(f"random pairs:    mine {nt_xent(z1, z2):.6f}   reference {nt_xent_reference(z1, z2):.6f}")
z2b = z1 + 0.05 * torch.randn(6, 16)
print(f"matching pairs:  mine {nt_xent(z1, z2b):.6f}   reference {nt_xent_reference(z1, z2b):.6f}")
print(f"chance level log(2N−1) = {math.log(11):.4f}")
""")

md(r"""
For random pairs the loss is near the chance level $\log(2N-1)$; for matched pairs it drops well below it.

### 2.3 Pretraining
Encoder $f$ = `SmallCNN` (64-d), projection head $g$ = MLP 64→128→64, Adam, $\tau=0.5$, batch 256, 12 epochs on 5000 images (228 steps). The paper uses LARS, batch 4096 and 800 epochs on ImageNet; this is a toy version of the same objective.
""")

code(r"""
torch.manual_seed(0)
enc = SmallCNN().to(device)
head = nn.Sequential(nn.Linear(enc.out_dim, 128), nn.ReLU(), nn.Linear(128, 64)).to(device)
opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=2e-3, weight_decay=1e-5)
EPOCHS_SIMCLR, BS = 12, 256                         # Colab GPU: SmallCNN(32), 100+ epochs on N_PRE = 50000
X_pre_d = X_pre.to(device)
simclr_losses, t0 = [], time.time()
for ep in range(EPOCHS_SIMCLR):
    perm = torch.randperm(N_PRE, device=device)
    for k in range(0, N_PRE - BS + 1, BS):
        xb = X_pre_d[perm[k:k + BS]]
        loss = nt_xent(head(enc(augment(xb))), head(enc(augment(xb))))  # @student: loss = ...  # TODO: two augmented views of xb -> enc -> head -> nt_xent
        opt.zero_grad(); loss.backward(); opt.step()
        simclr_losses.append(loss.item())
    if ep % 4 == 0 or ep == EPOCHS_SIMCLR - 1:
        print(f"epoch {ep:2d}  NT-Xent {np.mean(simclr_losses[-(N_PRE // BS):]):.3f}   ({time.time() - t0:.0f}s)")
print(f"chance level log(2·{BS}−1) = {math.log(2 * BS - 1):.3f}")
""")

md(r"""
### 2.4 Linear probe
Freeze the encoder, extract features (the 64-d $h = f(x)$, **before** the projection head), standardize them, and fit multinomial logistic regression with an L2 penalty:
$$\min_{W,b}\ \frac1N\sum_i \mathrm{CE}(W h_i + b,\ y_i) + \frac{\lambda}{2}\|W\|^2 .$$
We minimize it to convergence with L-BFGS; the check compares with `sklearn.linear_model.LogisticRegression`, whose objective is the same up to scaling ($C = 1/(\lambda N)$).
""")

code(r"""
@torch.no_grad()
def extract(f, X, bs=500):
    f.eval()
    out = torch.cat([f(X[k:k + bs].to(device)).float().cpu() for k in range(0, len(X), bs)])
    f.train()
    return out

def linear_probe(Htr, ytr, Hte, yte, wd=1e-3, iters=200):
    mu, sd = Htr.mean(0), Htr.std(0) + 1e-6
    Htr, Hte = (Htr - mu) / sd, (Hte - mu) / sd
    W = torch.zeros(Htr.shape[1], int(ytr.max()) + 1, requires_grad=True)
    b = torch.zeros(W.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        #>> the probe objective above: mean cross-entropy of (Htr W + b) plus (wd/2)·||W||²
        loss = F.cross_entropy(Htr @ W + b, ytr) + wd / 2 * (W ** 2).sum()
        #<<
        loss.backward()
        return loss
    opt.step(closure)
    acc = ((Hte @ W + b).argmax(1) == yte).float().mean().item()
    return acc, (W.detach(), b.detach(), mu, sd)

H_tr, H_te = extract(enc, X_probe), extract(enc, X_te)
acc_simclr, _ = linear_probe(H_tr, y_probe, H_te, y_te)
print(f"SimCLR features, linear probe test accuracy: {acc_simclr:.3f}")
""")

code(r"""
from sklearn.linear_model import LogisticRegression
mu, sd = H_tr.mean(0), H_tr.std(0) + 1e-6
sk = LogisticRegression(C=1 / (1e-3 * len(H_tr)), max_iter=3000).fit(((H_tr - mu) / sd).numpy(), y_probe.numpy())
print(f"sklearn LogisticRegression test accuracy:    {sk.score(((H_te - mu) / sd).numpy(), y_te.numpy()):.3f}")

torch.manual_seed(0)
rand_enc = SmallCNN().to(device)                     # same architecture, never trained
acc_random, _ = linear_probe(extract(rand_enc, X_probe), y_probe, extract(rand_enc, X_te), y_te)
acc_pixels, _ = linear_probe(X_probe.flatten(1), y_probe, X_te.flatten(1), y_te)
print(f"baselines: random-init SmallCNN {acc_random:.3f}, raw pixels {acc_pixels:.3f}, chance 0.100")
""")

md(r"""
The from-scratch probe and sklearn agree (0.439 vs 0.439). SimCLR features reach **0.439**; the same CNN at random init gives 0.296 and raw pixels 0.264. So 228 steps on 5000 unlabeled images add ≈ 14 points over the architecture alone. The NT-Xent loss fell from 5.64 to 5.00 (chance: 6.24), far from converged.

✏️ Set `min_area=1.0` (no cropping) in `augment` and rerun 2.3–2.4. The SimCLR paper (Fig. 5) found crop + color the one pair of augmentations that matters most; what happens to the probe accuracy without crops?
""")

md(r"""
## 3. MAE on a tiny ViT

A $32\times32$ image with $4\times4$ patches is a sequence of $L=64$ tokens of dimension $3\cdot4\cdot4=48$. MAE:
1. **Random masking**: draw noise $\sim U(0,1)$ per token, `argsort` it, keep the first 25% of the shuffled indices. `ids_restore = argsort(ids_shuffle)` undoes the shuffle later.
2. **Encoder** (128-d, 4 blocks) sees only the 16 visible tokens, so it costs a quarter of a full forward.
3. **Decoder** (64-d, 1 block): append mask tokens, unshuffle, add positions, predict 48 pixel values per token.
4. **Loss**: per-token MSE, averaged over the masked tokens only, $\;\mathcal{L} = \frac{\sum_\ell m_\ell \,\|\hat x_\ell - x_\ell\|^2/48}{\sum_\ell m_\ell}$.
""")

code(r"""
def patchify(x, p=4):
    # (B, C, H, W) -> (B, L, C·p·p); token order row-major over the patch grid, features ordered (c, i, j) like F.unfold
    B, C, H, W = x.shape
    #>> reshape to (B, C, H/p, p, W/p, p), permute to (B, H/p, W/p, C, p, p), flatten
    x = x.reshape(B, C, H // p, p, W // p, p).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(B, (H // p) * (W // p), C * p * p)
    #<<

def unpatchify(t, p=4, C=3):
    B, L, _ = t.shape; h = int(L ** 0.5)
    return t.reshape(B, h, h, C, p, p).permute(0, 3, 1, 4, 2, 5).reshape(B, C, h * p, h * p)

def random_masking(t, mask_ratio):
    # t: (B, L, D) -> visible tokens (B, L_keep, D), mask (B, L) with 1 = masked, ids_restore (B, L)
    B, L, D = t.shape
    n_keep = int(L * (1 - mask_ratio))
    #>> noise -> ids_shuffle (argsort) -> ids_restore (argsort of ids_shuffle); gather the first n_keep; build the 0/1 mask in original order
    noise = torch.rand(B, L, device=t.device)
    ids_shuffle = noise.argsort(dim=1)
    ids_restore = ids_shuffle.argsort(dim=1)
    ids_keep = ids_shuffle[:, :n_keep]
    t_vis = torch.gather(t, 1, ids_keep[..., None].expand(-1, -1, D))
    mask = torch.ones(B, L, device=t.device)
    mask[:, :n_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)
    #<<
    return t_vis, mask, ids_restore
""")

code(r"""
x = torch.rand(4, 3, 32, 32)
print("patchify vs F.unfold:  max diff =", (patchify(x) - F.unfold(x, 4, stride=4).transpose(1, 2)).abs().max().item())
print("unpatchify(patchify):  max diff =", (unpatchify(patchify(x)) - x).abs().max().item())
t = torch.randn(4, 64, 48)
t_vis, mask, ids_restore = random_masking(t, 0.75)
print("visible tokens:", t_vis.shape[1], " masked per image:", mask.sum(1).tolist())
print("visible tokens == t[mask == 0]:", torch.equal(t_vis.sort(1).values, torch.stack([t[b][mask[b] == 0] for b in range(4)]).sort(1).values))
""")

code(r"""
class TinyMAE(nn.Module):
    def __init__(self, L=64, pd=48, dim=128, depth=4, ddim=64, ddepth=1, heads=4):
        super().__init__()
        self.embed, self.pos = nn.Linear(pd, dim), nn.Parameter(torch.randn(1, L, dim) * 0.02)
        self.enc, self.norm = nn.Sequential(*[Block(dim, heads) for _ in range(depth)]), nn.LayerNorm(dim)
        self.dec_embed, self.mask_token = nn.Linear(dim, ddim), nn.Parameter(torch.zeros(1, 1, ddim))
        self.dec_pos = nn.Parameter(torch.randn(1, L, ddim) * 0.02)
        self.dec = nn.Sequential(*[Block(ddim, heads) for _ in range(ddepth)])
        self.dec_norm, self.out = nn.LayerNorm(ddim), nn.Linear(ddim, pd)
    def encode(self, x, mask_ratio=0.0):
        t = self.embed(patchify(x)) + self.pos
        mask = ids_restore = None
        if mask_ratio > 0:
            t, mask, ids_restore = random_masking(t, mask_ratio)
        return self.norm(self.enc(t)), mask, ids_restore
    def forward(self, x, mask_ratio=0.75):
        lat, mask, ids_restore = self.encode(x, mask_ratio)
        d = self.dec_embed(lat)
        B, Lv, D = d.shape
        d = torch.cat([d, self.mask_token.expand(B, ids_restore.shape[1] - Lv, D)], 1)
        d = torch.gather(d, 1, ids_restore[..., None].expand(-1, -1, D)) + self.dec_pos
        return self.out(self.dec_norm(self.dec(d))), mask

def mae_loss(pred, target, mask):
    #>> per-token MSE (mean over the 48 values), then average over masked tokens only
    per_token = ((pred - target) ** 2).mean(-1)
    return (per_token * mask).sum() / mask.sum()
    #<<
""")

code(r"""
pred, target, m = torch.randn(4, 64, 48), torch.randn(4, 64, 48), (torch.rand(4, 64) < 0.75).float()
print(f"mae_loss {mae_loss(pred, target, m):.6f}   F.mse_loss on masked tokens {F.mse_loss(pred[m.bool()], target[m.bool()]):.6f}")
""")

md(r"""
**Pretraining.** Inputs are channel-normalized images; the target is the normalized pixels of each patch. 15 epochs on the same 5000 images, batch 128, AdamW with a one-cycle schedule, random flips.
""")

code(r"""
MEAN, STD = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1), torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1)
def norm(x): return (x - MEAN.to(x.device)) / STD.to(x.device)

torch.manual_seed(0)
mae = TinyMAE().to(device)
opt = torch.optim.AdamW(mae.parameters(), lr=1.5e-3, weight_decay=0.05)
EPOCHS_MAE, BS = 15, 128                            # Colab GPU: 200+ epochs on N_PRE = 50000
steps = EPOCHS_MAE * (N_PRE // BS)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1.5e-3, total_steps=steps, pct_start=0.1)
mae_losses, t0 = [], time.time()
for ep in range(EPOCHS_MAE):
    perm = torch.randperm(N_PRE, device=device)
    for k in range(0, N_PRE - BS + 1, BS):
        xb = X_pre_d[perm[k:k + BS]]
        xb = torch.where(torch.rand(BS, 1, 1, 1, device=device) < 0.5, xb.flip(-1), xb)
        xb = norm(xb)
        pred, mask = mae(xb, 0.75)
        loss = mae_loss(pred, patchify(xb), mask)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        mae_losses.append(loss.item())
    if ep % 5 == 0 or ep == EPOCHS_MAE - 1:
        print(f"epoch {ep:2d}  masked MSE {np.mean(mae_losses[-(N_PRE // BS):]):.3f}   ({time.time() - t0:.0f}s)")
""")

code(r"""
torch.manual_seed(1)
mae.eval()
with torch.no_grad():
    xb = norm(X_te[:8].to(device))
    pred, mask = mae(xb, 0.75)
    m = mask[..., None]
    gray = patchify(norm(torch.full_like(xb, 0.5)))                  # masked patches shown as gray
    masked_in = unpatchify(patchify(xb) * (1 - m) + gray * m)
    recon = unpatchify(patchify(xb) * (1 - m) + pred * m)          # visible patches pasted back
mae.train()
def show(t): return (t.cpu() * STD + MEAN).clamp(0, 1).permute(0, 2, 3, 1).numpy()
fig, axes = plt.subplots(3, 8, figsize=(12, 4.8))
for row, (imgs, name) in enumerate([(show(masked_in), "input (75% masked)"), (show(recon), "MAE reconstruction"), (show(xb), "original")]):
    for i in range(8):
        axes[row, i].imshow(imgs[i]); axes[row, i].axis("off")
    axes[row, 0].set_title(name, loc="left", fontsize=9)
fig.suptitle("Tiny MAE on held-out CIFAR-10 test images"); plt.tight_layout(); plt.show()
""")

md(r"""
The masked MSE fell from 0.92 to 0.47 (predicting the dataset mean gives ≈ 1 in these normalized units). The reconstructions recover the colour and coarse layout of each masked region (sky, grass, the brown body of a horse) but no detail: each masked $4\times4$ patch comes out as a smooth block. This is a 0.9M-parameter model after 585 steps; the MAE paper's sharp reconstructions take ViT-L and 800 epochs on ImageNet.

**MAE probe features.** For the probe we encode all 64 tokens (no masking) and average them: a 128-d vector.
""")

code(r"""
class MAEFeatures(nn.Module):
    def __init__(self, mae): super().__init__(); self.mae = mae
    def forward(self, x): return self.mae.encode(norm(x))[0].mean(1)
acc_mae, _ = linear_probe(extract(MAEFeatures(mae), X_probe), y_probe, extract(MAEFeatures(mae), X_te), y_te)
print(f"MAE features, linear probe test accuracy: {acc_mae:.3f}")
""")

md(r"""
## 4. Frozen DINOv2 as a dense backbone

`facebook/dinov2-small` (ViT-S/14, 22M parameters, self-distillation on 142M curated images). A $224\times224$ image gives a $16\times16$ grid of 384-d patch tokens plus a CLS token. We freeze it and train only linear layers on top.

Data: 400 images of Oxford-IIIT Pets with **trimaps** (1 = pet, 2 = background, 3 = boundary), resized to $224\times224$; 300 for training the probe, 100 for testing. We call boundary pixels "pet". (Colab GPU: use the full 3680 trainval images.)
""")

code(r"""
from transformers import AutoModel
dino = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
IMNET_MEAN, IMNET_STD = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

@torch.no_grad()
def dino_tokens(X, size=224, bs=20):
    # X: (B,3,H,W) in [0,1] -> CLS (B,384), patch tokens (B, (size/14)^2, 384)
    cls, patches = [], []
    for k in range(0, len(X), bs):
        x = F.interpolate(X[k:k + bs].float(), size=size, mode="bilinear", align_corners=False, antialias=True)
        h = dino(pixel_values=((x - IMNET_MEAN) / IMNET_STD).to(device)).last_hidden_state.float().cpu()
        cls.append(h[:, 0]); patches.append(h[:, 1:])
    return torch.cat(cls), torch.cat(patches)

pets = torchvision.datasets.OxfordIIITPet("./data", split="trainval", target_types="segmentation", download=True)
N_PETS, N_PETS_TR = 400, 300                        # Colab GPU: N_PETS = len(pets)
idx = torch.randperm(len(pets), generator=torch.Generator().manual_seed(0))[:N_PETS]
imgs, tris = [], []
for i in idx.tolist():
    im, tri = pets[i]
    imgs.append(torch.tensor(np.array(im.convert("RGB").resize((224, 224)))).permute(2, 0, 1))
    tris.append(torch.tensor(np.array(tri.resize((224, 224), resample=0))))
P_img = torch.stack(imgs)                      # uint8 (400, 3, 224, 224)
P_fg = (torch.stack(tris) != 2).float()        # 1 = pet (incl. boundary), (400, 224, 224)
t0 = time.time()
_, P_tok = dino_tokens(P_img / 255.)
print("patch tokens:", tuple(P_tok.shape), f"({time.time() - t0:.0f}s)   pet pixel fraction: {P_fg.mean():.2f}")
""")

md(r"""
**Patch labels.** A patch is "pet" if more than half of its $14\times14$ pixels are pet: reshape the mask to $(B, 16, 14, 16, 14)$ and average over the two within-patch axes. The check is `F.avg_pool2d`.
""")

code(r"""
def patch_fraction(mask, p=14):
    # mask (B, H, W) -> fraction of 1s in each p x p patch, (B, H/p, W/p)
    B, H, W = mask.shape
    #>> reshape and average over the within-patch axes
    return mask.reshape(B, H // p, p, W // p, p).mean((2, 4))
    #<<

frac = patch_fraction(P_fg)
print("max |mine − avg_pool2d| =", (frac - F.avg_pool2d(P_fg[:, None], 14)[:, 0]).abs().max().item())
P_lab = (frac > 0.5).long().flatten(1)         # (400, 256)
""")

md(r"""
**The segmentation probe** is the same `linear_probe` as in section 2, now on 300·256 = 76,800 patch tokens: one 384×2 matrix. At test time we upsample the patch logits bilinearly to $224\times224$ and score pixel-level IoU of the pet class, $\mathrm{IoU} = |P\cap G|/|P\cup G|$, and mIoU over {background, pet}. The check is `sklearn.metrics.jaccard_score`.
""")

code(r"""
def iou(pred, gt, cls):
    #>> intersection over union of (pred == cls) and (gt == cls)
    p, g = pred == cls, gt == cls
    return ((p & g).sum() / (p | g).sum()).item()
    #<<

tr, te = slice(0, N_PETS_TR), slice(N_PETS_TR, N_PETS)
acc_patch, (W, b, mu, sd) = linear_probe(P_tok[tr].flatten(0, 1), P_lab[tr].flatten(), P_tok[te].flatten(0, 1), P_lab[te].flatten())
logits = ((P_tok[te] - mu) / sd) @ W + b                                   # (100, 256, 2)
logits = F.interpolate(logits.permute(0, 2, 1).reshape(-1, 2, 16, 16), size=224, mode="bilinear", align_corners=False)
pred_px, gt_px = logits.argmax(1), P_fg[te].long()
ious = [iou(pred_px, gt_px, c) for c in (0, 1)]
print(f"patch accuracy {acc_patch:.3f}   pixel accuracy {(pred_px == gt_px).float().mean():.3f}")
print(f"IoU background {ious[0]:.3f}  pet {ious[1]:.3f}  mIoU {np.mean(ious):.3f}")

from sklearn.metrics import jaccard_score
print("sklearn jaccard_score per class:", np.round(jaccard_score(gt_px.flatten().numpy(), pred_px.flatten().numpy(), average=None), 3))
""")

code(r"""
# PCA of the patch tokens (fit on all 400 images): the first 3 components as RGB
Z = P_tok.flatten(0, 1); Zc = Z - Z.mean(0)
_, _, V = torch.pca_lowrank(Zc, q=3, center=False)
pcs = (Zc @ V).reshape(N_PETS, 16, 16, 3)
pcs = (pcs - pcs.amin((0, 1, 2))) / (pcs.amax((0, 1, 2)) - pcs.amin((0, 1, 2)))
show_idx = [0, 1, 2, 3, 4]
fig, axes = plt.subplots(4, 5, figsize=(12, 10))
for j, i in enumerate(show_idx):
    k = N_PETS_TR + i
    axes[0, j].imshow(P_img[k].permute(1, 2, 0)); axes[1, j].imshow(P_fg[k], cmap="gray")
    axes[2, j].imshow(pred_px[i], cmap="gray"); axes[3, j].imshow(pcs[k])
    for a in axes[:, j]: a.axis("off")
for r, t in enumerate(["test image", "ground truth (pet = white)", "linear probe on DINOv2", "PCA of patch tokens → RGB"]):
    axes[r, 0].set_title(t, loc="left", fontsize=10)
fig.suptitle("Frozen DINOv2-small, one linear layer, 300 training images"); plt.tight_layout(); plt.show()
""")

md(r"""
One $384\times2$ linear layer gives patch accuracy 0.952, pixel accuracy 0.950, IoU 0.884 (pet) and 0.919 (background); sklearn agrees. The masks follow the outline at patch resolution (14 px), so thin parts are lost or blobby (the cat's tail, the dog's legs in column 4). The PCA row shows why a linear layer is enough: with no labels at all, the first three principal components of the patch tokens already colour the animal differently from the background.

**A second dense probe: where is this patch?** Depth needs depth labels; a free dense target is the patch's own grid position. Fit a linear regression from each patch token to its (row, col) and measure $R^2$ on the 100 test images. Baseline: the same regression from the raw $14\times14\times3$ pixels of the patch.
""")

code(r"""
rows, cols = torch.meshgrid(torch.arange(16.), torch.arange(16.), indexing="ij")
pos = torch.stack([rows.flatten(), cols.flatten()], 1)                      # (256, 2)

def position_probe(Ftr, Fte):
    # ridge regression with a bias column, closed form; returns R^2 on test and mean abs error in patches
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    A = torch.cat([(Ftr - mu) / sd, torch.ones(len(Ftr), 1)], 1)
    Yt = pos.repeat(len(Ftr) // 256, 1)
    Wr = torch.linalg.solve(A.T @ A + 1.0 * torch.eye(A.shape[1]), A.T @ Yt)
    P = torch.cat([(Fte - mu) / sd, torch.ones(len(Fte), 1)], 1) @ Wr
    Y = pos.repeat(len(Fte) // 256, 1)
    r2 = 1 - ((P - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
    return r2, (P - Y).abs().mean().item()

pix = patchify(P_img.float() / 255, p=14)                                   # (400, 256, 588)
for name, Fe in [("DINOv2 patch tokens", P_tok), ("raw patch pixels", pix)]:
    r2, mae_err = position_probe(Fe[tr].flatten(0, 1), Fe[te].flatten(0, 1))
    print(f"{name:20s} R² row {r2[0]:.3f}, col {r2[1]:.3f}   mean |error| {mae_err:.2f} patches")
""")

md(r"""
DINOv2 tokens: $R^2$ = 0.94 (row) and 0.91 (column), mean error 0.96 patches. Raw pixels: $R^2 \approx 0$, error 4.0 patches, i.e. no better than always predicting the centre. Every token still carries its grid position linearly after 12 blocks (it enters through the position embeddings). A **depth probe** is the same recipe with a depth map (e.g. NYU-Depth) as the per-patch regression target; it is one of the dense benchmarks in the DINOv2/v3 papers.

✏️ The segmentation probe was fit on 300 images. Refit it on 10 and on 30 (`tr = slice(0, 10)` …). How many labeled images does a frozen DINOv2 need for a usable pet mask? Compare with what training a U-Net from scratch would need.
""")

md(r"""
## 5. One probe, three pretraining objectives

The same `linear_probe` on the same 2000 / 1000 CIFAR-10 images, five feature extractors. For DINOv2 the $32\times32$ images are upsampled to $112\times112$ ($8\times8$ patches) and the feature is [CLS, mean patch token] (768-d), the DINOv2 linear-eval recipe.
""")

code(r"""
t0 = time.time()
c_tr, p_tr = dino_tokens(X_probe, size=112)
c_te, p_te = dino_tokens(X_te, size=112)
acc_dino, _ = linear_probe(torch.cat([c_tr, p_tr.mean(1)], 1), y_probe, torch.cat([c_te, p_te.mean(1)], 1), y_te)
print(f"DINOv2 features, linear probe test accuracy: {acc_dino:.3f}   ({time.time() - t0:.0f}s)")

res = {"raw pixels": acc_pixels, "random SmallCNN": acc_random, "SimCLR (ours, 5k imgs)": acc_simclr,
       "MAE (ours, 5k imgs)": acc_mae, "DINOv2-S (142M imgs)": acc_dino}
fig, ax = plt.subplots(figsize=(8, 3.4))
bars = ax.barh(list(res), list(res.values()), color=["0.6", "0.6", "C0", "C1", "C2"])
ax.bar_label(bars, fmt="%.3f", padding=3); ax.axvline(0.1, ls="--", c="k", lw=0.8)
ax.set(xlabel="CIFAR-10 linear-probe test accuracy (1000 images)", xlim=(0, 1), title="Same linear probe, different frozen features")
ax.invert_yaxis(); plt.tight_layout(); plt.show()
""")

md(r"""
DINOv2-S **0.822**, SimCLR 0.439, MAE 0.412, random CNN 0.296, pixels 0.264 (dashed line: chance). This is not a comparison of objectives at equal compute: DINOv2 saw 142M images, our encoders saw 5000 images for about a minute each. It shows the protocol (one frozen-feature probe for all extractors) and that both of our self-supervised encoders beat a random network of the same architecture. SimCLR vs MAE (2.7 points) is within noise here: the standard error of an accuracy near 0.43 on 1000 test images is ≈ 1.6 points. Random augmentations and masks are drawn differently on CPU, MPS and CUDA, so your numbers will move by a few points (a CPU run of this notebook gives SimCLR 0.467, MAE 0.438); the ranking pixels < random < SimCLR ≈ MAE ≪ DINOv2 is stable. At ImageNet scale, MAE is weaker than contrastive and self-distillation methods under a linear probe and stronger after fine-tuning.

✏️ MAE features are known to be weak under a linear probe but strong after fine-tuning (MAE paper, Table 3 / Fig. 9). Fine-tune the whole MAE encoder + a linear head for 5 epochs on the 2000 labeled images, and compare with fine-tuning the SimCLR encoder the same way.
""")

md(r"""
## 6. Superposition and sparse autoencoders

A layer with $m$ dimensions has room for $m$ orthogonal directions, but a model may need to represent many more features than that. If features are **sparse** (rarely active together), a model can store $n > m$ of them in **superposition**: almost-orthogonal directions that interfere only when two are active at once. Then single neurons (or single dimensions) are not interpretable, and we need another way to read the features out.

### 6.1 Toy model: 5 features in 2 dimensions
Elhage et al. (2022, *Toy Models of Superposition*): $x\in\mathbb{R}^5$, each $x_i = 0$ with probability $S$ (sparsity) and $\sim U(0,1)$ otherwise; feature $i$ has importance $I_i = 0.8^i$. The model compresses to $h = Wx \in \mathbb{R}^2$ and reconstructs $\hat x = \mathrm{ReLU}(W^\top h + b)$, trained on $\sum_i I_i (x_i - \hat x_i)^2$. Column $W_{:,i}$ is the direction of feature $i$ in the 2-d hidden space.
""")

code(r"""
def train_toy(S, n=5, m=2, steps=3000, seed=0):
    torch.manual_seed(seed)
    I = 0.8 ** torch.arange(n).float()
    W = nn.Parameter(torch.randn(m, n) * 0.1); b = nn.Parameter(torch.zeros(n))
    opt = torch.optim.Adam([W, b], lr=1e-2)
    for _ in range(steps):
        x = torch.rand(1024, n) * (torch.rand(1024, n) > S)
        x_hat = torch.relu(x @ W.T @ W + b)  # @student: x_hat = ...  # TODO: h = W x, x_hat = ReLU(W^T h + b), batched (x is (1024, n))
        loss = (I * (x - x_hat) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return W.detach(), loss.item()

sparsities = [0.0, 0.8, 0.95]
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, S in zip(axes, sparsities):
    W, L = train_toy(S)
    norms = W.norm(dim=0)
    for i in range(5):
        ax.arrow(0, 0, W[0, i], W[1, i], width=0.02, color=plt.cm.viridis(i / 4), length_includes_head=True)
        ax.text(1.12 * W[0, i], 1.12 * W[1, i], str(i), ha="center", va="center", fontsize=9)
    ax.set(xlim=(-1.4, 1.4), ylim=(-1.4, 1.4), aspect="equal", xlabel="hidden dim 1", ylabel="hidden dim 2",
           title=f"S = {S}: {int((norms > 0.5).sum())} of 5 features represented")
    print(f"S = {S}: loss {L:.4f}, ||W_i|| =", np.round(norms.numpy(), 2))
fig.suptitle("Columns of W (feature directions; 0 = most important)"); plt.tight_layout(); plt.show()
""")

md(r"""
- **S = 0 (dense):** only the two most important features get a direction (norm 1.0, orthogonal); features 2–4 are dropped (norm ≤ 0.03). This is what PCA would do.
- **S = 0.8:** four features. Pairs (0, 2) and (1, 4) point in nearly opposite directions: two features can share one axis if they are rarely active at the same time, because the ReLU removes the negative interference. Feature 3 is dropped.
- **S = 0.95:** all five features, arranged as a pentagon (72° apart), with norms 1.11–1.14.

So when features are sparse, the number of features a layer represents is not bounded by its width, and no hidden dimension corresponds to a single feature.

### 6.2 A sparse autoencoder on DINOv2 patch tokens
If real features are stored in superposition, we can try to recover them with a **sparse autoencoder** (SAE; Bricken et al. 2023, *Towards Monosemanticity*): an overcomplete dictionary of $k > d$ directions, with an L1 penalty so each token uses only a few of them,
$$f = \mathrm{ReLU}\big((x - b_d)W_e + b_e\big),\qquad \hat x = f W_d + b_d,\qquad \mathcal{L} = \|x-\hat x\|^2 + \lambda \|f\|_1 ,$$
with the rows of $W_d$ (the dictionary directions) kept at unit norm so the L1 term cannot be cheated by shrinking $f$ and growing $W_d$.
We train one on the 102,400 DINOv2 patch tokens from section 4: $d = 384$, $k = 768$ (2× overcomplete).
""")

code(r"""
class SAE(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        W = torch.randn(k, d); W = W / W.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(W.clone())             # (k, d), unit-norm rows
        self.W_enc = nn.Parameter(W.T.clone())           # (d, k)
        self.b_enc, self.b_dec = nn.Parameter(torch.zeros(k)), nn.Parameter(torch.zeros(d))
    def forward(self, x):
        #>> f = ReLU((x − b_dec) W_enc + b_enc), x_hat = f W_dec + b_dec
        f = torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        x_hat = f @ self.W_dec + self.b_dec
        #<<
        return x_hat, f

Zs = (Z - Z.mean(0)) / Z.std()                           # 102400 x 384, mean squared norm ≈ 384
torch.manual_seed(0)
sae = SAE(384, 768).to(device)
sae.b_dec.data = Zs.mean(0).to(device)
opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
LAM, SAE_STEPS, BS = 3.0, 2000, 512           # Colab GPU: k = 4·384, 20000 steps of 4096
Zs_d, t0 = Zs.to(device), time.time()
for step in range(SAE_STEPS):
    xb = Zs_d[torch.randint(0, len(Zs_d), (BS,), device=device)]
    x_hat, f = sae(xb)
    loss = ((x_hat - xb) ** 2).sum(1).mean() + LAM * f.abs().sum(1).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        sae.W_dec.data /= sae.W_dec.data.norm(dim=1, keepdim=True)
    if step % 500 == 0 or step == SAE_STEPS - 1:
        fve = 1 - ((x_hat - xb) ** 2).sum() / ((xb - xb.mean(0)) ** 2).sum()
        print(f"step {step:5d}  loss {loss.item():7.2f}  L0 {(f > 0).float().sum(1).mean():5.1f}  var. explained {fve:.3f}  ({time.time() - t0:.0f}s)")
""")

md(r"""
**Check.** Fraction of variance explained (FVE) and the mean number of active features per token (L0) on all tokens, and a reference point: PCA with the same number of components as the SAE's L0 (PCA uses the same $k$ directions for every token, the SAE picks its own $k$ per token from 768).
""")

code(r"""
with torch.no_grad():
    acts = torch.cat([sae(Zs_d[k:k + 8192])[1].cpu() for k in range(0, len(Zs_d), 8192)])
    recon = torch.cat([sae(Zs_d[k:k + 8192])[0].cpu() for k in range(0, len(Zs_d), 8192)])
fve = 1 - ((recon - Zs) ** 2).sum() / ((Zs - Zs.mean(0)) ** 2).sum()
L0 = (acts > 0).float().sum(1).mean().item()
freq = (acts > 0).float().mean(0)
_, _, Vp = torch.pca_lowrank(Zs - Zs.mean(0), q=int(round(L0)), center=False)
Zc_ = Zs - Zs.mean(0)
fve_pca = 1 - ((Zc_ - Zc_ @ Vp @ Vp.T) ** 2).sum() / (Zc_ ** 2).sum()
print(f"SAE: L0 = {L0:.1f} active of 768, FVE = {fve:.3f}, dead features (never active) = {(freq == 0).sum().item()}")
print(f"PCA with {int(round(L0))} components: FVE = {fve_pca:.3f}")
""")

md(r"""
**Top-activating patches.** For 6 randomly chosen live features (active on 0.1–5% of tokens), show the 6 patches with the highest activation, at most one per image, with one patch of context on each side (red box = the patch).
""")

code(r"""
def patch_crop(img_i, r, c, ctx=1, p=14):
    im = F.pad(P_img[img_i].float() / 255, (ctx * p,) * 4, value=1.0)
    return im[:, r * p:(r + 2 * ctx + 1) * p, c * p:(c + 2 * ctx + 1) * p].permute(1, 2, 0).numpy()

rng = np.random.default_rng(0)
cands = torch.nonzero((freq > 0.001) & (freq < 0.05)).flatten().numpy()
feats = rng.choice(cands, 6, replace=False)
fig, axes = plt.subplots(6, 6, figsize=(9, 9.6))
for row, j in enumerate(feats):
    order = acts[:, j].argsort(descending=True).numpy()
    seen, shown = set(), 0
    for t in order:
        img_i = int(t) // 256
        if img_i in seen: continue
        seen.add(img_i); r, c = divmod(int(t) % 256, 16)
        ax = axes[row, shown]; ax.imshow(patch_crop(img_i, r, c))
        ax.add_patch(plt.Rectangle((13.5, 13.5), 14, 14, fill=False, ec="red", lw=1.5)); ax.axis("off")
        shown += 1
        if shown == 6: break
    axes[row, 0].set_title(f"feature {j} (active on {100 * freq[j]:.1f}% of tokens)", loc="left", fontsize=9)
fig.suptitle("SAE on DINOv2-small patch tokens: top-activating patches of 6 random features"); plt.tight_layout(); plt.show()
""")

md(r"""
With $\lambda = 3$: on average 18 of 768 features are active per token, FVE = 0.73, no dead features. PCA with 18 components explains only 0.40: a sparse code chosen per token from an overcomplete dictionary describes the tokens much better than any fixed 18-dimensional subspace, as superposition predicts. Several of the 6 random features read off their top patches: 489 fires on a dark leg or body edge against a lighter background, 392 on pale fluffy fur, 236 on flat floor or wall texture, 651 on tan legs; 207 mixes unrelated patches. 400 images and 2000 steps is a very small SAE; more tokens and steps (Colab) give cleaner features.

✏️ Plot the activation map of one feature (`acts[:, j].reshape(400, 16, 16)`) over a few whole images. Then raise `LAM` by 3× and retrain: what happens to L0, FVE and the number of dead features?
""")

md(r"""
## 7. If time: why DINO needs centering and sharpening

DINO's loss is cross-entropy between a teacher distribution $p_t = \mathrm{softmax}((z_t - c)/\tau_t)$ and the student's $p_s = \mathrm{softmax}(z_s/\tau_s)$, with an EMA teacher. Nothing stops both networks from outputting the same distribution for every image. Two failure modes, two diagnostics:
- **collapse to one dimension**: every image puts its mass on the same output, so the batch-mean distribution $\bar p_t$ has entropy $\approx 0$;
- **collapse to uniform**: $p_t$ is flat for every image, so its per-image entropy is $\approx \log K$.
Healthy training has per-image entropy well below $\log K$ (sharp) and batch-mean entropy near $\log K$ (diverse). We run 150 steps of DINO on the CIFAR subset with $K=64$ outputs in three settings.
""")

code(r"""
def run_dino(center, tau_t, steps=150, K=64, tau_s=0.1, m=0.996, seed=0):
    torch.manual_seed(seed)
    student = nn.Sequential(SmallCNN(), nn.Linear(64, 256), nn.GELU(), nn.Linear(256, K)).to(device)
    teacher = nn.Sequential(SmallCNN(), nn.Linear(64, 256), nn.GELU(), nn.Linear(256, K)).to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=0.04)
    c = torch.zeros(K, device=device)
    H_img, H_mean = [], []
    for step in range(steps):
        xb = X_pre_d[torch.randint(0, N_PRE, (64,), device=device)]
        v1, v2 = augment(xb), augment(xb)
        with torch.no_grad():
            zt = teacher(torch.cat([v1, v2]))
            pt = F.softmax((zt - c) / tau_t, -1) if center else F.softmax(zt / tau_t, -1)
            if center: c = 0.9 * c + 0.1 * zt.mean(0)
        log_ps = F.log_softmax(student(torch.cat([v2, v1])) / tau_s, -1)      # each view predicts the other's teacher
        loss = -(pt * log_ps).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for pt_, ps_ in zip(teacher.parameters(), student.parameters()):
                pt_.mul_(m).add_((1 - m) * ps_)
            H_img.append(-(pt * pt.clamp_min(1e-12).log()).sum(-1).mean().item())
            pm = pt.mean(0); H_mean.append(-(pm * pm.clamp_min(1e-12).log()).sum().item())
    return H_img, H_mean

configs = {"sharpening only (τ_t=0.04, no centering)": (False, 0.04),
           "centering only (τ_t=1.0)": (True, 1.0),
           "centering + sharpening (τ_t=0.04)": (True, 0.04)}
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for name, (cen, tt) in configs.items():
    H_img, H_mean = run_dino(cen, tt)
    axes[0].plot(H_img, label=name); axes[1].plot(H_mean, label=name)
    print(f"{name:45s} final: per-image entropy {H_img[-1]:.2f}, batch-mean entropy {H_mean[-1]:.2f}  (log K = {math.log(64):.2f})")
for a, t in zip(axes, ["per-image teacher entropy (high = uniform)", "entropy of batch-mean teacher output (low = one dimension)"]):
    a.axhline(math.log(64), ls="--", c="k", lw=0.8); a.set(title=t, xlabel="step", ylabel="nats"); a.legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

md(r"""
- **Sharpening only:** both entropies fall steadily, to 1.55 (per image) and 1.91 (batch mean) nats after 150 steps and still falling. The batch-mean distribution is concentrating on a few outputs shared by all images: collapse to (almost) one dimension.
- **Centering only:** both entropies sit at $\log K = 4.16$ from the start (the orange curve is under the dashed line): every image gets the uniform distribution, and the loss carries no information.
- **Both:** batch-mean entropy 4.12 ≈ $\log K$ (all outputs used), per-image entropy 3.66, below $\log K$. At this scale the teacher is only mildly sharp; DINO trains for hundreds of epochs and warms $\tau_t$ up from 0.04 to 0.07.
""")

md(r"""
## Summary
- **SimCLR** = cross-entropy over the $2N\times2N$ cosine-similarity matrix of two augmented views; the augmentations decide what the features ignore. 5000 images, 228 steps: probe 0.44 vs 0.30 for the same CNN at random init.
- **MAE** = mask 75% of the patches, encode only the visible ones, regress the pixels of the rest, loss on masked tokens only.
- **Linear probe** = the standard measurement of frozen features; the same probe ranked five extractors here.
- **Frozen DINOv2 + one linear layer**: pet IoU 0.88 from 300 labeled images; patch position is linearly decodable ($R^2$ ≈ 0.93).
- **Superposition**: sparse features share dimensions (5 in 2 at S = 0.95). A **sparse autoencoder** recovers an overcomplete sparse code (L0 = 18, FVE 0.73 vs 0.40 for 18 PCA components) whose units can be inspected one at a time.

**Further watching:** NYU Deep Learning, LeCun & Canziani, energy-based and self-supervised learning lectures, https://atcold.github.io/NYU-DLSP21/ ; UvA DL notebooks (Tutorial 17, SimCLR), https://uvadlc-notebooks.readthedocs.io/en/latest/ ; Meta, V-JEPA 2, https://ai.meta.com/vjepa/ ; Bricken et al., *Towards Monosemanticity*, https://transformer-circuits.pub/2023/monosemantic-features .
""")

code(r"""
print(f"total runtime: {(time.time() - T0) / 60:.1f} min")
""")

for k, p in B.write(STEM).items():
    print(k, p)
print(make_recap(STEM, [("slides/lectures/p5.pdf", [23, 37, 40, 43, 49, 69, 72, 59, 61, 64, 80, 83, 77, 86])]))
