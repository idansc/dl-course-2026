"""Builds T02_cnns.ipynb (Tutorial 2, Deep Learning 89-6877, Fall 2026)."""
from nbkit import Builder
from recap import make_recap

STEM = "T02_cnns"
B = Builder()
md, code = B.md, B.code

md(rf"""
# Tutorial 2: convolutional networks, from im2col to ResNet
**Deep Learning 89-6877, Fall 2026.** TA: Tal Fiskus.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idansc/dl-course-2026/blob/main/tutorials/{STEM}.ipynb)
Recap slides: [{STEM}_recap.pdf](https://github.com/idansc/dl-course-2026/blob/main/tutorials/recap/{STEM}_recap.pdf)

Plan for today (≈ 70 min):
1. Lecture recap: convolution, padding/stride, pooling, AlexNet → VGG → ResNet → ResNeXt (10 min)
2. Convolution as one matrix multiply (im2col) (10 min)
3. Pooling (5 min)
4. Residual, bottleneck and grouped-conv blocks; what they cost in parameters (10 min)
5. A small ResNet-18 on CIFAR-10 (10 min, training runs in the background)
6. Translation equivariance: check it, and find where it breaks (10 min)
7. What is the CNN looking at? Grad-CAM and integrated gradients (15 min)
8. If time: a ConvNeXt block

Runs on CPU (Colab or laptop); a GPU makes section 5 faster. Cells marked ✏️ are for you to try.
<<STUDENT>>
**Before the tutorial:** fill in every `# TODO` (replace the `...`). Each one is followed by a check cell that compares your result with PyTorch. The solutions are presented in the tutorial.
<</STUDENT>>
""")

code(r"""
import math, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
torch.manual_seed(0); random.seed(0); np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)
""")

md(r"""
## 1. Lecture recap

**Convolution layer.** Input $C_{in}\times H\times W$, $C_{out}$ filters of size $C_{in}\times K\times K$. Each output is a dot product of one filter with one $C_{in}\times K\times K$ patch, plus a bias:
$$y[o,i,j] = b_o + \sum_{c}\sum_{u,v} w[o,c,u,v]\;x[c,\, sI+u-P,\, sJ+v-P].$$
Output size $H' = \lfloor (H + 2P - K)/S \rfloor + 1$. "Same" padding: $P=(K-1)/2$. Parameters $C_{out}(C_{in}K^2+1)$; multiply-adds $C_{out}C_{in}K^2 H'W'$ (lecture example: 10 filters 5×5 on 3×32×32, pad 2 → 10×32×32, 760 params, 768K MACs).

**Receptive field.** $L$ stacked $K\times K$ convs see $1+L(K-1)$ pixels; downsampling (stride, pooling) grows it faster.

**Pooling.** Max or average over $K\times K$ windows with stride $S$, per channel, no parameters. Max pooling gives some invariance to small shifts.

**Translation equivariance.** The same filter is applied at every position, so shifting the input shifts the feature map: $f(T_s x) = T_s f(x)$. This is the inductive bias that makes CNNs data-efficient on images.

**Architectures.**
- **AlexNet (2012)**: 5 conv + 3 FC; most memory and FLOPs in the convs, most parameters in the FC layers.
- **VGG (2014)**: only 3×3 convs (stride 1, pad 1) and 2×2 max pools; after each pool, double the channels, so every stage costs the same FLOPs.
- **ResNet (2015)**: deeper plain nets train *worse*. A residual block learns $F(x)$ and outputs $\mathrm{ReLU}(F(x)+x)$, so identity is easy to represent. Stages halve resolution and double channels. **Bottleneck** block: 1×1 (4C→C), 3×3 (C→C), 1×1 (C→4C), 17HWC² FLOPs vs 18HWC² for two 3×3 convs at C, while working on 4× more channels.
- **ResNeXt (2017)**: split the bottleneck's 3×3 into $G$ parallel groups (**grouped conv**, weight $C_{out}\times C_{in}/G\times K\times K$). At equal FLOPs, more groups give lower error.
- **ConvNeXt (2022, not in the slides)**: a ResNet modernized with Transformer design choices (7×7 depthwise conv, LayerNorm, inverted bottleneck, GELU); matches Swin Transformers on ImageNet. See section 8.
""")

md(r"""
## 2. Convolution as one matrix multiply (im2col)

`F.unfold` cuts the (padded) input into all $K\times K$ patches and stacks each as a column: shape $N\times (C_{in}K^2)\times L$, with $L=H'W'$ positions. The convolution is then a single matmul with the flattened filters, $(C_{out}\times C_{in}K^2)\cdot(C_{in}K^2\times L)$. This is how cuDNN-era libraries (and Caffe) implemented it.

**Grouped conv** ($G$ groups): the $C_{in}$ input channels and the $C_{out}$ filters are split into $G$ groups; group $g$'s filters only see group $g$'s channels. In im2col form, reshape the columns to $N\times G\times (C_{in}/G\cdot K^2)\times L$ and do $G$ independent matmuls (one batched matmul). Unfold orders rows channel-major, so the split on channels is a plain reshape.
""")

code(r"""
def conv2d(x, w, b=None, stride=1, padding=0, groups=1):
    # x: N x Cin x H x W,  w: Cout x Cin/groups x K x K
    N, Cin, H, W = x.shape
    Cout, Cg, K, _ = w.shape
    Ho = (H + 2 * padding - K) // stride + 1
    Wo = (W + 2 * padding - K) // stride + 1
    cols = F.unfold(x, K, padding=padding, stride=stride)          # N x (Cin*K*K) x (Ho*Wo)
    #>> reshape cols to N x G x (Cg*K*K) x L and w to G x Cout/G x (Cg*K*K); one matmul; reshape to N x Cout x Ho x Wo
    cols = cols.view(N, groups, Cg * K * K, Ho * Wo)
    w_mat = w.view(groups, Cout // groups, Cg * K * K)
    out = (w_mat.unsqueeze(0) @ cols).view(N, Cout, Ho, Wo)        # (1,G,Co/G,CgKK) @ (N,G,CgKK,L)
    #<<
    if b is not None:
        out = out + b.view(1, -1, 1, 1)
    return out
""")

md(r"""
**Check:** forward and all three gradients against `F.conv2d`, for several kernel/stride/padding/group settings (float64, so the differences should be ~1e-15).
""")

code(r"""
for K, S, P, G in [(3, 1, 0, 1), (3, 2, 1, 1), (5, 1, 2, 1), (1, 1, 0, 1), (3, 2, 1, 4)]:
    x = torch.randn(2, 8, 11, 11, dtype=torch.float64, requires_grad=True)
    w = torch.randn(12, 8 // G, K, K, dtype=torch.float64, requires_grad=True)
    b = torch.randn(12, dtype=torch.float64, requires_grad=True)
    mine = conv2d(x, w, b, stride=S, padding=P, groups=G)
    ref = F.conv2d(x, w, b, stride=S, padding=P, groups=G)
    g = torch.randn_like(ref)
    gm = torch.autograd.grad(mine, (x, w, b), g)
    gr = torch.autograd.grad(ref, (x, w, b), g)
    errs = [(mine - ref).abs().max().item()] + [(a - c).abs().max().item() for a, c in zip(gm, gr)]
    print(f"K={K} S={S} P={P} G={G}: out {tuple(ref.shape)}  max err  y {errs[0]:.1e}  dx {errs[1]:.1e}  dw {errs[2]:.1e}  db {errs[3]:.1e}")
""")

md(r"""
The gradients come for free: `unfold`, `view` and `@` are all differentiable, so autograd backpropagates through im2col. The backward of `unfold` is `fold`, which *sums* the gradient of every patch back into the pixels it came from (a pixel used by 9 patches receives 9 contributions, the "+=" rule from Tutorial 1).

✏️ How much memory does im2col use? For a 3×3 conv on a 64×56×56 feature map, compare `cols.numel()` with `x.numel()`. Why do modern libraries (Winograd, implicit GEMM) avoid materializing `cols`?
""")

md(r"""
**Past exam question (Moed B, 2026), backpropagation**

A 1-D ConvNet without nonlinearities. Figure, in words: five inputs $x_1,\dots,x_5$; a conv layer with a single filter $w=(w_1,w_2,w_3)$, no bias (stride 1, no padding), gives three outputs
$$z_1 = w_1x_1+w_2x_2+w_3x_3,\quad z_2 = w_1x_2+w_2x_3+w_3x_4,\quad z_3 = w_1x_3+w_2x_4+w_3x_5;$$
two average nodes $v_1 = \frac{z_1+z_2}{2}$, $v_2=\frac{z_2+z_3}{2}$; a fully connected output $\hat y = a_1v_1 + a_2v_2$. The loss is
$$L = \tfrac12(y-\hat y)^2 + \tfrac{\lambda}{2}\big(a_1^2+a_2^2+w_1^2+w_2^2+w_3^2\big),\qquad \lambda>0.$$
**(a)** Using backpropagation, give algebraic expressions for $\frac{\partial L}{\partial \hat y}$, $\frac{\partial L}{\partial a_1}$, $\frac{\partial L}{\partial a_2}$, $\frac{\partial L}{\partial z_2}$, $\frac{\partial L}{\partial w_2}$.
**(b)** At some point during training: $y=1,\ \hat y=3,\ a_1=a_2=1,\ x_2=1,\ x_3=0,\ x_4=1$, $\lambda=0.5$ and $w_2=-2$. To decrease the loss, should $w_2$ move toward $-3$ or toward $-1$? Explain.

<<STUDENT>>
✏️ Your answer:
<</STUDENT>>
<<SOLUTION>>
**(a)** Let $\delta = \frac{\partial L}{\partial \hat y} = \hat y - y$. Then $\frac{\partial L}{\partial a_i} = \delta v_i + \lambda a_i$.
$z_2$ feeds both averages (two paths, gradients add): $\frac{\partial L}{\partial z_2} = \delta\big(\frac{a_1}{2}+\frac{a_2}{2}\big)$; likewise $\frac{\partial L}{\partial z_1}=\delta\frac{a_1}{2}$, $\frac{\partial L}{\partial z_3}=\delta\frac{a_2}{2}$.
$w_2$ is shared by all three outputs ($\partial z_1/\partial w_2 = x_2$, $\partial z_2/\partial w_2 = x_3$, $\partial z_3/\partial w_2 = x_4$), so
$$\frac{\partial L}{\partial w_2} = \delta\Big(\frac{a_1}{2}x_2 + \frac{a_1+a_2}{2}x_3 + \frac{a_2}{2}x_4\Big) + \lambda w_2 .$$
**(b)** $\delta = 2$, $\frac{\partial L}{\partial w_2} = 2\big(\tfrac12 + 0 + \tfrac12\big) + 0.5\cdot(-2) = 1 > 0$. GD moves against the gradient, so $w_2$ decreases: **toward $-3$**.
The exam gives only the values the answer needs. The check picks $x_1 = x_5 = 3$, $w_1=w_3=1$, which make $\hat y = 3$, and compares with autograd through `F.conv1d`.
<</SOLUTION>>
""")

code(r"""
x = torch.tensor([[[3., 1., 0., 1., 3.]]], dtype=torch.float64)           # (N, C, L): x1..x5
w = torch.tensor([[[1., -2., 1.]]], dtype=torch.float64, requires_grad=True)
a = torch.tensor([1., 1.], dtype=torch.float64, requires_grad=True)
y, lam = 1.0, 0.5
z = F.conv1d(x, w)[0, 0]                                                 # (z1, z2, z3); conv1d is a cross-correlation, as in the exam
v = F.avg_pool1d(z.view(1, 1, 3), kernel_size=2, stride=1)[0, 0]        # (v1, v2)
y_hat = a @ v
L = 0.5 * (y - y_hat) ** 2 + lam / 2 * (a.pow(2).sum() + w.pow(2).sum())
L.backward()
x1, x2, x3, x4, x5 = x.flatten().tolist(); a1, a2 = a.tolist(); w2 = w[0, 0, 1].item()
#>> dL/dw2 from your formula in (a)
delta = y_hat.item() - y
dw2 = delta * (a1 / 2 * x2 + (a1 + a2) / 2 * x3 + a2 / 2 * x4) + lam * w2
#<<
print(f"y_hat = {y_hat.item():.1f}   dL/dw2: formula {dw2:.4f}, autograd {w.grad[0, 0, 1].item():.4f}  → move w2 toward {'-3' if dw2 > 0 else '-1'}")
""")

md(r"""
## 3. Pooling

Pooling is the same patch extraction followed by a reduction instead of a dot product: unfold each channel separately, then take the max or the mean over the $K^2$ entries of each patch.
""")

code(r"""
def pool2d(x, K, stride=None, mode="max"):
    stride = stride or K
    N, C, H, W = x.shape
    Ho, Wo = (H - K) // stride + 1, (W - K) // stride + 1
    cols = F.unfold(x, K, stride=stride).view(N, C, K * K, Ho * Wo)
    #>> reduce over the K*K patch entries (max or mean), reshape to N x C x Ho x Wo
    red = cols.max(dim=2).values if mode == "max" else cols.mean(dim=2)
    return red.view(N, C, Ho, Wo)
    #<<
""")

code(r"""
x = torch.randn(2, 3, 8, 8, dtype=torch.float64, requires_grad=True)
for mode, ref_fn in [("max", F.max_pool2d), ("avg", F.avg_pool2d)]:
    for K, S in [(2, 2), (3, 2)]:
        mine, ref = pool2d(x, K, S, mode), ref_fn(x, K, S)
        gm, = torch.autograd.grad(mine.sum(), x); gr, = torch.autograd.grad(ref.sum(), x)
        print(f"{mode} K={K} S={S}: max err  y {(mine - ref).abs().max().item():.1e}  dx {(gm - gr).abs().max().item():.1e}")

# the lecture's max-pooling example
x_lec = torch.tensor([[1., 1, 2, 4], [5, 6, 7, 8], [3, 2, 1, 0], [1, 2, 3, 4]]).view(1, 1, 4, 4)
print("lecture example, 2x2 max pool stride 2:\n", pool2d(x_lec, 2).squeeze())
""")

md(r"""
Max pooling routes the whole gradient to the arg-max of each window (the "max = router" pattern from Tutorial 1); average pooling spreads it as $1/K^2$. With overlapping windows ($K=3$, $S=2$) a pixel can win in two windows and receive two gradients; the check above covers that case.
""")

md(r"""
**Past exam question (Moed C, 2026)**

True or false: a max-pooling layer with a $2\times2$ window and stride 2 contributes no learned parameters, but it does pass gradients backward during the backward pass.

<<STUDENT>>
✏️ Your answer:
<</STUDENT>>
<<SOLUTION>>
**True.** It has nothing to learn (0 parameters), and its local gradient is 1 for the arg-max of each window and 0 elsewhere, so the upstream gradient is routed to the winners. Check on the lecture example: one 1 per window.
<</SOLUTION>>
""")

code(r"""
#>> count the parameters of nn.MaxPool2d(2, 2), then backprop the sum of its output on x_lec and print the input gradient
pool = nn.MaxPool2d(2, 2)
print("parameters:", sum(p.numel() for p in pool.parameters()))
xg = x_lec.clone().requires_grad_()
pool(xg).sum().backward()
print("dL/dx (1 at each window's max):\n", xg.grad.squeeze())
#<<
""")

md(r"""
## 4. Residual blocks, bottlenecks, and grouped convs

Below, three blocks written with `nn.Conv2d`/`nn.BatchNorm2d` and the same attribute names as `torchvision.models.resnet`, so that we can load torchvision's weights into ours and compare outputs.

- **Basic block** (ResNet-18/34): $\;y = \mathrm{ReLU}\big(\mathrm{BN}(\mathrm{conv}_{3\times3}(\mathrm{ReLU}(\mathrm{BN}(\mathrm{conv}_{3\times3}(x))))) + \mathrm{sc}(x)\big)$.
- The shortcut $\mathrm{sc}$ is the identity, except when the block changes resolution or width: then a 1×1 conv with the same stride, followed by BN (`downsample`).
- **Bottleneck** (ResNet-50+): 1×1 reduce → 3×3 → 1×1 expand (×4). **ResNeXt** = bottleneck whose 3×3 is a grouped conv with $G$ groups of width $d$ ("32×4d": 32 groups of 4 channels per 64 base channels).
""")

code(r"""
def shortcut(cin, cout, stride):
    if stride == 1 and cin == cout:
        return None
    return nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False);     self.bn2 = nn.BatchNorm2d(cout)
        self.downsample = shortcut(cin, cout, stride)

    def forward(self, x):
        #>> two conv-BN (ReLU after the first), add the shortcut (identity or self.downsample), final ReLU
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)
        #<<

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, cin, planes, stride=1, groups=1, base_width=64):
        super().__init__()
        width = int(planes * base_width / 64) * groups      # ResNet: width = planes; ResNeXt-32x4d: 4*32 = 2*planes
        cout = planes * self.expansion
        self.conv1 = nn.Conv2d(cin, width, 1, bias=False);                                 self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride, 1, groups=groups, bias=False);     self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, cout, 1, bias=False);                                self.bn3 = nn.BatchNorm2d(cout)
        self.downsample = shortcut(cin, cout, stride)

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        #>> 1x1 → BN → ReLU → 3x3 (grouped) → BN → ReLU → 1x1 → BN, then add identity and ReLU
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return F.relu(out + identity)
        #<<
""")

md(r"""
**Check:** copy torchvision's block weights into ours (randomizing the BN statistics first, so BN is not the identity) and compare outputs.
""")

code(r"""
from torchvision.models.resnet import BasicBlock as TVBasic, Bottleneck as TVBottle

def randomize_bn(m):
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.running_mean.normal_(0, 0.5); mod.running_var.uniform_(0.5, 2)
            mod.weight.data.uniform_(0.5, 1.5); mod.bias.data.normal_(0, 0.5)
    return m

torch.manual_seed(0)
cases = [
    ("basic, 64→64, stride 1",       BasicBlock(64, 64),                 TVBasic(64, 64)),
    ("basic, 64→128, stride 2",      BasicBlock(64, 128, 2),             TVBasic(64, 128, 2, downsample=shortcut(64, 128, 2))),
    ("bottleneck, 256→256",          Bottleneck(256, 64),                TVBottle(256, 64)),
    ("ResNeXt 32x4d, 256→512, s2",   Bottleneck(256, 128, 2, 32, 4),     TVBottle(256, 128, 2, downsample=shortcut(256, 512, 2), groups=32, base_width=4)),
]
for name, mine, ref in cases:
    randomize_bn(ref).eval(); mine.load_state_dict(ref.state_dict()); mine.eval()
    x = torch.randn(2, mine.conv1.in_channels, 16, 16)
    with torch.no_grad():
        print(f"{name:30s} out {tuple(ref(x).shape)}  max err {(mine(x) - ref(x)).abs().max().item():.1e}")
""")

md(r"""
**Parameter counts.** A conv with $G$ groups has $C_{out}\cdot\frac{C_{in}}{G}\cdot K^2$ weights (plus $C_{out}$ biases if used). At stride 1 its multiply-adds are that number times $H'W'$, so for these blocks the parameter ratio is also the FLOP ratio.
""")

code(r"""
def conv_params(cin, cout, k, groups=1, bias=False):
    return cout * (cin // groups) * k * k + (cout if bias else 0)  # @student: return ...  # TODO: number of parameters of an nn.Conv2d

for args in [(3, 10, 5, 1, True), (64, 64, 3, 1, False), (128, 128, 3, 32, False), (96, 96, 7, 96, True)]:
    ref = sum(p.numel() for p in nn.Conv2d(*args[:3], groups=args[3], bias=args[4]).parameters())
    print(f"Conv2d(cin={args[0]}, cout={args[1]}, k={args[2]}, groups={args[3]}, bias={args[4]}): yours {conv_params(*args)}, torch {ref}")
""")

md(r"""
**Past exam question (Moed C, 2026)**

A CNN classifies $64\times64\times3$ images. Layers, from input to output:
- CONV1: 16 filters, kernel $3\times3$, stride 1, padding 1, with bias.
- POOL1: max-pooling $2\times2$, stride 2, no padding.
- CONV2: 32 filters, kernel $3\times3$, stride 1, padding 1, with bias.
- POOL2: max-pooling $2\times2$, stride 2, no padding.
- FC-10: fully connected, 10 outputs, on the flattened output of POOL2, with bias.

Compute the **total number of learned parameters**. Show the computation for each layer separately, stating the number of filters, the depth of each feature map, and the bias.

<<STUDENT>>
✏️ Your answer:
<</STUDENT>>
<<SOLUTION>>
- CONV1: each filter is $3\times3\times3$ (input depth 3) $+1$ bias $=28$; $16\cdot 28 = 448$. Output $64\times64\times16$ (padding 1 keeps the size).
- POOL1: 0. Output $32\times32\times16$.
- CONV2: each filter is $3\times3\times16$ (depth 16) $+1 = 145$; $32\cdot145 = 4{,}640$. Output $32\times32\times32$.
- POOL2: 0. Output $16\times16\times32 = 8{,}192$ values.
- FC-10: $8192\cdot10+10 = 81{,}930$.

**Total $= 448 + 4{,}640 + 81{,}930 = 87{,}018$**, 94% of it in the FC layer.
<</SOLUTION>>
""")

code(r"""
#>> per-layer parameter counts with conv_params (and the flattened size after POOL2 for the FC layer)
p_conv1 = conv_params(3, 16, 3, bias=True)
p_conv2 = conv_params(16, 32, 3, bias=True)
p_fc = (16 * 16 * 32) * 10 + 10
#<<
net = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1), nn.MaxPool2d(2, 2), nn.Conv2d(16, 32, 3, 1, 1), nn.MaxPool2d(2, 2),
                    nn.Flatten(), nn.Linear(16 * 16 * 32, 10))
print("yours:", p_conv1, p_conv2, p_fc, "total", p_conv1 + p_conv2 + p_fc)
print("torch:", [sum(p.numel() for p in m.parameters()) for m in net if any(True for _ in m.parameters())],
      "total", sum(p.numel() for p in net.parameters()))
h = torch.zeros(1, 3, 64, 64)
for m in net:
    h = m(h); print(f"  {type(m).__name__:10s} → {tuple(h.shape[1:])}")
""")

code(r"""
def conv_only(m):   # parameters of the conv layers only (ignore BN)
    return sum(p.numel() for mod in m.modules() if isinstance(mod, nn.Conv2d) for p in mod.parameters())

C, HW = 256, 56 * 56
rows = [
    ("plain: two 3x3 convs at 256 ch", 2 * conv_params(C, C, 3),                                        None),
    ("basic block at 64 ch",           2 * conv_params(64, 64, 3),                                      BasicBlock(64, 64)),
    ("bottleneck 256-64-256",          conv_params(C, 64, 1) + conv_params(64, 64, 3) + conv_params(64, C, 1), Bottleneck(C, 64)),
    ("ResNeXt 256-(32x4d)-256",        conv_params(C, 128, 1) + conv_params(128, 128, 3, 32) + conv_params(128, C, 1), Bottleneck(C, 64, groups=32, base_width=4)),
]
print(f"{'block':34s} {'params':>9s} {'(module)':>9s} {'GMACs @56x56':>13s}")
for name, p, mod in rows:
    print(f"{name:34s} {p:9,d} {conv_only(mod) if mod else '':>9} {p * HW / 1e9:13.3f}")
""")

md(r"""
- At the 256-channel interface, a bottleneck costs **17× fewer** parameters/FLOPs than two plain 3×3 convs at 256 channels (69.6K vs 1.18M), and about the same as a basic block at 64 channels (69.6K vs 73.7K, the lecture's 17HWC² vs 18HWC²). That is why ResNet-50 is not much more expensive than ResNet-34 while being much deeper and wider at the block boundaries.
- ResNeXt-32x4d **doubles the inner width** (64 → 128 channels) at almost the same cost (70.1K): the 3×3 conv with 32 groups has $128\cdot 4 \cdot 9 = 4{,}608$ weights instead of $64\cdot 64\cdot 9 = 36{,}864$.

✏️ A depthwise conv is the extreme $G = C_{in} = C_{out}$. With `conv_params`, compare a 7×7 depthwise conv at 96 channels with a dense 3×3 conv at 96 channels.
""")

md(r"""
**Past exam question (Moed C, 2026)**

A $1\times1$ convolution (kernel $=1\times1$) is widely used in deep architectures, e.g. in the Inception block and in ResNet's bottleneck block. Assume an input feature map of size $H\times W\times C_{in}$ and a $1\times1$ conv layer with $C_{out}$ filters (no bias).

State the spatial size of the output and the number of learned parameters, and explain the main role of the $1\times1$ conv in the architecture: why is it used before an expensive $3\times3$ conv?

<<STUDENT>>
✏️ Your answer:
<</STUDENT>>
<<SOLUTION>>
(There is no official solution for this question; this is ours.) Output $H\times W\times C_{out}$ (stride 1; a $1\times1$ kernel needs no padding). Parameters $C_{in}\cdot C_{out}$. It is the same linear map $\mathbb{R}^{C_{in}}\to\mathbb{R}^{C_{out}}$ applied at every pixel: it mixes channels and does not look at neighbours. Placed before a $3\times3$ conv it **reduces the channel count**, so the $3\times3$ conv, whose cost is $9\,C_{in}C_{out}HW$, runs on few channels. At the 256-channel interface of the bottleneck above: a $3\times3$ conv $256\to256$ has 589,824 weights, while $1\times1$ ($256\to64$) followed by $3\times3$ ($64\to64$) has 16,384 + 36,864 = 53,248, 11× fewer, and the extra ReLU after the $1\times1$ adds a nonlinearity.
<</SOLUTION>>
""")

code(r"""
H_, W_, C_in, C_out = 56, 56, 256, 64
#>> parameter count of a 1x1 conv C_in → C_out without bias, and of 3x3 at 256 vs (1x1 256→64, then 3x3 64→64)
p_1x1 = C_in * C_out
p_direct, p_reduced = 9 * 256 * 256, 256 * 64 + 9 * 64 * 64
#<<
conv1x1 = nn.Conv2d(C_in, C_out, 1, bias=False)
print("1x1 conv: yours", p_1x1, "| torch", conv1x1.weight.numel(), "| output shape", tuple(conv1x1(torch.zeros(1, C_in, H_, W_)).shape))
print(f"3x3 at 256 ch: {p_direct:,}   1x1 reduce + 3x3 at 64 ch: {p_reduced:,}   ratio {p_direct / p_reduced:.1f}x")
""")

md(r"""
## 5. A small ResNet-18 on CIFAR-10

ResNet-18 = stem + 4 stages × 2 basic blocks; stage $i$ has $w\cdot 2^{i}$ channels and halves the resolution (except stage 1), then global average pooling and one linear layer. For 32×32 inputs the ImageNet stem (7×7 stride 2 + max pool) throws away too much resolution, so CIFAR versions use one 3×3 stride-1 conv. We use base width $w=16$ instead of 64 (16× fewer FLOPs) to fit a CPU.
""")

code(r"""
def make_stage(cin, cout, n_blocks, stride):
    #>> nn.Sequential of n_blocks BasicBlocks; only the first changes channels/stride
    blocks = [BasicBlock(cin, cout, stride)] + [BasicBlock(cout, cout) for _ in range(n_blocks - 1)]
    return nn.Sequential(*blocks)
    #<<

class ResNet18(nn.Module):
    def __init__(self, width=16, n_classes=10):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv2d(3, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU())
        self.layer1 = make_stage(w, w, 2, 1)
        self.layer2 = make_stage(w, 2 * w, 2, 2)
        self.layer3 = make_stage(2 * w, 4 * w, 2, 2)
        self.layer4 = make_stage(4 * w, 8 * w, 2, 2)
        self.fc = nn.Linear(8 * w, n_classes)

    def features(self, x):          # last conv feature map, N x 8w x H/8 x W/8
        return self.layer4(self.layer3(self.layer2(self.layer1(self.stem(x)))))

    def forward(self, x):
        return self.fc(self.features(x).mean((2, 3)))   # global average pooling + linear
""")

md(r"""
**Check:** at width 64 our four stages are exactly torchvision's `resnet18` stages: load its weights stage by stage and compare.
""")

code(r"""
torch.manual_seed(0)
tv = randomize_bn(torchvision.models.resnet18()).eval()
mine = ResNet18(width=64).eval()
x = torch.randn(2, 64, 32, 32)
with torch.no_grad():
    for name in ["layer1", "layer2", "layer3", "layer4"]:
        getattr(mine, name).load_state_dict(getattr(tv, name).state_dict())
        ym, yt = getattr(mine, name)(x), getattr(tv, name)(x)
        print(f"{name}: out {tuple(yt.shape)}  max err {(ym - yt).abs().max().item():.1e}")
        x = yt
n_tv = sum(p.numel() for p in tv.parameters()); n_mine = sum(p.numel() for p in mine.parameters())
print(f"params: torchvision resnet18 (ImageNet stem, 1000 classes) {n_tv/1e6:.2f}M, ours at width 64 (CIFAR stem, 10 classes) {n_mine/1e6:.2f}M, ours at width 16 {sum(p.numel() for p in ResNet18(16).parameters())/1e6:.2f}M")
""")

md(r"""
**Data.** 10,000 CIFAR-10 training images, 2,000 test images, per-channel normalization. **Augmentation**: random crop with 4-pixel zero padding and random horizontal flip, applied to each batch on the fly (the standard CIFAR recipe). Training: SGD with Nesterov momentum 0.9, weight decay 5e-4, one-cycle LR schedule with peak 0.1, batch 128, 5 epochs.

**On a Colab GPU** (Runtime → T4): set `N_TRAIN = 50_000`, `WIDTH = 64`, `EPOCHS = 30`. That is the standard CIFAR ResNet-18 and reaches ≈ 93–95% test accuracy in roughly 20–30 minutes.
""")

code(r"""
N_TRAIN, N_TEST, WIDTH, EPOCHS, BATCH = 10_000, 2_000, 16, 5, 128
train_set = torchvision.datasets.CIFAR10("./data", train=True, download=True)
test_set = torchvision.datasets.CIFAR10("./data", train=False, download=True)
classes = train_set.classes

def to_tensor(ds, n):
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(ds), generator=g)[:n]
    x = torch.tensor(ds.data[idx.numpy()]).permute(0, 3, 1, 2).float() / 255
    return x, torch.tensor(ds.targets)[idx]

x_train, y_train = to_tensor(train_set, N_TRAIN)
x_test, y_test = to_tensor(test_set, N_TEST)
MEAN = x_train.mean((0, 2, 3), keepdim=True); STD = x_train.std((0, 2, 3), keepdim=True)
x_train, x_test = (x_train - MEAN) / STD, (x_test - MEAN) / STD

def augment(x):
    flip = torch.rand(x.shape[0], 1, 1, 1) < 0.5
    x = torch.where(flip, x.flip(3), x)
    xp = F.pad(x, (4, 4, 4, 4))
    i, j = torch.randint(0, 9, (2, x.shape[0]))
    return torch.stack([xp[k, :, i[k]:i[k] + 32, j[k]:j[k] + 32] for k in range(x.shape[0])])

print(x_train.shape, x_test.shape, "classes:", classes)
""")

code(r"""
@torch.no_grad()
def accuracy(model, x, y, bs=500):
    model.eval()
    pred = torch.cat([model(x[i:i + bs].to(device)).argmax(1).cpu() for i in range(0, len(x), bs)])
    return (pred == y).float().mean().item()

torch.manual_seed(0)
model = ResNet18(WIDTH).to(device)
steps_per_epoch = N_TRAIN // BATCH
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, total_steps=EPOCHS * steps_per_epoch)
hist = {"loss": [], "train_acc": [], "test_acc": []}
t0 = time.time()
for ep in range(EPOCHS):
    model.train()
    perm = torch.randperm(N_TRAIN)
    for s in range(steps_per_epoch):
        idx = perm[s * BATCH:(s + 1) * BATCH]
        xb, yb = augment(x_train[idx]).to(device), y_train[idx].to(device)
        loss = F.cross_entropy(model(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        hist["loss"].append(loss.item())
    hist["train_acc"].append(accuracy(model, x_train[:2000], y_train[:2000]))
    hist["test_acc"].append(accuracy(model, x_test, y_test))
    print(f"epoch {ep + 1}: loss {np.mean(hist['loss'][-steps_per_epoch:]):.3f}  train acc {hist['train_acc'][-1]:.3f}  "
          f"test acc {hist['test_acc'][-1]:.3f}  ({time.time() - t0:.0f}s)")
_ = model.eval()
""")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot(hist["loss"], lw=0.8); ax[0].set(title="training loss (augmented batches)", xlabel="step", ylabel="cross-entropy")
ep = range(1, EPOCHS + 1)
ax[1].plot(ep, hist["train_acc"], marker="o", label="train (2k, no aug)")
ax[1].plot(ep, hist["test_acc"], marker="o", label="test (2k)")
ax[1].set(title=f"ResNet-18, width {WIDTH}, {N_TRAIN // 1000}k images", xlabel="epoch", ylabel="accuracy"); ax[1].legend()
plt.tight_layout(); plt.show()
""")

md(r"""
After 5 epochs on 10k images the width-16 ResNet-18 (0.70M parameters) reaches **62.8%** test accuracy (66.1% on training images), and both curves are still rising: this is an underfitted run limited by compute, not by data. On an Apple-silicon GPU (mps) the training took about a minute; on a 2-thread laptop CPU expect about 5–6 minutes, and numbers that differ by about one point (61.9% in our CPU run), since CPU and GPU kernels round differently.

✏️ Train once more with `augment` replaced by the identity. With only 10k images, which gap grows: train accuracy minus test accuracy, or the test accuracy itself?
""")

md(r"""
## 6. Translation equivariance: check it, and find where it breaks

Let $T_s$ shift an image by $s$ pixels. A layer $f$ is **equivariant** if $f(T_s x) = T_{s'} f(x)$ for some output shift $s'$, and **invariant** if $f(T_s x) = f(x)$. A convolution with stride 1 is equivariant with $s' = s$, *if* the shift does not move content across the border. We measure
$$\mathrm{err}(f, s, s') = \max\,|f(T_s x) - T_{s'} f(x)|$$
with `torch.roll` as $T$ (a circular shift, so nothing is lost at the border by the shift itself).
""")

code(r"""
def shift(x, s):                     # circular shift by s pixels right and s pixels down
    return torch.roll(x, shifts=(s, s), dims=(2, 3))

def equiv_error(f, x, s, s_out):
    #>> max |f(T_s x) − T_{s_out} f(x)|
    return (f(shift(x, s)) - shift(f(x), s_out)).abs().max().item()
    #<<

torch.manual_seed(0)
img = x_test[:1]                                       # one real CIFAR image, 3x32x32
w3 = torch.randn(8, 3, 3, 3)
conv_circ = lambda x: F.conv2d(F.pad(x, (1, 1, 1, 1), mode="circular"), w3)
conv_zero = lambda x: F.conv2d(x, w3, padding=1)
conv_s2   = lambda x: F.conv2d(F.pad(x, (1, 1, 1, 1), mode="circular"), w3, stride=2)
pool2     = lambda x: F.max_pool2d(x, 2)

tests = [("conv 3x3, circular padding", conv_circ, 3, 3),
         ("conv 3x3, zero padding",     conv_zero, 3, 3),
         ("conv 3x3 stride 2, shift 2", conv_s2,   2, 1),
         ("conv 3x3 stride 2, shift 1", conv_s2,   1, None),
         ("max pool 2x2, shift 2",      pool2,     2, 1),
         ("max pool 2x2, shift 1",      pool2,     1, None)]
for name, f, s, s_out in tests:
    if s_out is None:   # no integer output shift exists; report the best of 0 and 1
        e = min(equiv_error(f, img, s, 0), equiv_error(f, img, s, 1))
    else:
        e = equiv_error(f, img, s, s_out)
    print(f"{name:30s} input shift {s}: error {e:.2e}   (max |output| {f(img).abs().max().item():.1f})")
""")

md(r"""
- **Circular padding**: exact (float error only). Weight sharing *is* translation equivariance.
- **Zero padding**: the error (14.0) is as large as the outputs themselves. Where is it? The map below shows it is confined to a frame: the output border (row/column 0 and 31), where the shifted input now meets the zero padding with different content, and rows/columns 2–3, where the old border pixels landed after the shift. Everywhere else the error is exactly 0. Padding tells the network where the border is, so CNNs can and do encode absolute position.
- **Stride 2 / pooling 2**: a shift by 2 becomes an exact shift by 1 (error 0). A shift by 1 has no counterpart on the coarser output grid (aliasing): the error is 9.0 for the strided conv (outputs up to 10.4) and 1.5 for max pooling (outputs up to 2.0), spread over the whole map, not only the border.
""")

code(r"""
err_map = (conv_zero(shift(img, 3)) - shift(conv_zero(img), 3)).abs().amax(1)[0]
err_s2 = (conv_s2(shift(img, 1)) - conv_s2(img)).abs().amax(1)[0]
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
ax[0].imshow((img[0] * STD[0] + MEAN[0]).permute(1, 2, 0).clamp(0, 1)); ax[0].set_title(f"input ({classes[y_test[0]]})")
im = ax[1].imshow(err_map, cmap="magma"); ax[1].set_title("zero padding, shift 3:\n|f(Tx) − T f(x)|"); plt.colorbar(im, ax=ax[1])
im = ax[2].imshow(err_s2, cmap="magma"); ax[2].set_title("stride 2, shift 1:\n|f(Tx) − f(x)|"); plt.colorbar(im, ax=ax[2])
for a in ax: a.set(xlabel="x (pixels)", ylabel="y (pixels)")
plt.tight_layout(); plt.show()
""")

md(r"""
**The whole trained network.** ResNet-18 ends with global average pooling, so with stride 1 everywhere it would be exactly shift-*invariant* under circular shifts. It has three stride-2 stages (total stride 8), zero padding, and was trained with crops of up to 4 pixels. How often does its prediction change when we shift test images by 1–8 pixels? We use a real translation here (shifted with zero fill, i.e. the mean colour), which is also what random crops do.
""")

code(r"""
def translate(x, s):                 # shift right by s pixels, fill with zeros (= mean colour after normalization)
    return F.pad(x, (s, 0, 0, 0))[..., :x.shape[-1]] if s > 0 else x

@torch.no_grad()
def predict(x, bs=500):
    return torch.cat([model(x[i:i + bs].to(device)).argmax(1).cpu() for i in range(0, len(x), bs)])

base = predict(x_test)
shifts = list(range(0, 9))
changed = [(predict(translate(x_test, s)) != base).float().mean().item() for s in shifts]
changed_roll = [(predict(torch.roll(x_test, s, dims=3)) != base).float().mean().item() for s in shifts]
acc_s = [(predict(translate(x_test, s)) == y_test).float().mean().item() for s in shifts]
for s, c, cr, a in zip(shifts, changed, changed_roll, acc_s):
    print(f"shift {s}: prediction changed {100*c:5.1f}% (zero fill)  {100*cr:5.1f}% (circular)   accuracy {100*a:.1f}%")

plt.figure(figsize=(5.5, 3.5))
plt.plot(shifts, [100 * c for c in changed], marker="o", label="translate, zero fill")
plt.plot(shifts, [100 * c for c in changed_roll], marker="s", label="circular shift (roll)")
plt.axvline(4, color="gray", ls=":", label="max shift seen in training crops")
plt.xlabel("horizontal shift (pixels)"); plt.ylabel("% test predictions changed")
plt.title("Is the trained ResNet shift-invariant?"); plt.legend(); plt.tight_layout(); plt.show()
""")

md(r"""
- A **1-pixel** shift already changes **9.3%** of the predictions, although the network saw random shifts of up to 4 pixels during training. Global average pooling is invariant, but the three stride-2 stages in front of it alias, as in the single-layer test above.
- The fraction grows with the shift (13–14% at 3–4 pixels, 30% at 8) and accuracy drops from 62.8% to 53.9% at 8 pixels.
- Up to 5 pixels, zero fill and circular shift give the same numbers (within 1%), so the border content is not the main cause; beyond the 4-pixel training range zero fill is worse (30.2% vs 24.3% at 8).
- Equivariance is a property of one stride-1 conv layer. A full CNN is only approximately shift-invariant, and how much is learned from augmentation (Azulay & Weiss, 2019; Zhang, "Making convolutional networks shift-invariant again", 2019, adds blur before every downsampling).
""")

md(r"""
## 7. What is the CNN looking at?

Two ways to ask which pixels a prediction depends on, both computed from gradients of the class score $y^c$ (the logit, before softmax).

**Grad-CAM** (Selvaraju et al., 2017). Take the last conv feature map $A\in\mathbb{R}^{K\times h\times w}$ (here $128\times4\times4$). Weight each channel by its average gradient, sum, and keep the positive part:
$$\alpha_k^c = \frac{1}{hw}\sum_{i,j}\frac{\partial y^c}{\partial A_{k,ij}},\qquad L^c = \mathrm{ReLU}\Big(\sum_k \alpha_k^c A_k\Big),$$
then upsample $L^c$ to the image size. Cheap (one backward), coarse (the resolution of $A$).

**Integrated gradients** (Sundararajan et al., 2017). The plain gradient $\partial y^c/\partial x$ is local and saturates. IG integrates it along the straight path from a baseline $x'$ (here a black image) to $x$:
$$\mathrm{IG}_i(x) = (x_i - x'_i)\int_0^1 \frac{\partial y^c\big(x' + \alpha(x-x')\big)}{\partial x_i}\,d\alpha \;\approx\; (x_i - x'_i)\,\frac{1}{m}\sum_{k=1}^{m}\frac{\partial y^c}{\partial x_i}\Big|_{x'+\frac{k-1/2}{m}(x-x')} .$$
**Completeness** (fundamental theorem of calculus for path integrals): $\sum_i \mathrm{IG}_i(x) = y^c(x) - y^c(x')$. The Riemann sum satisfies it only approximately, so the gap tells us if $m$ is large enough.
""")

code(r"""
def grad_cam(model, x, target):
    # x: N x 3 x H x W, target: N class indices. Returns N x H x W maps.
    A = model.features(x)                                 # N x K x h x w
    scores = model.fc(A.mean((2, 3))).gather(1, target[:, None]).sum()
    #>> alpha = spatial mean of dScore/dA; cam = ReLU(sum_k alpha_k A_k); bilinear upsample to x's size
    grads, = torch.autograd.grad(scores, A)
    alpha = grads.mean((2, 3), keepdim=True)
    cam = F.relu((alpha * A).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    #<<
    return cam[:, 0].detach()
""")

md(r"""
**Check.** Our network is conv features → global average pooling → linear, the setting of the original CAM (Zhou et al., 2016), where the map is computed directly from the classifier weights: $\mathrm{CAM}^c = \mathrm{ReLU}(\sum_k W_{ck} A_k)$. Here $\partial y^c/\partial A_{k,ij} = W_{ck}/(hw)$, so Grad-CAM must equal CAM$/(hw)$ exactly.
""")

code(r"""
xb, yb = x_test[:8].to(device), y_test[:8].to(device)
gc = grad_cam(model, xb, yb)
with torch.no_grad():
    A = model.features(xb)
    cam = F.relu(torch.einsum("nk,nkij->nij", model.fc.weight[yb], A)) / (A.shape[2] * A.shape[3])
    cam = F.interpolate(cam[:, None], size=(32, 32), mode="bilinear", align_corners=False)[:, 0]
print(f"feature map {tuple(A.shape[1:])};  max |Grad-CAM − CAM/hw| = {(gc - cam).abs().max().item():.1e}  (map max {gc.max().item():.2f})")
""")

code(r"""
def integrated_gradients(model, x, target, baseline, steps=64):
    # x, baseline: 1 x 3 x H x W; returns 1 x 3 x H x W attributions
    alphas = (torch.arange(steps, device=x.device, dtype=x.dtype) + 0.5) / steps
    #>> build the path points baseline + alpha (x − baseline), gradient of the target logit at each, average, times (x − baseline)
    path = (baseline + alphas.view(-1, 1, 1, 1) * (x - baseline)).requires_grad_()
    score = model(path)[:, target].sum()
    grads, = torch.autograd.grad(score, path)
    return (x - baseline) * grads.mean(0, keepdim=True)
    #<<
""")

code(r"""
black = ((torch.zeros(1, 3, 1, 1) - MEAN) / STD).expand(1, 3, 32, 32).to(device)  # a black image, normalized
x0, c0 = x_test[:1].to(device), y_test[0].item()
with torch.no_grad():
    gap = (model(x0)[0, c0] - model(black)[0, c0]).item()
print(f"f(x) − f(baseline) = {gap:.4f}")
for m in [4, 16, 64, 256]:
    ig = integrated_gradients(model, x0, c0, black, steps=m)
    print(f"steps {m:3d}: sum of IG = {ig.sum().item():.4f}   completeness gap {abs(ig.sum().item() - gap):.4f}  ({100 * abs(ig.sum().item() - gap) / abs(gap):.2f}%)")
""")

md(r"""
The gap falls from 4.8% at 4 steps to 0.01% at 256, but not monotonically (0.25% at 16, 0.60% at 64): for a ReLU network the integrand is piecewise constant along the path, so the midpoint-rule error depends on where the steps fall relative to the kinks. Check completeness for every image you explain; the IG paper suggests increasing $m$ until the gap is within 5%. Below we use 128 steps.
""")

code(r"""
def show(x):                       # normalized tensor → HxWx3 image in [0,1]
    return (x.cpu() * STD[0] + MEAN[0]).permute(1, 2, 0).clamp(0, 1).numpy()

pred_all = predict(x_test)
picks = [i for i in range(len(y_test)) if pred_all[i] == y_test[i]][:6]       # first six correctly classified
fig, ax = plt.subplots(3, 6, figsize=(14, 7.2))
for col, i in enumerate(picks):
    x1, c1 = x_test[i:i + 1].to(device), y_test[i].item()
    cam1 = grad_cam(model, x1, torch.tensor([c1], device=device))[0].cpu()
    ig1 = integrated_gradients(model, x1, c1, black, steps=128)[0].abs().sum(0).cpu()
    ax[0, col].imshow(show(x1[0])); ax[0, col].set_title(classes[c1])
    ax[1, col].imshow(show(x1[0])); ax[1, col].imshow(cam1, cmap="jet", alpha=0.5)
    ax[2, col].imshow(ig1, cmap="gray_r", vmax=np.percentile(ig1.numpy(), 99))
    for r in range(3): ax[r, col].axis("off")
ax[0, 0].text(-6, 16, "image", rotation=90, va="center", ha="right")
ax[1, 0].text(-6, 16, "Grad-CAM", rotation=90, va="center", ha="right")
ax[2, 0].text(-6, 16, "|IG| (sum over RGB)", rotation=90, va="center", ha="right")
fig.suptitle("What is the CNN looking at? (true class, correctly predicted)"); plt.tight_layout(); plt.show()
""")

md(r"""
- **Grad-CAM** puts a single blob on the object in all six images (on the hull, not the sail, for the sailboat). The underlying map is 4×4, so it can say "lower centre" but not "the horse's legs". Note also that most CIFAR objects are centred: from these images alone we cannot tell an object detector from a centre prior.
- **IG** is pixel-level and noisy. It concentrates on edges and high-contrast parts of the object (the dog's head and body, the motorboat's deck), but a good share of the attribution also lands on the background. Both maps explain a 63%-accurate model trained for one minute, so they show what this model uses, not what a good classifier should use.

✏️ **Sanity check** (Adebayo et al., 2018, "Sanity checks for saliency maps"): re-initialize `model.layer4` with random weights (`copy.deepcopy` the model first) and recompute both maps. An explanation that barely changes is explaining the image, not the model.

✏️ IG depends on the baseline. Replace the black image with a blurred copy of `x0` (`torchvision.transforms.functional.gaussian_blur(x0, 11, 5.0)`). Does completeness still hold? Which pixels lose their attribution?
""")

md(r"""
## 8. If time: a ConvNeXt block

ConvNeXt (Liu et al., 2022) keeps the ResNet skeleton and changes the block: 7×7 **depthwise** conv ($G=C$) → LayerNorm over channels → 1×1 to $4C$ → GELU → 1×1 back to $C$ → add $x$. Spatial mixing (depthwise) and channel mixing (the 1×1 MLP) are separated, as in a Transformer block (attention, then MLP). Our `conv2d` from section 2 already handles the depthwise conv.
""")

code(r"""
class ConvNeXtBlock(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.dw = nn.Conv2d(C, C, 7, padding=3, groups=C)
        self.norm = nn.LayerNorm(C)
        self.pw1, self.pw2 = nn.Linear(C, 4 * C), nn.Linear(4 * C, C)   # 1x1 convs = Linear on the channel axis
    def forward(self, x):
        h = self.dw(x).permute(0, 2, 3, 1)                              # N H W C
        h = self.pw2(F.gelu(self.pw1(self.norm(h))))
        return x + h.permute(0, 3, 1, 2)

blk = ConvNeXtBlock(96)
x = torch.randn(2, 96, 14, 14)
print("depthwise via our im2col conv2d, max err:",
      (conv2d(x, blk.dw.weight, blk.dw.bias, padding=3, groups=96) - blk.dw(x)).abs().max().item())
print(f"ConvNeXt block at C=96: {sum(p.numel() for p in blk.parameters()):,} params "
      f"(depthwise 7x7: {conv_params(96, 96, 7, 96, True):,});  ResNet basic block at C=96: {sum(p.numel() for p in BasicBlock(96, 96).parameters()):,}")
""")

md(r"""
The 7×7 depthwise conv costs 4.8K parameters, fewer than one dense 1×1 conv at 96 channels (9.3K). Almost all of the block's parameters are in the channel MLP.
""")

md(r"""
## Summary
- A convolution is one matrix multiply on unfolded patches (im2col); grouped and depthwise convs are block-diagonal versions of the same matmul. Autograd through `unfold` gives exact gradients.
- Pooling = the same patches, reduced by max (gradient routed to the winner) or mean.
- ResNet's identity shortcut makes depth trainable; bottlenecks and grouped convs buy width and depth at fixed FLOPs (bottleneck 69.6K params vs 1.18M for two plain 3×3 convs at 256 channels; ResNeXt doubles the inner width for +0.7%).
- A ResNet-18 at 1/4 width reaches 62.8% on CIFAR-10 after 5 epochs on 10k images; the full-width model on all 50k images reaches ≈ 93–95% on a GPU.
- Convolution is exactly translation-equivariant with stride 1 and circular padding. Zero padding breaks it at the border; stride and pooling break it for shifts that are not multiples of the stride. The trained network changes 9% of its predictions under a 1-pixel shift.
- Grad-CAM (coarse, one backward, equals CAM for a GAP+linear head) and integrated gradients (pixel-level, satisfies completeness up to the Riemann-sum error) answer "where" differently; check explanations with completeness and randomization tests before trusting them.

**Further watching:** Stanford CS231n (2017), [lecture playlist](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv): Lecture 5 (Convolutional Neural Networks) and Lecture 9 (CNN Architectures). Course notes on convolution, im2col and architectures: [cs231n.stanford.edu](https://cs231n.stanford.edu).
""")

make_recap(STEM, [
    ("slides/lectures/p3.pdf", [56, 59, 71, 77, 86, 74, 78, 63]),
    ("slides/lectures/p4.pdf", [27, 34, 45, 48, 50, 64, 66]),
])
for k, p in B.write(STEM).items():
    print(k, p)
