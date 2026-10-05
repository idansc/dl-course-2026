"""Builds T01_neurons_to_backprop.ipynb (Tutorial 1, Deep Learning 89-6877, Fall 2026)."""
from nbkit import Builder
from recap import make_recap

make_recap("T01_neurons_to_backprop", [
    ("slides/lectures/p1.pdf", [17, 18, 29, 30, 32]),
    ("slides/lectures/p3.pdf", [11, 19, 25, 30, 35, 38, 39, 44, 49]),
])

B = Builder()
md, code = B.md, B.code

md(r"""
# Tutorial 1: from a neuron to backprop, built by hand
**Deep Learning 89-6877, Fall 2026.** TA: Tal Fiskus.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idansc/dl-course-2026/blob/main/tutorials/T01_neurons_to_backprop.ipynb) · [Recap slides](https://github.com/idansc/dl-course-2026/blob/main/tutorials/recap/T01_neurons_to_backprop_recap.pdf)

Plan for today (≈ 75 min):
0. Recap slides (15 min)
1. Lecture recap: neuron, perceptron, XOR, MLP, chain rule (10 min)
2. A perceptron, and why it fails on XOR (5 min)
3. A scalar autograd engine in ~40 lines (20 min)
4. The same MLP with tensors: manual backward vs `torch.autograd` (15 min)
5. Initial weights: why the scale of the random start matters (10 min)
6. If time: BatchNorm by hand, and the library version of everything

Runs on CPU (Colab or laptop). Cells marked ✏️ are for you to try.
<<STUDENT>>
**Before the tutorial:** fill in every `# TODO` (replace the `...`). Each one is followed by a check cell that prints the PyTorch answer next to yours. The solutions are presented in the tutorial.
<</STUDENT>>
""")

code(r"""
import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
torch.manual_seed(0); random.seed(0); np.random.seed(0)
""")

md(r"""
## 1. Lecture recap

**Neuron.** Input $x\in\mathbb{R}^d$, weights $w\in\mathbb{R}^d$, bias $b$: $\;\hat y = \sigma(w^\top x + b)$.

**Perceptron (Rosenblatt, 1958).** Step activation, labels $y\in\{0,1\}$, update on every example
$$\Delta w = \eta\,(y-\hat y)\,x .$$
No change when correct; when wrong, the hyperplane moves toward the misclassified point. It converges iff the data is linearly separable.

**XOR (Minsky & Papert, 1969).** No single hyperplane separates XOR → one hidden layer fixes it, but now we need a way to train the hidden weights.

**Training an MLP** = minimize a loss $L(\theta)$ by gradient descent, $\theta \leftarrow \theta - \eta \nabla_\theta L$.

**Backprop** = the chain rule on the computational graph. Every node only needs its *local* gradient:
$$\underbrace{\frac{\partial L}{\partial x}}_{\text{downstream}} = \underbrace{\frac{\partial z}{\partial x}}_{\text{local}} \cdot \underbrace{\frac{\partial L}{\partial z}}_{\text{upstream}}$$
Patterns: **add** copies the upstream gradient to both inputs; **mul** swaps the inputs; **max** routes it to the winner; a node used twice **sums** its gradients.
""")

md(r"""
## 2. Perceptron, and the XOR wall
""")

code(r"""
def perceptron(X, y, epochs=20, lr=1.0):
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(epochs):
        for xi, yi in zip(X, y):
            y_hat = float(w @ xi + b > 0)
            #>> perceptron update of w and b: Δw = η (y − ŷ) x
            w += lr * (yi - y_hat) * xi
            b += lr * (yi - y_hat)
            #<<
    return w, b

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
for name, y in [("AND", np.array([0,0,0,1.])), ("OR", np.array([0,1,1,1.])), ("XOR", np.array([0,1,1,0.]))]:
    w, b = perceptron(X, y)
    acc = ((X @ w + b > 0) == y).mean()
    print(f"{name}: w={w}, b={b:+.0f}, accuracy={acc:.2f}")
""")

md(r"""
AND and OR are learned exactly. On XOR the weights keep cycling and accuracy never reaches 1, however many epochs we give it (at best 3 of 4 points; here it ends at 2 of 4). We need a hidden layer, and therefore gradients through it.
""")

md(r"""
## 3. A scalar autograd engine

Each `Value` stores its number, its gradient, its parents, and a closure that applies the **local gradient × upstream** rule.
`backward()` visits the graph in reverse topological order, so every node has its full upstream gradient before it passes it on.
""")

code(r"""
class Value:
    def __init__(self, data, parents=()):
        self.data, self.grad = float(data), 0.0
        self._parents, self._backward = parents, lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def _backward():                      # add: distribute
            self.grad  += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def _backward():                      # mul: swap
            #>> local gradient of a*b w.r.t. each input, times out.grad (accumulate with +=)
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad
            #<<
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,))
        def _backward():
            #>> d tanh(x)/dx = 1 − tanh(x)^2
            self.grad += (1 - t * t) * out.grad
            #<<
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1 / (1 + math.exp(-self.data))
        out = Value(s, (self,))
        def _backward():
            #>> d σ(x)/dx = σ(x)(1 − σ(x))
            self.grad += s * (1 - s) * out.grad
            #<<
        out._backward = _backward
        return out

    def __pow__(self, k):                     # k is a plain number
        out = Value(self.data ** k, (self,))
        def _backward():
            #>> d x^k/dx = k x^(k−1)
            self.grad += k * self.data ** (k - 1) * out.grad
            #<<
        out._backward = _backward
        return out

    __radd__ = __add__
    __rmul__ = __mul__
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)

    def backward(self):
        #>> topological order of the graph (DFS over _parents), set self.grad = 1, call _backward in reverse order
        order, seen = [], set()
        def visit(v):
            if v not in seen:
                seen.add(v)
                for p in v._parents: visit(p)
                order.append(v)
        visit(self)
        self.grad = 1.0
        for v in reversed(order):
            v._backward()
        #<<

    def __repr__(self): return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
""")

md(r"""
**Check 1: the sigmoid neuron from the backprop lecture.** $f = \sigma(w_0x_0 + w_1x_1 + w_2)$ with $w_0=2, x_0=-1, w_1=-3, x_1=-2, w_2=-3$. The lecture got $f=0.73$ and a local sigmoid gradient of $(1-0.73)\cdot 0.73 = 0.20$.
""")

code(r"""
w0, x0, w1, x1, w2 = Value(2.0), Value(-1.0), Value(-3.0), Value(-2.0), Value(-3.0)
f = (w0 * x0 + w1 * x1 + w2).sigmoid()
f.backward()
print(f"f = {f.data:.2f}")
print(f"df/dw0 = {w0.grad:.2f}, df/dx0 = {x0.grad:.2f}, df/dw1 = {w1.grad:.2f}, df/dx1 = {x1.grad:.2f}, df/dw2 = {w2.grad:.2f}")

# same graph in PyTorch
t = {k: torch.tensor(v, requires_grad=True) for k, v in dict(w0=2., x0=-1., w1=-3., x1=-2., w2=-3.).items()}
torch.sigmoid(t['w0'] * t['x0'] + t['w1'] * t['x1'] + t['w2']).backward()
print("torch:   ", {k: round(v.grad.item(), 2) for k, v in t.items()})
""")

md(r"""
✏️ **Why `+=` and not `=` in every `_backward`?** Try `a = Value(3.0); b = a * a; b.backward()` with `=` instead. A node used twice must sum its gradients.
""")

code(r"""
a = Value(3.0); b = a * a; b.backward()
print(a.grad, "(should be d(a^2)/da = 6)")
""")

md(r"""
**Check 2: an MLP made of `Value`s learns XOR**, the function the perceptron could not.
""")

code(r"""
class Neuron:
    def __init__(self, n_in):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)
    def __call__(self, x):
        return sum((wi * xi for wi, xi in zip(self.w, x)), self.b).tanh()
    def params(self): return self.w + [self.b]

class MLP:
    def __init__(self, sizes):
        self.layers = [[Neuron(a) for _ in range(b)] for a, b in zip(sizes[:-1], sizes[1:])]
    def __call__(self, x):
        for layer in self.layers:
            x = [n(x) for n in layer]
        return x[0] if len(x) == 1 else x
    def params(self): return [p for layer in self.layers for n in layer for p in n.params()]

random.seed(1)
net = MLP([2, 4, 1])
X_xor = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_xor = [-1, 1, 1, -1]                      # tanh output, so targets in {-1, +1}

for step in range(200):
    loss = sum(((net(x) - y) ** 2 for x, y in zip(X_xor, y_xor)), Value(0.0))
    for p in net.params(): p.grad = 0.0     # zero grads, or they accumulate across steps
    loss.backward()
    for p in net.params(): p.data -= 0.05 * p.grad  # @student: for p in net.params(): ...  # TODO: gradient step, lr = 0.05
    if step % 50 == 0 or step == 199:
        print(f"step {step:3d}  loss {loss.data:.4f}")

print("predictions:", [round(net(x).data, 2) for x in X_xor], "targets:", y_xor)
""")

md(r"""
## 4. The same thing with tensors: manual backward

A 2-layer MLP for 10-class classification, written as matrix ops:
$$h = \mathrm{ReLU}(XW_1 + b_1),\qquad s = hW_2 + b_2,\qquad L = \tfrac{1}{N}\textstyle\sum_i -\log \mathrm{softmax}(s_i)_{y_i}.$$
Backward, using the same local × upstream rule but with matrices (shapes must match the forward):
$$\frac{\partial L}{\partial s} = \tfrac{1}{N}(\mathrm{softmax}(s) - Y),\quad
\frac{\partial L}{\partial W_2} = h^\top \frac{\partial L}{\partial s},\quad
\frac{\partial L}{\partial h} = \frac{\partial L}{\partial s} W_2^\top,\quad
\frac{\partial L}{\partial a_1} = \frac{\partial L}{\partial h}\odot \mathbb{1}[a_1>0],\quad
\frac{\partial L}{\partial W_1} = X^\top \frac{\partial L}{\partial a_1}.$$
""")

code(r"""
N, D, H, C = 64, 20, 32, 10
X = torch.randn(N, D, dtype=torch.float64)
y = torch.randint(0, C, (N,))
W1 = (torch.randn(D, H, dtype=torch.float64) * 0.1).requires_grad_()
b1 = torch.zeros(H, dtype=torch.float64, requires_grad=True)
W2 = (torch.randn(H, C, dtype=torch.float64) * 0.1).requires_grad_()
b2 = torch.zeros(C, dtype=torch.float64, requires_grad=True)

# forward
a1 = X @ W1 + b1
h = a1.clamp(min=0)
s = h @ W2 + b2
loss = F.cross_entropy(s, y)

# manual backward
with torch.no_grad():
    #>> ds, dW2, db2, dh, da1, dW1, db1 from the equations above
    ds = (torch.softmax(s, 1) - F.one_hot(y, C)) / N
    dW2, db2 = h.T @ ds, ds.sum(0)
    dh = ds @ W2.T
    da1 = dh * (a1 > 0)
    dW1, db1 = X.T @ da1, da1.sum(0)
    #<<

# autograd
loss.backward()
for name, mine, ref in [("W1", dW1, W1.grad), ("b1", db1, b1.grad), ("W2", dW2, W2.grad), ("b2", db2, b2.grad)]:
    print(f"{name}: max |manual − autograd| = {(mine - ref).abs().max().item():.2e}")
""")

md(r"""
✏️ Add a `tanh` layer between `h` and `s` and extend the manual backward. Which single line changes?
""")

md(r"""
## 5. Initial weights

**Background.** In 2006 the "deep learning" revival (Hinton & Salakhutdinov, lecture 1) trained deep nets by **layer-wise unsupervised pretraining** first, because deep nets trained from a random start mostly failed. Four years later, Glorot & Bengio (2010) showed that a large part of the failure was simply **the scale of the random initial weights**. Then He et al. (2015) derived the right scale for ReLU and trained a 30-layer plain network from scratch. Nobody pretrains for this reason anymore. The fix is one line: choose the init std.

**Why scale matters.** For a layer $a = Wx$ with $n_{in}$ inputs and i.i.d. weights of variance $\sigma^2$, $\;\mathrm{Var}(a) \approx n_{in}\,\sigma^2\,\mathrm{Var}(x)$. Over $L$ layers this multiplies $L$ times, so the activations (and the gradients going back) **explode** if $n_{in}\sigma^2 > 1$ and **vanish** if it is $< 1$.
- Xavier/Glorot (tanh): $\sigma^2 = 1/n_{in}$ (or $2/(n_{in}+n_{out})$)
- Kaiming/He (ReLU kills half the signal): $\sigma^2 = 2/n_{in}$

Let's see it: 10 layers, width 512; three choices of std with tanh, two with ReLU.
""")

code(r"""
def signal_stats(std_fn, depth=10, width=512, act=torch.tanh):
    # per-layer activation std (forward) and std of dL/d(layer input) (backward)
    x = torch.randn(1000, width)
    inputs, acts = [], []
    for _ in range(depth):
        x = x.detach().requires_grad_() if not inputs else x
        x.retain_grad(); inputs.append(x)
        W = torch.randn(width, width) * std_fn(width)
        x = act(x @ W)
        acts.append(x.std().item())
    x.backward(torch.randn_like(x))
    grads = [t.grad.std().item() for t in inputs]
    return acts, grads

inits = {
    "std = 0.01":              lambda n: 0.01,
    "std = 1":                 lambda n: 1.0,
    "Xavier: 1/sqrt(n)":       lambda n: 1 / math.sqrt(n),
}
layers = range(1, 11)
fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
for name, fn in inits.items():
    acts, grads = signal_stats(fn)
    axes[0].plot(layers, acts, marker="o", label=name)
    axes[1].plot(layers, grads, marker="o", label=name)
axes[0].set(title="tanh: activation std (forward)", xlabel="layer", yscale="log"); axes[0].legend()
axes[1].set(title="tanh: gradient std (backward)", xlabel="layer", yscale="log"); axes[1].legend()

relu_inits = {"Xavier: 1/sqrt(n)": lambda n: 1 / math.sqrt(n), "Kaiming: sqrt(2/n)": lambda n: math.sqrt(2 / n)}  # @student: relu_inits = {"Xavier: 1/sqrt(n)": lambda n: 1 / math.sqrt(n), "Kaiming: sqrt(2/n)": lambda n: None}  # TODO: the std He et al. derived for ReLU
for name, fn in relu_inits.items():
    axes[2].plot(layers, signal_stats(fn, act=torch.relu)[0], marker="o", label=name)
axes[2].set(title="ReLU: activation std (forward)", xlabel="layer", yscale="log"); axes[2].legend()
plt.tight_layout(); plt.show()
""")

md(r"""
- **std = 0.01**: activations shrink by ~$\sqrt{512}\cdot 0.01 \approx 0.23$ per layer and are ~$10^{-6}$ by layer 10. Gradients shrink the same way going back (middle plot), so the first layers barely learn.
- **std = 1**: $\sqrt{512}\approx 23$× too large. The forward plot *looks* fine (std ≈ 1), but only because the tanh units are saturated at ±1. The backward plot shows the problem: going back, each layer multiplies the gradient by $W^\top$ (≈23× too large) and by $1-\tanh^2$ (small for most units), and the product still grows, so layer 1 gets gradients ~$10^5$× larger than layer 10. One learning rate cannot suit both ends. Always look at both directions.
- **Xavier** keeps tanh stable; with ReLU it halves the variance each layer; **Kaiming**'s factor 2 fixes exactly that.

**And zero init?** Then every hidden unit computes the same function, receives the same gradient, and stays identical forever. Initial weights must be random to break the symmetry.
""")

code(r"""
torch.manual_seed(0)
net0 = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
for p in net0.parameters():
    nn.init.constant_(p, 0.3)                 # identical (non-zero) start for every unit
opt = torch.optim.SGD(net0.parameters(), lr=0.1)
Xt = torch.tensor([[0., 0], [0, 1], [1, 0], [1, 1]]); yt = torch.tensor([[-1.], [1], [1], [-1]])
for _ in range(500):
    opt.zero_grad(); F.mse_loss(net0(Xt), yt).backward(); opt.step()
print("hidden-layer weights after 500 steps (all rows identical):\n", net0[0].weight.data)
print("XOR predictions:", net0(Xt).squeeze().detach().numpy().round(2))
""")

md(r"""
✏️ Re-run the XOR `Value`-MLP from section 3 with `random.uniform(-1, 1)` replaced by `random.uniform(-0.001, 0.001)` and then by `random.uniform(-10, 10)`. What happens to the loss curve, and why?

PyTorch's `nn.Linear` already initializes with a Kaiming-style uniform scale, which is why "it just works" by default.
""")

md(r"""
## 6. If time: BatchNorm by hand

The other fix for bad scaling: normalize each feature over the batch, then let the network rescale it,
$$\hat x = \frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}},\qquad y = \gamma \hat x + \beta .$$
Backward (derive it on the board from the graph; three paths reach $x$: directly, through $\mu_B$, and through $\sigma_B^2$):
$$\frac{\partial L}{\partial x} = \frac{\gamma}{N\sqrt{\sigma_B^2+\epsilon}}\Big(N\,\frac{\partial L}{\partial y} - \sum_i \frac{\partial L}{\partial y_i} - \hat x \sum_i \frac{\partial L}{\partial y_i}\hat x_i\Big)$$
""")

code(r"""
def batchnorm_forward(x, gamma, beta, eps=1e-5):
    mu, var = x.mean(0), x.var(0, unbiased=False)
    x_hat = (x - mu) / torch.sqrt(var + eps)
    return gamma * x_hat + beta, (x_hat, var, eps)

def batchnorm_backward(dy, gamma, cache):
    x_hat, var, eps = cache
    N = dy.shape[0]
    dgamma, dbeta = (dy * x_hat).sum(0), dy.sum(0)
    dx = gamma / (N * torch.sqrt(var + eps)) * (N * dy - dy.sum(0) - x_hat * (dy * x_hat).sum(0))  # @student: dx = None  # TODO: the formula above
    return dx, dgamma, dbeta

x = torch.randn(32, 8, dtype=torch.float64, requires_grad=True)
gamma = torch.rand(8, dtype=torch.float64, requires_grad=True)
beta = torch.randn(8, dtype=torch.float64, requires_grad=True)
y_bn, cache = batchnorm_forward(x, gamma, beta)
dy = torch.randn_like(y_bn)
y_bn.backward(dy)
dx, dgamma, dbeta = batchnorm_backward(dy, gamma.detach(), tuple(c.detach() if torch.is_tensor(c) else c for c in cache))
print("dx     max err:", (dx - x.grad).abs().max().item())
print("dgamma max err:", (dgamma - gamma.grad).abs().max().item())
print("dbeta  max err:", (dbeta - beta.grad).abs().max().item())
""")

md(r"""
## Summary
- The perceptron rule learns any linearly separable function and nothing else; XOR needs a hidden layer.
- Backprop is just **local gradient × upstream gradient**, applied in reverse topological order, with **+=** for nodes that are reused.
- Manual matrix gradients match `torch.autograd` to ~1e-16: autograd is not magic, it is section 3 with tensors.
- **Initial weights** set the scale of every signal in the network: too small → vanishing, too large → saturation or explosion, all equal → symmetric units. Xavier (tanh) and Kaiming (ReLU) choose $\sigma^2 \propto 1/n_{in}$.

**Further watching:** Karpathy, *The spelled-out intro to neural networks and backpropagation: building micrograd* (section 3 follows it); Karpathy, *makemore part 3* (initialization and BatchNorm).
""")

for k, p in B.write("T01_neurons_to_backprop").items():
    print(k, p)
