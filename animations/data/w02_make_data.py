"""Precompute the real numbers used by w02_cnns.py.
    python data/w02_make_data.py
Writes:
  data/w02_resgrads.json  plain vs residual 30-layer MLP on a 2-D spiral: per-layer gradient norms at init
                          and training-loss curves (same init scheme, same optimizer, same data)
  data/w02_gradcam.npz    Grad-CAM of a pretrained torchvision ResNet-18 on grace_hopper.jpg, plus a
                          shortcut experiment: a tiny CNN trained on shapes whose label leaks through a corner tag
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path(__file__).parent
torch.manual_seed(0)
np.random.seed(0)


# ---------------------------------------------------------------- residual vs plain
def spiral(n=600, k=3, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(k):
        r = np.linspace(0.05, 1, n // k)
        t = np.linspace(c * 4, (c + 1) * 4, n // k) + rng.normal(0, 0.2, n // k)
        X.append(np.c_[r * np.sin(t), r * np.cos(t)])
        y += [c] * (n // k)
    return torch.tensor(np.concatenate(X), dtype=torch.float32), torch.tensor(y)


class Net(nn.Module):
    def __init__(self, depth=30, width=64, residual=False):
        super().__init__()
        self.inp = nn.Linear(2, width)
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.out = nn.Linear(width, 3)
        self.residual = residual

    def forward(self, x):
        h = self.inp(x)
        for L in self.layers:
            # plain: h <- ReLU(W h);  residual (pre-activation): h <- h + W ReLU(h)
            h = h + L(F.relu(h)) if self.residual else F.relu(L(h))
        return self.out(F.relu(h) if self.residual else h)


def res_experiment(depth=30, steps=600):
    X, y = spiral()
    out = {"depth": depth, "width": 64, "init": "PyTorch default (Kaiming-uniform, a=sqrt(5))",
           "task": "3-class 2-D spiral, 600 points",
           "optimizer": "SGD momentum 0.9, full batch, best lr of {0.01, 0.003, 0.001} per net"}
    for name, res in [("plain", False), ("residual", True)]:
        torch.manual_seed(1)
        net = Net(depth, residual=res)
        loss = F.cross_entropy(net(X), y)
        loss.backward()
        out[name + "_grad"] = [float(L.weight.grad.norm()) for L in net.layers]
        best = None
        for lr in (0.01, 0.003, 0.001):
            torch.manual_seed(1)
            net = Net(depth, residual=res)
            opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
            curve = []
            for s in range(steps):
                opt.zero_grad()
                loss = F.cross_entropy(net(X), y)
                loss.backward()
                opt.step()
                curve.append(float(loss.detach()))
            acc = float((net(X).argmax(1) == y).float().mean())
            if np.isfinite(curve[-1]) and (best is None or curve[-1] < best[1][-1]):
                best = (lr, curve, acc)
            print(name, lr, round(curve[-1], 4), acc, flush=True)
        out[name + "_lr"], out[name + "_loss"], out[name + "_acc"] = best
    (OUT / "w02_resgrads.json").write_text(json.dumps(out))
    print("plain grad first/last", out["plain_grad"][0], out["plain_grad"][-1], "acc", out["plain_acc"])
    print("res   grad first/last", out["residual_grad"][0], out["residual_grad"][-1], "acc", out["residual_acc"])


# ---------------------------------------------------------------- Grad-CAM
def gradcam(model, layer, x, cls=None):
    acts, grads = {}, {}
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    logits = model(x)
    if cls is None:
        cls = int(logits.argmax())
    model.zero_grad()
    logits[0, cls].backward()
    h1.remove(); h2.remove()
    a, g = acts["a"][0], grads["g"][0]
    w = g.mean(dim=(1, 2))
    cam = F.relu((w[:, None, None] * a).sum(0))
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().numpy(), cls, torch.softmax(logits, 1)[0, cls].item()


def real_gradcam():
    import matplotlib.cbook as cbook
    from PIL import Image
    import torchvision
    from torchvision.models import resnet18, ResNet18_Weights
    img = Image.open(cbook.get_sample_data("grace_hopper.jpg")).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, 0, (w - s) // 2 + s, s)).resize((224, 224))
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights).eval()
    x = weights.transforms()(img)[None]
    cams, names, probs = [], [], []
    with torch.no_grad():
        p = torch.softmax(model(x), 1)[0]
    top = p.argsort(descending=True)[:5].tolist()
    cats = weights.meta["categories"]
    for c in top:
        cam, _, pr = gradcam(model, model.layer4, x, c)
        cams.append(cam); names.append(cats[c]); probs.append(pr)
    print("resnet18 top-5:", [(cats[c], round(float(p[c]), 3)) for c in top])
    return np.asarray(img), np.stack(cams), names, probs


def shapes_data(n, tag_follows_label=True, seed=0, size=32):
    """circle (0) vs square (1), random position/size, noisy; a tag in the bottom-right corner leaks the label:
    a horizontal bar for "circle", a vertical bar for "square"."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 1, size, size), np.float32)
    y = rng.integers(0, 2, n)
    yy, xx = np.mgrid[:size, :size]
    for i in range(n):
        r = rng.integers(4, 7)
        cx, cy = rng.integers(8, size - 12, 2)
        if y[i] == 0:
            m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        else:
            m = (np.abs(xx - cx) <= r * 0.85) & (np.abs(yy - cy) <= r * 0.85)
        X[i, 0][m] = 1.0
        X[i, 0] += rng.normal(0, 0.3, (size, size))
        lab = y[i] if tag_follows_label else 1 - y[i]
        if lab == 0:
            X[i, 0, size - 4:size - 3, size - 8:size - 2] = 1.5   # horizontal bar -> "circle"
        else:
            X[i, 0, size - 8:size - 2, size - 4:size - 3] = 1.5   # vertical bar   -> "square"
    return torch.tensor(X), torch.tensor(y)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 32, 3, padding=1)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.relu(self.c2(x))
        x = self.c3(x)
        self.feat = x
        return self.fc(F.relu(x).mean((2, 3)))


def shortcut_experiment():
    torch.manual_seed(0)
    Xtr, ytr = shapes_data(3000, True, seed=1)
    model = Tiny()
    opt = torch.optim.Adam(model.parameters(), 3e-3)
    for ep in range(6):
        perm = torch.randperm(len(ytr))
        for i in range(0, len(ytr), 100):
            idx = perm[i:i + 100]
            opt.zero_grad()
            F.cross_entropy(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()
    model.eval()
    Xte, yte = shapes_data(1000, True, seed=2)
    Xsw, ysw = shapes_data(1000, False, seed=2)       # same shapes, tag swapped
    with torch.no_grad():
        acc_same = float((model(Xte).argmax(1) == yte).float().mean())
        acc_swap = float((model(Xsw).argmax(1) == ysw).float().mean())
    print("shortcut model: acc tag-consistent", acc_same, "tag-swapped", acc_swap)
    # share of Grad-CAM mass inside the tag corner (bottom-right 8x8 pixels = 4x4 CAM cells), over 300 test images
    def tag_mass(X):
        ms = []
        for j in range(300):
            cam, _, _ = gradcam(model, model.c3, X[j:j + 1])
            ms.append(cam[-4:, -4:].sum() / (cam.sum() + 1e-8))
        return float(np.mean(ms))
    mass_same, mass_swap = tag_mass(Xte), tag_mass(Xsw)
    print("Grad-CAM mass in tag corner (6.25% of the image): honest", mass_same, "swapped", mass_swap)
    # one example image (a circle) for Grad-CAM, in both tag conditions
    i = int((yte == 0).nonzero()[3])
    out = {}
    for key, X in [("same", Xte), ("swap", Xsw)]:
        x = X[i:i + 1].clone().requires_grad_(False)
        cam, cls, pr = gradcam(model, model.c3, x)
        out[key] = (X[i, 0].numpy(), cam, cls, pr)
    out["mass"] = (mass_same, mass_swap)
    return out, acc_same, acc_swap


if __name__ == "__main__":
    res_experiment()
    img, cams, names, probs = real_gradcam()
    sc, acc_same, acc_swap = shortcut_experiment()
    np.savez_compressed(OUT / "w02_gradcam.npz", img=img, cams=cams, names=np.array(names), probs=np.array(probs),
                        sc_img_same=sc["same"][0], sc_cam_same=sc["same"][1], sc_cls_same=sc["same"][2],
                        sc_prob_same=sc["same"][3], sc_img_swap=sc["swap"][0], sc_cam_swap=sc["swap"][1],
                        sc_cls_swap=sc["swap"][2], sc_prob_swap=sc["swap"][3],
                        sc_acc_same=acc_same, sc_acc_swap=acc_swap,
                        sc_mass_same=sc["mass"][0], sc_mass_swap=sc["mass"][1])
    print("saved")
