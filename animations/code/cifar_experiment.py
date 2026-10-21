"""Code for the lecture-2 slides "Loss curves: the loop on CIFAR-10" and "Swap the model, keep the loop":
the training loop from the opening slide, run for a linear classifier (3072 -> 10) and a 2-layer MLP (hidden 100).
45k/5k/10k split, pixels/255 minus the mean image, SGD momentum 0.9, weight decay 1e-4, batch 128, 20 epochs, seed 0.
Learning rate chosen on the validation set; test accuracy measured once. About 30 s on a laptop CPU.

Run:  python cifar_experiment.py   -> prints the numbers on the slides, writes data/cifar_results.json
Try:  another learning rate, more epochs, a wider hidden layer, or Adam instead of SGD.
"""
import json, pickle, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tutorials/data/cifar-10-batches-py"
if not DATA.exists():                                   # first run: download CIFAR-10 (170 MB) with torchvision
    import torchvision
    torchvision.datasets.CIFAR10(DATA.parent, download=True)
OUT = Path(__file__).with_name("data") / "cifar_results.json"

def load(name):
    with open(DATA / name, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    return d[b"data"].astype(np.float32) / 255.0, np.array(d[b"labels"], dtype=np.int64)

xs, ys = zip(*[load(f"data_batch_{i}") for i in range(1, 6)])
X, Y = np.concatenate(xs), np.concatenate(ys)
Xte, Yte = load("test_batch")
perm = np.random.default_rng(0).permutation(len(X))
tr, va = perm[:45000], perm[45000:]
mean = X[tr].mean(0)                                   # per-pixel mean image of the training split
Xtr, Ytr = torch.tensor(X[tr] - mean), torch.tensor(Y[tr])
Xva, Yva = torch.tensor(X[va] - mean), torch.tensor(Y[va])
Xte, Yte = torch.tensor(Xte - mean), torch.tensor(Yte)

@torch.no_grad()
def evaluate(model, x, y):
    s = model(x)
    return F.cross_entropy(s, y).item(), (s.argmax(1) == y).float().mean().item()

def linear():
    return torch.nn.Linear(3072, 10)


def mlp():
    return torch.nn.Sequential(torch.nn.Linear(3072, 100), torch.nn.ReLU(), torch.nn.Linear(100, 10))


def run(lr, epochs=20, batch=128, seed=0, make=linear):
    torch.manual_seed(seed)
    model = make()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "iter_loss": []}
    for epoch in range(epochs):
        order = torch.randperm(len(Xtr), generator=g)
        losses = []
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            scores = model(Xtr[idx])
            loss = F.cross_entropy(scores, Ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        hist["iter_loss"] += losses[::10]
        tl, ta = evaluate(model, Xtr, Ytr)
        vl, vacc = evaluate(model, Xva, Yva)
        for k, v in zip(["train_loss", "train_acc", "val_loss", "val_acc"], [tl, ta, vl, vacc]):
            hist[k].append(v)
    return model, hist

if __name__ == "__main__":
    t0 = time.time()
    res = {"setup": "CIFAR-10, 45k train / 5k val / 10k test, x/255 minus mean image, nn.Linear(3072,10), "
                    "SGD momentum 0.9, weight_decay 1e-4, batch 128, 20 epochs, seed 0", "runs": {}}
    best = None
    for lr in [1e-1, 1e-2, 1e-3, 1e-4]:
        model, h = run(lr)
        res["runs"][str(lr)] = h
        print(f"lr={lr:g}: train loss {h['train_loss'][-1]:.3f} acc {h['train_acc'][-1]:.3f} | "
              f"val loss {h['val_loss'][-1]:.3f} acc {h['val_acc'][-1]:.3f} | best val acc {max(h['val_acc']):.3f}")
        if best is None or h["val_acc"][-1] > best[1]:
            best = (lr, h["val_acc"][-1], model)
    lr, vacc, model = best
    res["best_lr"] = lr
    res["test_loss"], res["test_acc"] = evaluate(model, Xte, Yte)
    res["params_linear"] = sum(p.numel() for p in model.parameters())

    # same loop, 2-layer MLP (hidden 100, as on the lecture-3 slide)
    res["mlp"] = {"runs": {}}
    best = None
    for lr in [1e-1, 1e-2, 1e-3]:
        model, h = run(lr, make=mlp)
        res["mlp"]["runs"][str(lr)] = h
        print(f"MLP lr={lr:g}: train acc {h['train_acc'][-1]:.3f} | val acc {h['val_acc'][-1]:.3f} "
              f"(best {max(h['val_acc']):.3f} at epoch {1 + h['val_acc'].index(max(h['val_acc']))})")
        if best is None or h["val_acc"][-1] > best[1]:
            best = (lr, h["val_acc"][-1], model)
    lr, vacc, model = best
    res["mlp"]["best_lr"] = lr
    res["mlp"]["test_loss"], res["mlp"]["test_acc"] = evaluate(model, Xte, Yte)
    res["mlp"]["params"] = sum(p.numel() for p in model.parameters())
    print(f"MLP best lr {lr:g} (val {vacc:.3f}) -> test {res['mlp']['test_acc']:.3f}; params {res['mlp']['params']}")
    res["seconds"] = time.time() - t0
    print(f"linear: best lr {res['best_lr']:g} -> test acc {res['test_acc']:.3f}; total {res['seconds']:.0f}s")
    OUT.write_text(json.dumps(res))
