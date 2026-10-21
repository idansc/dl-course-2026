"""Code for the lecture-2 Muon slides: (1) how lopsided a real gradient matrix is, (2) what Muon's
orthogonalization does to it, (3) Muon vs SGD-momentum vs AdamW on our 2-layer MLP for CIFAR-10
(the same training loop as the opening slide; only the optimizer line changes).

Muon (Keller Jordan, 2024): github.com/KellerJordan/Muon, MIT License, Copyright (c) 2024 Keller Jordan.
zeropower_via_newtonschulz5 and muon_update below are copied from that repository; the only change is
float32 instead of bfloat16, because this runs on a laptop CPU.

Run:  python muon_cifar.py     -> prints the numbers on the slides, writes data/muon_results.json
Selection: learning rate and number of epochs chosen on the validation set (early stopping); test accuracy is
           read at the chosen epoch only.
Try:  more Newton-Schulz steps, Muon on both layers, other learning rates, weight decay for Muon.
"""
import json, time
from pathlib import Path
import torch
import torch.nn.functional as F
from cifar_experiment import Xtr, Ytr, Xva, Yva, Xte, Yte, evaluate, mlp   # data: 45k / 5k / 10k split

torch.set_num_threads(4)
OUT = Path(__file__).with_name("data") / "muon_results.json"


# ------------------------------------------------------------- Muon (from github.com/KellerJordan/Muon, MIT)
def zeropower_via_newtonschulz5(G, steps: int):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()                                      # bfloat16 on GPU in the original
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)   # spectral norm at most 1
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)                     # momentum buffer (an average of past gradients)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)   # orthogonalize: U S V^T -> ~U V^T
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


class SingleDeviceMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])


# ------------------------------------------------------------- the training loop, optimizer as a parameter
def train(make_opts, epochs=20, batch=128, seed=0, probe=False):
    torch.manual_seed(seed)
    model = mlp()                                      # Linear(3072, 100) -> ReLU -> Linear(100, 10)
    opts = make_opts(model)
    g = torch.Generator().manual_seed(seed)
    hist = {"val_acc": [], "train_loss": [], "test_acc": []}
    for epoch in range(epochs):
        order = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            loss = F.cross_entropy(model(Xtr[idx]), Ytr[idx])
            for o in opts:
                o.zero_grad()
            loss.backward()
            for o in opts:
                o.step()
        hist["train_loss"].append(evaluate(model, Xtr, Ytr)[0])
        hist["val_acc"].append(evaluate(model, Xva, Yva)[1])
        hist["test_acc"].append(evaluate(model, Xte, Yte)[1])      # recorded, never used for any choice
    return model, hist


def sgd(lr):
    return lambda m: [torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)]


def adamw(lr):
    return lambda m: [torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)]


def muon(lr):
    # Muon for the hidden 2-D weight matrix W1 (100 x 3072); AdamW for the output layer and the biases
    def make(m):
        hidden, rest = [m[0].weight], [m[0].bias, m[2].weight, m[2].bias]
        return [SingleDeviceMuon(hidden, lr=lr, momentum=0.95), torch.optim.AdamW(rest, lr=1e-3, weight_decay=0.01)]
    return make


def momentum_buffer():
    """W1's SGD-momentum buffer (100 x 3072) after one epoch of the training loop (lr 0.01, momentum 0.9, seed 0):
    the matrix Muon would orthogonalize."""
    torch.manual_seed(0)
    model = mlp()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    g = torch.Generator().manual_seed(0)
    order = torch.randperm(len(Xtr), generator=g)
    for i in range(0, len(order), 128):
        idx = order[i:i + 128]
        opt.zero_grad()
        F.cross_entropy(model(Xtr[idx]), Ytr[idx]).backward()
        opt.step()
    return opt.state[model[0].weight]["momentum_buffer"].clone()        # 100 x 3072


def gradient_spectrum(M=None):
    """Singular values of W1's SGD-momentum buffer after one epoch, before and after orthogonalization."""
    M = momentum_buffer() if M is None else M
    s = torch.linalg.svdvals(M)
    s_orth = torch.linalg.svdvals(zeropower_via_newtonschulz5(M, steps=5))
    energy = (s ** 2).cumsum(0) / (s ** 2).sum()
    return {"sv": (s / s[0]).tolist(), "sv_orth": s_orth.tolist(), "top1_energy": energy[0].item(),
            "top5_energy": energy[4].item(), "top10_energy": energy[9].item(), "shape": list(M.shape)}


if __name__ == "__main__":
    t0 = time.time()
    res = {"spectrum": gradient_spectrum(), "runs": {}}
    sp = res["spectrum"]
    print(f"W1 momentum {sp['shape']}: top-1 direction {100 * sp['top1_energy']:.0f}% of the energy, "
          f"top-5 {100 * sp['top5_energy']:.0f}%, top-10 {100 * sp['top10_energy']:.0f}%")
    print(f"after Newton-Schulz: singular values in [{min(sp['sv_orth']):.2f}, {max(sp['sv_orth']):.2f}]")
    for name, make, lrs in [("SGD + momentum", sgd, [0.01, 0.03]), ("AdamW", adamw, [3e-4, 1e-3, 3e-3]),
                            ("Muon + AdamW", muon, [0.001, 0.002, 0.005, 0.01, 0.02])]:
        best = None
        for lr in lrs:
            model, h = train(make(lr))
            ep = max(range(len(h["val_acc"])), key=lambda i: h["val_acc"][i])      # early stopping on validation
            print(f"{name:15s} lr={lr:<7g} best val {h['val_acc'][ep]:.3f} at epoch {ep + 1}, "
                  f"final val {h['val_acc'][-1]:.3f}  ({time.time() - t0:.0f}s)")
            res["runs"][f"{name}|{lr}"] = h
            if best is None or h["val_acc"][ep] > best[1]:
                best = (lr, h["val_acc"][ep], ep, h)
        lr, vacc, ep, h = best
        first50 = next((i + 1 for i, v in enumerate(h["val_acc"]) if v >= 0.50), None)
        res[name] = {"lr": lr, "epoch": ep + 1, "val_acc": vacc, "test_acc": h["test_acc"][ep],
                     "epochs_to_50": first50, "final_train_loss": h["train_loss"][-1], "val_acc_curve": h["val_acc"]}
        print(f"  -> chosen lr {lr:g}, epoch {ep + 1}: val {vacc:.3f}, test {h['test_acc'][ep]:.3f}; "
              f"50% val reached at epoch {first50}")
    res["seconds"] = time.time() - t0
    OUT.write_text(json.dumps(res))
    print(f"saved {OUT} ({res['seconds']:.0f}s)")
