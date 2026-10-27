"""Code for the lecture-3 slide "Plain vs residual on our loop": the degradation problem of He et al. (2016, Fig. 1 and
Fig. 6) on CIFAR-10, with the training loop of lecture 2. Four networks from He et al. sec. 4.2 (6n + 2 layers: a 3x3
conv, three stages of 2n 3x3 convs with 16 / 32 / 64 channels, global average pooling, a 10-way linear layer; batch
norm after every conv): plain-20, plain-56, ResNet-20, ResNet-56. The ResNets add identity shortcuts (option A: zero
channels when a stage doubles them, so plain and residual nets have the same parameters).
Same loop and data as cifar_experiment.py: 45k / 5k / 10k split, SGD momentum 0.9, weight decay 1e-4, batch 128,
20 epochs, seed 0; learning rate 0.1 as in the paper, no data augmentation. About 15 min on an M4 Max (MPS).

Run:  python resnet_degradation.py   -> prints the numbers on the slide, writes data/resnet_degradation.json
Try:  n = 18 (110 layers); plain nets without batch norm; data augmentation (random crop + flip) and 160 epochs to
      get the paper's test errors (8.75% for ResNet-20, 6.97% for ResNet-56).
"""
import json, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from cifar_experiment import Xte, Xtr, Xva, Yte, Ytr, Yva

OUT = Path(__file__).with_name("data") / "resnet_degradation.json"
DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


class Block(nn.Module):
    """Two 3x3 convs with batch norm; with residual=True the input is added before the last ReLU."""

    def __init__(self, cin, cout, stride, residual):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.residual, self.stride, self.pad = residual, stride, cout - cin

    def forward(self, x):
        out = self.b2(self.c2(F.relu(self.b1(self.c1(x)))))
        if self.residual:
            sc = x[:, :, ::self.stride, ::self.stride]                       # option A: subsample, then zero channels
            out = out + F.pad(sc, (0, 0, 0, 0, self.pad // 2, self.pad - self.pad // 2))
        return F.relu(out)


class Net(nn.Module):
    def __init__(self, n, residual):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        blocks, cin = [], 16
        for cout, stride in [(16, 1), (32, 2), (64, 2)]:
            for i in range(n):
                blocks.append(Block(cin, cout, stride if i == 0 else 1, residual))
                cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(64, 10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):                                   # the loop feeds flattened images, as in lecture 2
        h = self.blocks(self.stem(x.view(-1, 3, 32, 32)))
        return self.fc(h.mean((2, 3)))


@torch.no_grad()
def evaluate(model, x, y, chunk=1000):
    model.eval()
    loss = correct = 0.0
    for i in range(0, len(x), chunk):
        s = model(x[i:i + chunk].to(DEV))
        t = y[i:i + chunk].to(DEV)
        loss += F.cross_entropy(s, t, reduction="sum").item()
        correct += (s.argmax(1) == t).sum().item()
    model.train()
    return loss / len(x), correct / len(x)


def run(n, residual, lr=0.1, epochs=20, batch=128, seed=0):
    torch.manual_seed(seed)
    model = Net(n, residual).to(DEV)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    hist = {"train_loss": [], "train_err": [], "val_err": []}
    for epoch in range(epochs):
        order = torch.randperm(len(Xtr), generator=g)
        tot = wrong = 0.0
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            x, y = Xtr[idx].to(DEV), Ytr[idx].to(DEV)
            scores = model(x)
            loss = F.cross_entropy(scores, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            wrong += (scores.argmax(1) != y).sum().item()
        hist["train_loss"].append(tot / len(order))         # running training loss over the epoch (train mode)
        hist["train_err"].append(wrong / len(order))
        hist["val_err"].append(1 - evaluate(model, Xva, Yva)[1])
    hist["test_err"] = 1 - evaluate(model, Xte, Yte)[1]
    hist["params"] = sum(p.numel() for p in model.parameters())
    return hist


if __name__ == "__main__":
    t0 = time.time()
    res = {"setup": "CIFAR-10 45k/5k/10k, He et al. sec 4.2 nets (BN, option-A shortcuts), SGD momentum 0.9, "
                    "wd 1e-4, batch 128, lr 0.1, 20 epochs, seed 0, no augmentation", "device": DEV, "runs": {}}
    for name, n, residual in [("plain-20", 3, False), ("plain-56", 9, False), ("ResNet-20", 3, True),
                              ("ResNet-56", 9, True)]:
        h = run(n, residual)
        res["runs"][name] = h
        print(f"{name:9s} params {h['params']:,}  train err epoch 1 {100 * h['train_err'][0]:.1f}%  "
              f"epoch 20 {100 * h['train_err'][-1]:.1f}%  train loss {h['train_loss'][-1]:.3f}  "
              f"val err {100 * h['val_err'][-1]:.1f}%  test err {100 * h['test_err']:.1f}%  ({time.time() - t0:.0f}s)",
              flush=True)
    res["seconds"] = time.time() - t0
    OUT.write_text(json.dumps(res))
    print("saved", OUT)
