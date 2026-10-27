"""Code for the lecture-3 clip BatchNormSteps (slide "Batch normalization, step by step"): batch norm on real
activations. The lecture-2 MLP (3072 -> 100 -> 10) with a BatchNorm1d layer after the first linear layer, trained with
the lecture-2 loop on CIFAR-10. Then, for 3 of the 100 features:
  training: a batch of 8 test images -> per-feature mean and std over the batch -> x_hat = (x - mu) / sigma
            -> y = gamma * x_hat + beta (gamma, beta learned);
  test:     one image alone (PyTorch refuses batch statistics for a batch of one) -> the running mean and std that
            training accumulated replace the batch statistics.
Every number is checked against torch.nn.BatchNorm1d (train mode on the batch, eval mode on the single image).
Writes data/batchnorm_steps.json (the clip draws exactly these numbers). About 1 min on a laptop CPU.

Run:  python batchnorm_steps.py
Try:  momentum=0.5 in BatchNorm1d (the running mean then follows the last few batches only); a batch of 2 or 64
      (the batch statistics move away from / towards the running ones); BatchNorm2d after a conv layer (one mean
      and std per channel, over N x H x W).
"""
import copy, json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from cifar_experiment import Xte, Xtr, Xva, Ytr, Yva

OUT = Path(__file__).with_name("data") / "batchnorm_steps.json"
N, EPOCHS, LR = 8, 5, 0.1


def mlp_bn():
    return nn.Sequential(nn.Linear(3072, 100), nn.BatchNorm1d(100), nn.ReLU(), nn.Linear(100, 10))


def train(epochs=EPOCHS, lr=LR, batch=128, seed=0):
    """The lecture-2 loop; model.eval() only for the validation accuracy (eval mode = running statistics)."""
    torch.manual_seed(seed)
    model = mlp_bn()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            loss = F.cross_entropy(model(Xtr[idx]), Ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xva).argmax(1) == Yva).float().mean().item()
        print(f"epoch {epoch + 1}: val acc {acc:.3f}")
    return model, acc


def batch_norm_train(x, bn):
    """Training mode: statistics of this batch (biased variance, as PyTorch normalizes)."""
    mu, var = x.mean(0), x.var(0, unbiased=False)
    sigma = (var + bn.eps).sqrt()
    x_hat = (x - mu) / sigma
    return mu, sigma, x_hat, bn.weight * x_hat + bn.bias


def batch_norm_test(x, bn):
    """Test mode: the running statistics collected during training."""
    sigma = (bn.running_var + bn.eps).sqrt()
    x_hat = (x - bn.running_mean) / sigma
    return bn.running_mean, sigma, x_hat, bn.weight * x_hat + bn.bias


if __name__ == "__main__":
    model, acc = train()
    lin, bn = model[0], model[1]
    with torch.no_grad():
        z = lin(Xte[:N])                                   # pre-BN activations of 8 test images, N x 100
        z1 = lin(Xte[N:N + 1])                             # one more image, alone
        mu, sigma, z_hat, y = batch_norm_train(z, bn)
        check = copy.deepcopy(bn).train()(z)               # a copy: train mode would update the running stats
        assert torch.allclose(y, check, atol=1e-5)
        try:
            copy.deepcopy(bn).train()(z1)
            err = None
        except ValueError as e:                            # batch of one: no variance to normalize with
            err = str(e)
        rmu, rsigma, z1_hat, y1 = batch_norm_test(z1, bn)
        assert torch.allclose(y1, bn.eval()(z1), atol=1e-5)

    # 3 of the 100 features, picked by their running (whole training set) std: smallest, median, largest; picking by
    # this batch's statistics would exaggerate how far a batch of 8 strays from the running ones
    order = (bn.running_var + bn.eps).sqrt().argsort()
    feats = [order[0].item(), order[50].item(), order[-1].item()]
    r = lambda t: [round(v, 4) for v in t.tolist()]
    res = {"setup": f"lecture-2 MLP 3072-100-10 with BatchNorm1d after the first layer, CIFAR-10, SGD momentum 0.9, "
                    f"lr {LR}, wd 1e-4, batch 128, {EPOCHS} epochs, seed 0; val acc {acc:.3f}; batch = test images "
                    f"0-{N - 1}, single image = test image {N}",
           "features": feats, "eps": bn.eps, "momentum": bn.momentum, "val_acc": round(acc, 4),
           "batch": {"x": [r(row[feats]) for row in z], "mu": r(mu[feats]), "sigma": r(sigma[feats]),
                     "x_hat": [r(row[feats]) for row in z_hat], "gamma": r(bn.weight[feats]),
                     "beta": r(bn.bias[feats]), "y": [r(row[feats]) for row in y]},
           "test": {"x": r(z1[0, feats]), "running_mean": r(rmu[feats]), "running_sigma": r(rsigma[feats]),
                    "x_hat": r(z1_hat[0, feats]), "y": r(y1[0, feats]), "train_mode_error": err}}
    OUT.write_text(json.dumps(res, indent=1))
    print(f"features {feats}")
    for j, k in enumerate(feats):
        b = res["batch"]
        print(f"  feature {k}: batch mu {b['mu'][j]:+.2f} sigma {b['sigma'][j]:.2f} | gamma {b['gamma'][j]:.2f} "
              f"beta {b['beta'][j]:+.2f} | running mu {res['test']['running_mean'][j]:+.2f} "
              f"sigma {res['test']['running_sigma'][j]:.2f}")
    print("batch of one in train mode:", err)
    print("saved", OUT)
