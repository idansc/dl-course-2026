"""Code for the clip TrainingLoop (renders/w01/TrainingLoop.mp4): the training loop of lecture 2 run on CIFAR-10.
model = nn.Linear(3072, 10), softmax cross-entropy, SGD (lr 1e-3, momentum 0.9, weight decay 1e-4), batch 128,
20 epochs. Same split and seed as the lecture's results: 45k train / 5k val / 10k test, pixels/255 minus the mean image.

Run:  python training_loop.py      -> prints the numbers in the clip (minibatch loss at iterations 1, 11, 21, training
                                      loss after 20 epochs, test accuracy), saves figures/training_loop.png
Try:  train(data, lr=1e-1) or lr=1e-4; or swap the model for an MLP:
      nn.Sequential(nn.Linear(3072, 100), nn.ReLU(), nn.Linear(100, 10)) -- the loop stays the same.
"""
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(4)        # as in the run behind the slides (same thread count -> same numbers to the last digit)
DATA = Path(__file__).resolve().parents[2] / "tutorials/data/cifar-10-batches-py"
CLASSES = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def load_cifar10(root=DATA):
    """CIFAR-10 images as tensors N x 3 x 32 x 32, pixels/255 minus the mean training image."""
    if not root.exists():                                   # first run: download CIFAR-10, python version (163 MB)
        import torchvision
        torchvision.datasets.CIFAR10(root.parent, download=True)

    def batch(name):
        with open(root / name, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        return d[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255, np.array(d[b"labels"])

    xs, ys = zip(*[batch(f"data_batch_{i}") for i in range(1, 6)])
    x, y = np.concatenate(xs), np.concatenate(ys)
    x_test, y_test = batch("test_batch")
    perm = np.random.default_rng(0).permutation(len(x))      # fixed 45k / 5k train / validation split
    tr, va = perm[:45000], perm[45000:]
    mean = x[tr].mean(0)
    t = torch.tensor
    return dict(x_train=t(x[tr] - mean), y_train=t(y[tr]), x_val=t(x[va] - mean), y_val=t(y[va]),
                x_test=t(x_test - mean), y_test=t(y_test), mean=mean)


def evaluate(model, x, y):
    """mean cross-entropy and accuracy on a whole split"""
    with torch.no_grad():
        scores = model(x.flatten(1))
        return F.cross_entropy(scores, y).item(), (scores.argmax(1) == y).float().mean().item()


def train(data, lr=1e-3, epochs=20, batch_size=128, seed=0, watch=(0, 10, 20)):
    """The loop on the slide. Returns the model and its history: the loss of every 10th minibatch, train/val loss and
    accuracy and W after every epoch, and a closer look at the steps in `watch` (what the clip shows)."""
    torch.manual_seed(seed)
    model = torch.nn.Linear(3072, 10)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    shuffle = torch.Generator().manual_seed(seed)
    x_train, y_train = data["x_train"], data["y_train"]
    hist = dict(W_init=model.weight.detach().clone(), iter_loss=[], train_loss=[], train_acc=[], val_loss=[],
                val_acc=[], W=[], watched=[])
    step = 0
    for epoch in range(epochs):
        order = torch.randperm(len(x_train), generator=shuffle)
        for i in range(0, len(order), batch_size):
            x, y = x_train[order[i:i + batch_size]], y_train[order[i:i + batch_size]]     # one minibatch
            scores = model(x.flatten(1))            # 1 model: s = Wx + b for 128 images -> 128 x 10
            loss = F.cross_entropy(scores, y)       # 2 loss
            opt.zero_grad()
            loss.backward()                         # 3 gradient dL/dW (backprop)
            if step in watch:                       #   (for the clip: what this step looked like)
                hist["watched"].append(dict(step=step + 1, loss=loss.item(), x=x[:4].clone(), y=y[:4].clone(),
                                            scores=scores[0].detach().clone(), grad_W=model.weight.grad.clone()))
            opt.step()                              # 4 update W
            if step in watch:
                hist["watched"][-1]["W_after"] = model.weight.detach().clone()
            if (i // batch_size) % 10 == 0:
                hist["iter_loss"].append(loss.item())
            step += 1
        for split in ("train", "val"):              # 5 repeat; check the validation set after every epoch
            l, a = evaluate(model, data[f"x_{split}"], data[f"y_{split}"])
            hist[f"{split}_loss"].append(l)
            hist[f"{split}_acc"].append(a)
        hist["W"].append(model.weight.detach().clone())
    hist["iterations"], hist["iters_per_epoch"] = step, step // epochs
    return model, hist


def run():
    """Everything the clip shows: data, the trained model, its history and the test result (measured once)."""
    data = load_cifar10()
    model, hist = train(data)
    hist["test_loss"], hist["test_acc"] = evaluate(model, data["x_test"], data["y_test"])
    return data, model, hist


def to_image(x, mean):
    """one normalized image tensor 3 x 32 x 32 back to a uint8 picture 32 x 32 x 3"""
    return np.round(255 * (x.numpy() + mean)).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)


def templates(W):
    """W (10 x 3072) as 10 pictures 32 x 32 x 3, each row min-max scaled: one template per class"""
    im = W.reshape(10, 3, 32, 32).permute(0, 2, 3, 1).numpy()
    lo, hi = im.min(axis=(1, 2, 3), keepdims=True), im.max(axis=(1, 2, 3), keepdims=True)
    return np.round(255 * (im - lo) / (hi - lo)).astype(np.uint8)


def figure(hist, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ipe, E = hist["iters_per_epoch"], len(hist["train_loss"])
    per = len(hist["iter_loss"]) // E
    x_it = [e + 10 * j / ipe for e in range(E) for j in range(per)]
    fig = plt.figure(figsize=(12, 4.2))
    ax = fig.add_axes([0.06, 0.13, 0.42, 0.75])
    ax.plot(x_it, hist["iter_loss"], color="#EA580C", alpha=0.35, lw=0.8, label="minibatch loss (every 10th)")
    ax.plot(range(1, E + 1), hist["train_loss"], "o-", color="#EA580C", lw=2.5, ms=4, label="training loss")
    ax.plot(range(1, E + 1), hist["val_loss"], "o--", color="#0E7490", lw=1.5, ms=3, label="validation loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_xlim(0, E + 0.5)
    ax.set_xticks(range(0, E + 1, 5))
    ax.legend(frameon=False)
    ax.set_title(f"{hist['iterations']:,} iterations: training loss {hist['train_loss'][-1]:.2f}, "
                 f"test accuracy {100 * hist['test_acc']:.1f}%", fontsize=11)
    T = templates(hist["W"][-1])
    for k in range(10):
        a = fig.add_axes([0.53 + 0.09 * (k % 5), 0.52 - 0.42 * (k // 5), 0.085, 0.33])
        a.imshow(T[k])
        a.set_title(CLASSES[k], fontsize=10)
        a.axis("off")
    fig.text(0.755, 0.94, "W after 20 epochs: one template per class", ha="center", fontsize=11)
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    data, model, hist = run()
    for w in hist["watched"]:
        print(f"iteration {w['step']:>2}: minibatch loss {w['loss']:.3f}")
    print(f"{hist['iterations']:,} iterations ({hist['iters_per_epoch']} per epoch x 20 epochs): "
          f"training loss {hist['train_loss'][-1]:.2f}, validation accuracy {100 * hist['val_acc'][-1]:.1f}%, "
          f"test accuracy {100 * hist['test_acc']:.1f}%")
    out = Path(__file__).with_name("figures") / "training_loop.png"
    out.parent.mkdir(exist_ok=True)
    figure(hist, out)
    print("saved", out)
