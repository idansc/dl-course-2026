"""Code for the lecture-2 slide "Swap the model, keep the loop" (CNN row): the LeNet-5 of the lecture slide
(Conv 5x5/20 -> ReLU -> MaxPool 2 -> Conv 5x5/50 -> ReLU -> MaxPool 2 -> Linear 500 -> ReLU -> Linear 10, padding 2)
on CIFAR-10 (3 x 32 x 32 instead of MNIST's 1 x 28 x 28), trained with exactly the same loop as the linear
classifier and the MLP in cifar_experiment.py: SGD momentum 0.9, weight decay 1e-4, batch 128, 20 epochs, seed 0,
learning rate chosen on the validation set, test accuracy measured once. About 10-20 min per run on a laptop CPU
(seconds on a GPU).

Run:  python cifar_cnn.py      -> prints the numbers on the slide, writes data/cifar_cnn_results.json
Try:  fewer filters (Conv 5x5/6, /16 as in LeCun 1998), stride-2 convolutions instead of pooling, or Adam.
"""
import json, time
from pathlib import Path
import torch
from cifar_experiment import Xte, Yte, evaluate, run

torch.set_num_threads(8)
OUT = Path(__file__).with_name("data") / "cifar_cnn_results.json"


def lenet5():
    nn = torch.nn
    return nn.Sequential(
        nn.Unflatten(1, (3, 32, 32)),                         # the loop feeds flattened images; restore 3 x 32 x 32
        nn.Conv2d(3, 20, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2),      # 20 x 16 x 16
        nn.Conv2d(20, 50, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2),     # 50 x 8 x 8
        nn.Flatten(), nn.Linear(50 * 8 * 8, 500), nn.ReLU(), nn.Linear(500, 10))


if __name__ == "__main__":
    t0 = time.time()
    res = {"runs": {}}
    best = None
    for lr in [1e-2, 3e-2]:
        model, h = run(lr, make=lenet5)
        res["runs"][str(lr)] = h
        print(f"LeNet-5 lr={lr:g}: train acc {h['train_acc'][-1]:.3f} | val acc {h['val_acc'][-1]:.3f} "
              f"(best {max(h['val_acc']):.3f} at epoch {1 + h['val_acc'].index(max(h['val_acc']))})  "
              f"{time.time() - t0:.0f}s", flush=True)
        if best is None or h["val_acc"][-1] > best[1]:
            best = (lr, h["val_acc"][-1], model)
    lr, vacc, model = best
    res["best_lr"] = lr
    res["test_loss"], res["test_acc"] = evaluate(model, Xte, Yte)
    res["params"] = sum(p.numel() for p in model.parameters())
    res["seconds"] = time.time() - t0
    OUT.write_text(json.dumps(res))
    print(f"LeNet-5 best lr {lr:g} (val {vacc:.3f}) -> test {res['test_acc']:.3f}; params {res['params']:,}; "
          f"{res['seconds']:.0f}s")
