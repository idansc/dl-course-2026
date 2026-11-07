"""Code for the lecture-4 slide "Dropout": what nn.Dropout does in training and at test time, and where PyTorch's
Transformer layer applies it.

1. Training mode: every entry is zeroed with probability p and the others are divided by 1 - p (here 1 / 0.9 = 1.111),
   so the expected value is unchanged ("inverted dropout"). Evaluation mode: the identity.
2. Averaged over many random masks, the training-mode output equals the input.
3. nn.TransformerEncoderLayer (dropout=0.1 by default) applies dropout in four places: on the attention weights (inside
   nn.MultiheadAttention), on the attention output before the residual add, inside the MLP after the activation, and on
   the MLP output before the residual add. The 2017 paper (Section 5.4) names two: the output of each sublayer before
   the residual add, and the sum of the token embeddings and the positional encodings.
Note the two conventions for p: PyTorch's p is the probability of dropping a unit; Srivastava et al. (2014) use p for
the probability of keeping it.

Run:  python dropout_demo.py      (writes data/dropout_demo.json; the slide draws the two vectors from it)
"""
import json
from pathlib import Path

import torch
from torch import nn

OUT = Path(__file__).parent / "data" / "dropout_demo.json"

torch.manual_seed(0)
drop = nn.Dropout(p=0.1)
x = torch.ones(10)

drop.train()                                                   # model.train() sets this for every submodule
train = drop(x)
print("training mode:", train)
drop.eval()                                                    # model.eval()
test = drop(x)
print("eval mode:    ", test)

drop.train()
mean = torch.stack([drop(x) for _ in range(100_000)]).mean(0)
print("mean over 100,000 masks:", mean)
assert torch.allclose(mean, x, atol=0.01)

layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)      # the 2017 base model's sizes; dropout=0.1 by default
sites = [name for name, m in layer.named_modules() if isinstance(m, nn.Dropout)]
print("nn.Dropout modules in nn.TransformerEncoderLayer:", sites,
      "+ the attention weights inside self_attn, p =", layer.self_attn.dropout)
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps({"p": 0.1, "seed": 0, "train": [round(v, 4) for v in train.tolist()],
                           "eval": test.tolist(), "mean_over_masks": [round(v, 4) for v in mean.tolist()],
                           "transformer_layer_dropout_modules": sites, "attention_weight_dropout": layer.self_attn.dropout,
                           "torch": torch.__version__}, indent=1))
print("saved", OUT)
