"""Code for the lecture-3 slide "Count the parameters": one layer of each architecture, counted by formula and checked
against PyTorch. The same counts come up in the exams (exam bank, weeks 2-4).

Run:  python count_params.py
Try:  nhead=8 or a 200-token input for the Transformer block (same count: heads split d, and no weight depends on the
      sequence length); nn.LSTM(12, 20, bias=False) (the exam's 2,560); nn.GRU(32, 64) (three gates instead of four).
"""
import torch
import torch.nn as nn


def count(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    conv = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)       # AlexNet conv1 (lecture slide "AlexNet")
    conv_formula = 64 * (3 * 11 * 11) + 64                             # filters x (C_in x K x K) + one bias each
    assert count(conv) == conv_formula == 23_296

    lstm = nn.LSTM(input_size=32, hidden_size=64)                      # x in R^32, h in R^64
    lstm_weights = 4 * 64 * (32 + 64)                                  # four gates, each sees [h, x]
    assert count(lstm) == lstm_weights + 2 * 4 * 64 == 25_088          # PyTorch keeps two bias vectors per gate

    block = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128)
    attn_weights = 4 * 64 * 64                                         # W_Q, W_K, W_V, W_O: d x d each
    mlp_weights = 2 * 64 * 128                                         # d -> d_ff -> d
    biases_and_ln = (3 * 64 + 64) + (128 + 64) + 2 * (64 + 64)         # attention biases, MLP biases, 2 LayerNorms
    assert count(block) == attn_weights + mlp_weights + biases_and_ln == 33_472
    for T in (20, 200):                                                # the count does not depend on the length
        assert block(torch.randn(T, 1, 64)).shape == (T, 1, 64)

    print(f"Conv2d(3, 64, 11x11):          {count(conv):,} = 64 x (3 x 11 x 11) + 64")
    print(f"LSTM(32 -> 64):                {count(lstm):,} = 4 x 64 x (32 + 64) = {lstm_weights:,} weights + 512 biases")
    print(f"Transformer block (d=64, h=4): {count(block):,} = 4 x 64^2 + 2 x 64 x 128 = "
          f"{attn_weights + mlp_weights:,} weights + {biases_and_ln} biases and LayerNorm")
