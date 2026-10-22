"""Code for the clip ConvRunningExample (renders/w01/ConvRunningExample.mp4): the "Running example" slide of the
CNN part of lecture 2 (the classic UFLDL convolution schematic). A 3x3 filter slides over a 5x5 binary image one
step at a time (stride 1, no padding); each output is the sum of value x weight under the window. Same numbers as
the slide and the clip.

Run:  python conv_running_example.py   -> prints the 3x3 output and the check against PyTorch,
                                           saves figures/conv_running_example.png
Try:  conv2d(image, kernel, stride=2)  -> 2x2 output [[4, 4], [2, 4]] (every second position);
      conv2d(image, kernel, padding=1) -> 5x5 output, the same size as the input (zeros around the border).
"""
from pathlib import Path
import numpy as np

image = np.array([[1, 1, 1, 0, 0],
                  [0, 1, 1, 1, 0],
                  [0, 0, 1, 1, 1],
                  [0, 0, 1, 1, 0],
                  [0, 1, 1, 0, 0]])
kernel = np.array([[1, 0, 1],
                   [0, 1, 0],
                   [1, 0, 1]])


def conv2d(image, kernel, stride=1, padding=0):
    """Slide the kernel over the image; each output = sum of value x weight under the window.
    The kernel is not flipped: this is cross-correlation, which is what the slide, PyTorch's conv2d and every
    deep-learning library call convolution."""
    image = np.pad(image, padding)                      # `padding` rings of zeros around the image
    H, W = image.shape
    K = kernel.shape[0]
    H_out = (H - K) // stride + 1                       # (N - K) / stride + 1
    W_out = (W - K) // stride + 1
    out = np.zeros((H_out, W_out), dtype=image.dtype)
    for i in range(H_out):                              # output row
        for j in range(W_out):                          # output column
            total = 0
            for a in range(K):                          # row inside the window
                for b in range(K):                      # column inside the window
                    total += image[i * stride + a, j * stride + b] * kernel[a, b]     # value x weight
            out[i, j] = total
    return out


def check_with_torch(image, kernel, stride=1, padding=0):
    """Our loops vs torch.nn.functional.conv2d (also cross-correlation). Returns both outputs."""
    import torch
    import torch.nn.functional as F
    x = torch.tensor(image, dtype=torch.float32)[None, None]           # (batch 1, channel 1, H, W)
    w = torch.tensor(kernel, dtype=torch.float32)[None, None]          # (out channels 1, in channels 1, K, K)
    theirs = F.conv2d(x, w, stride=stride, padding=padding)[0, 0].numpy()
    return conv2d(image, kernel, stride, padding), theirs


def figure(path):
    """Static version of the clip: the window at the first position, the filter, the full output."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, green, red, weight, ink = "#C1E4F7", "#C7E8AC", "#FFBBB1", "#166534", "#1F2937"
    out = conv2d(image, kernel)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    def grid(x0, y0, n, size, fill, text, color, fontsize, name):
        """n x n cells, top-left corner at (x0, y0), name underneath"""
        for r in range(n):
            for c in range(n):
                ax.add_patch(plt.Rectangle((x0 + c * size, y0 - (r + 1) * size), size, size, facecolor=fill(r, c),
                                           edgecolor=ink, lw=1.2))
                ax.text(x0 + (c + 0.5) * size, y0 - (r + 0.5) * size, text(r, c), ha="center", va="center",
                        fontsize=fontsize, color=color)
        ax.text(x0 + n * size / 2, y0 - n * size - 0.3, name, ha="center", va="center", fontsize=11, color="#6B7280")

    inside = lambda r, c: r < 3 and c < 3                    # the window at its first position
    grid(0, 5, 5, 1, lambda r, c: green if inside(r, c) else blue,
         lambda r, c: f"{image[r, c]}×{kernel[r, c]}" if inside(r, c) else str(image[r, c]), ink, 13, "input")
    ax.add_patch(plt.Rectangle((0, 2), 3, 3, fill=False, edgecolor="#16A34A", lw=3))
    grid(7.75, 5.25, 3, 0.5, lambda r, c: green, lambda r, c: str(kernel[r, c]), weight, 11, "filter")
    ax.add_patch(plt.Rectangle((7.75, 3.75), 1.5, 1.5, fill=False, edgecolor="#16A34A", lw=2))
    grid(7, 3, 3, 1, lambda r, c: red, lambda r, c: str(out[r, c]), ink, 15, "output")
    ax.set_xlim(-0.1, 10.1)
    ax.set_ylim(-0.55, 5.35)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    ours, theirs = check_with_torch(image, kernel)
    print("output (stride 1, no padding):")
    print(ours)
    print(f"torch.nn.functional.conv2d: same output: {np.array_equal(ours, theirs)} "
          f"(max |difference| {np.abs(ours - theirs).max():g})")
    out = Path(__file__).with_name("figures") / "conv_running_example.png"
    out.parent.mkdir(exist_ok=True)
    figure(out)
    print("saved", out)
