"""Code for the lecture-4 slides on what a Vision Transformer learns: a real ViT-B/16 (google/vit-base-patch16-224,
pre-trained on ImageNet-21k, fine-tuned on ImageNet-1k at 224 x 224) on 130 ImageNet validation images (Imagenette
subset, 13 per class). Same analyses as Dosovitskiy et al. (ICLR 2021), Figures 6, 7 and 11, which used ViT-L/32 and
ViT-L/16 trained on the non-public JFT-300M.

1. Patch-embedding filters: the first 28 principal components of the 768 filters (each 16 x 16 x 3)
   -> figures/vit/filters.png
2. Position embeddings: cosine similarity between the position vector of each patch and those of all 196 patches
   (one 14 x 14 tile per patch) -> figures/vit/posemb.png
3. Mean attention distance, for each of the 12 heads in each of the 12 layers: for every query patch, the distance in
   pixels to every key patch, weighted by the attention weight (attention over the 196 patches, renormalized without
   [CLS]), averaged over the query patches and the images -> data/vit_inspect.json (the slide draws these numbers)
4. Attention rollout (Abnar and Zuidema, ACL 2020): average the heads, add the residual (0.5 A + 0.5 I), renormalize,
   multiply the 12 layers; the [CLS] row says where the class token reads from -> figures/vit/rollout_*.png.
   Rollout is the same for every class; the class-specific relevance of Chefer, Gur and Wolf (ICCV 2021) weights each
   attention map by the gradient of one class score -> figures/vit/relevance_{dog,bird}.png (a springer and a hen)
Also writes the tiles of the example image for the ViT diagram (figures/vit/example*.png).

Data: Imagenette 320 px validation images (fast.ai):
  curl -L https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz | tar xz -C data --include 'imagenette2-320/val/*'
Run:  python vit_inspect.py
"""
import json
from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image, ImageDraw

from vit_minimal import CLASSES, images

HERE = Path(__file__).parent
OUT = HERE / "data" / "vit_inspect.json"
FIGS = HERE / "figures" / "vit"
EXAMPLE = ("n02102040", "ILSVRC2012_val_00007032.JPEG")      # an English springer, for the ViT diagram
ROLLOUT = [("n02102040", 8), ("n03028079", 2), ("n03417042", 6), ("n03888257", 5)]
TWO = ("n02102040", 8)                                       # a springer and a hen in one photo


def to224(im):
    return im.resize((224, 224), Image.BICUBIC)


def save_filters(W, path, n=28, cols=7, scale=6, gap=6):
    """W: 768 filters x 3 x 16 x 16. The principal components of the filters, each scaled to [0, 1]."""
    X = W.reshape(W.shape[0], -1).numpy()
    X = X - X.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    rows = n // cols
    tile = 16 * scale
    canvas = Image.new("RGB", (cols * tile + (cols - 1) * gap, rows * tile + (rows - 1) * gap), "white")
    for k in range(n):
        v = Vt[k].reshape(3, 16, 16).transpose(1, 2, 0)
        v = (v - v.min()) / (v.max() - v.min())
        im = Image.fromarray((v * 255).astype(np.uint8)).resize((tile, tile), Image.NEAREST)
        canvas.paste(im, ((k % cols) * (tile + gap), (k // cols) * (tile + gap)))
    canvas.save(path)


def save_posemb(pos, path, every=2, cell=8, gap=10):
    """pos: 196 x D position vectors (no [CLS]). Tile (r, c) = cosine similarity of patch (r, c) with every patch;
    every second row and column of tiles is drawn (7 x 7 of the 14 x 14), as in the paper's Figure 7."""
    P = torch.nn.functional.normalize(pos, dim=1)
    S = (P @ P.T).numpy()                                                  # 196 x 196
    cmap = colormaps["viridis"]
    t, keep = 14 * cell, range(0, 14, every)
    n = len(keep)
    canvas = Image.new("RGB", (n * t + (n - 1) * gap, n * t + (n - 1) * gap), "white")
    for a, r in enumerate(keep):
        for b, c in enumerate(keep):
            rgb = (cmap((S[r * 14 + c].reshape(14, 14) + 1) / 2)[..., :3] * 255).astype(np.uint8)
            canvas.paste(Image.fromarray(rgb).resize((t, t), Image.NEAREST), (b * (t + gap), a * (t + gap)))
    canvas.save(path)
    near = [float(S[i, j]) for i in range(196) for j in range(196)
            if abs(i // 14 - j // 14) + abs(i % 14 - j % 14) == 1]
    far = [float(S[i, j]) for i in range(196) for j in range(196)
           if abs(i // 14 - j // 14) >= 7 and abs(i % 14 - j % 14) >= 7]
    return float(np.mean(near)), float(np.mean(far))


def rollout(att):
    """att: list of 12 tensors (heads x 197 x 197) for one image -> 14 x 14 map of where [CLS] reads."""
    R = torch.eye(197)
    for A in att:
        A = 0.5 * A.mean(0) + 0.5 * torch.eye(197)
        A = A / A.sum(-1, keepdim=True)
        R = A @ R
    m = R[0, 1:].reshape(14, 14)
    return ((m - m.min()) / (m.max() - m.min())).numpy()


def relevance(model, x, target):
    """Class-specific relevance (Chefer, Gur and Wolf, ICCV 2021): R = I; per layer R += mean_heads[(grad * A)+] R,
    with the gradient of the target class score w.r.t. the attention map A. [CLS] row -> 14 x 14 map."""
    out = model(pixel_values=x, output_attentions=True)
    grads = torch.autograd.grad(out.logits[0, target], out.attentions)
    R = torch.eye(197)
    for A, G in zip(out.attentions, grads):
        R = R + (G[0] * A[0]).clamp(min=0).mean(0).detach() @ R
    m = R[0, 1:].reshape(14, 14)
    return ((m - m.min()) / (m.max() - m.min())).numpy()


def overlay(img224, m, path):
    """Heat map on the image, as in his relevance-map slides: jet colours + image, rescaled to the maximum."""
    heat = Image.fromarray((m * 255).astype(np.uint8)).resize((224, 224), Image.BICUBIC)
    jet = colormaps["jet"](np.asarray(heat) / 255.0)[..., :3]
    cam = jet + np.asarray(img224) / 255.0
    Image.fromarray((cam / cam.max() * 255).astype(np.uint8)).save(path)


if __name__ == "__main__":
    from transformers import ViTForImageClassification, ViTImageProcessor
    FIGS.mkdir(parents=True, exist_ok=True)
    name = "google/vit-base-patch16-224"
    model = ViTForImageClassification.from_pretrained(name, attn_implementation="eager").eval()
    proc = ViTImageProcessor.from_pretrained(name)
    sd = model.state_dict()

    # 1-2: weights only
    save_filters(sd["vit.embeddings.patch_embeddings.projection.weight"], FIGS / "filters.png")
    near, far = save_posemb(sd["vit.embeddings.position_embeddings"][0, 1:], FIGS / "posemb.png")

    # 3: mean attention distance per layer and head
    data = images()
    centers = torch.tensor([((i // 14) + 0.5, (i % 14) + 0.5) for i in range(196)]) * 16
    dist = torch.cdist(centers, centers)                                 # 196 x 196, pixels
    total = torch.zeros(12, 12)
    cls_share = torch.zeros(12)
    maps = {}
    correct = 0
    for k in range(0, len(data), 26):
        chunk = data[k:k + 26]
        x = proc(images=[im for im, _, _ in chunk], return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            out = model(pixel_values=x, output_attentions=True)
        correct += (out.logits.argmax(1) == torch.tensor([y for _, y, _ in chunk])).sum().item()
        att = torch.stack(out.attentions, 1)                             # B x 12 layers x 12 heads x 197 x 197
        P = att[..., 1:, 1:]
        P = P / P.sum(-1, keepdim=True)
        total += (P * dist).sum(-1).mean(-1).sum(0)                      # 12 x 12, summed over the batch
        cls_share += att[..., 1:, 0].mean((-1, -2)).sum(0)               # how much patches read [CLS], per layer
        for b, (_, _, fname) in enumerate(chunk):
            maps[fname] = (rollout([a[b] for a in out.attentions]), out.logits[b].argmax().item())
    mad = total / len(data)
    res = {"model": name, "images": len(data), "top1_accuracy": correct / len(data),
           "mean_attention_distance_px": [[round(v, 2) for v in row] for row in mad.tolist()],   # [layer][head]
           "patch_reads_cls_share": [round(v, 4) for v in (cls_share / len(data)).tolist()],
           "posemb_cosine_neighbours": round(near, 3), "posemb_cosine_far": round(far, 3),
           "image_width_px": 224}

    # 4: rollout figures (class-agnostic), and class-specific relevance for the dog-and-hen image
    by_class = {}
    for im, y, fname in data:
        by_class.setdefault(y, []).append((im, fname))
    picked = []
    for wnid, j in ROLLOUT:
        im, fname = by_class[CLASSES[wnid]][j]
        m, pred = maps[fname]
        overlay(to224(im), m, FIGS / f"rollout_{wnid}.png")
        to224(im).save(FIGS / f"input_{wnid}.png")
        picked.append({"wnid": wnid, "file": fname, "pred": model.config.id2label[pred]})
    res["rollout_images"] = picked
    im, fname = by_class[CLASSES[TWO[0]]][TWO[1]]
    x = proc(images=[im], return_tensors="pt")["pixel_values"]
    with torch.no_grad():
        logits = model(pixel_values=x).logits[0]
    top5 = logits.softmax(0).topk(5)
    bird = max((7, 8), key=lambda c: logits[c].item())                    # ImageNet 7 = cock, 8 = hen
    res["two_objects"] = {"file": fname, "top5": [(model.config.id2label[i], round(p, 3))
                                                   for p, i in zip(top5.values.tolist(), top5.indices.tolist())],
                          "classes": {"dog": model.config.id2label[217], "bird": model.config.id2label[bird]}}
    overlay(to224(im), relevance(model, x, 217), FIGS / "relevance_dog.png")
    overlay(to224(im), relevance(model, x, bird), FIGS / "relevance_bird.png")

    ex = to224(Image.open(HERE / "data" / "imagenette2-320" / "val" / EXAMPLE[0] / EXAMPLE[1]).convert("RGB"))
    ex.save(FIGS / "example.png")
    grid = ex.copy()
    d = ImageDraw.Draw(grid)
    for t in range(1, 14):
        d.line([(t * 16, 0), (t * 16, 224)], fill=(255, 255, 255), width=1)
        d.line([(0, t * 16), (224, t * 16)], fill=(255, 255, 255), width=1)
    grid.save(FIGS / "example_grid.png")
    for r in range(3):                                                   # a 3 x 3 cut for the diagram (illustration)
        for c in range(3):
            ex.crop((c * 224 // 3, r * 224 // 3, (c + 1) * 224 // 3, (r + 1) * 224 // 3)).save(
                FIGS / f"example_tile_{r}{c}.png")

    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "mean_attention_distance_px"}, indent=1))
    for l, row in enumerate(mad.tolist()):
        print(f"layer {l + 1:2d}: min {min(row):5.1f}  median {sorted(row)[6]:5.1f}  max {max(row):5.1f} px")
    print("saved", OUT, "and", FIGS)
