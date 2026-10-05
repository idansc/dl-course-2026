"""Precompute real data for the week 5-7 clips (run once; outputs land next to this file).
  w05_weights.npz      : SmolLM2-135M layer 10 mlp.down_proj: 4096 sampled weights, one group of 128, int-k errors
  w06_collapse.json    : tiny real run: siamese (no stop-grad) vs BYOL-style EMA teacher, 2-D unit outputs
  w07_clip.json + w07_img*.png : CLIP ViT-B/32 similarities on 5 product photos + a zero-shot query
"""
import json, copy, io
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

D = Path(__file__).parent
HUB = Path.home() / ".cache/huggingface/hub"


def quant(w, bits, group=None):
    """symmetric absmax round-to-nearest; one scale per tensor or per group of `group` weights"""
    q = 2 ** (bits - 1) - 1
    if group is None:
        s = np.abs(w).max() / q
        return np.round(w / s) * s
    g = w.reshape(-1, group)
    s = np.abs(g).max(1, keepdims=True) / q
    return (np.round(g / s) * s).reshape(w.shape)


def weights():
    from safetensors import safe_open
    f = next(HUB.glob("models--HuggingFaceTB--SmolLM2-135M/snapshots/*/model.safetensors"))
    with safe_open(str(f), "pt") as s:
        W = s.get_tensor("model.layers.10.mlp.down_proj.weight").float().numpy()   # (576, 1536)
    flat = W.flatten()
    rng = np.random.default_rng(0)
    sample = rng.choice(flat, 4096, replace=False)
    group = W[7, 256:384]                                     # one contiguous group of 128 weights in a row
    bits = list(range(2, 9))
    rel = lambda a, b: float(np.sqrt(((a - b) ** 2).mean()) / np.sqrt((b ** 2).mean()))
    err_tensor = [rel(quant(flat, b), flat) for b in bits]
    err_group = [rel(quant(flat, b, 128), flat) for b in bits]
    np.savez(D / "w05_weights.npz", sample=sample, group=group, absmax=np.abs(flat).max(), std=flat.std(),
             bits=bits, err_tensor=err_tensor, err_group=err_group, shape=W.shape)
    print("weights", W.shape, "std", flat.std(), "absmax", np.abs(flat).max(), "group absmax", np.abs(group).max())
    print("err tensor", np.round(err_tensor, 4)); print("err group", np.round(err_group, 4))


def collapse():
    torch.manual_seed(0)
    N = 512
    ang = torch.rand(N) * 2 * np.pi
    X = torch.stack([torch.cos(ang), torch.sin(ang)], 1) * (1 + 0.3 * torch.randn(N, 1)).abs() + 0.05 * torch.randn(N, 2)
    aug = lambda x: x * (1 + 0.2 * torch.randn(x.shape[0], 1)) + 0.15 * torch.randn_like(x)
    mlp = lambda i, o: nn.Sequential(nn.Linear(i, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, o))
    show = torch.arange(0, N, 4)          # 128 points to draw
    snaps = [0, 2, 5, 10, 20, 40, 70, 100, 150, 200, 300, 400, 600, 800, 1000, 1250, 1500]
    res = {"input_angle": ang[show].tolist(), "snap_steps": snaps}
    for mode in ["siam", "byol"]:
        torch.manual_seed(1)
        f = mlp(2, 2)
        pred = mlp(2, 2) if mode == "byol" else None
        params = list(f.parameters()) + (list(pred.parameters()) if pred else [])
        opt = torch.optim.SGD(params, lr=1e-2, momentum=0.9)
        tgt = copy.deepcopy(f)
        spread, angles = [], []
        for s in range(1501):
            with torch.no_grad():
                z = F.normalize(f(X), dim=1)
                if s % 10 == 0:
                    spread.append(round(1 - z.mean(0).norm().item(), 4))
                if s in snaps:
                    zz = z[show]
                    angles.append(torch.atan2(zz[:, 1], zz[:, 0]).tolist())
            v1, v2 = aug(X), aug(X)
            if mode == "siam":
                loss = -F.cosine_similarity(f(v1), f(v2)).mean()
            else:
                p1, p2 = pred(f(v1)), pred(f(v2))
                with torch.no_grad():
                    t1, t2 = tgt(v1), tgt(v2)
                loss = -(F.cosine_similarity(p1, t2).mean() + F.cosine_similarity(p2, t1).mean()) / 2
            opt.zero_grad(); loss.backward(); opt.step()
            if mode == "byol":
                with torch.no_grad():
                    for a, b in zip(tgt.parameters(), f.parameters()):
                        a.mul_(0.99).add_(b, alpha=0.01)
        res[mode] = {"spread": spread, "angles": angles}
        print(mode, spread[::15])
    (D / "w06_collapse.json").write_text(json.dumps(res))


def clip():
    import pandas as pd
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    pq = sorted(HUB.glob("datasets--ashraq--fashion-product-images-small/snapshots/*/data/*.parquet"))[0]
    df = pd.read_parquet(pq)
    picks = [("Watches", "a wrist watch"), ("Sunglasses", "a pair of sunglasses"), ("Handbags", "a handbag"),
             ("Sports Shoes", "a running shoe"), ("Tshirts", "a t-shirt")]
    rows = []
    for art, _ in picks:
        rows.append(df[df.articleType == art].iloc[3])
    query = df[df.articleType == "Backpacks"].iloc[2]
    imgs = [Image.open(io.BytesIO(r.image["bytes"])).convert("RGB") for r in rows]
    qimg = Image.open(io.BytesIO(query.image["bytes"])).convert("RGB")
    caps = [f"a photo of {c}" for _, c in picks]
    classes = ["wrist watch", "backpack", "handbag", "running shoe", "t-shirt", "pair of sunglasses"]
    prompts = [f"a photo of a {c}" for c in classes]
    name = "openai/clip-vit-base-patch32"
    m, p = CLIPModel.from_pretrained(name).eval(), CLIPProcessor.from_pretrained(name)
    with torch.no_grad():
        emb = lambda o: F.normalize(o if torch.is_tensor(o) else o.pooler_output, dim=-1)
        ie = emb(m.get_image_features(**p(images=imgs + [qimg], return_tensors="pt")))
        te = emb(m.get_text_features(**p(text=caps + prompts, return_tensors="pt", padding=True)))
    S = (ie[:5] @ te[:5].T).numpy()
    Z = (ie[5:] @ te[5:].T).numpy()[0]
    scale = m.logit_scale.exp().item()
    for i, im in enumerate(imgs + [qimg]):
        im.save(D / f"w07_img{i}.png")
    out = {"captions": caps, "names": [r.productDisplayName for r in rows], "S": S.tolist(), "logit_scale": scale,
           "zs_classes": classes, "zs_prompts": prompts, "zs_cos": Z.tolist(), "query_name": query.productDisplayName}
    (D / "w07_clip.json").write_text(json.dumps(out, indent=1))
    print(np.round(S, 3)); print(np.round(Z, 3), scale)


if __name__ == "__main__":
    import sys
    for fn in (sys.argv[1:] or ['weights', 'collapse', 'clip']):
        globals()[fn]()
