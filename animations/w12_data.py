"""Precompute retrieval data for w12_agents.py (writes data/w12_retrieval.json). Uses the HF cache:
sentence-transformers/all-MiniLM-L6-v2 (dense, mean pooling) and answerdotai/answerai-colbert-small-v1 (ColBERT).

    python w12_data.py
"""
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

DATA = Path(__file__).parent / "data"
QUERY = "how do I fix a flat tire"
PASSAGES = [
    "Repairing a punctured bicycle wheel: remove the tube, find the hole, patch it.",
    "Flat tire prices: compare tire deals and flat-rate shipping.",
    "How to change a car tyre on the roadside in ten minutes.",
    "Slow leak? Patch kits and sealants seal small holes in tubes.",
    "The history of the Michelin company and its mascot.",
    "Flat-pack furniture: how do I fix a wobbly table?",
    "Best road bikes of 2026, reviewed.",
    "Baking sourdough bread at home.",
]
SHORT = ["repairing a punctured bicycle wheel", "flat tire prices, flat-rate shipping", "change a car tyre on the roadside",
         "patch kits for slow leaks", "history of Michelin", "flat-pack furniture: fix a wobbly table",
         "best road bikes of 2026", "baking sourdough bread"]


def words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def bm25(query, docs, k1=1.2, b=0.75):
    toks = [words(d) for d in docs]
    N, avg = len(docs), np.mean([len(t) for t in toks])
    df = Counter(w for t in toks for w in set(t))
    out, detail = [], []
    for t in toks:
        tf = Counter(t)
        s, parts = 0.0, {}
        for q in words(query):
            if tf[q] == 0:
                continue
            idf = math.log(1 + (N - df[q] + 0.5) / (df[q] + 0.5))
            v = idf * tf[q] * (k1 + 1) / (tf[q] + k1 * (1 - b + b * len(t) / avg))
            parts[q] = round(v, 3)
            s += v
        out.append(s)
        detail.append(parts)
    return out, detail


def dense():
    name = "sentence-transformers/all-MiniLM-L6-v2"
    tok, m = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).eval()
    enc = tok([QUERY] + PASSAGES, padding=True, return_tensors="pt")
    with torch.no_grad():
        h = m(**enc).last_hidden_state
    mask = enc["attention_mask"][..., None].float()
    e = torch.nn.functional.normalize((h * mask).sum(1) / mask.sum(1), dim=-1).numpy()
    return (e[1:] @ e[0]).tolist(), e.shape[1]


def colbert(query, passage):
    name = "answerdotai/answerai-colbert-small-v1"
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModel.from_pretrained(name).eval()
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
    f = safe_open(hf_hub_download(name, "model.safetensors"), "pt")
    W = f.get_tensor("linear.weight")                                   # 96 x 384 projection

    def enc(text, marker):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        ids = [tok.cls_token_id, tok.convert_tokens_to_ids(marker)] + ids + [tok.sep_token_id]
        with torch.no_grad():
            h = m(torch.tensor([ids])).last_hidden_state[0]
        e = torch.nn.functional.normalize(h @ W.T, dim=-1)
        keep = [i for i, t in enumerate(ids) if i >= 2 and t != tok.sep_token_id
                and re.match(r"^[#a-z0-9]+$", tok.convert_ids_to_tokens(t))]
        return [tok.convert_ids_to_tokens(ids[i]) for i in keep], e[keep].numpy()

    qt, qe = enc(query, "[unused0]")
    pt, pe = enc(passage, "[unused1]")
    S = qe @ pe.T
    return {"q_tokens": qt, "p_tokens": pt, "sim": S.round(4).tolist(), "maxsim": float(S.max(1).sum())}


if __name__ == "__main__":
    cos, dim = dense()
    bm, det = bm25(QUERY, PASSAGES)
    out = {"query": QUERY, "passages": PASSAGES, "short": SHORT, "cos": cos, "dim": dim, "bm25": bm, "bm25_terms": det,
           "colbert": [colbert("how do I fix a flat tire", PASSAGES[0]), colbert("how do I fix a flat tire", PASSAGES[1])]}
    for s, c, b in zip(SHORT, cos, bm):
        print(f"{c:6.3f} {b:6.3f}  {s}")
    for c in out["colbert"]:
        print(c["q_tokens"], c["p_tokens"], round(c["maxsim"], 3))
    (DATA / "w12_retrieval.json").write_text(json.dumps(out, indent=1))
