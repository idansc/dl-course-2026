"""Code for the lecture-4 slide "The attention sink" (replaces the SpAtten heatmap): where the real GPT-2 (124M
parameters, 12 layers x 12 heads) looks, layer by layer.

For layer l and head h, attention is a causal n x n matrix: row t = how query t splits its attention over keys 0..t
(the row sums to 1). The slide shows, for every layer, the row of the LAST token ("papers"), averaged over the 12
heads: how it splits its attention over the whole sentence. From layer 4 on, about half to three quarters of it goes
to the first token ("He"), whatever that token says: the attention sink (Xiao et al., ICLR 2024). Also saved: the share
of ALL queries' attention that lands on the first token, per layer. Writes data/attention_sink.json (the slide draws
exactly these numbers).

Run:  python attention_sink.py
Try:  another sentence: the sink stays on whatever token comes first.
"""
import json
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

SENTENCE = "He is a very famous researcher in the computer architecture area and has published many papers"
OUT = Path(__file__).with_name("data") / "attention_sink.json"

if __name__ == "__main__":
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
    ids = tok(SENTENCE, return_tensors="pt").input_ids                    # 1 x n
    with torch.no_grad():
        att = model(ids, output_attentions=True).attentions               # 12 tensors, each 1 x 12 heads x n x n
    A = torch.stack(att)[:, 0].mean(1)                                    # layers x n x n, averaged over heads
    last = A[:, -1, :]                                                    # layers x n: where the last token looks
    first_share = A[:, :, 0].mean(1)                                      # all queries: share on the first token
    tokens = [tok.decode(t).strip() for t in ids[0]]
    res = {"model": "gpt2 (124M)", "sentence": SENTENCE, "tokens": tokens,
           "last_query": [[round(v, 4) for v in row.tolist()] for row in last],
           "first_token_share": [round(v, 3) for v in first_share.tolist()]}
    OUT.write_text(json.dumps(res, indent=1))
    for l in range(len(last)):
        print(f"layer {l + 1:2d}: '{tokens[-1]}' puts {last[l, 0]:.0%} of its attention on '{tokens[0]}'; "
              f"all queries together {first_share[l]:.0%}")
    print("saved", OUT)
