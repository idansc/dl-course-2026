"""Code for the lecture-3 clip AutoregressiveDecoding (slide "GPT-2 showed - autoregressive decoding is all you need"):
the real GPT-2 (124M parameters, Radford et al. 2019) continues "Deep learning is" one token at a time. Each step is one
forward pass over everything so far -> a softmax over the 50,257 tokens of the vocabulary -> the most likely token
(greedy decoding) -> appended to the input -> repeat. Writes the top-5 tokens and their probabilities at every step to
data/gpt2_decoding.json (the clip draws exactly these numbers). Needs `transformers`; GPT-2 is downloaded once (500 MB).

Run:  python gpt2_decoding.py
Try:  sampling instead of the argmax (torch.multinomial(p, 1)), a temperature (logits / T), or 40 steps: greedy
      decoding soon repeats itself.
"""
import json
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

PROMPT, STEPS, TOP = "Deep learning is", 9, 5
OUT = Path(__file__).with_name("data") / "gpt2_decoding.json"

if __name__ == "__main__":
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    ids = tok(PROMPT, return_tensors="pt").input_ids                      # 1 x n
    res = {"model": "gpt2 (124M)", "vocab": tok.vocab_size, "prompt": [tok.decode(t) for t in ids[0]], "steps": []}
    for step in range(STEPS):
        with torch.no_grad():
            p = model(ids).logits[0, -1].softmax(-1)                    # next-token distribution, 50,257 numbers
        top = p.topk(TOP)
        nxt = top.indices[0]                                            # greedy: the most likely token
        res["steps"].append({"top": [[tok.decode(i), round(v.item(), 4)] for v, i in zip(top.values, top.indices)],
                             "chosen": tok.decode(nxt)})
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)                   # append it and run again
        print(f"step {step + 1}: " + "  ".join(f"{t!r} {v:.2f}" for t, v in res["steps"][-1]["top"]))
    res["text"] = tok.decode(ids[0])
    OUT.write_text(json.dumps(res, indent=1))
    print(res["text"])
    print("saved", OUT)
