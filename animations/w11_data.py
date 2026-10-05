"""Precompute real model outputs for w11_post_training.py (writes data/w11_*.json). Needs the HF cache
(SmolLM2-135M, SmolLM2-135M-Instruct, Qwen2.5-0.5B-Instruct); runs on CPU.

    python w11_data.py pipeline   # one prompt through a base model and its instruct (SFT + DPO) version
    python w11_data.py grpo       # G=8 sampled completions per prompt, verifiable rewards, one real GRPO step
"""
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA = Path(__file__).parent / "data"
torch.set_num_threads(6)


def chat_ids(tok, prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                   return_tensors="pt", return_dict=True)["input_ids"]


def final_number(text):
    nums = re.findall(r"-?\d[\d,]*", text.replace("\\", " "))
    return nums[-1].replace(",", "") if nums else None


def pipeline():
    prompt = "What is 17 × 24?"
    out = {"prompt": prompt}
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    m = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    ids = tok(prompt, return_tensors="pt").input_ids
    g = m.generate(ids, max_new_tokens=40, do_sample=False)
    out["base"] = tok.decode(g[0][ids.shape[1]:], skip_special_tokens=True)
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    m = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    ids = chat_ids(tok, prompt)
    g = m.generate(ids, max_new_tokens=60, do_sample=False)
    out["instruct"] = tok.decode(g[0][ids.shape[1]:], skip_special_tokens=True)
    print(json.dumps(out, indent=1))
    (DATA / "w11_pipeline.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


def seq_logprob(m, ids, n_prompt):
    """Mean per-token log-prob of the completion part, per row (padding = -100 labels)."""
    logits = m(ids["input_ids"], attention_mask=ids["attention_mask"]).logits[:, :-1]
    lab = ids["input_ids"][:, 1:]
    lp = torch.log_softmax(logits.float(), -1).gather(-1, lab[..., None])[..., 0]
    mask = ids["attention_mask"][:, 1:].clone().float()
    mask[:, : n_prompt - 1] = 0
    return (lp * mask).sum(1) / mask.sum(1), (lp * mask).sum(1)


def grpo(G=8, seed=0):
    name = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "right"
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    res = {}
    for key, prompt, answer in [("mixed", "What is 17 × 24? End with the final number.", "408"),
                                ("hard", "What is 7919 × 6421? End with the final number.", str(7919 * 6421))]:
        torch.manual_seed(seed)
        ids = chat_ids(tok, prompt)
        n_p = ids.shape[1]
        gen = m.generate(ids.repeat(G, 1), max_new_tokens=160, do_sample=True, temperature=1.0, top_p=1.0,
                         pad_token_id=tok.eos_token_id)
        comps = [tok.decode(g[n_p:], skip_special_tokens=True) for g in gen]
        finals = [final_number(c) for c in comps]
        r = np.array([1.0 if f == answer else 0.0 for f in finals])
        trunc = [bool((g[n_p:] == tok.eos_token_id).sum() == 0) for g in gen]   # hit max_new_tokens: no final answer
        res[key] = {"prompt": prompt, "answer": answer, "completions": comps, "finals": finals, "rewards": r.tolist(),
                    "truncated": trunc}
        print(key, finals, r)
        # one real GRPO step (single on-policy step, so the PPO ratio is 1 and clipping is inactive):
        #   L = -(1/G) sum_i A_i * mean_t log pi(y_i,t | x, y_i,<t);  plain SGD, step size picked from a small sweep
        if r.std() > 0:
            A = (r - r.mean()) / (r.std() + 1e-6)
            am = torch.ones_like(gen)                     # exact sampled tokens; mask everything after the first EOS
            for i in range(G):
                eos = (gen[i, n_p:] == tok.eos_token_id).nonzero()
                if len(eos):
                    am[i, n_p + eos[0, 0] + 1:] = 0
            batch = {"input_ids": gen, "attention_mask": am}
            lp_mean, lp_sum = seq_logprob(m, batch, n_p)
            loss = -(torch.tensor(A, dtype=torch.float32) * lp_mean).mean()
            m.zero_grad(); loss.backward()
            params = [q for q in m.parameters() if q.grad is not None]
            sweep = {}
            for lr in [1e-5, 3e-5, 1e-4, 3e-4]:
                with torch.no_grad():
                    for q in params: q.sub_(lr * q.grad)
                    new_mean, new_sum = seq_logprob(m, batch, n_p)
                    for q in params: q.add_(lr * q.grad)
                dl = (new_mean - lp_mean).detach().numpy()
                sweep[lr] = (new_mean.tolist(), new_sum.tolist())
                print("lr", lr, "dlogp/token", np.round(dl, 3), "sign agreement", np.mean(np.sign(dl) == np.sign(A)))
            res[key].update(adv=A.tolist(), logp_before=lp_sum.detach().tolist(), logp_tok_before=lp_mean.detach().tolist(),
                            sweep={str(k): v for k, v in sweep.items()})
    (DATA / "w11_grpo.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    what = sys.argv[1:] or ["pipeline", "grpo"]
    if "pipeline" in what: pipeline()
    if "grpo" in what: grpo()
