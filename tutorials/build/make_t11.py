"""Builds T11_post_training.ipynb (Tutorial 11, Deep Learning 89-6877, Fall 2026)."""
from nbkit import Builder
from recap import make_recap

STEM = "T11_post_training"
B = Builder()
md, code = B.md, B.code

md(r"""
# Tutorial 11: post-training a small LM: LoRA, SFT, DPO, GRPO
**Deep Learning 89-6877, Fall 2026.** TA: Tal Fiskus.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idansc/dl-course-2026/blob/main/tutorials/T11_post_training.ipynb)
Recap slides: [T11_post_training_recap.pdf](https://github.com/idansc/dl-course-2026/blob/main/tutorials/recap/T11_post_training_recap.pdf)

Plan for today (≈ 75 min):
1. Lecture recap: scaling laws, SFT, LoRA, RLHF/DPO, policy gradients → PPO → GRPO, reasoning (12 min)
2. LoRA from scratch: merge check, injection into SmolLM2, comparison with `peft` (12 min)
3. SFT with loss masking on prompt tokens (10 min)
4. DPO from policy and reference log-probs (10 min)
5. From policy gradient to GRPO, one line at a time, then GRPO with a verifiable reward (20 min)
6. Reward hacking: optimizing against a flawed verifier (10 min)
7. If time: what the PPO clip does to the gradient

Model: `HuggingFaceTB/SmolLM2-135M-Instruct` (135M parameters). Task: arithmetic questions with a checkable answer.
Everything runs on a laptop CPU in under 10 minutes. Cells marked ✏️ are for you to try.
The printed numbers come from one run on an Apple-silicon GPU (`mps`). Sampling differs across devices, so your numbers will differ by a few points (a CPU run: GRPO 0.48 → 0.70 instead of 0.75); the text notes where a conclusion changed.
<<STUDENT>>
**Before the tutorial:** fill in every `# TODO` (replace the `...`). Each one is followed by a check cell that compares your code with a library result or a direct formula. The solutions are presented in the tutorial.
<</STUDENT>>
""")

code(r"""
import math, random, re, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

def seed_all(s):
    torch.manual_seed(s); random.seed(s); np.random.seed(s)
seed_all(0)
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)
T_START = time.time()
""")

md(r"""
## 1. Lecture recap

**The pipeline.** Pretraining (next-token prediction on web text) → **SFT** (instruction data) → **preference tuning** (RLHF or DPO) → **RLVR** (RL with verifiable rewards, for reasoning).

**Scaling laws.** Test loss is a power law in parameters $N$ and tokens $D$: $\;L(N,D) = E + A/N^{\alpha} + B/D^{\beta}$. Training compute is $C \approx 6ND$ FLOPs. For a fixed $C$, Chinchilla's optimum is $D \approx 20N$. Production models train far past it (cheaper inference per quality), e.g. our model below.

**SFT = behavior cloning.** Maximize $\log \pi_\theta(y\mid x)=\sum_t \log\pi_\theta(y_t\mid x,y_{<t})$ on demonstrations $(x,y)$. The loss is computed on response tokens only; the prompt is context, not a target.

**LoRA.** Freeze $W\in\mathbb{R}^{d\times k}$, learn a low-rank update: $\;h = Wx + \frac{\alpha}{r}BAx$, $B\in\mathbb{R}^{d\times r}$, $A\in\mathbb{R}^{r\times k}$, $r\ll\min(d,k)$. $B=0$ at init, so training starts exactly at the pretrained model. After training, merge: $W' = W + \frac{\alpha}{r}BA$ (zero inference cost). **QLoRA**: the same, with $W$ stored in 4-bit NF4.

**Reward model (Bradley-Terry).** $p(y_w \succ y_l) = \sigma(r(x,y_w) - r(x,y_l))$, trained with $-\log\sigma(r_w - r_l)$.

**RLHF objective.** $\max_\theta\; \mathbb{E}_{y\sim\pi_\theta}[r(x,y)] - \beta\,\mathrm{KL}(\pi_\theta\,\|\,\pi_{\text{ref}})$, optimized with PPO (policy, value model, reward model, reference: 4 models).

**DPO.** The optimum of that objective satisfies $r(x,y) = \beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} + \text{const}$. Plug it into Bradley-Terry:
$$\mathcal{L}_{\text{DPO}} = -\log\sigma\Big(\beta\log\tfrac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\tfrac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\Big).$$
No reward model, no sampling: a supervised loss on preference pairs.

**Policy gradient → PPO → GRPO.** REINFORCE: $\nabla J = \mathbb{E}[(r-b)\nabla\log\pi_\theta(y|x)]$. PPO adds an importance ratio, clipping and a learned value baseline (critic). **GRPO** samples $G$ answers per prompt and uses the group as the baseline: $\hat A_i = (r_i - \text{mean}(r))/\text{std}(r)$. No critic. With a programmatic checker as the reward (math answer, unit tests) this is **RLVR** (DeepSeek-R1). Section 5 derives each step.

**Reasoning and test-time compute.** Models trained with RLVR learn to write a long chain of thought (CoT) before the answer; response length grows during RL. More tokens at test time (longer CoT, best-of-$N$, majority vote) buy accuracy.

**CoT faithfulness and monitoring.** A CoT is text the model produces, not a trace of its computation. Models use hints in the prompt and often do not mention them in the CoT (Turpin et al., 2023; Chen et al., 2025, *Reasoning models don't always say what they think*). Still, a CoT is the one place where intent can be read: a monitor model reading the CoT catches reward hacking far better than one reading only actions (Baker et al., 2025). Putting the monitor into the reward teaches the policy to hide intent while still hacking, so monitorability is fragile (Korbak et al., 2025). Section 6 shows the hacking half on a small scale.
""")

code(r"""
# Where does our model sit relative to Chinchilla? (SmolLM2 model card: 135M params, 2T training tokens)
N, D = 135e6, 2e12
print(f"Chinchilla-optimal tokens for 135M params: {20 * N / 1e9:.1f}B")
print(f"SmolLM2-135M saw {D / 1e12:.0f}T tokens = {D / N:,.0f} tokens per parameter ({D / (20 * N):.0f}x past Chinchilla)")
print(f"training compute C ≈ 6ND = {6 * N * D:.2e} FLOPs")
""")

md(r"""
### The model and the task

Prompts look like `Q: What is 47 plus 38?\nA:`. We want the answer in a fixed, checkable format: ` The answer is 85.` followed by the end token.
`make_problem(kind)` returns `(prompt, correct_answer)`. SFT trains on one-digit additions, subtractions and products; the held-out task for SFT, and the RL task, is **two-digit addition** (`add2`).
""")

code(r"""
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
for p in model.parameters():
    p.requires_grad_(False)
EOS = tok.eos_token_id
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, eos = {tok.eos_token!r}")
print(model.model.layers[0].self_attn)

FMT = "Q: What is {a} {op} {b}?\nA:"
def make_problem(kind):
    if kind == "add1": a, b = random.randint(0, 9), random.randint(0, 9); return FMT.format(a=a, op="plus", b=b), a + b
    if kind == "sub":  a = random.randint(5, 20); b = random.randint(0, a); return FMT.format(a=a, op="minus", b=b), a - b
    if kind == "mul":  a, b = random.randint(2, 9), random.randint(2, 9); return FMT.format(a=a, op="times", b=b), a * b
    if kind == "add2": a, b = random.randint(10, 99), random.randint(10, 99); return FMT.format(a=a, op="plus", b=b), a + b

def target(ans):
    return f" The answer is {ans}."

def parse_answer(text):
    m = re.search(r"The answer is (-?\d+)\.", text)
    return int(m.group(1)) if m else None

@torch.no_grad()
def generate(prompts, sample=False, n=1, max_new=10):
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    kw = dict(do_sample=True, temperature=1.0, top_k=0, top_p=1.0) if sample else dict(do_sample=False)
    out = model.generate(**enc, max_new_tokens=max_new, num_return_sequences=n, pad_token_id=EOS, **kw)
    return out, enc, tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)

def evaluate(kind="add2", n=64, sample=False, seed=123, max_new=10):
    rng_state = random.getstate(); random.seed(seed)
    data = [make_problem(kind) for _ in range(n)]
    random.setstate(rng_state)
    _, _, outs = generate([p for p, _ in data], sample=sample, max_new=max_new)
    fmt = np.mean([parse_answer(o) is not None for o in outs])
    acc = np.mean([parse_answer(o) == a for o, (_, a) in zip(outs, data)])
    return fmt, acc, outs

fmt, acc, outs = evaluate("add2")
print(f"base model on add2: format {fmt:.2f}, accuracy {acc:.2f}; first outputs: {outs[:3]}")
""")

md(r"""
The base instruct model does not follow our format at all (it emits the end token right away). SFT will fix the format; RL will then improve the accuracy.

## 2. LoRA from scratch

A `LoRALinear` wraps a frozen `nn.Linear`. Forward: $\;y = Wx + b + \frac{\alpha}{r}\,B(Ax)$, computed as two thin matmuls (never form $BA$ during training).
Init as in the paper and in `peft`: $A$ Kaiming-uniform, $B=0$. The `enabled` flag switches the adapter off, which gives us the **reference model for free** in sections 4 to 6 (the frozen weights are the reference).
""")

code(r"""
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        dev = base.weight.device
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=dev))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
        self.enabled = True

    def forward(self, x):
        y = self.base(x)
        if self.enabled:
            #>> add the low-rank update: scale * B(Ax), as two matmuls
            y = y + self.scale * (x @ self.A.T) @ self.B.T
            #<<
        return y

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        lin = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None,
                        device=self.base.weight.device)
        #>> W' = W + scale * B @ A (copy the bias too)
        lin.weight.copy_(self.base.weight + self.scale * self.B @ self.A)
        if self.base.bias is not None:
            lin.bias.copy_(self.base.bias)
        #<<
        lin.requires_grad_(False)
        return lin

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")

def inject_lora(model, r=8, alpha=16):
    for layer in model.model.layers:
        attn = layer.self_attn
        for name in TARGETS:
            #>> replace attn.<name> by a LoRALinear wrapping it (setattr / getattr)
            setattr(attn, name, LoRALinear(getattr(attn, name), r, alpha))
            #<<

def remove_lora(model, merge):
    # merge=True: bake BA into W; merge=False: drop the adapter, back to the frozen weights
    for layer in model.model.layers:
        attn = layer.self_attn
        for name in TARGETS:
            m = getattr(attn, name)
            setattr(attn, name, m.merged() if merge else m.base)

def set_lora(model, on):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.enabled = on

def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]
""")

md(r"""
**Check 1.** (a) At init ($B=0$) the LoRA model equals the base model exactly. (b) With a random $B$, the merged model gives the same logits as the unmerged one. (c) The number of trainable parameters.
""")

code(r"""
x_ids = tok(["Q: What is 3 plus 4?\nA: The answer is 7.", "LoRA is a low-rank adapter."], return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    logits_base = model(**x_ids).logits

inject_lora(model)
with torch.no_grad():
    print("(a) init, max |LoRA − base|:", (model(**x_ids).logits - logits_base).abs().max().item())
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.B.normal_(0, 0.02)                  # pretend we trained
    logits_lora = model(**x_ids).logits
print("    the adapter changed the logits by", (logits_lora - logits_base).abs().max().item())

n_lora = sum(p.numel() for p in trainable(model))
n_all = sum(p.numel() for p in model.parameters())
print(f"(c) trainable: {n_lora:,} of {n_all:,} ({100 * n_lora / n_all:.2f}%)")
""")

md(r"""
**Check 2: against `peft`.** Same rank, alpha and target modules; copy our $A,B$ into `peft`'s `lora_A`/`lora_B` and compare logits and parameter counts. Then merge ours and compare again.
""")

code(r"""
from peft import LoraConfig, get_peft_model
ref_model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
peft_model = get_peft_model(ref_model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=list(TARGETS)))
n_peft = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
print(f"trainable params: ours {n_lora:,}, peft {n_peft:,}")

with torch.no_grad():
    for i, layer in enumerate(model.model.layers):
        for name in TARGETS:
            ours = getattr(layer.self_attn, name)
            theirs = getattr(peft_model.base_model.model.model.layers[i].self_attn, name)
            theirs.lora_A["default"].weight.copy_(ours.A)
            theirs.lora_B["default"].weight.copy_(ours.B)
    logits_peft = peft_model(**x_ids).logits
print("max |ours − peft| logits:   ", (logits_lora - logits_peft).abs().max().item())

remove_lora(model, merge=True)
with torch.no_grad():
    logits_merged = model(**x_ids).logits
print("max |merged − unmerged|:    ", (logits_merged - logits_lora).abs().max().item())
del ref_model, peft_model

# restore the original weights for the rest of the tutorial
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device).requires_grad_(False)
""")

md(r"""
Ours and `peft` agree exactly; merged vs unmerged differ by float32 round-off (below 1e-4 on logits of size ~10), since $(W+sBA)x$ and $Wx+sB(Ax)$ sum in a different order.

**Memory, the reason LoRA and QLoRA exist.** Full fine-tuning with Adam in mixed precision costs ≈16 bytes per parameter (weights, gradients, two Adam moments, fp32 master copy). LoRA keeps the frozen weights only (2 bytes in BF16) plus 16 bytes per *adapter* parameter. QLoRA stores the frozen weights in 4-bit (0.5 byte).
""")

code(r"""
def gb(n_bytes): return n_bytes / 1e9
P7 = 7e9
lora_frac = n_lora / n_all                   # same fraction as our 135M model, as a rough guide
print(f"7B full fine-tune (Adam, mixed precision): {gb(16 * P7):6.1f} GB")
print(f"7B LoRA  (bf16 frozen + adapters):         {gb(2 * P7 + 16 * lora_frac * P7):6.1f} GB")
print(f"7B QLoRA (4-bit frozen + adapters):        {gb(0.5 * P7 + 16 * lora_frac * P7):6.1f} GB   (activations not included)")
""")

md(r"""
✏️ Which matrices get adapted matters more than $r$. Re-run check 1 with `TARGETS = ("q_proj", "v_proj")` (the original LoRA paper's choice): how many trainable parameters now? Why do `k_proj` and `v_proj` have fewer adapter parameters than `q_proj` here? (Hint: print `model.model.layers[0].self_attn` above; SmolLM2 uses grouped-query attention.)

*Further reading:* Stanford CME 295 (2025), [Lecture 4: SFT and LoRA](https://cme295.stanford.edu/slides/fall25-cme295-lecture4.pdf).

## 3. SFT with loss masking

Each example is `prompt + response + <eos>`. The labels are the input ids with the **prompt positions set to −100**, the value `F.cross_entropy` (and the HF loss) ignores. Without the mask the model also learns to generate questions, and the loss is dominated by the (long, easy-to-predict) prompt.
The end token must be a target: it is how the model learns to stop.
""")

code(r"""
def build_sft_batch(prompts, responses):
    ids, labels = [], []
    for p, r in zip(prompts, responses):
        p_ids = tok(p).input_ids
        r_ids = tok(r).input_ids + [EOS]
        #>> the sequence is p_ids + r_ids; labels are -100 on the prompt and the token ids on the response
        ids.append(p_ids + r_ids)
        labels.append([-100] * len(p_ids) + r_ids)
        #<<
    L = max(len(x) for x in ids)                         # right-pad
    att = [[1] * len(x) + [0] * (L - len(x)) for x in ids]
    ids = [x + [EOS] * (L - len(x)) for x in ids]
    labels = [x + [-100] * (L - len(x)) for x in labels]
    return (torch.tensor(ids, device=device), torch.tensor(att, device=device), torch.tensor(labels, device=device))

def token_logprobs(model, ids, att, labels):
    # log π(token_t | tokens_<t) at every labelled position; zeros elsewhere. Returns (logp, mask), shape (B, T-1).
    pos = (att.cumsum(-1) - 1).clamp(min=0)              # correct positions for left- and right-padded batches
    logits = model(input_ids=ids, attention_mask=att, position_ids=pos).logits[:, :-1].float()
    tgt = labels[:, 1:]
    mask = tgt != -100
    #>> log_softmax over the vocabulary, pick the target token (gather), zero the unmasked positions
    logp = torch.log_softmax(logits, -1).gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    logp = logp * mask
    #<<
    return logp, mask
""")

md(r"""
**Check 3.** The unmasked labels decode to exactly the response plus the end token, and our masked NLL equals the HF loss (`model(..., labels=...)`), which does its own shift and masking.
""")

code(r"""
ps, rs = ["Q: What is 3 plus 4?\nA:", "Q: What is 12 minus 5?\nA:"], [target(7), target(7)]
ids, att, lab = build_sft_batch(ps, rs)
for i in range(2):
    print(repr(tok.decode(lab[i][lab[i] != -100])), "| prompt tokens masked:", (lab[i] == -100).sum().item() - (1 - att[i]).sum().item())
with torch.no_grad():
    logp, mask = token_logprobs(model, ids, att, lab)
    ours = -(logp.sum() / mask.sum()).item()
    hf = model(input_ids=ids, attention_mask=att, labels=lab).loss.item()
print(f"masked NLL: ours {ours:.6f}, HF {hf:.6f}, diff {abs(ours - hf):.1e}")
""")

md(r"""
**Training.** LoRA ($r=8$) on all attention projections, AdamW, lr $10^{-3}$, 30 steps of 16 examples drawn from one-digit `add1`/`sub`/`mul`. Then we **merge** the adapter: the merged model is our SFT model and the reference for everything after.
""")

code(r"""
seed_all(0)
inject_lora(model)
opt = torch.optim.AdamW(trainable(model), lr=1e-3)
sft_losses = []
model.train()
for step in range(30):
    data = [make_problem(random.choice(["add1", "sub", "mul"])) for _ in range(16)]
    ids, att, lab = build_sft_batch([p for p, _ in data], [target(a) for _, a in data])
    logp, mask = token_logprobs(model, ids, att, lab)
    loss = -(logp.sum() / mask.sum())
    opt.zero_grad(); loss.backward(); opt.step()
    sft_losses.append(loss.item())
model.eval()
remove_lora(model, merge=True)          # the SFT model = base + merged adapter

plt.figure(figsize=(5, 3))
plt.plot(sft_losses); plt.yscale("log")
plt.title("SFT loss (response tokens only)"); plt.xlabel("step"); plt.ylabel("NLL per token")
plt.tight_layout(); plt.show()

for kind in ["add1", "mul", "add2"]:
    fmt, acc, outs = evaluate(kind)
    print(f"SFT model on {kind}: format {fmt:.2f}, accuracy {acc:.2f}   e.g. {outs[0]!r}")
""")

md(r"""
Thirty steps teach the format completely (format 1.00), on all tasks including the unseen two-digit additions. One-digit arithmetic is (nearly) solved; on `add2` the SFT model is right on 48% of the questions. SFT taught *how to answer*; the arithmetic itself is what pretraining left behind. That gap is what RL will work on.

✏️ Set the labels to the full input ids (no mask) and retrain. Compare the first-step loss and the add2 format rate.

## 4. DPO

**Pairs from the model's own mistakes.** We sample the SFT model on add2 prompts and keep the wrong answers as $y_l$; $y_w$ is the correct answer. (In RLHF a human or a reward model picks the winner; here the verifier does.)

The DPO loss needs four sequence log-probs per pair: policy and reference, on $y_w$ and $y_l$. A sequence log-prob is the sum of `token_logprobs` over the response.
$$\mathcal{L} = -\log\sigma\big(\beta[(\log\pi_\theta(y_w) - \log\pi_{\text{ref}}(y_w)) - (\log\pi_\theta(y_l) - \log\pi_{\text{ref}}(y_l))]\big)$$
The two bracketed terms are the **implicit rewards** $\hat r = \beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$.
""")

code(r"""
def dpo_loss(pol_w, pol_l, ref_w, ref_l, beta=0.1):
    # all inputs: sequence log-probs, shape (n_pairs,)
    #>> implicit rewards r_w, r_l = beta * (policy − reference); loss = −log σ(r_w − r_l), averaged
    r_w, r_l = beta * (pol_w - ref_w), beta * (pol_l - ref_l)
    loss = -F.logsigmoid(r_w - r_l).mean()
    #<<
    return loss, r_w.detach(), r_l.detach()
""")

md(r"""
**Check 4 on toy numbers.** $\log\pi_\theta(y_w)=-5,\ \log\pi_{\text{ref}}(y_w)=-6,\ \log\pi_\theta(y_l)=-7,\ \log\pi_{\text{ref}}(y_l)=-6.5,\ \beta=0.1$: the margin is $0.1\,(1-(-0.5)) = 0.15$ and the loss is $\log(1+e^{-0.15})$.
""")

code(r"""
loss, _, _ = dpo_loss(torch.tensor([-5.]), torch.tensor([-7.]), torch.tensor([-6.]), torch.tensor([-6.5]), beta=0.1)
print(f"ours {loss.item():.6f}   direct {math.log1p(math.exp(-0.15)):.6f}")
# a pair the policy already prefers more than the reference does gives loss < log 2; the reverse gives > log 2
print("at policy = reference the loss is log 2 =", round(math.log(2), 6),
      "->", round(dpo_loss(*[torch.tensor([-3.])] * 4)[0].item(), 6))
""")

md(r"""
**Past exam question (Moed C, 2026)**

In **Direct Preference Optimization (DPO)**, a language model $\pi_\theta$ is trained on pairs of answers $(y_w, y_l)$ to a prompt $x$, where $y_w$ is preferred by humans over $y_l$. The loss is
$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)}\Big[\log \sigma\Big(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\Big)\Big],$$
where $\pi_{\text{ref}}$ is a fixed reference model (usually the model after SFT) and $\beta$ is a temperature parameter.

Which of the following statements about DPO is **correct**? **(More than one answer may be correct.)**
1. DPO trains the model directly from human preference pairs, without an RL algorithm (such as PPO), by analytically deriving an equivalent loss function.
2. A high value of $\beta$ constrains $\pi_\theta$ to stay closer to $\pi_{\text{ref}}$, in a role similar to the KL-regularization term in traditional RLHF.
3. DPO requires training a separate, explicit reward model before the optimization stage of $\pi_\theta$.
4. DPO is suitable only for classification tasks and cannot be used for autoregressive generative models.
<<STUDENT>>
✏️ Your answer:
<</STUDENT>>

**Numeric check of statement 2.** Two possible answers, $\pi_{\text{ref}} = (0.5, 0.5)$. Preference data follow Bradley-Terry with rewards $r = (1, 0)$: answer 1 wins a comparison with probability $p = \sigma(1) = 0.73$. Minimize the expected DPO loss, $p\,\mathcal L(y_1 \succ y_2) + (1-p)\,\mathcal L(y_2 \succ y_1)$, over the policy's two logits for several $\beta$, using `dpo_loss` from above. RLHF's KL-regularized optimum is $\pi^* \propto \pi_{\text{ref}}\,e^{r/\beta}$, i.e. $\pi^*(y_1) = \sigma(1/\beta)$ here.
""")

code(r"""
p_win = 1 / (1 + math.exp(-1.0))                    # sigma(r_1 - r_2) with r = (1, 0)
ref_lp = torch.log(torch.tensor([0.5, 0.5]))
print(" beta   pi(y1) DPO   pi(y1) RLHF optimum   KL(pi || pi_ref)")
for beta in [0.25, 0.5, 1.0, 2.0, 4.0]:
    logits = torch.zeros(2, requires_grad=True)
    opt_toy = torch.optim.Adam([logits], lr=0.05)
    for _ in range(600):
        lp = torch.log_softmax(logits, 0)
        #>> expected DPO loss: p_win × dpo_loss(y1 wins) + (1 − p_win) × dpo_loss(y2 wins), each pair as tensors of shape (1,)
        loss_w, _, _ = dpo_loss(lp[0:1], lp[1:2], ref_lp[0:1], ref_lp[1:2], beta)
        loss_l, _, _ = dpo_loss(lp[1:2], lp[0:1], ref_lp[1:2], ref_lp[0:1], beta)
        loss = p_win * loss_w + (1 - p_win) * loss_l
        #<<
        opt_toy.zero_grad(); loss.backward(); opt_toy.step()
    pi = torch.softmax(logits.detach(), 0)
    kl = (pi * (pi.log() - ref_lp)).sum().item()
    print(f"{beta:5.2f}   {pi[0]:.4f}        {1 / (1 + math.exp(-1 / beta)):.4f}               {kl:.4f}")
""")

md(r"""
<<SOLUTION>>
**Answer: 1 and 2** (as in the official solution).
1. True. The KL-regularized RLHF objective has a closed-form optimum, $\pi^*(y|x) \propto \pi_{\text{ref}}(y|x)\,e^{r(x,y)/\beta}$. Solving it for $r$ and substituting into the Bradley-Terry likelihood gives a supervised loss on $\pi_\theta$: no PPO, no sampling during training.
2. True. The table shows it: the DPO minimizer equals the RLHF optimum $\sigma(1/\beta)$ for every $\beta$, and its KL to $\pi_{\text{ref}}$ falls from 0.60 at $\beta=0.25$ to 0.0078 at $\beta=4$. $\beta$ is exactly the KL coefficient of the RLHF objective DPO is derived from.
3. False. The reward is implicit, $\hat r = \beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$; `dpo_loss` takes only four log-probs. Not needing a reward model is DPO's main practical advantage.
4. False. $\pi_\theta(y|x)$ is the product of the token probabilities of an autoregressive LM; this notebook trains a causal LM with it, as Llama 3, Mistral/Zephyr and Tülu did.
<</SOLUTION>>
""")

code(r"""
def seq_logprob(model, prompts, responses):
    ids, att, lab = build_sft_batch(prompts, responses)
    logp, _ = token_logprobs(model, ids, att, lab)
    return logp.sum(1)

seed_all(7)
pairs = []
while len(pairs) < 96:
    data = [make_problem("add2") for _ in range(32)]
    _, _, outs = generate([p for p, _ in data], sample=True)
    for (p, a), o in zip(data, outs):
        guess = parse_answer(o)
        if guess is not None and guess != a:
            pairs.append((p, target(a), target(guess)))
train_pairs, test_pairs = pairs[:80], pairs[80:96]
print(len(pairs), "pairs, e.g.", train_pairs[0])

def pref_accuracy(pairs, beta=0.1):
    # fraction of pairs where the implicit reward ranks y_w above y_l
    with torch.no_grad():
        ps = [p for p, _, _ in pairs]
        pol_w, pol_l = seq_logprob(model, ps, [w for _, w, _ in pairs]), seq_logprob(model, ps, [l for _, _, l in pairs])
        set_lora(model, False)
        ref_w, ref_l = seq_logprob(model, ps, [w for _, w, _ in pairs]), seq_logprob(model, ps, [l for _, _, l in pairs])
        set_lora(model, True)
    _, r_w, r_l = dpo_loss(pol_w, pol_l, ref_w, ref_l, beta)
    return (r_w > r_l).float().mean().item()

inject_lora(model)                       # fresh adapter on the SFT model; adapter off = reference
opt = torch.optim.AdamW(trainable(model), lr=1e-3)
hist = {"loss": [], "r_w": [], "r_l": []}
for step in range(30):
    batch = random.sample(train_pairs, 8)
    ps = [p for p, _, _ in batch]
    ws, ls = [w for _, w, _ in batch], [l for _, _, l in batch]
    pol = seq_logprob(model, ps + ps, ws + ls)
    with torch.no_grad():
        set_lora(model, False); ref = seq_logprob(model, ps + ps, ws + ls); set_lora(model, True)
    loss, r_w, r_l = dpo_loss(pol[:8], pol[8:], ref[:8], ref[8:])
    opt.zero_grad(); loss.backward(); opt.step()
    hist["loss"].append(loss.item()); hist["r_w"].append(r_w.mean().item()); hist["r_l"].append(r_l.mean().item())

fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
ax[0].plot(hist["loss"]); ax[0].axhline(math.log(2), ls="--", c="gray", label="log 2 (policy = reference)")
ax[0].set(title="DPO loss", xlabel="step", ylabel="loss"); ax[0].legend()
ax[1].plot(hist["r_w"], label=r"chosen $\hat r_w$"); ax[1].plot(hist["r_l"], label=r"rejected $\hat r_l$")
ax[1].axhline(0, c="gray", lw=0.8)
ax[1].set(title=r"implicit rewards $\beta\log\pi_\theta/\pi_{ref}$ (batch mean)", xlabel="step", ylabel="reward"); ax[1].legend()
plt.tight_layout(); plt.show()

print(f"preference accuracy: train {pref_accuracy(train_pairs):.2f}, held-out pairs {pref_accuracy(test_pairs):.2f}")
fmt, acc, _ = evaluate("add2"); print(f"DPO model, greedy add2: format {fmt:.2f}, accuracy {acc:.2f}")
remove_lora(model, merge=False)          # back to the SFT model
""")

md(r"""
<<SOLUTION>>
What the run shows (numbers above):
- The loss drops from $\log 2$ to ≈0.13 and 98% of the training pairs are ranked correctly; the ranking also transfers to the 16 held-out pairs (94%).
- Almost all of the margin comes from pushing the **rejected** answers down (to ≈ −3.4); the chosen implicit reward stays near zero and even dips below it. The loss only constrains the *difference*, and lowering a specific wrong answer is the easy way to increase it.
- Generation accuracy on add2 does not reliably improve: 0.48 → 0.53 in this run (format 0.98), while a CPU run of the same cell (different samples, hence different pairs) went *down* to 0.28. Making 80 specific wrong answers unlikely does not make the right answer the *most likely* one on new questions, and where the freed probability mass goes is not controlled by the loss. Compare with GRPO next, which learns from its own fresh samples.
<</SOLUTION>>
<<STUDENT>>
Look at the two plots and the printed numbers: which log-prob moves more, chosen or rejected? Did generation accuracy on add2 change?
<</STUDENT>>

*Further reading:* Stanford CME 295 (2025), [Lecture 5: RLHF, PPO, DPO](https://cme295.stanford.edu/slides/fall25-cme295-lecture5.pdf); for DPO on diffusion models (Diffusion-DPO), Stanford CME 296 (2026), [Lecture 6](https://cme296.stanford.edu/slides/spring26-cme296-lecture6.pdf).

## 5. From policy gradient to GRPO

One line at a time. Each line keeps what the previous one had and fixes one problem. Every symbol here appears in the code below.

**Setup.** Prompt $q$, sampled answer $o=(o_1,\dots,o_T)\sim\pi_\theta(\cdot\mid q)$, reward $r(q,o)\in\{0,1\}$ from a verifier. Goal: $J(\theta)=\mathbb{E}_{o\sim\pi_\theta}[r(q,o)]$.

**Step 1, the policy gradient.** We cannot backprop through sampling, but $\nabla_\theta \pi = \pi\,\nabla_\theta\log\pi$ gives
$$\nabla J = \mathbb{E}_{o\sim\pi_\theta}\big[\,r(q,o)\,\nabla_\theta\log\pi_\theta(o\mid q)\big],\qquad \log\pi_\theta(o\mid q)=\textstyle\sum_t\log\pi_\theta(o_t\mid q,o_{<t}).$$
*Intuition:* SFT on your own samples, weighted by their reward. Rewarded answers become more likely. The log-probs are `token_logprobs`, the same function SFT used.

**Step 2, a baseline.** $\mathbb{E}_{o}[\nabla\log\pi_\theta(o)] = \nabla\sum_o\pi_\theta(o) = \nabla 1 = 0$, so subtracting any $b$ that does not depend on $o$ keeps the gradient unbiased:
$$\nabla J = \mathbb{E}\big[(r - b)\,\nabla\log\pi_\theta(o\mid q)\big].$$
*Intuition:* with $b=0$ and rewards in $\{0,1\}$, wrong answers get no push at all and every correct one is pushed by the same amount, however easy the question. With $b$ = the expected reward, better-than-usual answers go up and worse ones go down. Same mean, much lower variance.

**Step 3, the group as the baseline (GRPO).** PPO learns $b$ with a value network (a second LLM-sized model). GRPO samples $G$ answers $o_1..o_G$ to the **same** $q$ and uses their mean, then also divides by their std:
$$\hat A_i = \frac{r_i - \mathrm{mean}(r_1..r_G)}{\mathrm{std}(r_1..r_G)}.$$
*Intuition:* "was this answer better than my other attempts at this question?" A question the model always solves, or never solves, has all $r_i$ equal, so $\hat A_i = 0$: it teaches nothing. The signal comes from questions at the edge of the model's ability.

**Step 4, an importance ratio.** The samples come from the policy before the update, $\pi_{\text{old}}$. Per token,
$$\rho_t = \frac{\pi_\theta(o_t\mid q,o_{<t})}{\pi_{\text{old}}(o_t\mid q,o_{<t})} = \exp(\log\pi_\theta - \log\pi_{\text{old}}),\qquad \text{surrogate } \rho_t\,\hat A_i .$$
At $\theta=\theta_{\text{old}}$, $\nabla\rho_t = \nabla\log\pi_\theta(o_t)$, so the surrogate has exactly the gradient of step 2. It only matters when we take several gradient steps on one batch of samples (PPO's "epochs"). In our loop there is one step per batch, so `old_lp = lp.detach()` and $\rho_t = 1$ in value.

**Step 5, the clip.** $\;\min\big(\rho_t\hat A_i,\ \mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\,\hat A_i\big)$.
*Intuition:* once a token's probability has already moved by more than a factor $1\pm\epsilon$ in the direction $\hat A$ asks for, its gradient becomes zero. A cheap trust region: one lucky batch cannot move the policy far. (Section 7 plots it.)

**Step 6, stay close to the reference.** Subtract $\beta\,\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$, estimated per sampled token with $\delta_t = \log\pi_{\text{ref}}(o_t) - \log\pi_\theta(o_t)$:
$$k_3 = e^{\delta_t} - \delta_t - 1 \;\ge 0 ,\qquad \mathbb{E}_{o_t\sim\pi_\theta}[k_3] = \mathrm{KL}(\pi_\theta\|\pi_{\text{ref}}).$$
*Intuition:* a penalty that grows when the policy drifts from the SFT model, which keeps the format and the language intact. The reference is our SFT model with the adapter switched off.

**All together (the GRPO loss we minimize).** Average over tokens of each answer, then over the $G$ answers:
$$\mathcal{L} = -\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\Big[\min\big(\rho_{i,t}\hat A_i,\ \mathrm{clip}(\rho_{i,t},1\!-\!\epsilon,1\!+\!\epsilon)\hat A_i\big) - \beta\,k_{3,i,t}\Big].$$
The verifier is a regex plus an integer comparison. With it as the reward this is RLVR.

**Training setup.** 30 steps; each step: 4 add2 questions × $G=8$ answers, one AdamW step (lr $5\cdot10^{-4}$) on a fresh LoRA adapter, gradient norm clipped to 1, $\epsilon=0.2$, $\beta=0.04$.
""")

code(r"""
def verifier(text, answer):
    #>> 1.0 if the parsed answer equals the correct one, else 0.0 (unparseable = wrong)
    return 1.0 if parse_answer(text) == answer else 0.0
    #<<

def group_advantages(rewards, G, eps=1e-4):
    # rewards: (n_prompts * G,), grouped consecutively
    #>> reshape to (n_prompts, G), subtract the group mean, divide by the group std (+eps), flatten back
    r = rewards.view(-1, G)
    adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + eps)
    return adv.view(-1)
    #<<

def grpo_loss(lp, old_lp, ref_lp, adv, mask, eps=0.2, beta=0.04):
    # lp, old_lp, ref_lp, mask: (n, T) per-token; adv: (n,) one advantage per answer
    #>> ratio, clipped surrogate (min of the two), k3 KL; per-token loss = −(surrogate − beta·k3)
    ratio = torch.exp(lp - old_lp)
    A = adv[:, None]
    surrogate = torch.min(ratio * A, torch.clamp(ratio, 1 - eps, 1 + eps) * A)
    delta = ref_lp - lp
    k3 = torch.exp(delta) - delta - 1
    per_token = -(surrogate - beta * k3)
    #<<
    return ((per_token * mask).sum(1) / mask.sum(1)).mean()   # mean over each answer's tokens, then over answers
""")

md(r"""
**Check 5.** The verifier on a few strings; the advantages against NumPy's z-score; the GRPO loss against the formula above written as explicit Python loops, on random numbers with $\rho\neq 1$ so the clip is active.
""")

code(r"""
for text, ans in [(" The answer is 85.", 85), (" The answer is 85.", 84), (" 85", 85), (" The answer is 85. Or 86.", 85)]:
    print(f"{text!r:30} correct={ans}: reward {verifier(text, ans)}")

r = torch.tensor([1., 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0])
G = 4
ref_adv = np.concatenate([(g - g.mean()) / (g.std(ddof=1) + 1e-4) for g in r.numpy().reshape(-1, G)])
print("advantages:", group_advantages(r, G).numpy().round(2), " max diff vs numpy:", np.abs(group_advantages(r, G).numpy() - ref_adv).max())

torch.manual_seed(0)
n, T = 4, 5
lp_t = -torch.rand(n, T); old_t = lp_t + 0.3 * torch.randn(n, T); ref_t = lp_t + 0.3 * torch.randn(n, T)
adv_t = torch.randn(n); lens = [5, 3, 4, 2]
mask_t = torch.tensor([[t < L for t in range(T)] for L in lens])
lp_t, old_t, ref_t = lp_t * mask_t, old_t * mask_t, ref_t * mask_t
direct = 0.0
for i in range(n):
    s = 0.0
    for t in range(lens[i]):
        rho = math.exp(lp_t[i, t] - old_t[i, t]); A = adv_t[i].item(); d = (ref_t[i, t] - lp_t[i, t]).item()
        s += min(rho * A, min(max(rho, 0.8), 1.2) * A) - 0.04 * (math.exp(d) - d - 1)
    direct += -s / lens[i] / n
print(f"GRPO loss: ours {grpo_loss(lp_t, old_t, ref_t, adv_t, mask_t).item():.6f}, direct {direct:.6f}")
""")

md(r"""
**Past exam question (Moed C, 2026)**

**Group Relative Policy Optimization (GRPO)** is used to train reasoning models (such as DeepSeek-R1). For each prompt, a group of $G$ answers $\{y_1, \ldots, y_G\}$ is sampled, rewards $\{r_1, \ldots, r_G\}$ are obtained, and a relative advantage is computed for each sample:
$$A_i = \frac{r_i - \mathrm{mean}(r)}{\mathrm{std}(r)}.$$
**Explain why normalizing by $\mathrm{std}(r)$ can be problematic for *very hard* questions, where almost all samples fail** ($r_i \approx 0$ for almost all $i$). Propose a simple fix.
<<STUDENT>>
✏️ Your answer:
<</STUDENT>>

Compute it with `group_advantages` from above (it adds $\epsilon = 10^{-4}$ to the std) on four very hard groups: all 8 answers fail; 1 of 8 succeeds; 1 of 16 succeeds; and all 8 fail but one gets a tiny partial reward of 0.01 (e.g. a format bonus or a noisy judge). Compare with the advantage without the std, $r_i - \mathrm{mean}(r)$.
""")

code(r"""
groups = {"all 8 fail": torch.zeros(8),
          "1 of 8 succeeds": torch.tensor([1.] + [0.] * 7),
          "1 of 16 succeeds": torch.tensor([1.] + [0.] * 15),
          "8 fail, one gets 0.01": torch.tensor([0.01] + [0.] * 7)}
for name, r in groups.items():
    #>> A = group_advantages(r, G=len(r)); A_nostd = r − r.mean(); A_noeps = (r − mean) / std without the epsilon
    A = group_advantages(r, len(r))
    A_nostd = r - r.mean()
    A_noeps = (r - r.mean()) / r.std()
    #<<
    print(f"{name:22s} with std: first {A[0].item():6.2f}, others {A[1].item():6.2f}   without eps: first {A_noeps[0].item():6.2f}"
          f"   without std (r − mean): first {A_nostd[0].item():6.3f}, others {A_nostd[1].item():6.3f}")
""")

md(r"""
<<SOLUTION>>
**Answer** (the solution files have no official solution for this question; this is ours).
- **All answers fail:** $\mathrm{std}(r)=0$ and $A_i = 0/0$ (NaN without the $\epsilon$, printed above); with the $\epsilon$, $A_i=0$. The question costs $G$ rollouts and gives no gradient.
- **Almost all fail:** dividing by a small std **amplifies** the rare success. One success in 8 gets $A=+2.47$, one in 16 gets $+3.75$: the rarer the success, the larger its push, so the update is dominated by a few hard (or, symmetrically, very easy) questions, and a lucky or spuriously-rewarded answer (right answer, wrong reasoning) is reinforced the most. This is the question-level difficulty bias analyzed by Liu et al. (2025, Dr. GRPO).
- **Std normalization is scale-free:** a reward difference of 0.01 (last row) gets almost the same advantage as a real success: $+2.41$ ($+2.47$ without the $\epsilon$, exactly as for a success). On hard questions where the only variation is reward noise or a small partial credit, GRPO turns noise into a full-size learning signal.
- **Fix:** drop the division, $A_i = r_i - \mathrm{mean}(r)$ (Dr. GRPO): then a group whose rewards barely differ gives advantages near 0 (0.009 in the last row) and one success in 16 gets 0.94, not 3.75. Alternatives: divide by the std of the whole batch instead of the group, or a floor $\max(\mathrm{std}, c)$; and skip groups with identical rewards, resampling new prompts instead (DAPO's dynamic sampling).
<</SOLUTION>>
""")

md(r"""
**Rollouts.** `rollout` samples $G$ answers per prompt at temperature 1 and builds the tensors the loss needs: the full sequences, an attention mask that stops after the first end token, and labels that are −100 on the prompt and after the end.
""")

code(r"""
@torch.no_grad()
def rollout(prompts, G, max_new=10):
    out, enc, _ = generate(prompts, sample=True, n=G, max_new=max_new)
    P = enc.input_ids.shape[1]
    resp = out[:, P:]
    is_eos = (resp == EOS).int()
    keep = (is_eos.cumsum(1) - is_eos) == 0                      # response tokens up to and including the first eos
    att = torch.cat([enc.attention_mask.repeat_interleave(G, 0), keep.long()], 1)
    labels = torch.cat([torch.full_like(enc.input_ids.repeat_interleave(G, 0), -100), resp.masked_fill(~keep, -100)], 1)
    texts = tok.batch_decode(resp.masked_fill(~keep, EOS), skip_special_tokens=True)
    return out, att, labels, texts

def grpo_train(reward_fn, steps=30, n_prompts=4, G=8, lr=5e-4, max_new=10, log_every=10, seed=1):
    seed_all(seed)
    inject_lora(model)
    opt = torch.optim.AdamW(trainable(model), lr=lr)
    hist = {"reward": [], "acc": [], "len": [], "zero_groups": [], "eval": []}
    for step in range(steps):
        data = [make_problem("add2") for _ in range(n_prompts)]
        answers = [a for _, a in data for _ in range(G)]
        ids, att, lab, texts = rollout([p for p, _ in data], G, max_new)
        rewards = torch.tensor([reward_fn(t, a) for t, a in zip(texts, answers)], device=device)
        adv = group_advantages(rewards, G)

        lp, mask = token_logprobs(model, ids, att, lab)
        with torch.no_grad():
            set_lora(model, False); ref_lp, _ = token_logprobs(model, ids, att, lab); set_lora(model, True)
        loss = grpo_loss(lp, lp.detach(), ref_lp, adv, mask)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable(model), 1.0)
        opt.step()

        hist["reward"].append(rewards.mean().item())
        hist["acc"].append(np.mean([verifier(t, a) for t, a in zip(texts, answers)]))
        hist["len"].append(mask.sum(1).float().mean().item())
        hist["zero_groups"].append((rewards.view(-1, G).std(1) == 0).float().mean().item())
        if (step + 1) % log_every == 0:
            fmt, acc, _ = evaluate("add2", max_new=max_new)
            hist["eval"].append((step + 1, acc))
            print(f"step {step + 1:3d}  batch reward {np.mean(hist['reward'][-log_every:]):.2f}  held-out greedy acc {acc:.2f}  e.g. {texts[0]!r}")
    return hist

_, acc_sft, _ = evaluate("add2")
print(f"SFT model, held-out greedy add2 accuracy: {acc_sft:.2f}")
grpo_hist = grpo_train(verifier)
""")

code(r"""
def smooth(x, k=5):
    return np.convolve(x, np.ones(k) / k, mode="valid")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
ax[0].plot(grpo_hist["reward"], alpha=0.35, label="batch mean (32 answers)")
ax[0].plot(range(4, len(grpo_hist["reward"])), smooth(grpo_hist["reward"]), lw=2, label="moving average (5)")
ax[0].set(title="GRPO: verifier reward on training rollouts", xlabel="step", ylabel="reward = accuracy", ylim=(0, 1)); ax[0].legend()
steps_e = [0] + [s for s, _ in grpo_hist["eval"]]
ax[1].plot(steps_e, [acc_sft] + [a for _, a in grpo_hist["eval"]], marker="o")
ax[1].set(title="held-out add2, greedy decoding", xlabel="step", ylabel="accuracy", ylim=(0, 1))
plt.tight_layout(); plt.show()
print(f"fraction of groups with zero advantage (all G answers equally rewarded): {np.mean(grpo_hist['zero_groups']):.2f}")
""")

md(r"""
<<SOLUTION>>
Thirty updates of 32 rollouts raise held-out greedy accuracy from 0.48 (SFT) to 0.62, 0.72, 0.75 at steps 10, 20, 30. The training reward (left) is not a good progress measure here: each step sees only 4 new questions of varying difficulty, so even its moving average goes down after step 23 while the held-out accuracy on 64 fixed questions keeps rising. About half of the groups (printed) had identical rewards and contributed no gradient: those questions were too easy or too hard for the current policy.

RL on a small batch is fragile. With lr $10^{-3}$ and no gradient clipping, the same run reached 0.66 at step 20 and then collapsed to 0.12 at step 30.
<</SOLUTION>>
<<STUDENT>>
Read the two plots: how much did held-out accuracy move compared with the SFT model? How noisy is the per-step reward, and why (how many questions per step)?
<</STUDENT>>

### What did RL change? pass@1 vs pass@8

Sample 8 answers per held-out question from the SFT model (adapter off) and from the GRPO model (adapter on). pass@1 = mean accuracy of one sample; pass@8 = at least one of the 8 is right; maj@8 = the most common answer is right (majority vote, a simple form of test-time compute).
""")

code(r"""
@torch.no_grad()
def pass_at_k(k=8, n=64, seed=321):
    rng_state = random.getstate(); random.seed(seed)
    data = [make_problem("add2") for _ in range(n)]
    random.setstate(rng_state)
    _, _, outs = generate([p for p, _ in data], sample=True, n=k)
    correct = np.array([verifier(o, a) for o, (_, a) in zip(outs, [d for d in data for _ in range(k)])]).reshape(n, k)
    votes = [max(set(g), key=g.count) for g in np.array([parse_answer(o) for o in outs], dtype=object).reshape(n, k).tolist()]
    maj = np.mean([v == a for v, (_, a) in zip(votes, data)])
    return correct.mean(), correct.max(1).mean(), maj

seed_all(5)
set_lora(model, False); sft_stats = pass_at_k()
set_lora(model, True);  grpo_stats = pass_at_k()
for name, s in [("SFT", sft_stats), ("GRPO", grpo_stats)]:
    print(f"{name:5}: pass@1 {s[0]:.2f}   maj@8 {s[2]:.2f}   pass@8 {s[1]:.2f}")

x = np.arange(3); w = 0.35
plt.figure(figsize=(6, 3.2))
plt.bar(x - w / 2, [sft_stats[0], sft_stats[2], sft_stats[1]], w, label="SFT")
plt.bar(x + w / 2, [grpo_stats[0], grpo_stats[2], grpo_stats[1]], w, label="SFT + GRPO")
plt.xticks(x, ["pass@1", "maj@8", "pass@8"]); plt.ylim(0, 1); plt.ylabel("accuracy")
plt.title("add2, 8 samples per question (T = 1)"); plt.legend(); plt.tight_layout(); plt.show()
""")

md(r"""
<<SOLUTION>>
pass@1 rose by 0.27 (0.38 → 0.65), pass@8 by 0.16 (0.70 → 0.86). Most of the gain is **sharpening**: the SFT model already had a correct answer among its 8 samples on 70% of the questions, and RLVR made it the likely one. The gap pass@8 − pass@1 shrank from 0.32 to 0.21. Yue et al. (2025) report the same pattern at scale: RL-trained reasoning models win at pass@1 and their base models catch up at large $k$.

Majority vote (maj@8) stays within a few points of pass@1 (0.39 vs 0.38 and 0.64 vs 0.65 here; a CPU run gave 0.47 vs 0.43 and 0.72 vs 0.65). A vote returns the model's most likely answer, so it fixes errors caused by sampling noise, not errors the model makes systematically; this model's arithmetic mistakes are mostly of the second kind.
<</SOLUTION>>
""")

md(r"""
**Past exam question (Moed B, 2026)**

What are the main difficulties in training **reasoning** models with standard **supervised learning**? Explain how training with **Reinforcement Learning (RL)** helps to deal with these difficulties. Refer to the type of **reward** commonly used in training such models.
<<STUDENT>>
✏️ Your answer:
<</STUDENT>>
<<SOLUTION>>
**Answer** (following the official solution).
- **Supervised learning is hard for reasoning:** (i) one problem has many valid reasoning paths, and SFT rewards copying one particular written path token by token, not reaching the right answer; (ii) full, high-quality reasoning traces are expensive to obtain at scale; (iii) imitation is bounded by the demonstrations: it does not encourage *discovering* better solution strategies, and the model never learns from its own mistakes, since it is trained only on states the demonstrator visited.
- **RL** rewards the *outcome* of the model's own attempts, not agreement with a fixed path. The model explores different solutions; those that reach a correct answer are reinforced, wrong ones pushed down (section 5).
- **Reward:** a **verifiable reward**, computed by a program: the final answer matches the reference for math, the unit tests pass for code (RLVR). It needs no human labels per sample and no trace annotations, and it is hard to fool if the checker is exact; section 6 shows what happens when the checker is not.
- In this notebook: SFT on demonstrations (section 3) taught the answer format, but not two-digit addition; GRPO with the verifier then raised held-out accuracy, mostly by making answers the model could already sometimes produce more likely (pass@1 vs pass@8 above).
<</SOLUTION>>
""")

md(r"""
✏️ Re-run `grpo_train(verifier, lr=1e-3)` with the `clip_grad_norm_` line removed and watch for the collapse. Then PPO-style reuse: take 2 gradient steps on each rollout batch (compute `old_lp` once, before the first step, with `torch.no_grad()`). Print the fraction of tokens where the ratio is clipped in the second step.

*Further reading:* Stanford CME 295 (2025), [Lecture 6: reasoning and GRPO](https://cme295.stanford.edu/slides/fall25-cme295-lecture6.pdf); GRPO for image generation (Flow-GRPO), Stanford CME 296 (2026), [Lecture 6](https://cme296.stanford.edu/slides/spring26-cme296-lecture6.pdf).

On a Colab GPU: `steps=200, n_prompts=16`, and move to three-digit additions or GSM8K-style word problems with a `<think>` format.

## 6. Reward hacking: a flawed verifier

RL optimizes the reward you wrote, not the one you meant. Here the grader checks the **format** and prefers **longer** answers ("shows more work"), the verbosity bias that LLM judges and reward models have. It never checks the number:
$$r_{\text{flawed}}(o) = \mathbb{1}[\text{format ok}]\cdot\frac{\text{len}(o)}{20 \text{ characters}}.$$
We start again from the SFT model and run the same GRPO loop with 16 new tokens allowed, tracking the true accuracy with the real verifier. The learning rate is $10^{-3}$ (twice section 5's) so that the whole exploit shows within 30 steps; at $5\cdot10^{-4}$ the same thing happens, more slowly.
""")

code(r"""
def flawed_verifier(text, answer):
    return float(parse_answer(text) is not None) * len(text) / 20

remove_lora(model, merge=False)          # back to the SFT model
hack_hist = grpo_train(flawed_verifier, steps=30, max_new=16, lr=1e-3)
""")

code(r"""
fig, ax = plt.subplots(1, 3, figsize=(15, 3.4))
ax[0].plot(hack_hist["reward"], marker=".", c="C3"); ax[0].set(title="proxy: flawed verifier reward", xlabel="step", ylabel="reward")
ax[1].plot(hack_hist["acc"], marker=".", c="C2", label="flawed-verifier run")
ax[1].plot(grpo_hist["acc"], c="C0", alpha=0.5, label="section 5 run (true verifier)")
ax[1].set(title="true accuracy of the training rollouts", xlabel="step", ylabel="accuracy", ylim=(0, 1)); ax[1].legend()
ax[2].plot(hack_hist["len"], marker=".", c="C4"); ax[2].set(title="response length", xlabel="step", ylabel="tokens (incl. eos)")
plt.tight_layout(); plt.show()

seed_all(11)
data = [make_problem("add2") for _ in range(4)]
_, _, outs = generate([p for p, _ in data], sample=True, max_new=16)
for (p, a), o in zip(data, outs):
    print(f"{p.splitlines()[0]:28} correct {a:4d}  ->  {o!r}")
fmt, acc, _ = evaluate("add2", max_new=16); print(f"held-out greedy add2: format {fmt:.2f}, accuracy {acc:.2f}")
remove_lora(model, merge=False)
""")

md(r"""
<<SOLUTION>>
The run splits into two phases (see the plots and the samples):
1. **Longer numbers (steps 0 to 12).** The cheapest way to add characters inside the format is a longer answer, so the policy writes many-digit numbers (` The answer is 10811111.` at step 10). True accuracy is 0 from step 6 on, while the proxy barely moves. The dip at step 12 is the policy overshooting: answers that hit the 16-token limit lose the final period, fail the format check and get reward 0.
2. **Text after the answer (from step ≈15).** The policy learns to keep writing after the period (` The answer is 768. The answer is short for answer. The`). Every answer uses the full 16 tokens from step 22, and the proxy climbs to 2.6, 2.8× its starting value.

At the end the held-out format rate is 1.00 and the accuracy 0.02: the grader is fully satisfied and the task is lost. The KL penalty ($\beta=0.04$) did not prevent it. The same mechanism, at scale, is why RLVR rewards are checked against the answer, why length normalization in GRPO matters (Dr. GRPO, DAPO), and why labs monitor chains of thought for signs of gaming (recap, CoT monitoring).
<</SOLUTION>>
<<STUDENT>>
Describe what the policy learned, in two phases. What happened to the true accuracy while the proxy reward went up?
<</STUDENT>>

✏️ Fix the grader without removing the length term: `reward = verifier(text, answer) + 0.1 * flawed_verifier(text, answer)`. Does the policy still hack? What does that say about mixing a correctness reward with a style reward?

## 7. If time: what the clip does to the gradient

For one token with advantage $\hat A$, plot the clipped surrogate $\min(\rho\hat A, \mathrm{clip}(\rho,1-\epsilon,1+\epsilon)\hat A)$ and its gradient with respect to $\rho$ (autograd). This is the lecture's PPO-Clip figure.
""")

code(r"""
rho = torch.linspace(0.5, 1.5, 200, requires_grad=True)   # 200 points: avoids landing exactly on the kinks
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
for A, c in [(1.0, "C0"), (-1.0, "C3")]:
    surr = torch.min(rho * A, torch.clamp(rho, 0.8, 1.2) * A)
    g, = torch.autograd.grad(surr.sum(), rho)
    ax[0].plot(rho.detach(), surr.detach(), c=c, label=f"A = {A:+.0f}")
    ax[1].plot(rho.detach(), g, c=c, label=f"A = {A:+.0f}")
for a in ax:
    a.axvline(0.8, ls=":", c="gray"); a.axvline(1.2, ls=":", c="gray"); a.legend(); a.set_xlabel(r"ratio $\rho = \pi_\theta / \pi_{old}$")
ax[0].set(title="clipped surrogate", ylabel="objective"); ax[1].set(title="gradient d(surrogate)/dρ", ylabel="gradient")
plt.tight_layout(); plt.show()
print(f"total runtime: {time.time() - T_START:.0f} s")
""")

md(r"""
For $\hat A>0$ the gradient is $\hat A$ until $\rho=1+\epsilon$, then zero: no further push on a token that is already more likely. For $\hat A<0$ it is zero below $1-\epsilon$. On the "wrong" side (e.g. $\hat A>0$, $\rho<1-\epsilon$) the gradient stays on: the clip never blocks a correction.

## Summary
- **LoRA** = a frozen $W$ plus $\frac{\alpha}{r}BA$; under 1% of the parameters trainable here; matches `peft` and merges into $W$ exactly (up to float round-off). The adapter-off switch gives the reference model with no extra memory.
- **SFT** with the loss on response tokens only (labels −100 on the prompt) taught the answer format in 30 steps, also on an unseen task, but not the two-digit arithmetic (48%).
- **DPO** fit the offline preference pairs quickly, almost entirely by lowering the rejected answers; generation accuracy did not reliably follow (0.48 → 0.53 here, → 0.28 in a CPU run).
- **GRPO** = policy gradient + group baseline + ratio/clip + KL. With a verifiable reward it raised held-out accuracy from 0.48 to 0.75 in 30 steps, mostly by sharpening (pass@1 0.38 → 0.65, pass@8 0.70 → 0.86).
- **Reward hacking:** with a grader that checks format and length but not correctness, the same algorithm drove the proxy up 2.8× and true accuracy to ≈0, first by writing longer numbers, then by filler text.

**Further watching:** Karpathy, [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g); Karpathy, [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI); Stanford [CS336: Language Modeling from Scratch](https://cs336.stanford.edu) (scaling laws, alignment, RL); Berkeley [CS285: Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse/) (policy gradients, PPO).

**Further reading:** Stanford CME 295 (2025) lectures [4 (SFT, LoRA)](https://cme295.stanford.edu/slides/fall25-cme295-lecture4.pdf), [5 (RLHF, PPO, DPO)](https://cme295.stanford.edu/slides/fall25-cme295-lecture5.pdf), [6 (reasoning, GRPO)](https://cme295.stanford.edu/slides/fall25-cme295-lecture6.pdf); Stanford CME 296 (2026) [lecture 6 (Flow-GRPO, Diffusion-DPO)](https://cme296.stanford.edu/slides/spring26-cme296-lecture6.pdf).
""")

for k, p in B.write(STEM).items():
    print(k, p)

L11 = "slides/2026-updates/L11_p11_with_scaling_laws_insert_2026.pdf"
L11A = "slides/2026-updates/L11a_rl_foundations_2026.pdf"
L11B = "slides/2026-updates/L11b_efficient_models_2026.pdf"
P12 = "slides/lectures/p12.pdf"
print("recap", make_recap(STEM, [
    (L11, [2, 7, 20]), (L11A, [21]), (L11B, [52, 54]),
    (L11, [46, 51, 54, 63]), (L11A, [36, 49]), (P12, [28, 55, 76]),
]))
