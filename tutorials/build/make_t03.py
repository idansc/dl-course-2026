"""Builds T03_sequences.ipynb (Tutorial 3, Deep Learning 89-6877, Fall 2026)."""
from nbkit import Builder
from recap import make_recap

STEM = "T03_sequences"
B = Builder()
md, code = B.md, B.code

md(rf"""
# Tutorial 3: sequences, from tokens to RNNs to state-space models
**Deep Learning 89-6877, Fall 2026.** TA: Tal Fiskus.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idansc/dl-course-2026/blob/main/tutorials/{STEM}.ipynb)
Recap slides: [{STEM}_recap.pdf](https://github.com/idansc/dl-course-2026/blob/main/tutorials/recap/{STEM}_recap.pdf)

Plan for today (≈ 75 min + recap):
1. Lecture recap: tokens, RNN, BPTT, LSTM, language models, SSMs and linear attention (12 min)
2. A byte-level BPE tokenizer trained on Tiny Shakespeare (10 min)
3. RNN and LSTM cells by hand, matched to `nn.LSTM`; a character-level LM and temperature sampling (17 min)
4. A diagonal linear SSM and linear attention: recurrent form = convolution = scan; cost vs. softmax attention (18 min)
5. Gradients through time: how far back does the loss reach? (8 min)
6. Factor Graph Attention: attention over several modalities; a synthetic VQA task (12 min)
7. If time: a GRU cell by hand

Runs on CPU (Colab or laptop) in a few minutes. Cells marked ✏️ are for you to try.
<<STUDENT>>
**Before the tutorial:** fill in every `# TODO` (replace the `...`). Each one is followed by a check cell that compares your answer with PyTorch (or with another form of the same computation). The solutions are presented in the tutorial.
<</STUDENT>>
""")

code(r"""
import math, re, time, urllib.request, warnings
warnings.filterwarnings("ignore", message="IProgress not found")
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
torch.manual_seed(0); np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)

DATA = Path("data"); DATA.mkdir(exist_ok=True)
path = DATA / "tinyshakespeare.txt"
if not path.exists():
    urllib.request.urlretrieve("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", path)
text = path.read_text()
split = int(0.9 * len(text))
train_text, val_text = text[:split], text[split:]
print(f"{len(text):,} characters; first 200:\n{text[:200]}")
""")

md(r"""
## 1. Lecture recap

**Tokens.** A model sees integer ids, not text. Characters give a tiny vocabulary but long sequences; words give a huge vocabulary with unknown words. **Byte-pair encoding (BPE)** sits between them: start from the 256 bytes, repeatedly merge the most frequent adjacent pair into a new token. Any string is still encodable (worst case: raw bytes).

**Vanilla (Elman) RNN.** One hidden vector, the same weights at every step:
$$h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b),\qquad y_t = W_{hy}h_t .$$

**Autoregressive language model.** $p(x_1,\dots,x_T)=\prod_t p(x_t\mid x_{<t})$. Train with teacher forcing: input $x_{1:T-1}$, target $x_{2:T}$, loss $=\frac1T\sum_t -\log p(x_{t+1}\mid x_{\le t})$. Generate by sampling one token at a time and feeding it back, with temperature $\tau$: $p \propto \exp(\text{logits}/\tau)$.

**Backprop through time (BPTT).** Unroll the graph over $T$ steps and backprop as usual; the shared $W$ sums its gradient over steps. Truncated BPTT carries the hidden state forward but backprops only through the last $k$ steps. The gradient reaching step $t$ from step $T$ is a product of Jacobians
$$\frac{\partial h_T}{\partial h_t}=\prod_{s=t+1}^{T}\mathrm{diag}\big(1-h_s^2\big)\,W_{hh},$$
which **explodes** if the largest singular value of $W_{hh}$ is $>1$ (fix: gradient clipping) and **vanishes** if it is $<1$ (fix: change the architecture).

**LSTM.** Two states, cell $c_t$ and hidden $h_t$; four gates from one matrix multiply:
$$\begin{pmatrix} i\\ f\\ o\\ g\end{pmatrix}=\begin{pmatrix}\sigma\\ \sigma\\ \sigma\\ \tanh\end{pmatrix}\Big(W\begin{pmatrix}h_{t-1}\\ x_t\end{pmatrix}+b\Big),\qquad c_t=f\odot c_{t-1}+i\odot g,\qquad h_t=o\odot\tanh(c_t).$$
From $c_t$ to $c_{t-1}$ the gradient is only multiplied elementwise by $f$, with no $W$: an additive "highway", like a ResNet skip. The GRU is a cheaper two-gate variant.

**State-space models (S4 → Mamba).** A *linear* recurrence with a fixed-size state, three views of one map:
- recurrent (inference): $h_k=\bar A h_{k-1}+\bar B x_k,\; y_k = C h_k$, $O(1)$ per token, no KV cache;
- convolutional (training): $y=\bar K * x$ with $\bar K=(C\bar B,\,C\bar A\bar B,\,C\bar A^2\bar B,\dots)$, parallel over $T$ (FFT);
- Mamba makes $\bar A,\bar B,C$ functions of the input (*selective*), which breaks the convolution; training uses a **parallel scan** instead.

**Linear attention** drops the softmax: $y_t=\sum_{s\le t}(q_t^\top k_s)v_s = S_tq_t$ with $S_t=S_{t-1}+v_tk_t^\top$, the same kind of recurrent state (Mamba-2 makes the link exact). A fixed-size state cannot store every past token, so exact recall is weaker than attention → **hybrids**: mostly linear/SSM layers plus a few full-attention layers (Jamba, Nemotron-H, Qwen3-Next 3:1, Kimi Linear).

**Attention over several modalities (Factor Graph Attention).** Each modality (image regions, question words, ...) gets a softmax belief over its entities from a learned, weighted sum of potentials: a unary score per entity and learned interaction grids with the other modalities, each collapsed by a learned marginal. Equations in section 6.
""")

# ---------------------------------------------------------------- BPE
md(r"""
## 2. A byte-level BPE tokenizer

Training loop (the GPT-2 recipe, minus details):
1. Split the text into "words" with a regex (a word keeps its leading space), so merges never cross word boundaries. Count each distinct word once with its frequency: much faster than scanning the raw text.
2. Represent each word as a tuple of byte values (0–255).
3. Repeat: count all adjacent pairs (weighted by word frequency), take the most frequent pair $(a,b)$, give it the next id, and replace every $(a,b)$ by that id.

The ordered list of merges *is* the tokenizer.
""")

code(r"""
PAT = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")

def merge(ids, pair, new_id):
    # replace every occurrence of `pair` in the tuple `ids` by `new_id`, left to right
    out, i = [], 0
    #>> scan left to right; on a match append new_id and skip two, else copy one id
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    #<<
    return tuple(out)

def train_bpe(text, n_merges):
    words = {tuple(w.encode("utf-8")): c for w, c in Counter(PAT.findall(text)).items()}
    merges = {}                                   # (a, b) -> new id; insertion order = merge rank
    for k in range(n_merges):
        stats = Counter()
        for w, c in words.items():
            for p in zip(w, w[1:]):
                stats[p] += c
        if not stats:
            break
        pair = max(stats, key=stats.get)
        merges[pair] = 256 + k
        words = {(merge(w, pair, 256 + k) if pair[0] in w else w): c for w, c in words.items()}
    return merges

t0 = time.time()
merges = train_bpe(train_text, 500)
vocab = {i: bytes([i]) for i in range(256)}
for (a, b), i in merges.items():
    vocab[i] = vocab[a] + vocab[b]
print(f"500 merges in {time.time() - t0:.1f}s; vocab size {len(vocab)}")
print("first 15 merges:", [vocab[i].decode() for i in list(merges.values())[:15]])
print("longest tokens: ", sorted((vocab[i].decode() for i in merges.values()), key=len)[-8:])
""")

md(r"""
**Encoding** a new word: start from its bytes, and repeatedly apply the merge with the **lowest rank** (learned earliest) among the adjacent pairs present, until no pair is in the table. This reproduces the order in which training built the tokens.
""")

code(r"""
def encode_word(word, merges):
    ids = tuple(word.encode("utf-8"))
    while len(ids) >= 2:
        #>> pick the adjacent pair with the smallest merge id (math.inf if not a merge); stop if it is not in merges, else merge it
        pair = min(zip(ids, ids[1:]), key=lambda p: merges.get(p, math.inf))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
        #<<
    return ids

def encode(text, merges):
    cache, out = {}, []
    for w in PAT.findall(text):
        if w not in cache:
            cache[w] = encode_word(w, merges)
        out.extend(cache[w])
    return out

def decode(ids):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")
""")

md(r"""
**Check:** the round trip must be exact on held-out text (including Unicode the training set never saw), and we compare token counts with GPT-2's tokenizer (50,257 tokens, trained on web text).
""")

code(r"""
ids = encode(val_text, merges)
assert decode(ids) == val_text, "round trip failed"
odd = "Wherefore art thou, Röntgen? 🙂"
assert decode(encode(odd, merges)) == odd
print("round trip OK.  example:", " | ".join(decode([i]) for i in encode("To be, or not to be, that is the question:", merges)))

from transformers import AutoTokenizer
gpt2 = AutoTokenizer.from_pretrained("gpt2", model_max_length=10**8)  # we only count tokens, no 1024 limit
n_gpt2 = len(gpt2(val_text)["input_ids"])
print(f"held-out text: {len(val_text):,} chars | our BPE (756 tokens): {len(ids):,} tokens "
      f"({len(val_text) / len(ids):.2f} chars/token) | GPT-2 (50,257): {n_gpt2:,} ({len(val_text) / n_gpt2:.2f} chars/token)")
""")

code(r"""
# compression vs. vocabulary size: encode with only the first k merges
ks = [0, 25, 50, 100, 200, 300, 400, 500]
cpt = [len(val_text) / len(encode(val_text, dict(list(merges.items())[:k]))) for k in ks]
plt.figure(figsize=(5.5, 3.5))
plt.plot([256 + k for k in ks], cpt, marker="o", label="our BPE (Shakespeare)")
plt.axhline(len(val_text) / n_gpt2, color="gray", ls="--", label="GPT-2, 50,257 tokens")
plt.xlabel("vocabulary size"); plt.ylabel("characters per token"); plt.title("BPE compression on held-out Shakespeare")
plt.legend(); plt.grid(alpha=.3); plt.show()
""")

md(r"""
The first merges are space+letter and frequent letter pairs (` t`, `he`, ` a`, `ou`), then whole common words (` the`) and, specific to this corpus, speaker names (`GLOUCESTER`). Words the merges did not reach are split into pieces (` qu | est | ion`). The first 50 merges lift compression from 1 to 1.4 characters per token; the last 100 add only 0.07. With 756 tokens we reach 2.1 characters per token; GPT-2 reaches 3.1 with 66× more tokens, much of them spent on web text this corpus never uses. Sequence length is what attention pays for ($O(T^2)$), so the tokenizer is also a compute choice.

✏️ Encode a Hebrew sentence with our tokenizer and with GPT-2 and count tokens per character. Why do both fall back to (roughly) bytes?
""")

# ---------------------------------------------------------------- RNN / LSTM
md(r"""
## 3. RNN and LSTM cells by hand

PyTorch convention (row vectors, separate input and hidden biases):
$$h' = \tanh(xW_{ih}^\top + b_{ih} + hW_{hh}^\top + b_{hh}).$$
For the LSTM, `weight_ih` is $(4H\times D)$ with the gates stacked in the order **i, f, g, o**.
""")

code(r"""
def rnn_cell(x, h, W_ih, W_hh, b_ih, b_hh):
    #>> one vanilla RNN step (PyTorch convention above)
    return torch.tanh(x @ W_ih.T + b_ih + h @ W_hh.T + b_hh)
    #<<

def lstm_cell(x, state, W_ih, W_hh, b_ih, b_hh):
    h, c = state
    gates = x @ W_ih.T + b_ih + h @ W_hh.T + b_hh
    i, f, g, o = gates.chunk(4, dim=-1)
    #>> apply σ / tanh to the gates, then c' = f⊙c + i⊙g and h' = o⊙tanh(c')
    i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
    c = f * c + i * g
    h = o * torch.tanh(c)
    #<<
    return h, c
""")

code(r"""
torch.manual_seed(0)
D, H, Bsz = 10, 20, 4
x, h0, c0 = torch.randn(Bsz, D), torch.randn(Bsz, H), torch.randn(Bsz, H)

ref = nn.RNNCell(D, H)
print("RNN cell  max |mine − torch| =", (rnn_cell(x, h0, ref.weight_ih, ref.weight_hh, ref.bias_ih, ref.bias_hh) - ref(x, h0)).abs().max().item())

ref = nn.LSTMCell(D, H)
h1, c1 = lstm_cell(x, (h0, c0), ref.weight_ih, ref.weight_hh, ref.bias_ih, ref.bias_hh)
h1_ref, c1_ref = ref(x, (h0, c0))
print("LSTM cell max |h − torch| =", (h1 - h1_ref).abs().max().item(), "  |c − torch| =", (c1 - c1_ref).abs().max().item())
""")

md(r"""
A full LSTM layer is the cell in a loop over time. We name the parameters exactly like `nn.LSTM` (`weight_ih_l0`, ...) so `load_state_dict` copies the weights across, and compare the whole output sequence and final state.
""")

code(r"""
class MyLSTM(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        k = 1 / math.sqrt(H)
        self.weight_ih_l0 = nn.Parameter(torch.empty(4 * H, D).uniform_(-k, k))
        self.weight_hh_l0 = nn.Parameter(torch.empty(4 * H, H).uniform_(-k, k))
        self.bias_ih_l0 = nn.Parameter(torch.empty(4 * H).uniform_(-k, k))
        self.bias_hh_l0 = nn.Parameter(torch.empty(4 * H).uniform_(-k, k))
        self.H = H

    def forward(self, x, state=None):            # x: (B, T, D), batch_first
        B, T, _ = x.shape
        h, c = state if state is not None else (x.new_zeros(B, self.H), x.new_zeros(B, self.H))
        outs = []
        for t in range(T):
            h, c = lstm_cell(x[:, t], (h, c), self.weight_ih_l0, self.weight_hh_l0, self.bias_ih_l0, self.bias_hh_l0)
            outs.append(h)
        return torch.stack(outs, 1), (h, c)

torch.manual_seed(0)
ref = nn.LSTM(D, H, batch_first=True)
mine = MyLSTM(D, H)
mine.load_state_dict(ref.state_dict())
xs = torch.randn(Bsz, 30, D)
out, (hT, cT) = mine(xs)
out_ref, (hT_ref, cT_ref) = ref(xs)
print("outputs max diff:", (out - out_ref).abs().max().item())
print("h_T     max diff:", (hT - hT_ref[0]).abs().max().item(), "  c_T max diff:", (cT - cT_ref[0]).abs().max().item())
""")

md(r"""
**Past exam question (Moed B, 2026)**

Network: INPUT → LSTM-20 → FC-10.
- INPUT: sequence length 15, feature dimension 12, i.e. input $15\times12$ with $x_t\in\mathbb{R}^{12}$.
- LSTM-20: 20 hidden units; assume the LSTM cell has 4 gates.
- FC-10: 10 output units, applied to the last hidden state.
- No bias anywhere.

Fill in the table (activation dimensions and number of learned parameters for INPUT $15\times12$ / 0, LSTM-20, FC-10). Show the LSTM parameter computation explicitly.

<<STUDENT>>
✏️ Your answer:
<</STUDENT>>
<<SOLUTION>>
Each of the 4 gates has $W\in\mathbb{R}^{20\times12}$ for the input and $U\in\mathbb{R}^{20\times20}$ for the previous hidden state: LSTM $= 4(12\cdot20 + 20\cdot20) = 4\cdot640 = \mathbf{2{,}560}$; output $15\times20$ (all steps) or $20$ (last state). FC-10: $20\cdot10 = \mathbf{200}$; output $10$. The sequence length does not enter the parameter count: the weights are shared over time.
<</SOLUTION>>
""")

code(r"""
#>> LSTM and FC parameter counts by formula (no bias)
p_lstm = 4 * (12 * 20 + 20 * 20)
p_fc = 20 * 10
#<<
lstm_q, fc_q = nn.LSTM(12, 20, bias=False, batch_first=True), nn.Linear(20, 10, bias=False)
out_q, (h_q, _) = lstm_q(torch.zeros(1, 15, 12))
print(f"LSTM-20: yours {p_lstm}, torch {sum(p.numel() for p in lstm_q.parameters())}, outputs {tuple(out_q.shape[1:])} / last state {tuple(h_q.shape[2:])}")
print(f"FC-10  : yours {p_fc}, torch {sum(p.numel() for p in fc_q.parameters())}, output {tuple(fc_q(h_q[0]).shape[1:])}")
""")

md(r"""
**Past exam question (Moed C, 2026)**

An LSTM layer with $d_h = 64$ hidden units runs on an input sequence with $x_t\in\mathbb{R}^{32}$. Every gate has a bias. **How many learned parameters does the layer have?** Show the computation and explain the contribution of each gate.

<<STUDENT>>
✏️ Your answer:
<</STUDENT>>
<<SOLUTION>>
Four gates (input $i$, forget $f$, output $o$, candidate $g$), each $\sigma\text{ or }\tanh(Wx_t + Uh_{t-1} + b)$ with $W\in\mathbb{R}^{64\times32}$ (2,048), $U\in\mathbb{R}^{64\times64}$ (4,096), $b\in\mathbb{R}^{64}$ (64): 6,208 per gate, $4\cdot 64\cdot(32+64+1) = \mathbf{24{,}832}$.
`nn.LSTM` reports **25,088**: PyTorch (like cuDNN) keeps two bias vectors per gate, $b_{ih}$ and $b_{hh}$, i.e. $4\cdot64 = 256$ extra parameters that are redundant (only their sum matters). Both numbers are right under their convention; the exam's (one bias per gate) is 24,832.
<</SOLUTION>>
""")

code(r"""
d_x, d_h = 32, 64
#>> parameter count with one bias vector per gate
p_lstm = 4 * d_h * (d_x + d_h + 1)
#<<
lstm_q = nn.LSTM(d_x, d_h)
print("yours (one bias per gate):", p_lstm)
print("nn.LSTM:", {n: tuple(p.shape) for n, p in lstm_q.named_parameters()}, "total", sum(p.numel() for p in lstm_q.parameters()))
print("nn.LSTM without the redundant b_hh:", sum(p.numel() for n, p in lstm_q.named_parameters() if n != "bias_hh_l0"))
""")

md(r"""
### A character-level language model

Same math, but for speed we train with `nn.LSTM` (fused kernels) now that we know what it computes. 65 distinct characters; a uniform guess costs $\ln 65 = 4.17$ nats per character. Teacher forcing: random windows of 100 characters, the target is the window shifted by one. Gradient clipping at norm 1 guards against the exploding case from the recap.
""")

code(r"""
chars = sorted(set(text)); V = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}; itos = {i: ch for ch, i in stoi.items()}
data = torch.tensor([stoi[ch] for ch in text])
train_data, val_data = data[:split], data[split:]

def get_batch(d, B=64, T=100):
    ix = torch.randint(len(d) - T - 1, (B,))
    x = torch.stack([d[i:i + T] for i in ix]); y = torch.stack([d[i + 1:i + T + 1] for i in ix])
    return x.to(device), y.to(device)

class CharLM(nn.Module):
    def __init__(self, V, emb=64, hidden=256):
        super().__init__()
        self.emb, self.rnn, self.head = nn.Embedding(V, emb), nn.LSTM(emb, hidden, batch_first=True), nn.Linear(hidden, V)
    def forward(self, idx, state=None):
        out, state = self.rnn(self.emb(idx), state)
        return self.head(out), state

@torch.no_grad()
def val_loss(model, n=20):
    model.eval()
    l = sum(F.cross_entropy(model(x)[0].reshape(-1, V), y.reshape(-1)).item() for x, y in (get_batch(val_data) for _ in range(n))) / n
    model.train(); return l

torch.manual_seed(0)
lm = CharLM(V).to(device)
opt = torch.optim.AdamW(lm.parameters(), lr=3e-3)
STEPS = 800                                     # Colab GPU: hidden=512, 2 layers, 5000 steps
losses, t0 = [], time.time()
print(f"step    0  val loss {val_loss(lm):.3f}  (ln V = {math.log(V):.3f})")
for step in range(1, STEPS + 1):
    x, y = get_batch(train_data)
    logits, _ = lm(x)
    loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
    opt.step(); losses.append(loss.item())
    if step % 200 == 0:
        print(f"step {step:4d}  train {np.mean(losses[-50:]):.3f}  val loss {val_loss(lm):.3f}  ({time.time() - t0:.0f}s)")

plt.figure(figsize=(5.5, 3.3)); plt.plot(losses, lw=.7)
plt.axhline(math.log(V), color="gray", ls="--", label="uniform guess, ln 65")
plt.xlabel("step"); plt.ylabel("cross-entropy (nats/char)"); plt.title("Char-level LSTM LM, training loss"); plt.legend(); plt.show()
""")

md(r"""
**Sampling.** Feed the prompt once to get the state, then loop: divide the last logits by the temperature $\tau$, softmax, draw one character, feed it back with the carried state. $\tau\to0$ is greedy argmax; $\tau=1$ is the model's distribution; $\tau>1$ flattens it.
""")

code(r"""
@torch.no_grad()
def sample(model, prompt, n=250, temperature=1.0):
    model.eval()
    idx = torch.tensor([[stoi[c] for c in prompt]], device=device)
    logits, state = model(idx)
    out = list(prompt)
    for _ in range(n):
        #>> probs = softmax(last logits / temperature); nxt = one draw from probs (torch.multinomial), shape (1,)
        probs = F.softmax(logits[0, -1] / temperature, dim=-1)
        nxt = torch.multinomial(probs, 1)
        #<<
        out.append(itos[nxt.item()])
        logits, state = model(nxt.view(1, 1), state)
    return "".join(out)

@torch.no_grad()
def greedy(model, prompt, n=250):
    idx = torch.tensor([[stoi[c] for c in prompt]], device=device)
    logits, state = model(idx); out = list(prompt)
    for _ in range(n):
        nxt = logits[0, -1].argmax().view(1, 1)
        out.append(itos[nxt.item()]); logits, state = model(nxt, state)
    return "".join(out)

print("τ → 0 equals greedy argmax:", sample(lm, "ROMEO:", temperature=1e-4) == greedy(lm, "ROMEO:"))
""")

code(r"""
for tau in [0.5, 1.0, 1.5]:
    torch.manual_seed(1)
    print(f"----- temperature {tau} -----\n{sample(lm, 'ROMEO:', temperature=tau)}\n")
""")

md(r"""
<<SOLUTION>>
After 800 steps the validation loss is ≈1.6 nats/char (from 4.17). The model has the format: speaker names in capitals followed by a colon, line breaks, short lines. At $\tau=0.5$ almost every word is real and the phrasing is generic ("What is the world, that"); at $\tau=1$ invented words appear; at $\tau=1.5$ there are capitals inside words and broken names (`MERCUTI:`). The lecture's "train more" slides show the same progression with training time.
<</SOLUTION>>
✏️ Swap `nn.LSTM` for `nn.RNN` in `CharLM`, train the same 800 steps, and compare the validation loss. Then train on our BPE ids instead of characters: what changes in the loss value and why can't you compare it directly to the character-level number?
""")

# ---------------------------------------------------------------- SSM
md(r"""
## 4. Linear recurrences: SSMs and linear attention

**Diagonal linear SSM** (S4D/Mamba style, already discretized). Each of the $D$ input channels has its own $N$-dimensional state with diagonal decay $a\in(0,1)^{D\times N}$:
$$h_t = a\odot h_{t-1} + b\odot x_t,\qquad y_t=\textstyle\sum_n c_n h_{t,n}.$$
Unrolling, $h_t=\sum_{s\le t}a^{t-s}b\,x_s$, so $y = K * x$ with kernel $K_k=\sum_n c_n a_n^k b_n$ (one causal convolution per channel). The recurrence never mixes channels: a full model puts linear layers and gating between SSM layers.
""")

code(r"""
def ssm_recurrent(x, a, b, c):
    # x: (B, T, D); a, b, c: (D, N), or a: (B, T, D, N) when it changes with time (selective)
    B, T, D = x.shape
    h = x.new_zeros(B, D, b.shape[-1]); ys = []
    for t in range(T):
        a_t = a[:, t] if a.dim() == 4 else a
        #>> h_t = a_t ⊙ h_{t−1} + b ⊙ x_t (x_t broadcast over the state dim N); append y_t = Σ_n c ⊙ h_t
        h = a_t * h + b * x[:, t, :, None]
        ys.append((c * h).sum(-1))
        #<<
    return torch.stack(ys, 1)

def ssm_kernel(a, b, c, T):
    k = torch.arange(T, dtype=a.dtype)
    #>> K[k, d] = Σ_n c[d,n] · a[d,n]^k · b[d,n], shape (T, D)
    return (c * b * a[None] ** k[:, None, None]).sum(-1)
    #<<

def causal_conv_fft(x, K):
    # y[:, t, d] = Σ_{s≤t} K[t−s, d] x[:, s, d]; zero-pad to 2T so the circular FFT convolution does not wrap around
    T = x.shape[1]
    return torch.fft.irfft(torch.fft.rfft(x, n=2 * T, dim=1) * torch.fft.rfft(K, n=2 * T, dim=0), n=2 * T, dim=1)[:, :T]
""")

code(r"""
torch.manual_seed(0)
Bsz, T, D, N = 2, 256, 4, 8
x = torch.randn(Bsz, T, D, dtype=torch.float64)
a = torch.exp(-torch.exp(torch.empty(D, N, dtype=torch.float64).uniform_(math.log(1e-3), math.log(1e-1))))  # decays 0.90–0.999
b, c = torch.randn(D, N, dtype=torch.float64), torch.randn(D, N, dtype=torch.float64)

y_rec = ssm_recurrent(x, a, b, c)
K = ssm_kernel(a, b, c, T)
y_fft = causal_conv_fft(x, K)
# library reference: grouped conv1d (cross-correlation, so flip the kernel and left-pad)
y_conv = F.conv1d(F.pad(x.transpose(1, 2), (T - 1, 0)), K.T.flip(-1).unsqueeze(1), groups=D).transpose(1, 2)
print("recurrent vs FFT conv    max diff:", (y_rec - y_fft).abs().max().item())
print("recurrent vs F.conv1d    max diff:", (y_rec - y_conv).abs().max().item())
""")

md(r"""
**Selective SSM (Mamba).** If $a_t$ depends on the input $x_t$, the kernel is different at every step and the convolution view is gone. The recurrence $h_t=a_th_{t-1}+u_t$ (with $u_t=b\,x_t$) is still parallel, because composing two steps is again a step of the same form:
$$(a_1,u_1)\circ(a_2,u_2) = (a_2a_1,\; a_2u_1+u_2).$$
This operator is associative, so a **scan** computes all prefixes in $\log_2 T$ vectorized rounds (Hillis–Steele): in round $r$, every position combines with the one $2^r$ steps back.
""")

code(r"""
def scan(A, U):
    # all h_t for h_t = A_t ⊙ h_{t−1} + U_t, h_0 = 0; time is dim 1; log2(T) rounds
    off = 1
    while off < A.shape[1]:
        #>> U_t ← A_t ⊙ U_{t−off} + U_t and A_t ← A_t ⊙ A_{t−off} for t ≥ off (first `off` positions unchanged); use the OLD A in both
        U = torch.cat([U[:, :off], A[:, off:] * U[:, :-off] + U[:, off:]], 1)
        A = torch.cat([A[:, :off], A[:, off:] * A[:, :-off]], 1)
        #<<
        off *= 2
    return U

# input-dependent decay, a toy version of Mamba's selection: a_t = σ(w ⊙ x_t + 3)
w = torch.randn(D, N, dtype=torch.float64)
a_sel = torch.sigmoid(w * x[..., None] + 3)                   # (B, T, D, N)
y_rec_sel = ssm_recurrent(x, a_sel, b, c)
h_scan = scan(a_sel, b * x[..., None])                         # (B, T, D, N)
y_scan = (c * h_scan).sum(-1)
print(f"selective: recurrent vs scan max diff: {(y_rec_sel - y_scan).abs().max().item():.2e}  ({math.ceil(math.log2(T))} rounds instead of {T} steps)")
print(f"time-invariant: scan vs recurrent max diff: {((c * scan(a.expand(Bsz, T, D, N), b * x[..., None])).sum(-1) - y_rec).abs().max().item():.2e}")
""")

md(r"""
**Linear attention** is the same object with a matrix-valued state. With decay $\gamma\in(0,1]$ ($\gamma=1$: plain linear attention; $\gamma<1$: RetNet / a scalar-decay Mamba-2 head):
$$S_t=\gamma S_{t-1}+v_tk_t^\top,\quad y_t=S_tq_t \qquad\Longleftrightarrow\qquad Y=\big((QK^\top)\odot M\big)V,\quad M_{ts}=\begin{cases}\gamma^{t-s}& s\le t\\ 0 & s>t\end{cases}$$
Left: recurrent, $O(d^2)$ state, $O(1)$ per token. Right: parallel, like attention without the softmax.
""")

code(r"""
def linattn_recurrent(q, k, v, gamma=1.0):
    B, T, d = q.shape
    S = q.new_zeros(B, d, d); ys = []
    for t in range(T):
        #>> S_t = γ S_{t−1} + v_t k_tᵀ (outer product, (B, d, d)); append y_t = S_t q_t, shape (B, d)
        S = gamma * S + v[:, t, :, None] * k[:, t, None, :]
        ys.append((S @ q[:, t, :, None]).squeeze(-1))
        #<<
    return torch.stack(ys, 1)

def linattn_parallel(q, k, v, gamma=1.0):
    T = q.shape[1]
    t = torch.arange(T)
    #>> decay mask M[t, s] = γ^(t−s) for s ≤ t, else 0 (torch.where)
    M = torch.where(t[:, None] >= t[None, :], gamma ** (t[:, None] - t[None, :]).to(q.dtype), torch.zeros((), dtype=q.dtype))
    #<<
    return ((q @ k.transpose(1, 2)) * M) @ v

torch.manual_seed(0)
q, k, v = (torch.randn(2, 128, 16, dtype=torch.float64) for _ in range(3))
for gamma in [1.0, 0.9]:
    print(f"γ = {gamma}: recurrent vs parallel max diff {(linattn_recurrent(q, k, v, gamma) - linattn_parallel(q, k, v, gamma)).abs().max().item():.2e}")
""")

md(r"""
Neither form above is what fast kernels use: the recurrent form is a Python loop over $T$, the parallel form builds a $T\times T$ matrix. Real implementations (Mamba-2, GLA, Gated DeltaNet) are **chunked**: parallel attention-like form inside chunks of $C$ tokens, the recurrent state carried between chunks. Cost $O(TCd + Td^2)$, linear in $T$.
""")

code(r"""
def linattn_chunked(q, k, v, C=64):
    B, T, d = q.shape
    S = q.new_zeros(B, d, d); out = []
    mask = torch.tril(torch.ones(C, C, dtype=q.dtype))
    for i in range(0, T, C):
        qc, kc, vc = q[:, i:i + C], k[:, i:i + C], v[:, i:i + C]
        m = mask[:qc.shape[1], :qc.shape[1]]
        out.append(((qc @ kc.transpose(1, 2)) * m) @ vc + qc @ S.transpose(1, 2))   # within chunk + from earlier chunks
        S = S + vc.transpose(1, 2) @ kc                                               # Σ v kᵀ over the chunk
    return torch.cat(out, 1)

print("chunked vs parallel max diff:", (linattn_chunked(q, k, v, C=32) - linattn_parallel(q, k, v)).abs().max().item())
""")

md(r"""
**Cost vs. sequence length.** Causal softmax attention (materialized $T\times T$ scores, as in a naive implementation), chunked linear attention, and the SSM via FFT convolution, on CPU, one sequence, $d=64$. The right panel is memory at **generation** time: attention keeps a KV cache that grows with every token; the linear layers keep a fixed state.
""")

code(r"""
def softmax_attn(q, k, v):
    T, d = q.shape[1], q.shape[2]
    s = (q @ k.transpose(1, 2)) / math.sqrt(d)
    s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
    return s.softmax(-1) @ v

def bench(fn, *args, reps=5):
    fn(*args); best = math.inf
    for _ in range(reps):
        t0 = time.perf_counter(); fn(*args); best = min(best, time.perf_counter() - t0)
    return best

d, Nst = 64, 16
Ts = [256, 512, 1024, 2048, 4096, 8192, 16384]
res = {"softmax attention": [], "linear attention (chunked)": [], "SSM (FFT conv)": []}
a_b = torch.rand(d, Nst) * 0.1 + 0.9; b_b, c_b = torch.randn(d, Nst), torch.randn(d, Nst)
with torch.no_grad():
    for T in Ts:
        q, k, v = (torch.randn(1, T, d) for _ in range(3))
        res["softmax attention"].append(bench(softmax_attn, q, k, v) if T <= 8192 else np.nan)
        res["linear attention (chunked)"].append(bench(linattn_chunked, q, k, v))
        res["SSM (FFT conv)"].append(bench(lambda x: causal_conv_fft(x, ssm_kernel(a_b, b_b, c_b, x.shape[1])), v))
        print(f"T={T:6d}  " + "  ".join(f"{n}: {r[-1] * 1e3:7.1f} ms" for n, r in res.items()))

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for n, r in res.items():
    ax[0].plot(Ts, r, marker="o", label=n)
ax[0].set(xscale="log", yscale="log", xlabel="sequence length T", ylabel="forward time (s)", title="Training-style forward pass, CPU")
ax[0].legend(); ax[0].grid(alpha=.3, which="both")
pos = np.array([256, 1024, 4096, 16384, 65536, 262144])
L = 32                                                         # layers, fp16 bytes
ax[1].plot(pos, 2 * L * pos * d * 2 / 2**20, marker="o", label="attention: KV cache  2·L·T·d")
ax[1].plot(pos, np.full(len(pos), L * d * d * 2 / 2**20), marker="o", label="linear attention: state  L·d²")
ax[1].plot(pos, np.full(len(pos), L * d * Nst * 2 / 2**20), marker="o", label="SSM: state  L·D·N")
ax[1].set(xscale="log", yscale="log", xlabel="tokens generated so far", ylabel="MiB (one head / 64 channels)", title="Memory while generating (32 layers, fp16)")
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, which="both")
plt.tight_layout(); plt.show()
""")

md(r"""
<<SOLUTION>>
Softmax attention's time grows at least 4× per doubling of $T$ at the top end (slope ≥ 2 on the log-log plot: $O(T^2)$, worse once the $T\times T$ matrix no longer fits in cache); chunked linear attention grows ≈2× per doubling ($O(T)$) and is more than 10× faster from $T=4096$ on. (Wall-clock timings on a laptop are noisy; read the slopes, not single points.) The FFT SSM ($O(T\log T)$, including building the kernel) is *slower* than softmax attention up to $T=2048$ and faster at $T=8192$, with the crossover in between: at short lengths one dense matrix multiply is hard to beat, which is why short-context models never needed these layers. During generation the gap is in memory: the KV cache grows linearly without bound, the linear-attention / SSM state is constant.
<</SOLUTION>>
The price of the fixed state: it cannot store every past token, so exact recall (copying, needle-in-a-haystack) degrades. Production hybrids keep ~1 full-attention layer in 4–8.

✏️ In the selective SSM, what does the model do when it drives $a_t\to0$ at some step? And $a_t\to1$? Relate to the LSTM forget gate.
""")

# ---------------------------------------------------------------- vanishing gradients
md(r"""
## 5. Gradients through time: how far back does the loss reach?

The recap says vanilla RNN gradients vanish because $\partial h_T/\partial h_t$ is a product of $T-t$ Jacobians. Let's measure it. Same input sequence ($T=100$), loss = a random linear readout of the **last** state only, and record $\lVert\partial L/\partial h_t\rVert$ for every step $t$ (`retain_grad` on each state). Untrained models, PyTorch default initialization:
- vanilla RNN, $H=64$;
- LSTM, $H=64$, and the same LSTM with the **forget-gate bias set to 1 or 3** (Gers et al. 2000; Jozefowicz et al. 2015 recommend 1), a one-line change;
- the diagonal SSM from section 4 ($16$ channels × $4$ states $= 64$ numbers), decays $a\in[0.90, 0.999]$.

For the LSTM we take the gradient w.r.t. the cell $c_t$ (the highway) and w.r.t. $h_t$.
""")

code(r"""
torch.manual_seed(0)
T, Din, H, Bsz = 100, 16, 64, 32
xs = torch.randn(Bsz, T, Din, requires_grad=True)   # so every state is part of the autograd graph

def grad_profile(step, state, pick):
    # unroll, keep every state, backprop a readout of the LAST state, return ||dL/dstate_t|| for t = 1..T
    kept = []
    for t in range(T):
        state = step(xs[:, t], state)
        s = pick(state); s.retain_grad(); kept.append(s)
    (kept[-1] * torch.randn_like(kept[-1])).sum().backward()
    g = torch.tensor([s.grad.norm().item() for s in kept])
    return (g / g[-1]).numpy()                 # relative to the gradient at the last step

rnn = nn.RNNCell(Din, H)
lstm = nn.LSTMCell(Din, H)
def with_forget_bias(cell, value):
    new = nn.LSTMCell(Din, H); new.load_state_dict(cell.state_dict())
    with torch.no_grad():
        new.bias_ih[H:2 * H] = value; new.bias_hh[H:2 * H] = 0.0  # @student: pass  # TODO: forget-gate bias = value (gate order i, f, g, o; set it in bias_ih, zero that gate in bias_hh)
    return new
Nst = 4
a_s = torch.exp(-torch.exp(torch.empty(Din, Nst).uniform_(math.log(1e-3), math.log(1e-1))))
b_s = torch.randn(Din, Nst)

z = lambda: torch.zeros(Bsz, H)
lstm_step = lambda cell: (lambda x, s: lstm_cell(x, s, cell.weight_ih, cell.weight_hh, cell.bias_ih, cell.bias_hh))
profiles = {
    "vanilla RNN": grad_profile(lambda x, h: rnn_cell(x, h, rnn.weight_ih, rnn.weight_hh, rnn.bias_ih, rnn.bias_hh), z(), lambda s: s),
    "LSTM, ∂L/∂h": grad_profile(lstm_step(lstm), (z(), z()), lambda s: s[0]),
    "LSTM, ∂L/∂c": grad_profile(lstm_step(lstm), (z(), z()), lambda s: s[1]),
    "LSTM forget bias 1, ∂L/∂c": grad_profile(lstm_step(with_forget_bias(lstm, 1.0)), (z(), z()), lambda s: s[1]),
    "LSTM forget bias 3, ∂L/∂c": grad_profile(lstm_step(with_forget_bias(lstm, 3.0)), (z(), z()), lambda s: s[1]),
    "diagonal SSM": grad_profile(lambda x, h: a_s * h + b_s * x[..., None], torch.zeros(Bsz, Din, Nst), lambda s: s),
}
dist = np.arange(T)[::-1]                      # T − t: steps between the state and the loss
plt.figure(figsize=(7, 4.2))
for name, g in profiles.items():
    plt.plot(dist, g, label=name, ls="--" if "∂h" in name else "-")
plt.yscale("log"); plt.ylim(1e-14, 10)
plt.xlabel("distance from the loss, T − t (steps)"); plt.ylabel("‖∂L/∂state_t‖ / ‖∂L/∂state_T‖")
plt.title("How far back does the gradient reach? (untrained, default init)"); plt.legend(fontsize=8); plt.grid(alpha=.3)
plt.show()
for name, g in profiles.items():
    print(f"{name:28s} relative gradient at distance 10: {g[-11]:.1e}   at 50: {g[-51]:.1e}   at 99: {g[0]:.1e}")
""")

md(r"""
<<SOLUTION>>
- **Vanilla RNN**: the gradient shrinks by a factor ≈0.5 per step: $10^{-3}$ at 10 steps back, $5\cdot10^{-14}$ at 50, and it underflows to exactly 0 in float32 by 99. Whatever happened 20 steps ago cannot affect learning.
- **LSTM at default init is not much better** ($5\cdot10^{-3}$ at 10 steps, $10^{-11}$ at 50). The cell path is free of $W$, but it is still multiplied by $f_t$ at every step, and with biases near 0 the forget gate sits at $\sigma(0)\approx0.5$. The "uninterrupted gradient flow" of the lecture holds only when $f\approx1$.
- **Forget bias 1** ($\sigma(1)=0.73$): $10^{-1}$ at 10 steps, $10^{-8}$ at 99. **Forget bias 3** ($\sigma(3)=0.95$): the gradient stays within a factor of 2 of its value at the loss over all 100 steps. One bias value changes the reach by 20 orders of magnitude.
- **Diagonal SSM**: no nonlinearity and decays $a\in[0.90,0.999]$, so the gradient through channel $n$ is exactly $a_n^{T-t}$; the slow channels dominate and the total is still 0.55 at 99 steps. This is the reason S4/Mamba parametrize $a$ with timescales spread over orders of magnitude.

These are untrained profiles; training moves the gates. They still decide whether the first gradient steps can see long-range structure at all.
<</SOLUTION>>
The practical rules that follow: initialize the LSTM forget bias to 1 (some libraries do it for you; PyTorch does not); clip gradients for the exploding case; and if a task needs dependencies hundreds of steps long, use a gated or state-space recurrence (or attention), not a vanilla RNN.

✏️ Scale the vanilla RNN's `weight_hh` by 3 (`rnn.weight_hh.data *= 3`) and re-run. Does the gradient still vanish? What does `tanh` saturation do to the Jacobian $\mathrm{diag}(1-h^2)W_{hh}$?
""")

md(r"""
**Past exam question (Moed C, 2026), backpropagation through time**

A vanilla RNN with a scalar hidden state runs on an input sequence of length 3. Figure, in words: a chain $h_0=0 \to h_1 \to h_2 \to h_3$ with weight $w_h$ on every recurrent edge; input $x_t$ enters $h_t$ with weight $w_x$; the output $\hat y$ is computed from $h_3$ with weight $v$. For $t\in\{1,2,3\}$:
$$z_t = w_xx_t + w_hh_{t-1},\qquad h_t=\tanh(z_t),\qquad h_0=0,\qquad \hat y = v\,h_3,\qquad \tanh'(z) = 1-\tanh^2(z),$$
$$L = \tfrac12(y-\hat y)^2 + \tfrac{\lambda}{2}\big(w_x^2+w_h^2+v^2\big),\qquad \lambda>0.$$
The weights $w_x, w_h, v$ are shared across time steps.

**(a)** Give algebraic expressions for $\frac{\partial L}{\partial \hat y}$, $\frac{\partial L}{\partial v}$, $\frac{\partial L}{\partial h_3}$, $\frac{\partial L}{\partial h_2}$, and $\frac{\partial L}{\partial w_h}$ (sum the contributions of every step in which $w_h$ appears).

**(b)** The same architecture is extended to a long sequence of $T\gg3$ steps, $\hat y = v\,h_T$. In BPTT, $\partial L/\partial w_h$ contains the product $\prod_{t=k+1}^{T}(1-h_t^2)\,w_h$ on the way from $h_T$ back to $h_k$ ($k\ll T$). Under which conditions on $w_h$ and on the hidden states $\{h_t\}$ does this product decay exponentially with the sequence length (vanishing gradient)? What does this mean in practice for training the RNN on long sequences? Propose **one architectural change** that reduces the problem and explain briefly how it works.

<<STUDENT>>
✏️ Your answer:
<</STUDENT>>
<<SOLUTION>>
**(a)** $\frac{\partial L}{\partial\hat y} = \hat y - y$, $\;\frac{\partial L}{\partial v} = (\hat y-y)h_3 + \lambda v$, $\;\frac{\partial L}{\partial h_3} = (\hat y-y)v$, $\;\frac{\partial L}{\partial h_2} = (\hat y-y)\,v\,(1-h_3^2)\,w_h$.
With $\delta_t = \partial L/\partial z_t$: $\delta_3 = (\hat y-y)v(1-h_3^2)$, $\delta_2 = \delta_3 w_h(1-h_2^2)$, $\delta_1 = \delta_2 w_h (1-h_1^2)$, and since $\partial z_t/\partial w_h = h_{t-1}$,
$$\frac{\partial L}{\partial w_h} = \delta_3h_2 + \delta_2h_1 + \delta_1h_0 + \lambda w_h = \delta_3h_2+\delta_2h_1+\lambda w_h\quad(h_0=0).$$
**(b)** $0 < 1-h_t^2\le1$, so the product decays exponentially when $|w_h|(1-h_t^2) < 1$ for most steps: always if $|w_h|<1$, and also for larger $|w_h|$ when the states saturate ($|h_t|\to1$, so $1-h_t^2\to0$). In practice the gradient from the loss at step $T$ does not reach early steps, so the RNN cannot learn long-range dependencies (section 5 measured ~$10^{-3}$ after 10 steps for a vanilla RNN). Fix: an **LSTM/GRU**. The cell state is updated additively, $c_t = f_t\odot c_{t-1} + i_t\odot g_t$, so $\partial c_t/\partial c_{t-1} = f_t$, with no $w_h$ and no $\tanh'$; with the forget gate near 1 the gradient survives hundreds of steps (forget-bias curves above).
<</SOLUTION>>
""")

code(r"""
torch.manual_seed(1)
xs_q = torch.randn(3, dtype=torch.float64)
w_x, w_h, v_q = (torch.randn((), dtype=torch.float64, requires_grad=True) for _ in range(3))
y_q, lam = 0.7, 0.1
hs = [torch.zeros((), dtype=torch.float64)]
for t in range(3):
    hs.append(torch.tanh(w_x * xs_q[t] + w_h * hs[-1]))
y_hat = v_q * hs[3]
L = 0.5 * (y_q - y_hat) ** 2 + lam / 2 * (w_x ** 2 + w_h ** 2 + v_q ** 2)
L.backward()
h0, h1, h2, h3 = (h.item() for h in hs); wh, v_ = w_h.item(), v_q.item(); r = y_hat.item() - y_q
#>> dL/dv and dL/dw_h from your formulas in (a)
dv = r * h3 + lam * v_
d3 = r * v_ * (1 - h3 ** 2)
d2 = d3 * wh * (1 - h2 ** 2)
d1 = d2 * wh * (1 - h1 ** 2)
dwh = d3 * h2 + d2 * h1 + d1 * h0 + lam * wh
#<<
print(f"dL/dv  : formula {dv:.10f}   autograd {v_q.grad.item():.10f}")
print(f"dL/dw_h: formula {dwh:.10f}   autograd {w_h.grad.item():.10f}")
""")

# ---------------------------------------------------------------- FGA
md(r"""
## 6. Factor Graph Attention: attention over several modalities

Lecture 7's attention attends over **one** set (encoder states, image regions) given a query. In visual question answering (VQA) and visual dialog there are several sets, and each should be attended in the context of the others. **Factor Graph Attention** (FGA; Schwartz, Schwing, Hazan, CVPR 2019) treats each set as a node of a factor graph.

**Setup.** Modalities ("utilities") $U_1,\dots,U_M$; modality $i$ has $n_i$ entities with embeddings $\hat u\in\mathbb{R}^{d_i}$ (49 image regions, 15 question words, 100 candidate answers, ...). FGA outputs, per modality, a **belief** $b_i$ (a distribution over its $n_i$ entities) and an **attended vector** $a_i=\sum_u b_i(u)\,\hat u$.

**Potentials** (one score per entity each):
- **Unary**, is the entity salient on its own: $\psi_i(u)=v^\top\mathrm{relu}(V\hat u)$, $V\in\mathbb{R}^{d_i\times d_i}$, $v\in\mathbb{R}^{d_i}$.
- **Pairwise** between modalities $i$ and $j$. Each side gets its own learned projection, $L\in\mathbb{R}^{d\times d_i}$, $R\in\mathbb{R}^{d\times d_j}$, $d=\max(d_i,d_j)$, and the interaction grid is the cosine
$$C_{ij}(u,w)=\Big\langle \tfrac{L\hat u}{\lVert L\hat u\rVert},\tfrac{R\hat w}{\lVert R\hat w\rVert}\Big\rangle\in\mathbb{R}^{n_i\times n_j},$$
batch-normalized over the flattened $n_in_j$ grid. A **learned marginal** collapses it into a message per entity, $\mu_{j\to i}(u)=\sum_{w}W_w\,\tilde C_{ij}(u,w)+c$, i.e. `Linear(n_j → 1)` applied to each row; the other side gets `Linear(n_i → 1)` applied to each column. One grid, two messages.
- **Self**: the same construction with $j=i$ (own $L$, $R$), an entity in the context of its own modality.

**Belief.** Stack the $K$ potentials that reach modality $i$ and mix them with a learned, bias-free weight vector:
$$b_i(u)=\operatorname{softmax}_u\Big(\textstyle\sum_{k=1}^{K}w_k\,\psi_k(u)\Big),\qquad \texttt{Linear(K → 1, bias=False)}.$$

Every stage is learned: the factors produce the grid ($L$, $R$), the marginal collapses it ($W$), the mix combines the potentials ($w$). Replacing any of them by a fixed operation (a frozen similarity matrix, a max or mean over the grid) is a different, weaker model; the BatchNorm on the grid matters too (VisDial ablation: MRR 0.6301 without, 0.6525 with).

**High order.** The precursor, High-Order Attention (NeurIPS 2017), adds a **ternary** factor for multiple-choice VQA: $T(x,y,z)=\sum_d\tilde x_d\tilde y_d\tilde z_d$ over projected, normalized question words, regions and candidate answers, batch-normalized, with three learned marginals (`Linear(n_y n_z → 1)`, ...). It scores triples that no pair can see, at $n_xn_yn_z$ values per example.

**Bridge to self-attention (Tutorial 4).** The self-interaction grid $\langle L\hat u, R\hat w\rangle$ over one modality $X$ is exactly $QK^\top$ with $Q=XL^\top$, $K=XR^\top$ (cosine instead of a $1/\sqrt d$ scale). Self-attention **keeps the full grid**: a softmax per row $u$ gives $n$ distributions and a new vector per entity, a sequence in, a sequence out. FGA **marginalizes** the grid to one score per entity: one distribution per modality and one pooled vector, which is what a VQA classifier consumes. Cross-attention is the pairwise grid kept per row in the same way.

We implement the three pieces with the parameter names of the reference code (github.com/idansc/fga), check them against a naive loop, then train on a synthetic VQA task. Layout everywhere: `(batch, entities, channels)`.
""")

code(r"""
def l2_normalize(x, eps=1e-12):
    # x / ||x|| per row; an all-zero row (a masked entity) stays 0 with zero gradient (F.normalize gives NaN gradients there)
    n2 = x.pow(2).sum(-1, keepdim=True)
    return x * torch.where(n2 > eps, torch.rsqrt(n2.clamp_min(eps)), torch.zeros_like(n2))

class Unary(nn.Module):
    # ψ(u) = vᵀ relu(V û):  (B, n, d) -> (B, n)
    def __init__(self, d, dropout=0.5):
        super().__init__()
        self.embed, self.feature_reduce, self.dropout = nn.Linear(d, d), nn.Linear(d, 1), dropout   # V, v

    def forward(self, X):
        #>> relu(V û), dropout on that hidden layer (F.dropout(..., training=self.training)), then vᵀ(·); squeeze the last axis
        h = F.dropout(F.relu(self.embed(X)), p=self.dropout, training=self.training)
        return self.feature_reduce(h).squeeze(-1)
        #<<

class Pairwise(nn.Module):
    # cosine grid of learned projections -> BatchNorm over the flattened grid -> learned marginal per side
    def __init__(self, dx, nx, dy=None, ny=None, self_interaction=False):
        super().__init__()
        dy, ny = (dx, nx) if dy is None else (dy, ny)
        e = max(dx, dy)
        self.nx, self.ny = nx, ny
        self.embed_X, self.embed_Y = nn.Linear(dx, e), nn.Linear(dy, e)    # L and R
        self.normalize_S = nn.BatchNorm1d(nx * ny)
        self.margin_X = nn.Linear(ny, 1)                                    # message to X: collapses each row (n_Y values)
        if not self_interaction:
            self.margin_Y = nn.Linear(nx, 1)                                # message to Y: collapses each column (n_X values)

    def forward(self, X, Y=None):
        # X: (B, nx, dx); Y: (B, ny, dy), or None for the self-interaction (Y = X). Returns ψ_X (B, nx) [and ψ_Y (B, ny)]
        #>> S = cosine grid (B, nx, ny) of the projected, l2-normalized X and Y; BatchNorm1d over the flattened grid, reshape back
        x = l2_normalize(self.embed_X(X))
        y = l2_normalize(self.embed_Y(X if Y is None else Y))
        S = x @ y.transpose(1, 2)
        S = self.normalize_S(S.reshape(-1, self.nx * self.ny)).view(-1, self.nx, self.ny)
        #<<
        #>> ψ_X = margin_X applied to every row of S, shape (B, nx)
        psi_X = self.margin_X(S).squeeze(-1)
        #<<
        if Y is None:
            return psi_X
        #>> ψ_Y = margin_Y applied to every column of S, shape (B, ny)
        psi_Y = self.margin_Y(S.transpose(1, 2)).squeeze(-1)
        #<<
        return psi_X, psi_Y
""")

md(r"""
**Check:** Unary and Pairwise against explicit loops over entities. Eval mode, with random BatchNorm running statistics and affine parameters so the normalization is part of what is checked.
""")

code(r"""
torch.manual_seed(0)
Bsz, nq, nr, dq, dr = 4, 5, 9, 12, 20                      # 5 words of width 12, 9 regions of width 20
Xq, Xr = torch.randn(Bsz, nq, dq), torch.randn(Bsz, nr, dr)

un = Unary(dr).eval()
V, bV, v, bv = un.embed.weight, un.embed.bias, un.feature_reduce.weight[0], un.feature_reduce.bias[0]
loop_un = torch.tensor([[(v @ torch.relu(V @ Xr[b, j] + bV) + bv).item() for j in range(nr)] for b in range(Bsz)])
print("Unary    vs loop  max diff:", (un(Xr) - loop_un).abs().max().item())

pw = Pairwise(dq, nq, dr, nr)
bn = pw.normalize_S
with torch.no_grad():
    bn.running_mean.uniform_(-.3, .3); bn.running_var.uniform_(.5, 2); bn.weight.uniform_(.5, 2); bn.bias.uniform_(-1, 1)
pw.eval()
psi_q, psi_r = pw(Xq, Xr)
L, bL, R, bR = pw.embed_X.weight, pw.embed_X.bias, pw.embed_Y.weight, pw.embed_Y.bias
S_loop = torch.zeros(Bsz, nq, nr)
with torch.no_grad():
    for b in range(Bsz):
        for i in range(nq):
            for j in range(nr):
                l, r = L @ Xq[b, i] + bL, R @ Xr[b, j] + bR
                k = i * nr + j                                   # index in the flattened grid
                S_loop[b, i, j] = ((l @ r) / (l.norm() * r.norm()) - bn.running_mean[k]) / torch.sqrt(bn.running_var[k] + bn.eps) * bn.weight[k] + bn.bias[k]
    loop_q = torch.stack([torch.stack([pw.margin_X.weight[0] @ S_loop[b, i] + pw.margin_X.bias[0] for i in range(nq)]) for b in range(Bsz)])
    loop_r = torch.stack([torch.stack([pw.margin_Y.weight[0] @ S_loop[b, :, j] + pw.margin_Y.bias[0] for j in range(nr)]) for b in range(Bsz)])
print("Pairwise vs loop  max diff: ψ_question", (psi_q - loop_q).abs().max().item(), "  ψ_image", (psi_r - loop_r).abs().max().item())

ps = Pairwise(dr, nr, self_interaction=True).eval()           # self-interaction: one output, no margin_Y
print("self-interaction output shape:", tuple(ps(Xr).shape))
""")

md(r"""
**The factor graph.** One `Unary` per modality, one `Pairwise` per pair of modalities (keys `"i_j"`), optionally one self-interaction per modality (key `"self_i"`), and one bias-free `Linear(K → 1)` per modality to mix its $K$ potentials. Potential order per modality: unary, self, pairwise (as in the reference code).
""")

code(r"""
class FGA(nn.Module):
    def __init__(self, dims, sizes, use_pairwise=True, use_self=False, dropout=0.5):
        super().__init__()
        M = len(dims)
        self.use_pairwise, self.use_self = use_pairwise, use_self
        self.un_models = nn.ModuleList([Unary(d, dropout) for d in dims])
        self.pp_models = nn.ModuleDict()
        for i in range(M):
            for j in range(i, M):
                if i == j and use_self:
                    self.pp_models[f"self_{i}"] = Pairwise(dims[i], sizes[i], self_interaction=True)
                elif i != j and use_pairwise:
                    self.pp_models[f"{i}_{j}"] = Pairwise(dims[i], sizes[i], dims[j], sizes[j])
        K = 1 + use_self + use_pairwise * (M - 1)                      # potentials reaching each modality
        self.reduce_potentials = nn.ModuleList([nn.Linear(K, 1, bias=False) for _ in dims])

    def forward(self, *U):
        # U: one (B, n_i, d_i) tensor per modality -> attended vectors [(B, d_i)], beliefs [(B, n_i)]
        M = len(U)
        pots = [[self.un_models[i](U[i])] for i in range(M)]
        if self.use_self:
            for i in range(M):
                pots[i].append(self.pp_models[f"self_{i}"](U[i]))
        if self.use_pairwise:
            for i in range(M):
                for j in range(i + 1, M):
                    p_i, p_j = self.pp_models[f"{i}_{j}"](U[i], U[j])
                    pots[i].append(p_i); pots[j].append(p_j)
        att, beliefs = [], []
        for i in range(M):
            #>> stack the K potentials to (B, n_i, K); mix with reduce_potentials[i]; softmax over entities -> b (B, n_i); a = Σ_u b(u)·û -> (B, d_i)
            stack = torch.stack(pots[i], -1)
            b = F.softmax(self.reduce_potentials[i](stack).squeeze(-1), dim=-1)
            a = (b.unsqueeze(-1) * U[i]).sum(1)
            #<<
            att.append(a); beliefs.append(b)
        return att, beliefs
""")

code(r"""
torch.manual_seed(0)
fga = FGA([dq, dr], [nq, nr], use_self=True).eval()
(a_q, a_r), (b_q, b_r) = fga(Xq, Xr)
with torch.no_grad():                                          # image belief by hand: unary, self, pairwise-from-question
    pots_r = [fga.un_models[1](Xr), fga.pp_models["self_1"](Xr), fga.pp_models["0_1"](Xq, Xr)[1]]
    w = fga.reduce_potentials[1].weight[0]
    logits = sum(w[k] * pots_r[k] for k in range(3))
    b_hand = logits.exp() / logits.exp().sum(-1, keepdim=True)
print("image belief vs by hand max diff:", (b_r - b_hand).abs().max().item(),
      "  attended vector:", (a_r - (b_hand[..., None] * Xr).sum(1)).abs().max().item())
print("beliefs sum to 1:", torch.allclose(b_q.sum(-1), torch.ones(Bsz)), torch.allclose(b_r.sum(-1), torch.ones(Bsz)))
""")

md(r"""
**Optional check against the reference implementation.** Clone https://github.com/idansc/fga next to this notebook (or set `FGA_SRC` to its `src/` directory). The cell loads only the `fga.attention` subpackage, copies our weights into the reference modules, and compares outputs; without the repo it prints a note and moves on.
""")

code(r"""
import importlib.util, os, sys
ref = None
try:
    for root in [os.environ.get("FGA_SRC", ""), "fga/src"]:
        init = Path(root) / "fga" / "attention" / "__init__.py"
        if root and init.exists():
            spec = importlib.util.spec_from_file_location("fga_ref", init, submodule_search_locations=[str(init.parent)])
            ref = importlib.util.module_from_spec(spec); sys.modules["fga_ref"] = ref; spec.loader.exec_module(ref)
            break
except Exception as e:
    print("reference import failed:", repr(e)); ref = None

if ref is None:
    print("reference repo not available")
else:
    r_un = ref.Unary(dr).eval(); r_un.load_state_dict(un.state_dict())
    print("Unary    vs reference max diff:", (un(Xr) - r_un(Xr)).abs().max().item())
    r_pw = ref.Pairwise(dq, nq, dr, nr); r_pw.load_state_dict(pw.state_dict())
    pw.train(); r_pw.train()                                   # train mode: BatchNorm uses batch statistics
    (p1, p2), (r1, r2) = pw(Xq, Xr), r_pw(Xq, Xr)
    print("Pairwise vs reference max diff (train-mode BN):", max((p1 - r1).abs().max().item(), (p2 - r2).abs().max().item()))
    pw.eval()
    r_fga = ref.FactorGraphAttention(embed_dims=[dq, dr], num_entities=[nq, nr], use_self=True).eval()
    r_fga.load_state_dict(fga.state_dict())
    r_att, r_bel = r_fga(Xq, Xr, return_weights=True)
    print("FGA      vs reference max diff: beliefs", max((b_q - r_bel[0]).abs().max().item(), (b_r - r_bel[1]).abs().max().item()),
          "  attended", max((a_q - r_att[0]).abs().max().item(), (a_r - r_att[1]).abs().max().item()))
""")

md(r"""
### A synthetic VQA task: does the image attention need the question?

**Data.** An "image" is 9 regions on a 3×3 grid. Each region has one of 6 colors and one of 6 shapes; its feature is $e_\text{color}+e_\text{shape}+\varepsilon$, with fixed random $e\in\mathbb{R}^{32}$ (a frozen "detector") and noise $\varepsilon\sim\mathcal N(0,1.5^2 I)$. Two question templates of 5 words:
- `what color is the <shape>`: exactly one region has that shape; the answer is its color;
- `which shape is <color> ?`: exactly one region has that color; the answer is its shape.

12 answer classes (6 within each question type, so chance is 1/6). Every image contains each of the other attributes several times, so the answer needs the one region the question names.

**Model.** Learned word embeddings (32-d) for the question, the region features as they are, FGA over {question, image}, then an MLP on $[a_Q, a_I]$. Two attention variants:
- (a) **unary only**: $b_I(u)\propto\exp(w\,\psi_I(u))$; the image belief cannot depend on the question;
- (b) **FGA, unary + pairwise** question↔image: $b_I(u)\propto\exp(w_1\psi_I(u)+w_2\,\mu_{Q\to I}(u))$.

20,000 training / 4,000 test questions, 12 epochs of Adam, 3 seeds per variant (a few seconds each on CPU).
""")

code(r"""
COLORS = ["red", "green", "blue", "yellow", "purple", "orange"]
SHAPES = ["circle", "square", "triangle", "star", "heart", "moon"]
WORDS = ["what", "color", "is", "the", "which", "shape", "?"] + SHAPES + COLORS
W2I = {w: i for i, w in enumerate(WORDS)}
ANSWERS = COLORS + SHAPES
DV, NREG, NQW, NOISE = 32, 9, 5, 1.5
g = torch.Generator().manual_seed(123)
E_COLOR, E_SHAPE = torch.randn(6, DV, generator=g), torch.randn(6, DV, generator=g)

def make_vqa(n, seed):
    rng = np.random.default_rng(seed)
    col, shp = rng.integers(0, 6, (n, NREG)), rng.integers(0, 6, (n, NREG))
    qtype, key, pos = rng.integers(0, 2, n), rng.integers(0, 6, n), rng.integers(0, NREG, n)
    q, y = np.zeros((n, NQW), int), np.zeros(n, int)
    for k in range(n):
        attr = shp if qtype[k] == 0 else col                   # the attribute the question names
        attr[k] = rng.choice([a for a in range(6) if a != key[k]], NREG)
        attr[k, pos[k]] = key[k]                               # ... appears in exactly one region
        if qtype[k] == 0:
            q[k] = [W2I[w] for w in ["what", "color", "is", "the", SHAPES[key[k]]]]; y[k] = col[k, pos[k]]
        else:
            q[k] = [W2I[w] for w in ["which", "shape", "is", COLORS[key[k]], "?"]]; y[k] = 6 + shp[k, pos[k]]
    col, shp = torch.tensor(col), torch.tensor(shp)
    img = E_COLOR[col] + E_SHAPE[shp] + NOISE * torch.randn(n, NREG, DV, generator=torch.Generator().manual_seed(seed))
    return dict(img=img, q=torch.tensor(q), y=torch.tensor(y), pos=torch.tensor(pos), col=col, shp=shp, qtype=torch.tensor(qtype))

vqa_train, vqa_test = make_vqa(20000, 1), make_vqa(4000, 2)
print("example:", " ".join(WORDS[i] for i in vqa_test["q"][0]), "->", ANSWERS[vqa_test["y"][0]])
print("regions:", [f"{COLORS[c]} {SHAPES[s]}" for c, s in zip(vqa_test["col"][0].tolist(), vqa_test["shp"][0].tolist())])
""")

code(r"""
class TinyVQA(nn.Module):
    def __init__(self, use_pairwise):
        super().__init__()
        self.words = nn.Embedding(len(WORDS), DV)
        self.att = FGA([DV, DV], [NQW, NREG], use_pairwise=use_pairwise, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(2 * DV, 64), nn.ReLU(), nn.Linear(64, len(ANSWERS)))

    def forward(self, img, q):
        (a_q, a_i), (b_q, b_i) = self.att(self.words(q), img)
        return self.head(torch.cat([a_q, a_i], -1)), b_q, b_i

def train_vqa(use_pairwise, seed, epochs=12, bs=128):
    torch.manual_seed(seed)
    model = TinyVQA(use_pairwise)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    d = vqa_train
    for ep in range(epochs):
        model.train()
        for idx in torch.randperm(len(d["y"])).split(bs):
            loss = F.cross_entropy(model(d["img"][idx], d["q"][idx])[0], d["y"][idx])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        logits, b_q, b_i = model(vqa_test["img"], vqa_test["q"])
    acc = (logits.argmax(-1) == vqa_test["y"]).float().mean().item()
    hit = (b_i.argmax(-1) == vqa_test["pos"]).float().mean().item()     # image belief peaks on the named region
    return model, acc, hit

t0 = time.time()
vqa_models, vqa_res = {}, {}
for name, use_pw in [("(a) unary only", False), ("(b) FGA unary + pairwise", True)]:
    runs = [train_vqa(use_pw, seed) for seed in range(3)]
    vqa_models[name] = runs[0][0]
    vqa_res[name] = np.array([[r[1], r[2]] for r in runs])
    acc, hit = vqa_res[name][:, 0], vqa_res[name][:, 1]
    print(f"{name:26s} test accuracy {acc.mean():.3f} ± {acc.std():.3f}  (seeds: {', '.join(f'{a:.3f}' for a in acc)})"
          f"   image belief peaks on the named region: {hit.mean():.3f}")
# what the image alone gives: the most frequent color (shape) in the image, read from the true labels
attr = torch.where(vqa_test["qtype"][:, None] == 0, vqa_test["col"], vqa_test["shp"])
guess = F.one_hot(attr, 6).sum(1).argmax(1) + 6 * vqa_test["qtype"]
print(f"chance 1/6 = {1/6:.3f}; 'most frequent attribute in the image' = {(guess == vqa_test['y']).float().mean().item():.3f}; "
      f"region hit by chance 1/9 = {1/9:.3f}; {time.time() - t0:.0f}s")
""")

code(r"""
show = [0, 1, 2, 3]
d = vqa_test
with torch.no_grad():
    beliefs = {name: m(d["img"][show], d["q"][show]) for name, m in vqa_models.items()}
fig, ax = plt.subplots(3, len(show), figsize=(3.6 * len(show), 9.4), gridspec_kw=dict(height_ratios=[1, 1, .55]))
for c, k in enumerate(show):
    words = [WORDS[i] for i in d["q"][k]]
    for r, name in enumerate(vqa_models):
        logits, b_q, b_i = beliefs[name]
        im = ax[r, c].imshow(b_i[c].view(3, 3).numpy(), cmap="viridis", vmin=0, vmax=1)
        for j in range(NREG):
            ax[r, c].text(j % 3, j // 3, f"{COLORS[d['col'][k, j]]}\n{SHAPES[d['shp'][k, j]]}", ha="center", va="center", fontsize=7,
                          color="black" if b_i[c, j] > .5 else "white", fontweight="bold" if j == d["pos"][k] else "normal")
        ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        ax[r, c].set_title(f"{name}\nanswer: {ANSWERS[logits[c].argmax()]}", fontsize=8)
    ax[0, c].set_title(f"Q: {' '.join(words)}\n(true: {ANSWERS[d['y'][k]]})\n\n" + ax[0, c].get_title(), fontsize=8)
    b_q = beliefs["(b) FGA unary + pairwise"][1][c]
    ax[2, c].bar(range(NQW), b_q.numpy()); ax[2, c].set_xticks(range(NQW)); ax[2, c].set_xticklabels(words, fontsize=8)
    ax[2, c].set_ylim(0, 1); ax[2, c].set_title("(b) question belief", fontsize=8)
ax[0, 0].set_ylabel("image belief, (a)"); ax[1, 0].set_ylabel("image belief, (b)"); ax[2, 0].set_ylabel("belief")
fig.colorbar(im, ax=ax[:2].ravel().tolist(), shrink=.6, label="attention on region")
fig.suptitle("Image belief over the 3×3 regions (bold = the region the question names)", y=.995)
plt.show()
""")

md(r"""
<<SOLUTION>>
- **(a) unary only**: 0.308 ± 0.005 test accuracy over 3 seeds. Its image belief peaks on the named region for 11.8% of questions, chance level (1/9): $\psi_I$ sees one region at a time, so the belief is the same whatever the question asks (top row, nearly uniform). The classifier is left with the pooled image, from which it can read how often each color and shape occurs. Guessing the most frequent color (shape) of the image, from the true labels, scores 0.345; (a) is just below that, about twice chance.
- **(b) FGA, unary + pairwise**: 0.940 ± 0.006, and the image belief peaks on the named region for 96.4% of questions. The only addition is one learned word×region grid with its two marginals: the message $\mu_{Q\to I}$ makes the image attention a function of the question. The belief is not always peaked: in the second and fourth examples a third or more of it sits on a wrong region, and the answer is still right.
- **The question belief of (b) is not on the keyword.** For `what color is the <shape>` it splits between `the` and the shape word; for `which shape is <color> ?` (third example) most of it is on `shape`. The keyword has already acted through the pairwise message to the image; what the classifier still needs from $a_Q$ is the question type (answer with a color or a shape), and `the` and `shape` each occur in one template only. Attention maps show what the downstream task uses, not what a human would highlight.
<</SOLUTION>>
✏️ In `Pairwise.forward`, replace the learned marginal of the question→image message by a max over the words, `psi_Y = S.max(dim=1).values` (keep `margin_X`), retrain variant (b) for 3 seeds and compare accuracy, region hit rate and the image beliefs. Then print `vqa_models["(b) FGA unary + pairwise"].att.pp_models["0_1"].margin_Y.weight` for the learned version: which word positions does the marginal weight? With a max, what must the projections $L$, $R$ do instead for the rows of `what`, `is`, `the`?
""")

# ---------------------------------------------------------------- GRU
md(r"""
## 7. If time: a GRU cell by hand

PyTorch's GRU (gate order **r, z, n** in `weight_ih` / `weight_hh`):
$$r=\sigma(W_{ir}x+b_{ir}+W_{hr}h+b_{hr}),\quad z=\sigma(W_{iz}x+b_{iz}+W_{hz}h+b_{hz}),$$
$$n=\tanh\big(W_{in}x+b_{in}+r\odot(W_{hn}h+b_{hn})\big),\qquad h'=(1-z)\odot n+z\odot h.$$
One state, two gates; $z$ plays the forget gate and $1-z$ the input gate, tied.
""")

code(r"""
def gru_cell(x, h, W_ih, W_hh, b_ih, b_hh):
    gi, gh = x @ W_ih.T + b_ih, h @ W_hh.T + b_hh
    i_r, i_z, i_n = gi.chunk(3, -1); h_r, h_z, h_n = gh.chunk(3, -1)
    #>> r, z, n and h' from the equations above (note r multiplies the hidden part of n, including its bias)
    r, z = torch.sigmoid(i_r + h_r), torch.sigmoid(i_z + h_z)
    n = torch.tanh(i_n + r * h_n)
    return (1 - z) * n + z * h
    #<<

torch.manual_seed(0)
ref = nn.GRUCell(10, 20); x, h = torch.randn(4, 10), torch.randn(4, 20)
print("GRU cell max |mine − torch| =", (gru_cell(x, h, ref.weight_ih, ref.weight_hh, ref.bias_ih, ref.bias_hh) - ref(x, h)).abs().max().item())
""")

md(r"""
## Summary
- **BPE** learns an ordered list of merges from frequent byte pairs; encoding replays them by rank. The round trip is lossless for any string; on held-out Shakespeare 756 tokens give 2.1 characters per token, GPT-2's 50k give 3.1.
- An **LSTM layer** is a loop over a cell with four gates from one matrix multiply; ours matches `nn.LSTM` to float precision once the weights are copied.
- An **autoregressive LM** is trained with teacher forcing (targets = inputs shifted by one) and sampled one token at a time; temperature trades diversity for coherence.
- A **linear recurrence** (diagonal SSM, linear attention) has three equivalent forms: recurrent (constant memory per token), convolution (time-invariant only, FFT), and parallel scan (also for input-dependent, *selective* decay). Linear in $T$, but a fixed state cannot remember everything, hence attention + SSM **hybrids**.
- **Gradients through time** shrink geometrically in a vanilla RNN; the LSTM cell path and slowly decaying SSM states carry them much further, and the forget-gate bias controls how far.
- **Factor Graph Attention** gives every modality a belief from a learned, bias-free mix of potentials: unary $v^\top\mathrm{relu}(V\hat u)$ and cosine grids of learned projections, batch-normalized and collapsed by learned marginals. On the synthetic VQA task, unary-only attention cannot depend on the question (0.31 accuracy, chance 0.17); the question↔image pairwise factor finds the named region and reaches 0.94. The self-interaction grid is $QK^\top$; self-attention keeps it per row instead of marginalizing it.

**Further watching:**
- Stanford CS231n (2017), Lecture 10: Recurrent Neural Networks: https://www.youtube.com/watch?v=6niqTuYFZLQ
- Karpathy, *makemore part 1* (bigram character LM): https://www.youtube.com/watch?v=PaCmpygFfXo
- Karpathy, *makemore part 2* (MLP character LM): https://www.youtube.com/watch?v=TCH_1BHY58I
- Karpathy, *Let's build the GPT Tokenizer* (section 2 follows it): https://www.youtube.com/watch?v=zduSFxRajkE

**Further reading (section 6):**
- Schwartz, Schwing, Hazan, *Factor Graph Attention*, CVPR 2019: https://arxiv.org/abs/1904.05880
- Schwartz, Schwing, Hazan, *High-Order Attention Models for Visual Question Answering*, NeurIPS 2017: https://arxiv.org/abs/1711.04323
- Code (PyTorch, `fga.attention`): https://github.com/idansc/fga
""")

for k, p in B.write(STEM).items():
    print(k, p)

# recap slides, in teaching order: RNN, BPTT, LM, gradient flow, LSTM, then SSMs / linear attention / hybrids,
# then FGA (p7: co-attention VQA, factorized belief, unary, learned interaction, message = learned marginal)
print("recap", make_recap(STEM, [
    ("slides/lectures/p6.pdf", [10, 18, 74, 34, 39, 54, 58, 63, 64, 65]),
    ("slides/2026-updates/L8b_architectures_looped_2026.pdf", [49, 50, 51, 52]),
    ("slides/lectures/p7.pdf", [37, 40, 42, 46, 48]),
]))
