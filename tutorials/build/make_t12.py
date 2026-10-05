"""Builds T12_agents_retrieval.ipynb (Tutorial 12, Deep Learning 89-6877, Fall 2026)."""
from nbkit import Builder
from recap import make_recap

STEM = "T12_agents_retrieval"
PDF = "slides/2026-updates/L12b_agents_harness_arcagi3_2026.pdf"
B = Builder()
md, code = B.md, B.code

md(rf"""
# Tutorial 12: retrieval, RAG and a minimal agent harness
**Deep Learning 89-6877, Fall 2026.** TA: Tal Fiskus.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idansc/dl-course-2026/blob/main/tutorials/{STEM}.ipynb)
· [Recap slides (PDF)](https://github.com/idansc/dl-course-2026/blob/main/tutorials/recap/{STEM}_recap.pdf)

Plan for today (≈ 60 min):
1. Lecture recap: agent loop, harness, search + verification; retrieval and RAG (10 min)
2. Sparse vs dense retrieval: BM25 from scratch and mean-pooled embeddings (12 min)
3. Late interaction: ColBERT's MaxSim (8 min)
4. RAG: a 0.5B model with and without retrieved passages (8 min)
5. A minimal agent harness: JSON tool calls, a loop, a step budget (12 min)
6. pass@k and verification: sampling more only pays with a verifier (10 min)
7. If time: a cross-encoder reranker

Everything uses small models (MiniLM encoders, a 33M ColBERT, Qwen2.5-0.5B-Instruct) and runs on a laptop CPU in about 7 minutes. Cells marked ✏️ are for you to try.
<<STUDENT>>
**Before the tutorial:** fill in every `# TODO` (replace the `...`). Each one is followed by a check cell that compares your result with a library implementation or a naive loop. The solutions are presented in the tutorial.
<</STUDENT>>
""")

code(r"""
import importlib.util, subprocess, sys
for pkg, mod in [("rank_bm25", "rank_bm25"), ("sentence-transformers", "sentence_transformers")]:
    if importlib.util.find_spec(mod) is None:      # Colab has sentence-transformers; rank_bm25 is a 10 kB package
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
""")

code(r"""
import ast, collections, itertools, json, math, operator, random, re, time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
torch.manual_seed(0); random.seed(0); np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
T0 = time.time()
print("device:", device)
""")

md(r"""
## 1. Lecture recap

**Agent = model + harness.** An LLM agent is a policy in an environment: observation $o_t$ in, action $a_t$ out, repeat. The model chooses the action (a tool call, a click, text); the **harness** turns observations (files, web pages, tool outputs, game frames) into the prompt and parses the model's text into an action. The harness also holds:
- **loop and planning**: observe → think → act (ReAct: interleave a reasoning step with each action), sub-agents;
- **tools**: *function calling*. Each tool is described by a JSON schema; the model emits `{"name": ..., "arguments": {...}}`; the harness executes it and appends the result to the context. The **Model Context Protocol** (MCP, 2024) standardizes how tools and data sources are exposed to any model: one MCP server per tool provider instead of one integration per (model, tool) pair;
- **context and memory**: what enters the prompt. Long runs exceed the context window, so harnesses **compact** it (summarize old turns, drop stale tool outputs, keep notes in files) and keep a step budget;
- **permissions and sandbox**, **verification** (tests, judges, retries), **routing** (which model, how much effort).

Pieces of the harness migrate into the model over time: chain of thought began as a prompt and became reasoning models; tool use is now trained in.

**Search + verification at inference time.** Sample $n$ candidates and keep the best. With $c$ correct samples out of $n$, the unbiased estimate of "at least one of $k$ is correct" is
$$\text{pass@}k = 1 - \binom{n-c}{k}\Big/\binom{n}{k}.$$
pass@$k$ becomes accuracy only if something picks the correct sample: a verifier (unit tests, a math checker, a reward model, process supervision of each step). Without one, majority vote is the default. For agents, reliability is the opposite question: **pass^$k$** (τ-bench) $=\binom{c}{k}/\binom{n}{k}$, the probability that *all* $k$ attempts succeed.

**Retrieval** (CME 295, lecture 7). The model's knowledge is frozen at pretraining and its context is finite, so we fetch evidence at inference time. Two stages: **candidate retrieval** over the whole corpus (maximize recall; cheap) and **reranking** of a short list (maximize precision; expensive).
- **Sparse (BM25)**: a query and a document are bags of words; the score sums, over query terms, an IDF weight times a saturating term frequency
$$\mathrm{BM25}(q,d)=\sum_{t\in q}\mathrm{idf}(t)\,\frac{\mathrm{tf}(t,d)\,(k_1+1)}{\mathrm{tf}(t,d)+k_1\big(1-b+b\,\frac{|d|}{\mathrm{avgdl}}\big)},\qquad \mathrm{idf}(t)=\log\!\Big(1+\frac{N-n_t+0.5}{n_t+0.5}\Big).$$
Exact word matching; no training; strong on rare names and numbers.
- **Dense (bi-encoder)**: one vector per text, $e = \mathrm{normalize}(\mathrm{meanpool}(\mathrm{Encoder}(x)))$; score $= \cos(e_q, e_d)$. Matches paraphrases; documents are embedded once, offline; search is one matrix product (or an approximate nearest-neighbour index).
- **Late interaction (ColBERT)**: keep one vector per token and score $s(q,d)=\sum_i \max_j q_i^\top d_j$ (**MaxSim**). Documents are still encoded offline, but storage grows by the number of tokens.
- **Cross-encoder reranker**: feed `[query; document]` jointly through a transformer, output one relevance score. Most accurate, but needs one forward pass per (query, document) pair, so it only reranks a short list.
- **RAG** (Lewis et al., 2020): retrieve top-$k$ passages, put them in the prompt, generate. **Long context vs retrieval**: putting the whole corpus in the prompt costs tokens and attention, often exceeds the window, and models get distracted by irrelevant text; retrieval keeps the prompt short but fails when the retriever misses.
- Metrics: recall@$k$ (is the gold passage in the top $k$), MRR (mean of $1/\text{rank}$ of the first relevant passage), nDCG@$k$.

**Evaluation** (CME 295, lecture 8). Agent benchmarks score *outcomes*: SWE-bench (do the repository's tests pass after the agent's patch), τ-bench (database state after a tool-using conversation, reported as pass^$k$), ARC-AGI-3 (interactive games without instructions; score = action efficiency relative to humans, $\min(1.15, (h/a)^2)$ per level; humans 100%, frontier models below 1% at launch). Tool-calling failures fall into three places: the call (no tool, a hallucinated tool, wrong arguments), the tool (error, empty output), and the final answer (ignores the tool output).
""")

md(r"""
## 2. Sparse vs dense retrieval

**Corpus.** SQuAD v1.1 validation: Wikipedia paragraphs, each with crowd-written questions. We take the first 10 paragraphs of each of the 48 articles (480 passages) and 200 questions about them. Each question has exactly one gold passage, so recall@$k$ is well defined. Paragraphs from the same article are hard negatives: same topic, different facts.
""")

code(r"""
squad = load_dataset("rajpurkar/squad", split="validation")
per_article = collections.defaultdict(list)
for r in squad:
    if r["context"] not in per_article[r["title"]]:
        per_article[r["title"]].append(r["context"])
corpus = [c for title in per_article for c in per_article[title][:10]]
pid = {c: i for i, c in enumerate(corpus)}
qa = [dict(q=r["question"], gold=pid[r["context"]], answers=r["answers"]["text"]) for r in squad if r["context"] in pid]
random.Random(0).shuffle(qa)
qa = qa[:200]
questions, gold = [x["q"] for x in qa], np.array([x["gold"] for x in qa])
print(f"{len(corpus)} passages, {np.mean([len(c.split()) for c in corpus]):.0f} words on average; {len(qa)} questions")
print("Q:", questions[0]); print("gold passage:", corpus[gold[0]][:300], "...")
""")

md(r"""
### 2.1 BM25 from scratch

Precompute a document × vocabulary weight matrix $W_{dt} = \mathrm{idf}(t)\cdot\frac{\mathrm{tf}(t,d)(k_1+1)}{\mathrm{tf}(t,d)+k_1(1-b+b|d|/\mathrm{avgdl})}$. Then the score of a query is the sum of the columns of its words: a sparse dot product between a bag-of-words query and a sparse document vector. This is exactly what an inverted index computes, just stored densely here (480 × ~10k).
""")

code(r"""
def tokenize(s):
    return re.findall(r"\w+", s.lower())

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.vocab = {w: i for i, w in enumerate(sorted({w for d in docs for w in d}))}
        tf = np.zeros((len(docs), len(self.vocab)))
        for j, d in enumerate(docs):
            for w in d:
                tf[j, self.vocab[w]] += 1
        N, dl = len(docs), tf.sum(1, keepdims=True)
        #>> document frequency n_t, idf (Lucene form, always > 0), and the weight matrix W (docs × vocab)
        n_t = (tf > 0).sum(0)
        self.idf = np.log(1 + (N - n_t + 0.5) / (n_t + 0.5))
        self.W = self.idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / dl.mean()))
        #<<

    def scores(self, query_tokens):
        cols = [self.vocab[w] for w in query_tokens if w in self.vocab]   # repeated query words count twice
        return self.W[:, cols].sum(1)

docs_tok = [tokenize(c) for c in corpus]
bm25 = BM25(docs_tok)
S_bm25 = np.stack([bm25.scores(tokenize(q)) for q in questions])     # (200 questions, 480 passages)
print("vocabulary:", len(bm25.vocab), " nonzero weights per passage:", (bm25.W > 0).sum(1).mean().round(1))
""")

md(r"""
**Check** against `rank_bm25`. Its `BM25Okapi` uses a different IDF ($\log\frac{N-n+0.5}{n+0.5}$, floored for very common words), so we hand it our IDF; everything else (tf saturation, length normalization, $k_1=1.5$, $b=0.75$) must match.
""")

code(r"""
from rank_bm25 import BM25Okapi
ref = BM25Okapi(docs_tok)
inv_vocab = {i: w for w, i in bm25.vocab.items()}
ref.idf = {inv_vocab[i]: v for i, v in enumerate(bm25.idf)}
S_ref = np.stack([ref.get_scores(tokenize(q)) for q in questions])
print("max |ours − rank_bm25| =", np.abs(S_bm25 - S_ref).max())
""")

md(r"""
### 2.2 Recall@k
""")

code(r"""
def recall_at_k(S, gold, k):
    #>> fraction of queries whose gold passage is among the k highest scores (S: queries × passages)
    topk = np.argsort(-S, axis=1)[:, :k]
    return (topk == gold[:, None]).any(1).mean()
    #<<

def mrr(S, gold):
    rank = (S > S[np.arange(len(gold)), gold][:, None]).sum(1) + 1     # 1 + number of passages scored higher
    return (1 / rank).mean()

from sklearn.metrics import top_k_accuracy_score
for k in (1, 5, 10):
    print(f"BM25 recall@{k:<2} = {recall_at_k(S_bm25, gold, k):.3f}   sklearn top-k accuracy = "
          f"{top_k_accuracy_score(gold, S_bm25, k=k, labels=np.arange(len(corpus))):.3f}")
print(f"BM25 MRR = {mrr(S_bm25, gold):.3f}")
""")

md(r"""
### 2.3 Dense retrieval: mean pooling + cosine

Encoder: `sentence-transformers/all-MiniLM-L6-v2` (22M parameters, 384-d), trained contrastively on 1B sentence pairs. A batch is padded to its longest text, so the mean must run **only over real tokens**: $e = \sum_j m_j h_j / \sum_j m_j$ with the attention mask $m$. Then L2-normalize, so cosine similarity is a dot product.
""")

code(r"""
enc_name = "sentence-transformers/all-MiniLM-L6-v2"
enc_tok = AutoTokenizer.from_pretrained(enc_name)
enc = AutoModel.from_pretrained(enc_name).to(device).eval()

@torch.no_grad()
def embed(texts, bs=64, max_length=256):
    out = []
    for i in range(0, len(texts), bs):
        batch = enc_tok(texts[i:i + bs], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        H = enc(**batch).last_hidden_state                          # (B, L, 384)
        m = batch["attention_mask"].unsqueeze(-1).float()           # (B, L, 1)
        #>> e: masked mean over the token axis, then L2-normalize → (B, 384)
        e = (H * m).sum(1) / m.sum(1).clamp(min=1e-9)
        e = F.normalize(e, dim=-1)
        #<<
        out.append(e.cpu())
    return torch.cat(out)

t = time.time()
D_dense, Q_dense = embed(corpus), embed(questions)
S_dense = (Q_dense @ D_dense.T).numpy()                             # cosine similarity
print(f"embedded {len(corpus) + len(questions)} texts in {time.time() - t:.1f}s;  D_dense: {tuple(D_dense.shape)}")
""")

md(r"""
**Check 1**: the `sentence-transformers` library (same model, its own pooling code). **Check 2**: a short and a long text embedded together (padded) vs alone. Without the mask, padding tokens enter the mean.
""")

code(r"""
from sentence_transformers import SentenceTransformer
st = SentenceTransformer(enc_name, device=device)
ref = torch.from_numpy(st.encode(corpus[:32] + questions[:32], normalize_embeddings=True))
print("max |ours − sentence-transformers| =", (embed(corpus[:32] + questions[:32]) - ref).abs().max().item())

pair = [questions[0], corpus[0]]
together, alone = embed(pair), torch.cat([embed([pair[0]]), embed([pair[1]])])
print("padded batch vs one at a time:      ", (together - alone).abs().max().item())
with torch.no_grad():                                               # the bug: mean over all positions, padding included
    b = enc_tok(pair, padding=True, return_tensors="pt").to(device)
    naive = F.normalize(enc(**b).last_hidden_state.mean(1), dim=-1).cpu()
print("cosine(question, its embedding without the mask):", F.cosine_similarity(naive[0], alone[0], dim=0).item())
""")

code(r"""
print(f"{'':6s}{'R@1':>7s}{'R@5':>7s}{'R@10':>7s}{'MRR':>7s}")
for name, S in [("BM25", S_bm25), ("dense", S_dense)]:
    print(f"{name:6s}" + "".join(f"{recall_at_k(S, gold, k):7.3f}" for k in (1, 5, 10)) + f"{mrr(S, gold):7.3f}")
r_b, r_d = np.argsort(-S_bm25, 1)[:, 0] == gold, np.argsort(-S_dense, 1)[:, 0] == gold
print(f"top-1 correct: both {np.sum(r_b & r_d)}, only BM25 {np.sum(r_b & ~r_d)}, only dense {np.sum(~r_b & r_d)}, neither {np.sum(~r_b & ~r_d)}")
i = int(np.flatnonzero(r_b & ~r_d)[0])
print("\nBM25 right, dense wrong:", questions[i], "\n  dense top-1:", corpus[np.argmax(S_dense[i])][:160], "...")
""")

md(r"""
On SQuAD, BM25 beats the dense MiniLM encoder at top-1 (0.80 vs 0.74), and dense catches up from recall@5 on (0.985 vs 0.970 at recall@10). SQuAD questions were written by annotators *looking at the passage*, so they reuse its words, which is BM25's best case (DPR, Karpukhin et al. 2020, reported the same on SQuAD). The two fail on different questions: 35 questions only BM25 gets right at top-1, 23 only dense. That is the motivation for **hybrid** search (see the if-time section).

✏️ Rewrite the printed question as a paraphrase that avoids the passage's words (e.g. synonyms) and compare the rank of its gold passage under BM25 and dense.
""")

md(r"""
## 3. Late interaction: MaxSim

One vector per passage compresses ~130 tokens into 384 numbers. ColBERT (Khattab & Zaharia, 2020) keeps all token vectors and scores
$$s(q,d) = \sum_{i=1}^{|q|} \max_{j \le |d|}\; q_i^\top d_j ,$$
so each query token finds its best-matching passage token. Passages are padded to a common length; padded positions must never win the max.
""")

code(r"""
def pad_stack(seqs):
    # list of (L_i, dim) tensors → (n, L_max, dim) and a boolean mask (n, L_max)
    L = max(s.shape[0] for s in seqs)
    X, M = torch.zeros(len(seqs), L, seqs[0].shape[1]), torch.zeros(len(seqs), L, dtype=torch.bool)
    for i, s in enumerate(seqs):
        X[i, :len(s)], M[i, :len(s)] = s, True
    return X, M

def maxsim_scores(Q_list, D_pad, D_mask):
    # Q_list: list of (|q|, dim) query token embeddings; returns (num_queries, num_passages)
    S = torch.empty(len(Q_list), D_pad.shape[0])
    for i, q in enumerate(Q_list):
        #>> sim[n, a, j] = q_a · d_{n,j}; set padded j to -inf; max over j, sum over a; store the row in S[i]
        sim = torch.einsum("ad,njd->naj", q, D_pad)
        sim = sim.masked_fill(~D_mask[:, None, :], float("-inf"))
        S[i] = sim.max(-1).values.sum(-1)
        #<<
    return S

def maxsim_naive(q, d):
    return sum(max(float(q[a] @ d[j]) for j in range(len(d))) for a in range(len(q)))
""")

md(r"""
First with **MiniLM's own token embeddings** (the outputs before mean pooling, L2-normalized). This model was never trained for MaxSim.
""")

code(r"""
@torch.no_grad()
def token_embed(texts, bs=64, max_length=256):
    out = []
    for i in range(0, len(texts), bs):
        b = enc_tok(texts[i:i + bs], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        H = F.normalize(enc(**b).last_hidden_state, dim=-1).cpu()
        out += [H[j][b["attention_mask"][j].bool().cpu()] for j in range(H.shape[0])]
    return out

D_tok, Q_tok = token_embed(corpus), token_embed(questions)
D_pad, D_mask = pad_stack(D_tok)
S_maxsim_minilm = maxsim_scores(Q_tok, D_pad, D_mask).numpy()

# check against the double loop on a few (query, passage) pairs
err = max(abs(S_maxsim_minilm[a, n] - maxsim_naive(Q_tok[a], D_tok[n])) for a in range(3) for n in range(0, len(corpus), 60))
print("max |vectorized − naive loop| =", err)
""")

md(r"""
Now a model **trained** for late interaction: `answerdotai/answerai-colbert-small-v1` (33M parameters, the ColBERTv2 recipe). Three details from the ColBERT paper: a linear layer projects each token to 96 dimensions; a marker token (`[unused0]` for queries, `[unused1]` for passages) follows `[CLS]`; queries are padded to 32 tokens with `[MASK]`, and those mask positions also produce query vectors (*query augmentation*: learned soft expansion terms). The checkpoint stores the linear layer next to the BERT weights, so we load it by hand.
""")

code(r"""
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
cb_name = "answerdotai/answerai-colbert-small-v1"
cb_tok = AutoTokenizer.from_pretrained(cb_name)
cb = AutoModel.from_pretrained(cb_name).to(device).eval()
cb_proj = load_file(hf_hub_download(cb_name, "model.safetensors"))["linear.weight"].to(device)   # (96, 384)
Q_MARK, D_MARK = cb_tok.convert_tokens_to_ids(["[unused0]", "[unused1]"])

@torch.no_grad()
def colbert_encode(texts, is_query, bs=32):
    out = []
    for i in range(0, len(texts), bs):
        tx = [". " + t for t in texts[i:i + bs]]                     # position 1 is overwritten by the marker
        if is_query:
            b = cb_tok(tx, padding="max_length", truncation=True, max_length=32, return_tensors="pt")
        else:
            b = cb_tok(tx, padding=True, truncation=True, max_length=300, return_tensors="pt")
        ids, am = b["input_ids"].clone(), b["attention_mask"]
        ids[:, 1] = Q_MARK if is_query else D_MARK
        if is_query:
            ids[ids == cb_tok.pad_token_id] = cb_tok.mask_token_id  # query augmentation
        H = F.normalize(cb(input_ids=ids.to(device), attention_mask=am.to(device)).last_hidden_state @ cb_proj.T, dim=-1).cpu()
        out += [H[j] if is_query else H[j][am[j].bool()] for j in range(H.shape[0])]
    return out

t = time.time()
D_cb, Q_cb = colbert_encode(corpus, False), colbert_encode(questions, True)
D_cb_pad, D_cb_mask = pad_stack(D_cb)
S_colbert = maxsim_scores(Q_cb, D_cb_pad, D_cb_mask).numpy()
print(f"ColBERT encoding + scoring: {time.time() - t:.1f}s")
""")

code(r"""
methods = {"BM25": S_bm25, "dense (MiniLM, mean-pool)": S_dense,
           "MaxSim on MiniLM tokens (not trained for it)": S_maxsim_minilm, "ColBERT (trained for MaxSim)": S_colbert}
print(f"{'':38s}{'R@1':>7s}{'R@5':>7s}{'R@10':>7s}{'MRR':>7s}")
for name, S in methods.items():
    print(f"{name:38s}" + "".join(f"{recall_at_k(S, gold, k):7.3f}" for k in (1, 5, 10)) + f"{mrr(S, gold):7.3f}")

n_vec = sum(len(d) for d in D_cb)
print(f"\nindex size: dense {D_dense.numel() * 4 / 1e6:.2f} MB (1 × 384 per passage),"
      f" ColBERT {n_vec * 96 * 4 / 1e6:.2f} MB ({n_vec / len(corpus):.0f} × 96 per passage)")

ks = np.arange(1, 21)
plt.figure(figsize=(6.5, 4))
for name, S in methods.items():
    plt.plot(ks, [recall_at_k(S, gold, k) for k in ks], marker="o", ms=3, label=name)
plt.xlabel("k"); plt.ylabel("recall@k"); plt.title("Retrieval on 480 SQuAD passages (200 questions)")
plt.xticks([1, 5, 10, 15, 20]); plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.show()
""")

md(r"""
- **Token-level matching helps even without training for it**: MaxSim over MiniLM's token vectors beats the mean-pooled vector from the *same forward pass* (recall@1 0.88 vs 0.74). Mean pooling throws away which words matched; MaxSim keeps a soft per-word match, which is what SQuAD rewards.
- The **trained ColBERT** is best, but only by one question here (recall@1 0.89 vs 0.88; both 0.995 at recall@5). On a corpus with less word overlap between questions and passages, training for MaxSim matters more.
- The price is storage: one vector per token, ~40× the dense index here (29 MB vs 0.74 MB) even after projecting to 96-d. ColBERTv2 compresses each vector to 1–2 bits with residual quantization; PLAID/WARP make the search fast.

✏️ Remove the `[MASK]` query augmentation in `colbert_encode` (pad queries with `padding=True` instead, as for passages) and re-run the table. How much does recall@1 change?
""")

md(r"""
## 4. RAG: retrieve, augment, generate

Generator: `Qwen/Qwen2.5-0.5B-Instruct`. For 20 questions we compare **closed-book** (question only) with **RAG** (top-3 ColBERT passages in the prompt). An answer counts as correct if a gold answer string appears in it (lower-cased, punctuation removed): lenient, but enough to see the gap.
""")

code(r"""
lm_name = "Qwen/Qwen2.5-0.5B-Instruct"
lm_tok = AutoTokenizer.from_pretrained(lm_name, padding_side="left")
lm = AutoModelForCausalLM.from_pretrained(lm_name, dtype=torch.float32).to(device).eval()

@torch.no_grad()
def generate(conversations, max_new_tokens=32, tools=None, **sampling):
    # conversations: list of message lists; greedy unless sampling kwargs are given
    prompts = [lm_tok.apply_chat_template(m, tools=tools, tokenize=False, add_generation_prompt=True) for m in conversations]
    b = lm_tok(prompts, return_tensors="pt", padding=True).to(device)
    gen = dict(do_sample=False, temperature=None, top_p=None, top_k=None) if not sampling else dict(do_sample=True, **sampling)
    out = lm.generate(**b, max_new_tokens=max_new_tokens, pad_token_id=lm_tok.eos_token_id, **gen)
    return lm_tok.batch_decode(out[:, b["input_ids"].shape[1]:], skip_special_tokens=True)

def norm(s):
    return " ".join(re.sub(r"[^\w\s]", " ", s.lower()).split())

def is_correct(pred, answers):
    return any(norm(a) in norm(pred) for a in answers)

SYS = "Answer the question with a short phrase. No explanation."
def rag_messages(q, passages):
    ctx = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    return [{"role": "system", "content": SYS}, {"role": "user", "content": f"Passages:\n{ctx}\n\nQuestion: {q}"}]

rag_set = qa[:20]
top3 = [np.argsort(-S_colbert[i])[:3] for i in range(20)]
t = time.time()
closed = sum((generate([[{"role": "system", "content": SYS}, {"role": "user", "content": x["q"]}] for x in rag_set[i:i + 10]],
                       max_new_tokens=16) for i in range(0, 20, 10)), [])
rag = sum((generate([rag_messages(x["q"], [corpus[j] for j in top3[k]]) for k, x in enumerate(rag_set[i:i + 10], start=i)],
                    max_new_tokens=16) for i in range(0, 20, 10)), [])
print(f"generation time: {time.time() - t:.0f}s")
acc_closed = np.array([is_correct(p, x["answers"]) for p, x in zip(closed, rag_set)])
acc_rag = np.array([is_correct(p, x["answers"]) for p, x in zip(rag, rag_set)])
hit = np.array([x["gold"] in top3[k] for k, x in enumerate(rag_set)])
print(f"closed-book accuracy: {acc_closed.mean():.2f}")
print(f"RAG (top-3 ColBERT):  {acc_rag.mean():.2f}   gold passage retrieved for {hit.sum()}/20;"
      f" accuracy when retrieved {acc_rag[hit].mean():.2f}, when missed {acc_rag[~hit].mean() if (~hit).any() else float('nan'):.2f}")
for k in range(4):
    print(f"\nQ: {rag_set[k]['q']}   gold: {rag_set[k]['answers'][0]}\n  closed-book: {closed[k].strip()}\n  RAG:         {rag[k].strip()}")
""")

md(r"""
**Long context instead of retrieval?** Count the tokens.
""")

code(r"""
n_corpus = sum(len(lm_tok(c).input_ids) for c in corpus)
n_rag = np.mean([len(lm_tok(lm_tok.apply_chat_template(rag_messages(x["q"], [corpus[j] for j in top3[k]]), tokenize=False)).input_ids)
                 for k, x in enumerate(rag_set)])
print(f"whole corpus: {n_corpus:,} tokens   (Qwen2.5-0.5B context window: {lm.config.max_position_embeddings:,})")
print(f"RAG prompt:   {n_rag:,.0f} tokens on average")
""")

md(r"""
Closed-book, the 0.5B model gets 1 of 20 right: many SQuAD questions are underspecified without their passage ("What did he ..."), and a 0.5B model stores few facts. With three retrieved passages the same model gets 16 of 20 (0.80): ColBERT found the gold passage for 19 of 20 questions, so almost all remaining errors are reading errors. Reading errors include a wrong span (the fourth example: one term out of three, and not the right one) and answers whose wording the string match rejects.

The whole 480-passage corpus (79k tokens) is 2.4× the model's 32k context window, and every call would pay for all of it; the top-3 prompt (~600 tokens) is ~130× shorter. With a 1M-token model the corpus would fit, but cost per call and distraction by irrelevant passages remain.

✏️ Replace ColBERT's top-3 with BM25's top-3, and with the gold passage plus two random passages ("oracle retrieval"). Which part of the gap is retrieval and which is reading?
""")

md(r"""
## 5. A minimal agent harness

The model sees tool descriptions as JSON schemas (Qwen's chat template puts them in the system prompt) and emits
```
<tool_call>
{"name": "calculator", "arguments": {"expression": "1234 * 5678"}}
</tool_call>
```
The harness: **parse** the call(s) (a reply may contain several `<tool_call>` blocks: parallel function calling), **execute** them, **append** each result as a `tool` message, and **loop** until the model answers in plain text or the **step budget** runs out. Malformed calls are not fatal: the error goes back to the model as the tool result, so it can retry.

Two tools: a calculator (safe: an AST walk over numbers and `+ − × / ** %`, never `eval`) and search over our corpus (ColBERT top-1).
""")

code(r"""
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.FloorDiv: operator.floordiv}
def calculator(expression):
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in OPS: return OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in OPS: return OPS[type(n.op)](ev(n.operand))
        raise ValueError(f"unsupported expression: {expression!r}")
    return ev(ast.parse(str(expression), mode="eval").body)

def search(query):
    q = colbert_encode([query], True)
    return corpus[int(maxsim_scores(q, D_cb_pad, D_cb_mask)[0].argmax())]

TOOLS = {"calculator": calculator, "search": search}
TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "calculator", "description": "Evaluate an arithmetic expression, e.g. '(2016 - 1856) * 2'.",
     "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "search", "description": "Search an encyclopedia. Returns the most relevant paragraph.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]
print(calculator("1234 * 5678 - 999"), "|", search("When was Nikola Tesla born?")[:120], "...")
""")

code(r"""
class ToolCallError(Exception):
    pass

def parse_tool_calls(text):
    # all <tool_call> blocks in a reply, as [(name, arguments), ...]; [] means a final answer; raises ToolCallError if any is malformed
    bodies = re.findall(r"<tool_call>(.*?)(?:</tool_call>|$)", text, flags=re.S)
    calls = []
    for body in bodies:
        #>> json-decode the body; require a known "name" and a dict "arguments" (raise ToolCallError with a helpful message otherwise); append (name, arguments) to calls
        try:
            call = json.loads(body.strip())
        except json.JSONDecodeError as e:
            raise ToolCallError(f"invalid JSON in tool call ({e.msg})")
        if not isinstance(call, dict) or call.get("name") not in TOOLS:
            raise ToolCallError(f"unknown tool {call.get('name') if isinstance(call, dict) else call!r}; available: {list(TOOLS)}")
        args = call.get("arguments", {})
        if not isinstance(args, dict):
            raise ToolCallError("'arguments' must be a JSON object")
        calls.append((call["name"], args))
        #<<
    return calls

def run_agent(task, policy, max_steps=5, system="You are a helpful assistant."):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    for step in range(max_steps):
        reply = policy(messages)
        messages.append({"role": "assistant", "content": reply})
        #>> parse the reply. No calls → return (reply, messages). Malformed → append "Error: ..." as a "tool" message. Otherwise run each TOOLS[name](**args), catching exceptions as errors, and append str(result) as a "tool" message per call
        try:
            calls = parse_tool_calls(reply)
        except ToolCallError as e:
            messages.append({"role": "tool", "content": f"Error: {e}"})
            continue
        if not calls:
            return reply, messages
        for name, args in calls:
            try:
                result = TOOLS[name](**args)
            except Exception as e:
                result = f"Error: {type(e).__name__}: {e}"
            messages.append({"role": "tool", "content": str(result)})
        #<<
    return None, messages                                           # step budget exhausted

def show(messages):
    for m in messages[1:]:
        body = m["content"] if len(m["content"]) < 300 else m["content"][:300] + " ..."
        print(f"[{m['role']}] {body.strip()}")
""")

md(r"""
**Check 1**: the parser on well-formed and malformed calls. **Check 2**: the loop with a *scripted* policy (no model), whose replies cover each branch: broken JSON, an unknown tool, a tool that raises, a valid call, and a final answer.
""")

code(r"""
tc = lambda s: f"<tool_call>\n{s}\n</tool_call>"
assert parse_tool_calls("The answer is 42.") == []
assert parse_tool_calls(tc('{"name": "calculator", "arguments": {"expression": "2+2"}}')) == [("calculator", {"expression": "2+2"})]
assert parse_tool_calls(tc('{"name": "search", "arguments": {"query": "Tesla"}}') + tc('{"name": "calculator", "arguments": {"expression": "1+1"}}')) \
    == [("search", {"query": "Tesla"}), ("calculator", {"expression": "1+1"})]                          # parallel calls
for bad in [tc('{"name": "calculator", "arguments": {"expression": "2+2"}'),   # missing brace
            tc('{"name": "python", "arguments": {}}'),                          # hallucinated tool
            tc('{"name": "search", "arguments": "Tesla"}')]:                    # arguments not an object
    try:
        parse_tool_calls(bad); raise AssertionError("should have raised")
    except ToolCallError as e:
        print("ToolCallError:", e)

script = iter([tc('{"name": "calculator" "arguments": {}}'),
               tc('{"name": "browser", "arguments": {"url": "x"}}'),
               tc('{"name": "calculator", "arguments": {"expression": "import os"}}'),
               tc('{"name": "calculator", "arguments": {"expression": "6 * 7"}}'),
               "The answer is 42."])
answer, trace = run_agent("What is 6 times 7?", lambda msgs: next(script))
roles = [m["role"] for m in trace]
assert answer == "The answer is 42." and roles == ["system", "user"] + ["assistant", "tool"] * 4 + ["assistant"], roles
assert trace[-2]["content"] == "42" and all(trace[i]["content"].startswith("Error") for i in (3, 5, 7))
show(trace)
print("\nscripted-policy check passed")

_, trace = run_agent("loop forever", lambda msgs: tc('{"name": "calculator", "arguments": {"expression": "1+1"}}'), max_steps=3)
print("budget check: stopped after", sum(m["role"] == "assistant" for m in trace), "model calls")
""")

md(r"""
Now with the real model as the policy. The system prompt tells it to use the tools instead of its own memory or mental arithmetic.
""")

code(r"""
AGENT_SYS = ("You are a helpful assistant with tools. Use search for any factual question and calculator for any arithmetic; "
             "do not rely on memory. Call one tool at a time. When you have the result, give a short final answer.")
qwen_policy = lambda msgs: generate([msgs], max_new_tokens=80, tools=TOOL_SCHEMAS)[0]

tasks = ["In which village was Nikola Tesla born?",
         "In which city was Nikola Tesla born?",
         "How many years before 2026 was Martin Luther born?"]
t = time.time()
for task in tasks:
    answer, trace = run_agent(task, qwen_policy, max_steps=5, system=AGENT_SYS)
    print("=" * 100); show(trace)
    print(f"→ final answer: {answer!r}   ({sum(m['role'] == 'tool' for m in trace)} tool calls)")
print(f"\nagent time: {time.time() - t:.0f}s")
""")

md(r"""
Three traces, three outcomes, sorted by where they fail (the CME 295 lecture 8 taxonomy):
- **"In which village was Nikola Tesla born?"**: the query retrieves the birth paragraph and the answer (Smiljan) is grounded in it. Success.
- **"In which city ..."**: one word different; the model searches for "birthplace", retrieval returns the article's opening paragraph, which has no birthplace, and the model answers "Seattle", while claiming it came from the search. The harness executed everything correctly; the failure is a retrieval miss followed by an ungrounded answer. A better harness would re-search or answer "not found".
- **Two-step question**: the model searches and reads 1483, then does the subtraction itself instead of calling the calculator ("59 years = 2026 − 1967") and runs out of its 80-token reply. A 0.5B model does not chain tools: after one tool result it answers. (Qwen2.5-1.5B also fails this task, differently: it skips the search and subtracts a wrong remembered year, 1479. Chaining tools reliably comes with larger, agent-trained models.)

The harness code is correct in every case (the scripted check proves it); what changes the outcome is the model and what the tools return. Same model, different harness choices (the search query, how many passages search returns, whether the answer is checked against the tool output) change the result: the lecture's "the harness matters as much as the model".

✏️ Add a third tool, `lookup(passage_id)`, and change `search` to return the top-3 passage ids with their first sentence only. This is a small *agent–computer interface* change of the kind SWE-agent studied: what the tool returns decides how much context each step costs.
""")

md(r"""
## 6. pass@k and verification

Inference-time search in its simplest form: sample $n$ answers at temperature $T>0$, then pick one. Task: linear equations $ax+b=c$ with an integer solution. The model is asked for a one-line derivation (`5x = 110 → x = 22`), which it gets right most of the time but not always.

Three ways to turn $n=10$ samples into one answer:
- **pass@k (oracle)**: correct if any of $k$ samples is correct. An upper bound; needs the answer key.
- **majority vote** (self-consistency): the most frequent answer. No key needed.
- **verifier**: substitute the candidate into the equation and keep the first that satisfies it. No key needed; *checking* a solution is easier than *finding* it, the same asymmetry as unit tests for code.
""")

code(r"""
rng = random.Random(1)
problems = []
for _ in range(12):
    x, a, b = rng.randint(2, 30), rng.randint(6, 19), rng.randint(10, 99) * rng.choice([-1, 1])
    problems.append(dict(a=a, b=b, c=a * x + b, x=x))
eq = lambda p: f"{p['a']}x {'+' if p['b'] >= 0 else '-'} {abs(p['b'])} = {p['c']}"
EQ_SYS = "Solve the equation for x. Answer in one line, in exactly this format:\n3x + 7 = 31 → 3x = 24 → x = 8"
msgs = [[{"role": "system", "content": EQ_SYS}, {"role": "user", "content": eq(p)}] for p in problems]

def parse_x(text):
    m = re.findall(r"\bx\s*=\s*(-?\d+(?:\.\d+)?)", text)
    return float(m[-1]) if m else None

n = 10
t = time.time()
greedy = [parse_x(o) for o in generate(msgs, max_new_tokens=24)]
samples = []
for i in range(0, len(problems), 2):                                # 2 problems × 10 samples per batch
    outs = generate(msgs[i:i + 2], max_new_tokens=24, temperature=1.0, top_p=1.0, top_k=0, num_return_sequences=n)
    samples += [[parse_x(o) for o in outs[j * n:(j + 1) * n]] for j in range(2)]
print(f"sampling time: {time.time() - t:.0f}s")
correct = np.array([[s == p["x"] for s in ss] for p, ss in zip(problems, samples)])   # (12, 10)
for p, g, ss, c in zip(problems, greedy, samples, correct):
    print(f"{eq(p):18s} x={p['x']:<3d} greedy={g!s:6s} correct {c.sum():2d}/10  samples={[None if s is None else (int(s) if s == int(s) else s) for s in ss]}")
""")

md(r"""
**The unbiased pass@k estimator** (Chen et al., 2021, Codex). With $c$ correct among $n$ samples, the probability that a random subset of $k$ contains at least one correct sample is $1-\binom{n-c}{k}/\binom{n}{k}$. (Estimating it as $1-(1-c/n)^k$ is biased.) Check it by enumerating all $\binom{n}{k}$ subsets.
""")

code(r"""
def pass_at_k(n, c, k):
    #>> 1 − C(n−c, k) / C(n, k)   (equals 1 when n − c < k)
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)
    #<<

def pass_hat_k(n, c, k):                                            # τ-bench pass^k: all k attempts succeed
    return math.comb(c, k) / math.comb(n, k)

row = correct[int(np.argmin(np.abs(correct.sum(1) - 5)))]          # a problem with a mixed record
for k in (1, 3, 5):
    brute = np.mean([row[list(s)].any() for s in itertools.combinations(range(n), k)])
    print(f"k={k}: formula {pass_at_k(n, row.sum(), k):.4f}   enumeration {brute:.4f}")
""")

code(r"""
def verify(p, x):
    #>> does x satisfy a·x + b = c?  (the verifier never sees p["x"])
    return x is not None and abs(p["a"] * x + p["b"] - p["c"]) < 1e-9
    #<<

def majority(ss):
    votes = collections.Counter(s for s in ss if s is not None)
    return votes.most_common(1)[0][0] if votes else None

def verified(p, ss):
    ok = [s for s in ss if verify(p, s)]
    return ok[0] if ok else majority(ss)                            # fall back to the vote if nothing verifies

assert verify(dict(a=3, b=7, c=31), 8) and not verify(dict(a=3, b=7, c=31), 24) and not verify(dict(a=3, b=7, c=31), None)
ks = np.arange(1, n + 1)
c_per = correct.sum(1)
pk = [np.mean([pass_at_k(n, c, k) for c in c_per]) for k in ks]
phk = [np.mean([pass_hat_k(n, c, k) for c in c_per]) for k in ks]
acc = dict(greedy=np.mean([g == p["x"] for g, p in zip(greedy, problems)]),
           majority=np.mean([majority(ss) == p["x"] for ss, p in zip(samples, problems)]),
           verifier=np.mean([verified(p, ss) == p["x"] for ss, p in zip(samples, problems)]))
print({k: round(v, 3) for k, v in acc.items()}, f" pass@1 = {pk[0]:.3f}, pass@10 = {pk[-1]:.3f}")

plt.figure(figsize=(6.5, 4))
plt.plot(ks, pk, marker="o", label="pass@k (oracle picks)")
plt.plot(ks, phk, marker="s", label="pass^k (all k correct)")
plt.axhline(acc["verifier"], color="C2", ls="--", label=f"verifier over 10 samples ({acc['verifier']:.2f})")
plt.axhline(acc["majority"], color="C3", ls=":", label=f"majority vote over 10 ({acc['majority']:.2f})")
plt.axhline(acc["greedy"], color="gray", ls="-.", label=f"greedy ({acc['greedy']:.2f})")
plt.xlabel("k (samples)"); plt.ylabel("fraction of problems"); plt.ylim(0, 1.05)
plt.title("Qwen2.5-0.5B on 12 linear equations, n = 10 samples at T = 1")
plt.legend(fontsize=8, loc="lower left"); plt.grid(alpha=.3); plt.tight_layout(); plt.show()
""")

md(r"""
(Sampling differs across CPU, CUDA and MPS, so your counts will differ a little from the numbers below.)

- **pass@k** climbs from 0.53 at $k=1$ to 0.92 at $k=10$: for 11 of 12 equations at least one sample is right. The model "can" solve them; it is unreliable.
- **Majority vote** reaches 0.67, above greedy (0.58) but far below pass@10. It fails when the model's errors are systematic: for `17x − 39 = 131` most samples give the same wrong 10.57 (the model computes 131 + 39 = 180 instead of 170), and for `12x − 97 = 263` the majority is 31.67.
- **The verifier** reaches 0.92 = pass@10 exactly: with a perfect checker, every problem with at least one correct sample is solved. This is the slide's point: pass@k turns into accuracy only with a good verifier. Unit tests play this role for code, a reward model or PRM for open-ended reasoning, where checking is imperfect and the gap reopens.
- **pass^k** falls to 0 at $k=10$ (no equation was solved 10 times out of 10). Reliability, what a user of an agent experiences, is a different and much lower number than pass@k.

✏️ Replace the programmatic verifier by an *LLM-as-a-judge* (CME 295 lecture 8): ask the same 0.5B model "Is x = … a solution of …? Answer yes or no." for each candidate and keep the first "yes". How close does it get to the programmatic verifier?
""")

md(r"""
## 7. If time: a cross-encoder reranker

Second stage of retrieval: rerank BM25's top 10 with `cross-encoder/ms-marco-MiniLM-L6-v2`, which reads query and passage *together* (full cross-attention between them) and outputs one relevance logit. 10 forward passes per query instead of one matrix product, so it only runs on a short list. Recall@10 of the first stage caps what reranking can reach.
""")

code(r"""
ce_name = "cross-encoder/ms-marco-MiniLM-L6-v2"
ce_tok = AutoTokenizer.from_pretrained(ce_name)
ce = AutoModelForSequenceClassification.from_pretrained(ce_name).to(device).eval()

n_q, K = 60, 10
S_rerank = np.full((n_q, len(corpus)), -np.inf)
t = time.time()
with torch.no_grad():
    for i in range(n_q):
        cand = np.argsort(-S_bm25[i])[:K]
        b = ce_tok([questions[i]] * K, [corpus[j] for j in cand], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        S_rerank[i, cand] = ce(**b).logits[:, 0].cpu().numpy()
print(f"reranked {n_q} × {K} pairs in {time.time() - t:.0f}s")

def rrf(*score_mats, k=60):
    # reciprocal rank fusion: sum over systems of 1 / (k + rank)
    out = np.zeros_like(score_mats[0])
    for S in score_mats:
        ranks = np.argsort(np.argsort(-S, 1), 1) + 1
        out += 1 / (k + ranks)
    return out

g = gold[:n_q]
for name, S in [("BM25", S_bm25[:n_q]), ("dense", S_dense[:n_q]), ("hybrid RRF(BM25, dense)", rrf(S_bm25[:n_q], S_dense[:n_q])),
                ("BM25 top-10 → cross-encoder", S_rerank), ("ColBERT", S_colbert[:n_q])]:
    print(f"{name:30s} R@1 {recall_at_k(S, g, 1):.2f}   R@5 {recall_at_k(S, g, 5):.2f}")
print(f"(BM25 recall@10 on these {n_q} questions: {recall_at_k(S_bm25[:n_q], g, 10):.2f})")
""")

md(r"""
- The cross-encoder lifts BM25's recall@1 from 0.80 to 0.88 on these 60 questions, at 10 transformer passes per query. ColBERT alone is still better here (0.95): a reranker cannot recover passages the first stage missed, and BM25's recall@10 is 0.95.
- **Reciprocal rank fusion** of BM25 and dense lands between the two at recall@1 (0.77) and is best of the three at recall@5 (0.95 vs 0.93). Fusion helps the candidate list, where the two systems' complementary misses matter; at top-1 the weaker system dilutes the stronger one.

✏️ Weight the fusion, `rrf` with BM25 counted twice, or rerank the RRF top-10 instead of the BM25 top-10.
""")

md(r"""
## Summary
- **BM25** is a sparse dot product between a bag-of-words query and IDF-weighted, length-normalized, saturating term frequencies. No training, and on SQuAD it beats a 22M dense encoder at top-1 (0.80 vs 0.74).
- **Dense retrieval** needs a masked mean; padding tokens in the mean silently change the embedding (cosine 0.59 to the correct vector). Sparse and dense fail on different questions, but rank fusion of the two helped only the top-5 list, not top-1; a cross-encoder reranker did help top-1.
- **Late interaction** (MaxSim over token vectors) keeps the per-word match that mean pooling discards: the best first-stage retriever here (recall@1 0.89 vs 0.80 for BM25), at ~40× the index size of a single vector.
- **RAG** takes a 0.5B model from 0.05 to 0.80 on 20 SQuAD questions; with recall@3 near 1, the remaining errors are reading errors, and the whole corpus would not fit in its context.
- An **agent harness** is a loop: parse a JSON tool call, execute, append the result, repeat under a step budget. Malformed calls go back to the model as errors instead of crashing the loop.
- **pass@k** grows with $k$ (0.53 → 0.92 at $k=10$), but only a **verifier** converts it into accuracy (0.92); majority vote reaches 0.67, greedy 0.58. pass^k measures the opposite, reliability, and falls to 0.

**Further reading:** Stanford CME 295 (Fall 2025), lecture 7 *Agentic LLMs* (RAG, reranking, retrieval metrics, tool calling, MCP, ReAct; sections 1–5 follow it), https://cme295.stanford.edu/slides/fall25-cme295-lecture7.pdf ; lecture 8 *LLM evaluation* (LLM-as-a-judge, tool-calling failure modes, SWE-bench, τ-bench and pass^k), https://cme295.stanford.edu/slides/fall25-cme295-lecture8.pdf ; Khattab & Zaharia, *ColBERT*, https://arxiv.org/abs/2004.12832 ; Lewis et al., *Retrieval-Augmented Generation*, https://arxiv.org/abs/2005.11401 ; Yao et al., *ReAct*, https://arxiv.org/abs/2210.03629 ; ARC Prize, ARC-AGI-3, https://arcprize.org .

**On a Colab GPU:** use the full 2,067 SQuAD validation passages and all 10,570 questions, `Qwen2.5-1.5B-Instruct` or `Qwen2.5-3B-Instruct` as reader and agent (tool calls become much more reliable), and $n=64$ samples per equation.
""")

code(r"""
print(f"total runtime: {(time.time() - T0) / 60:.1f} min")
""")

for k, p in B.write(STEM).items():
    print(k, p)
print("recap", make_recap(STEM, [(PDF, [2, 3, 4, 5, 8, 9, 10, 12, 14, 15, 17, 18])]))
