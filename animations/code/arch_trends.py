"""Code for the lecture-4 slide "Architecture variations, 2025-26": the core choices of 16 open-weight LLMs released
between December 2024 and October 2026, read from each model's own files instead of from blog posts.

The 2017-2024 table on the slide before it is Stanford CS336's (Hashimoto and Liang, Spring 2025, lecture 3). This
script extends it with the same columns (norm, where the norm sits, serial or parallel layers, positions, activation)
plus two that became common later (QK-norm, the attention pattern). For every model it downloads config.json from the
Hugging Face Hub and reads the modeling code of the transformers library:
  - norm type and placement, parallel layers: the decoder layer's forward() (the order of norm, sublayer, residual add)
  - QK-norm: whether the attention module normalizes q and k (and the config does not switch it off)
  - positions: rope settings per layer (NoPE layers, partial rotary dimensions, a learned relative bias)
  - activation: hidden_act and whether the MLP is gated
  - attention: query / key-value heads, MLA, sparse indexing, and the layer pattern (local : global, linear : full)
The values printed are compared with the table on the slide (TABLE); the script fails if any cell differs.
Models were picked from S. Raschka's LLM Architecture Gallery (119 models on 4 Nov 2026), one or two per lab.

Run:  python arch_trends.py          (needs network access; ~1 MB of config files; no model weights)
Uses: transformers 5.18.0 (installed) for 14 models; for Muse Spark and Mistral Large 4, which are newer than 5.18,
      modeling_*.py from the transformers main branch (fetched 7 Nov 2026). Gated repos (Gemma 3, Llama 4, Muse
      Spark) are read from Unsloth's public copies, which carry the same config.json.
Writes data/arch_trends.json.
"""
import json
import os
import re
import urllib.request
from collections import Counter
from fractions import Fraction
from pathlib import Path

import transformers
from huggingface_hub import hf_hub_download

HERE = Path(__file__).parent
OUT = HERE / "data" / "arch_trends.json"
CACHE = HERE / "data" / "arch_trends_cache"
TF = Path(transformers.__file__).parent / "models"
MAIN = "https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/{0}/modeling_{0}.py"

# (row label, release month, Hub repo with the config, transformers model type)
MODELS = [("DeepSeek V3", "Dec 2024", "deepseek-ai/DeepSeek-V3", "deepseek_v3"),
          ("Gemma 3 (27B)", "Mar 2025", "unsloth/gemma-3-27b-it", "gemma3"),
          ("Llama 4 Maverick", "Apr 2025", "unsloth/Llama-4-Maverick-17B-128E-Instruct", "llama4"),
          ("Qwen3 (235B)", "Apr 2025", "Qwen/Qwen3-235B-A22B", "qwen3_moe"),
          ("GLM-4.5", "Jul 2025", "zai-org/GLM-4.5", "glm4_moe"),
          ("gpt-oss (120B)", "Aug 2025", "openai/gpt-oss-120b", "gpt_oss"),
          ("MiniMax M2", "Oct 2025", "MiniMaxAI/MiniMax-M2", "minimax_m2"),
          ("OLMo 3 (32B)", "Nov 2025", "allenai/Olmo-3-32B-Think", "olmo3"),
          ("Qwen3.5 (397B)", "Feb 2026", "Qwen/Qwen3.5-397B-A17B", "qwen3_5_moe"),
          ("Nemotron 3 Super", "Mar 2026", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", "nemotron_h"),
          ("Gemma 4 (31B)", "Apr 2026", "google/gemma-4-31B-it", "gemma4"),
          ("DeepSeek V4-Pro", "Apr 2026", "deepseek-ai/DeepSeek-V4-Pro", "deepseek_v4"),
          ("Command A+", "May 2026", "CohereLabs/command-a-plus-05-2026-w4a4", "cohere2_moe"),
          ("Inkling", "Jul 2026", "thinkingmachines/Inkling", "inkling"),
          ("Muse Spark", "Oct 2026", "unsloth/Muse-Spark-Open-816B-A42B", "muse_spark"),
          ("Mistral Large 4", "Oct 2026", "mistralai/Mistral-Large-4-1T-A52B", "mistral_large4")]

COLUMNS = ["Norm", "Norm position", "Layers", "Position", "Activation", "QK-norm", "Attention"]
# the table on the slide (row label -> the 7 cells); the script checks that the files say the same
TABLE = {
    "DeepSeek V3":      ["RMSNorm", "pre", "serial", "RoPE, ⅓ of dims", "SwiGLU", "✗", "MLA"],
    "Gemma 3 (27B)":    ["RMSNorm", "pre + post", "serial", "RoPE", "GeGLU", "✓", "GQA, 5:1 local:global"],
    "Llama 4 Maverick": ["RMSNorm", "pre", "serial", "RoPE; NoPE every 4th", "SwiGLU", "✗", "GQA, 3:1 local:global"],
    "Qwen3 (235B)":     ["RMSNorm", "pre", "serial", "RoPE", "SwiGLU", "✓", "GQA"],
    "GLM-4.5":          ["RMSNorm", "pre", "serial", "RoPE, ½ of dims", "SwiGLU", "✓", "GQA"],
    "gpt-oss (120B)":   ["RMSNorm", "pre", "serial", "RoPE", "SwiGLU", "✗", "GQA, 1:1 local:global"],
    "MiniMax M2":       ["RMSNorm", "pre", "serial", "RoPE, ½ of dims", "SwiGLU", "✓", "GQA"],
    "OLMo 3 (32B)":     ["RMSNorm", "post only", "serial", "RoPE", "SwiGLU", "✓", "GQA, 3:1 local:global"],
    "Qwen3.5 (397B)":   ["RMSNorm", "pre", "serial", "RoPE, ¼ of dims", "SwiGLU", "✓", "3:1 linear:GQA"],
    "Nemotron 3 Super": ["RMSNorm", "pre", "serial", "NoPE", "ReLU²", "✗", "5:1 Mamba-2:GQA"],
    "Gemma 4 (31B)":    ["RMSNorm", "pre + post", "serial", "RoPE, global ¼ of dims", "GeGLU", "✓", "GQA, 5:1 local:global"],
    "DeepSeek V4-Pro":  ["RMSNorm", "pre", "serial, 4 streams", "RoPE, ⅛ of dims", "SwiGLU", "✓", "compressed + sparse"],
    "Command A+":       ["LayerNorm", "pre", "parallel", "RoPE local, NoPE global", "SwiGLU", "✗", "GQA, 3:1 local:global"],
    "Inkling":          ["RMSNorm", "pre", "serial", "learned relative bias", "SwiGLU", "✓", "GQA, 5:1 local:global"],
    "Muse Spark":       ["RMSNorm", "pre + post", "serial", "RoPE local, NoPE global", "SwiGLU", "✓", "GQA, 3:1 local:global"],
    "Mistral Large 4":  ["RMSNorm", "pre", "serial", "RoPE, ½ of dims; NoPE every 4th", "SwiGLU", "✗", "MLA, sparse"],
}
FRAC = {Fraction(1, 2): "½", Fraction(1, 3): "⅓", Fraction(1, 4): "¼", Fraction(1, 8): "⅛"}


# ---------------------------------------------------------------- reading the files
def config(repo):
    c = json.loads(Path(hf_hub_download(repo, "config.json", cache_dir=CACHE)).read_text())
    return c["text_config"] if isinstance(c.get("text_config"), dict) else c


def modeling(mtype):
    local = TF / mtype / f"modeling_{mtype}.py"
    if local.exists():
        return local.read_text()
    path = CACHE / f"modeling_{mtype}.py"                    # newer than the installed transformers
    if not path.exists():
        path.write_text(urllib.request.urlopen(MAIN.format(mtype), timeout=60).read().decode())
    return path.read_text()


def class_src(src, pattern):
    """Source of the first class whose name matches `pattern` (the text decoder layer, the attention module)."""
    for name in re.findall(r"^class (\w+)\(", src, re.M):
        if re.fullmatch(pattern, name):
            return re.search(rf"^class {name}\(.*?(?=^class |\Z)", src, re.M | re.S).group(0)
    raise KeyError(pattern)


def forward_lines(cls):
    fw = re.search(r"    def forward\(.*?(?=\n    def |\Z)", cls, re.S).group(0)
    return [ln.strip() for ln in fw.split("\n") if ln.strip() and not ln.strip().startswith("#")]


SUB = re.compile(r"self\.(self_attn|linear_attn|mixer|mlp|feed_forward)\(")
NORM = re.compile(r"self\.(\w*norm\w*)\(")
ADD = re.compile(r"residual \+|residual_gate\(|torch\.matmul\(")


# ---------------------------------------------------------------- the seven columns
def norm_type(c, layer):
    if c.get("norm_type") == "layer_norm":                      # Cohere: its own LayerNorm (mean and variance)
        return "LayerNorm"
    first = re.search(r"self\.(?:input_layernorm|norm|post_attention_layernorm) = (\w+)", layer).group(1)
    return "LayerNorm" if "LayerNorm" in first and "RMS" not in first else "RMSNorm"


def norm_position(layer):
    """pre: a norm runs before a sublayer; post: a norm runs on a sublayer's output before the residual add."""
    pre = post = False
    after_sub = False
    for ln in forward_lines(layer):
        s, n, a = SUB.search(ln), NORM.search(ln), ADD.search(ln)
        if s and n and n.start() > s.start():                  # self.self_attn(self.input_layernorm(x)): pre
            pre = True
        elif n and not after_sub and "per_layer" not in ln:
            pre = True
        elif n and after_sub:
            post = True
        if s:
            after_sub = True
        if a:
            after_sub = False
    return "pre + post" if pre and post else "post only" if post else "pre"


def layers(c, layer):
    if any(re.search(r"residual \+ \w+ \+ \w+", ln) for ln in forward_lines(layer)):
        return "parallel"                                       # y = x + Attn(LN x) + MLP(LN x)
    if c.get("hc_mult", 1) > 1:
        return f"serial, {c['hc_mult']} streams"                # several residual streams, mixed after each sublayer
    return "serial"


def frac(x):
    return FRAC[Fraction(x).limit_denominator(16)]


def position(c, src, attn):
    if "d_rel" in c:                                            # Inkling: a learned bias per distance, no rotation
        return "learned relative bias"
    if "apply_rotary_pos_emb(" not in attn and "rotary" not in attn:
        return "NoPE"                                           # Nemotron-H: the Mamba layers carry the order
    parts = []
    lt = c.get("layer_types") or []
    rope_local_only = ("sliding_window is not None" in attn and "apply_rotary" in attn) or (
        c.get("layer_rope_theta") and all((th == 0) == (t == "full_attention")
                                          for th, t in zip(c["layer_rope_theta"], lt)))
    if "qk_rope_head_dim" in c:                                 # MLA-style: a few rotated dims next to unrotated ones
        whole = c.get("qk_nope_head_dim", 0) + c["qk_rope_head_dim"] if "kv_lora_rank" in c else c["head_dim"]
        parts.append(f"{frac(Fraction(c['qk_rope_head_dim'], whole))} of dims")
    elif c.get("rotary_dim") and c.get("head_dim") and c["rotary_dim"] < c["head_dim"]:
        parts.append(f"{frac(Fraction(c['rotary_dim'], c['head_dim']))} of dims")
    else:
        rp = c.get("rope_parameters") or {}
        prf = c.get("partial_rotary_factor") or rp.get("partial_rotary_factor")
        glob = rp.get("full_attention") if isinstance(rp.get("full_attention"), dict) else None
        if prf and prf < 1:
            parts.append(f"{frac(prf)} of dims")
        elif glob and glob.get("partial_rotary_factor", 1) < 1:
            parts.append(f"global {frac(glob['partial_rotary_factor'])} of dims")
    nope = c.get("no_rope_layers")
    if nope and not all(nope):
        gaps = {i for i, v in enumerate(nope) if not v}
        k = len(nope) // len(gaps)
        assert gaps == {i for i in range(len(nope)) if i % k == k - 1}, gaps
        nope_part = f"NoPE every {k}th"
    else:
        nope_part = None
    if rope_local_only:
        return "RoPE local, NoPE global"
    return ", ".join(["RoPE"] + parts) + (f"; {nope_part}" if nope_part else "")


def activation(c, src, mtype):
    act = c.get("mlp_hidden_act") or c.get("hidden_activation") or c.get("hidden_act")
    if act is None:                                             # not in config.json: the configuration class default
        conf = (TF / mtype / f"configuration_{mtype}.py").read_text()
        act = re.search(r'hidden_act: str = "(\w+)"', conf).group(1)
    ffn = [re.search(rf"^class {n}\(.*?(?=^class |\Z)", src, re.M | re.S).group(0)
           for n in re.findall(r"^class (\w+(?:MLP|Mlp|Experts))\(", src, re.M) if "Vision" not in n and "Audio" not in n]
    gated = c.get("use_gated_activation", True) and any(re.search(r"self\.(gate_proj|gate_up_proj)\s*=", f) for f in ffn)
    if act == "relu2" and not gated:
        return "ReLU²"
    if act.startswith("gelu") and gated:
        return "GeGLU"
    if act in ("silu", "swish") and gated:
        return "SwiGLU"
    raise ValueError(act)


def qk_norm(c, attn):
    if c.get("use_qk_norm") is False:                           # the code has it, this model switches it off
        return "✗"
    q = re.search(r"(query_states|\bq)\s*=\s*self\.(q_norm|qk_norm|q_b_norm|q_layernorm)\(", attn) or \
        re.search(r"self\.(q_norm|qk_norm)\(\s*query_states", attn)
    return "✓" if q else "✗"


def every(flags):
    """The repeating pattern of a layer list: one global (True) layer every k layers -> "(k-1):1"."""
    idx = [i for i, f in enumerate(flags) if f]
    k = Counter(b - a for a, b in zip(idx, idx[1:])).most_common(1)[0][0]
    return f"{k - 1}:1"


def attention(c):
    h = c["num_attention_heads"]
    kv = c.get("num_key_value_heads", h)
    base = "MLA" if "kv_lora_rank" in c else "MQA" if kv == 1 else "GQA" if kv < h else "MHA"
    lt = Counter(c.get("layer_types") or [])
    if "hybrid_override_pattern" in c:                          # Nemotron-H: M = Mamba-2, * = attention, E = MoE
        p = Counter(c["hybrid_override_pattern"])
        return f"{p['M'] // p['*']}:1 Mamba-2:{base}"
    if "linear_attention" in lt:
        return f"{every([t == 'full_attention' for t in c['layer_types']])} linear:{base}"
    if "compress_ratios" in c:                                  # DeepSeek V4: compressed KV blocks + top-k selection (1 K=V head)
        return "compressed + sparse"
    if "index_topk" in c:
        return f"{base}, sparse"
    if "local_layer_ids" in c:                                  # Inkling: the local layers are listed in the config
        ids = json.loads(c["local_layer_ids"]) if isinstance(c["local_layer_ids"], str) else c["local_layer_ids"]
        return f"{base}, {every([i not in ids for i in range(c['num_hidden_layers'])])} local:global"
    if c.get("sliding_window_pattern"):                         # Gemma 3: every 6th layer is global
        k = c["sliding_window_pattern"]
        return f"{base}, {k - 1}:1 local:global"
    if c.get("attention_chunk_size") and c.get("no_rope_layers"):  # Llama 4: chunked RoPE layers, global NoPE layers
        return f"{base}, {every([not v for v in c['no_rope_layers']])} local:global"
    if lt.get("sliding_attention"):
        return f"{base}, {every([t == 'full_attention' for t in c['layer_types']])} local:global"
    return base


def ffn(c):
    """Not a column on the slide (mixture of experts is not part of this lecture), counted for the notes."""
    n = max(c.get(k) or 0 for k in ("num_experts", "n_routed_experts", "num_local_experts", "moe_num_experts"))
    return f"MoE, {n} experts" if n > 1 else "dense"


def row(name, repo, mtype):
    c, src = config(repo), modeling(mtype)
    layer = class_src(src, r"\w+(TextDecoderLayer|DecoderLayer|Block)(?<!VisionBlock)(?<!MoeBlock)")
    names = [n for n in re.findall(r"^class (\w+Attention)\(", src, re.M)
             if not re.search(r"Vision|Audio|Indexer|Linear|Cross|Mamba|Delta", n)]
    attn = class_src(src, next((n for n in names if "Text" in n), names[0]))
    return [norm_type(c, layer), norm_position(layer), layers(c, layer), position(c, src, attn), activation(c, src, mtype),
            qk_norm(c, attn), attention(c)]


if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    rows, bad = [], []
    for name, when, repo, mtype in MODELS:
        cells = row(name, repo, mtype)
        rows.append({"model": name, "released": when, "repo": repo, **dict(zip(COLUMNS, cells)), "FFN": ffn(config(repo))})
        flag = "" if cells == TABLE[name] else "   <-- differs from the slide: " + str(TABLE[name])
        print(f"{name:18s} {when}  " + " | ".join(cells) + flag)
        if flag:
            bad.append(name)
    share = {
        "RMSNorm": sum(r["Norm"] == "RMSNorm" for r in rows),
        "a norm before each sublayer": sum(r["Norm position"] != "post only" for r in rows),
        "serial layers": sum(r["Layers"].startswith("serial") for r in rows),
        "SwiGLU": sum(r["Activation"] == "SwiGLU" for r in rows),
        "QK-norm": sum(r["QK-norm"] == "✓" for r in rows),
        "RoPE on every dim of every layer": sum(r["Position"] == "RoPE" for r in rows),
        "full attention in every layer": sum(r["Attention"] in ("GQA", "MHA", "MLA") for r in rows),
        "mixture of experts (not on the slide)": sum(r["FFN"].startswith("MoE") for r in rows)}
    print({k: f"{v}/{len(rows)}" for k, v in share.items()})
    OUT.write_text(json.dumps({"columns": COLUMNS, "rows": rows, "counts": share, "n": len(rows),
                               "transformers": transformers.__version__}, indent=1, ensure_ascii=False))
    assert not bad, bad
    print("saved", OUT)
