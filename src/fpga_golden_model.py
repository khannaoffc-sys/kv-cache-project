"""
fpga_golden_model.py — Software golden model for the FPGA KV eviction unit.

Phase 6 — Software side (stop here before writing Verilog).

What this script does:
1. Loads Qwen2.5-0.5B-Instruct, prefills a 512-token prompt.
2. Inspects the cache object and prints key tensor shapes for layers 0 and 10.
3. Reports the KnormPress scoring direction (from reading installed kvpress source).
4. Implements the Python golden model:
   - Per-tensor symmetric INT8 key quantization (scale = max(|key|) / 127)
   - Integer sum-of-squares per token (as a proxy for INT8 L2 norm)
   - Deterministic tie-break: stable sort by (score, token_index)
   - Top-k selection
5. Compares selected indices with KnormPress at ratios 0.3 / 0.5 / 0.7,
   per layer and per head, reporting Jaccard overlap.
6. Saves test vectors as .mem hex files for $readmemh in Verilog simulation.

Results are printed to stdout — copy them into PROGRESS.md before writing Verilog.

Usage (laptop, CPU-only — runs in ~2-4 min on 0.5B model):
    python src/fpga_golden_model.py

Output files:
    results/fpga/layer0_keys_int8.mem       — INT8 key hex for layer 0 head 0
    results/fpga/layer0_scores_int32.mem    — INT32 scores for layer 0 head 0
    results/fpga/layer0_indices_topk.mem    — expected keep indices (ratio 0.5)
    results/fpga/layer10_keys_int8.mem
    results/fpga/layer10_scores_int32.mem
    results/fpga/layer10_indices_topk.mem
"""

import sys
import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent.parent / "results" / "fpga"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"\n{'='*65}")
print("  Phase 6 — FPGA KV Eviction Golden Model")
print(f"  Model: {MODEL_NAME}")
print(f"{'='*65}\n")

print("Loading model (0.5B, float32 on CPU) ...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
model.eval()
print("Model loaded.\n")

# ─── 1. Prefill 512-token prompt ──────────────────────────────────────────────
PROMPT = (
    "The key-value cache is a fundamental optimization in transformer inference. "
    "During the prefill phase every input token is processed in parallel through "
    "the full self-attention mechanism. The resulting key and value tensors for "
    "each layer are stored so they do not need to be recomputed during the "
    "subsequent token-by-token generation phase. Without caching generating each "
    "new token would require O(n squared) attention over all previously seen tokens. "
    "The KV cache reduces this to O(n) per new token at the cost of memory. "
    "Token eviction methods score each cached token by its historical attention "
    "weight or key norm and permanently discard the lowest-scoring tokens when "
    "the cache exceeds a budget reducing memory by 30 to 70 percent. "
    "Quantization methods represent stored keys and values in fewer bits typically "
    "INT4 or INT8 reducing memory by 2 to 4 times. Hybrid systems combine eviction "
    "and quantization. The kvpress library provides a unified API for experimenting "
    "with many strategies. Benchmarks such as LongBench and RULER measure accuracy "
    "after compression. The key norm criterion in KnormPress computes the L2 norm "
    "of each key vector and evicts tokens whose keys have the largest norms. "
    "For FPGA implementation we need the integer approximation of this criterion. "
) * 3  # repeat to get > 512 tokens

ids = tok(PROMPT, return_tensors="pt", max_length=512, truncation=True)
prompt_tokens = ids["input_ids"].shape[-1]
assert prompt_tokens >= 512, f"Got only {prompt_tokens} tokens — need >= 512"

print(f"Prompt tokens: {prompt_tokens}  (>= 512 ✓)")

with torch.no_grad():
    out = model(**ids, use_cache=True)

pkv = out.past_key_values  # tuple of (key, value) per layer
num_layers = len(pkv)

# ─── 2. Inspect cache object and print shapes ─────────────────────────────────
print(f"\n{'─'*60}")
print("  Cache object inspection")
print(f"{'─'*60}")
print(f"  type(past_key_values)  : {type(pkv).__name__}")
print(f"  num_layers             : {num_layers}")

for layer_idx in [0, 10]:
    k = pkv.layers[layer_idx].keys  # (batch, num_kv_heads, seq_len, head_dim)
    print(
        f"  layer {layer_idx:>2} key shape   : {tuple(k.shape)}"
        f"  → (batch={k.shape[0]}, num_kv_heads={k.shape[1]}, "
        f"seq_len={k.shape[2]}, head_dim={k.shape[3]})"
    )
    print(f"  layer {layer_idx:>2} key dtype   : {k.dtype}")

# Model config info
cfg = model.config
NUM_KV_HEADS = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
HEAD_DIM = cfg.hidden_size // cfg.num_attention_heads
SEQ_LEN = pkv.layers[0].keys.shape[2]

print(f"\n  Model config:")
print(f"    num_hidden_layers   : {cfg.num_hidden_layers}")
print(f"    num_attention_heads : {cfg.num_attention_heads}")
print(f"    num_kv_heads        : {NUM_KV_HEADS}")
print(f"    head_dim            : {HEAD_DIM}")
print(f"    seq_len (prefilled) : {SEQ_LEN}")

# ─── 3. KnormPress scoring direction (read from installed source) ──────────────
print(f"\n{'─'*60}")
print("  KnormPress scoring (read from installed kvpress source)")
print(f"{'─'*60}")
print("""
  Source: venv/lib/site-packages/kvpress/presses/knorm_press.py (lines 45-46)
  Code:
      score_dtype = torch.promote_types(keys.dtype, torch.float32)
      return -torch.linalg.vector_norm(keys, dim=-1, dtype=score_dtype)

  Scoring direction:
    - Score = NEGATIVE L2 norm of the key vector per token (shape: batch, kv_heads, seq_len)
    - compute_n_kept() determines how many to keep from compression_ratio
    - ScorerPress.compress() calls scores.topk(n_kept, dim=-1)
    - topk on NEGATIVE norms selects the LEAST negative values = SMALLEST norms

  ✓ KnormPress KEEPS tokens with SMALL key-L2-norms.
  ✓ KnormPress EVICTS tokens with LARGE key-L2-norms.

  FPGA implication:
    Compute sum-of-squares per token (≈ squared L2 norm, avoids sqrt).
    Keep tokens with SMALLEST sum-of-squares.
    Tie-break: prefer earlier token index (deterministic stable sort).
""")

# ─── 4. Python golden model ───────────────────────────────────────────────────
print(f"{'─'*60}")
print("  Python INT8 golden model")
print(f"{'─'*60}")


def int8_quantize_per_tensor(key_tensor: torch.Tensor) -> tuple:
    """Per-tensor symmetric INT8 quantization.

    Args:
        key_tensor: float32 tensor of shape (num_kv_heads, seq_len, head_dim)
    Returns:
        (q_int8, scale) where q_int8 is int8 tensor, scale is float scalar
    """
    max_abs = key_tensor.abs().max().item()
    if max_abs == 0:
        return torch.zeros_like(key_tensor, dtype=torch.int8), 1.0
    scale = max_abs / 127.0
    q = (key_tensor / scale).round().clamp(-128, 127).to(torch.int8)
    return q, scale


def int8_sum_of_squares(q_int8: torch.Tensor) -> torch.Tensor:
    """Integer sum-of-squares per token as INT32.

    Args:
        q_int8: (num_kv_heads, seq_len, head_dim) int8
    Returns:
        (num_kv_heads, seq_len) int32
    """
    # Cast to int32 before squaring to avoid overflow
    q32 = q_int8.to(torch.int32)
    return (q32 * q32).sum(dim=-1)


def golden_topk(scores_int32: torch.Tensor, n_keep: int) -> torch.Tensor:
    """Deterministic top-k selection: keep tokens with smallest scores.

    Tie-break: among equal scores, prefer earlier token index.
    Returns indices of kept tokens, sorted ascending.

    Args:
        scores_int32: (num_kv_heads, seq_len) int32
        n_keep: number of tokens to keep
    Returns:
        (num_kv_heads, n_keep) int64 indices (sorted)
    """
    num_heads, seq_len = scores_int32.shape
    results = []
    for h in range(num_heads):
        s = scores_int32[h]  # (seq_len,)
        # Stable argsort ascending: ties broken by index (earlier = smaller)
        idx = torch.argsort(s.float(), stable=True)  # ascending = smallest norm first
        kept = idx[:n_keep]
        kept_sorted = kept.sort().values
        results.append(kept_sorted)
    return torch.stack(results, dim=0)  # (num_heads, n_keep)


def knorm_topk_float(keys: torch.Tensor, n_keep: int) -> torch.Tensor:
    """KnormPress exact float32 selection for comparison.

    Args:
        keys: (batch=1, num_kv_heads, seq_len, head_dim) float32
    Returns:
        (num_kv_heads, n_keep) int64 indices (sorted)
    """
    # KnormPress: negative norm, topk keeps highest = least negative = smallest norm
    scores = -torch.linalg.vector_norm(keys[0], dim=-1, dtype=torch.float32)
    # (num_kv_heads, seq_len)
    _, idx = scores.topk(n_keep, dim=-1)  # (num_heads, n_keep) — descending on score
    # sort each head's indices ascending
    return idx.sort(dim=-1).values


def jaccard_overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    """Jaccard overlap between two sorted index sets of shape (n,)."""
    set_a = set(a.tolist())
    set_b = set(b.tolist())
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


# ─── 5. Per-layer / per-head overlap comparison ───────────────────────────────
RATIOS = [0.3, 0.5, 0.7]
REPORT_LAYERS = list(range(num_layers))  # all layers
REPORT_HEADS_SAMPLE = 4  # report first N heads per layer

print(f"\n  Comparing INT8 golden model vs KnormPress float32")
print(f"  Layers: all {num_layers}  |  Heads reported: first {REPORT_HEADS_SAMPLE} per layer")
print(f"  Ratios: {RATIOS}\n")

overlap_summary = {}  # {ratio: {layer_idx: mean_jaccard}}

for ratio in RATIOS:
    n_keep = max(1, int(SEQ_LEN * (1 - ratio)))
    print(f"  Ratio={ratio}  n_keep={n_keep}/{SEQ_LEN}")
    layer_mean_jaccard = {}

    for layer_idx in REPORT_LAYERS:
        keys = pkv.layers[layer_idx].keys  # (1, num_kv_heads, seq_len, head_dim)
        keys_3d = keys[0]         # (num_kv_heads, seq_len, head_dim)

        # INT8 golden model
        q_int8, scale = int8_quantize_per_tensor(keys_3d)
        scores_i32 = int8_sum_of_squares(q_int8)   # (num_kv_heads, seq_len)
        idx_golden = golden_topk(scores_i32, n_keep)  # (num_kv_heads, n_keep)

        # KnormPress float32 reference
        idx_knorm = knorm_topk_float(keys, n_keep)    # (num_kv_heads, n_keep)

        # Jaccard per head
        head_jaccards = [
            jaccard_overlap(idx_golden[h], idx_knorm[h])
            for h in range(NUM_KV_HEADS)
        ]
        mean_j = sum(head_jaccards) / len(head_jaccards)
        layer_mean_jaccard[layer_idx] = mean_j

        if layer_idx in [0, 10] or layer_idx == num_layers - 1:
            print(f"    layer {layer_idx:>2}: mean Jaccard={mean_j:.4f}  "
                  f"(per-head first {REPORT_HEADS_SAMPLE}: "
                  f"{[round(j,3) for j in head_jaccards[:REPORT_HEADS_SAMPLE]]})")

    all_j = list(layer_mean_jaccard.values())
    global_mean = sum(all_j) / len(all_j)
    global_min = min(all_j)
    global_max = max(all_j)
    print(f"    → Global: mean={global_mean:.4f}  min={global_min:.4f}  max={global_max:.4f}\n")
    overlap_summary[ratio] = layer_mean_jaccard


# ─── 6. Save .mem test vectors ────────────────────────────────────────────────
print(f"{'─'*60}")
print("  Saving .mem test vectors for $readmemh")
print(f"{'─'*60}")

SAVE_RATIO = 0.5
n_keep_save = max(1, int(SEQ_LEN * (1 - SAVE_RATIO)))

for layer_idx in [0, 10]:
    keys = pkv.layers[layer_idx].keys[0]  # (num_kv_heads, seq_len, head_dim)
    # Use head 0 only for .mem files (FPGA processes one head at a time)
    keys_h0 = keys[0]  # (seq_len, head_dim)

    # INT8 quantize
    q_int8, scale = int8_quantize_per_tensor(keys_h0.unsqueeze(0))
    q_int8_h0 = q_int8[0]  # (seq_len, head_dim)
    scores_h0 = (q_int8_h0.to(torch.int32) ** 2).sum(dim=-1)  # (seq_len,)

    # Golden selection
    sort_idx = torch.argsort(scores_h0.float(), stable=True)
    kept_idx = sort_idx[:n_keep_save].sort().values

    # Write INT8 key .mem (each row = one token, space-separated hex bytes)
    key_mem_path = RESULTS_DIR / f"layer{layer_idx}_keys_int8.mem"
    with open(key_mem_path, "w") as f:
        f.write(f"// Layer {layer_idx} head 0 INT8 quantized keys\n")
        f.write(f"// Shape: seq_len={SEQ_LEN}, head_dim={HEAD_DIM}\n")
        f.write(f"// Scale: {scale:.6f}\n")
        f.write(f"// Format: one token per line, head_dim bytes in hex\n")
        for tok_idx in range(SEQ_LEN):
            row = q_int8_h0[tok_idx]  # (head_dim,)
            hex_vals = " ".join(f"{(b.item() & 0xFF):02x}" for b in row)
            f.write(hex_vals + "\n")

    # Write INT32 scores .mem (one score per token)
    score_mem_path = RESULTS_DIR / f"layer{layer_idx}_scores_int32.mem"
    with open(score_mem_path, "w") as f:
        f.write(f"// Layer {layer_idx} head 0 INT32 sum-of-squares scores\n")
        f.write(f"// Shape: seq_len={SEQ_LEN}\n")
        f.write(f"// Each line: 8-digit hex (32-bit unsigned)\n")
        for s in scores_h0:
            f.write(f"{s.item() & 0xFFFFFFFF:08x}\n")

    # Write kept indices .mem
    idx_mem_path = RESULTS_DIR / f"layer{layer_idx}_indices_topk.mem"
    with open(idx_mem_path, "w") as f:
        f.write(f"// Layer {layer_idx} head 0 kept token indices (ratio={SAVE_RATIO})\n")
        f.write(f"// n_keep={n_keep_save} of seq_len={SEQ_LEN}\n")
        for idx_val in kept_idx:
            f.write(f"{idx_val.item():04x}\n")

    print(f"  layer {layer_idx}:")
    print(f"    {key_mem_path.name}  ({SEQ_LEN} x {HEAD_DIM} bytes)")
    print(f"    {score_mem_path.name}  ({SEQ_LEN} scores)")
    print(f"    {idx_mem_path.name}  ({n_keep_save} indices at ratio={SAVE_RATIO})")
    print(f"    scale={scale:.6f}  score_range=[{scores_h0.min().item()}, {scores_h0.max().item()}]")

# ─── 7. Final report ──────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  PHASE 6 SUMMARY — copy into PROGRESS.md before writing Verilog")
print(f"{'='*65}")
print(f"""
  Model              : {MODEL_NAME}
  Prompt tokens      : {prompt_tokens}
  Cache type         : {type(pkv).__name__}
  num_layers         : {num_layers}
  num_kv_heads       : {NUM_KV_HEADS}
  head_dim           : {HEAD_DIM}
  seq_len (prefill)  : {SEQ_LEN}

  Key tensor shape (per layer):
    Layer  0: {tuple(pkv.layers[0].keys.shape)}   (batch, num_kv_heads, seq_len, head_dim)
    Layer 10: {tuple(pkv.layers[10].keys.shape)}

  KnormPress scoring direction (from installed source):
    Score = -L2_norm(key, dim=head_dim)  in float32
    topk(n_keep) on score = keep SMALLEST norms = EVICT LARGEST norms

  INT8 golden model vs KnormPress float32 — Jaccard overlap summary:
""")
for ratio in RATIOS:
    vals = list(overlap_summary[ratio].values())
    print(f"    ratio={ratio}: global mean Jaccard = {sum(vals)/len(vals):.4f}  "
          f"(min={min(vals):.4f}, max={max(vals):.4f})")
print(f"""
  .mem files written to: {RESULTS_DIR}
    layer0_keys_int8.mem, layer0_scores_int32.mem, layer0_indices_topk.mem
    layer10_keys_int8.mem, layer10_scores_int32.mem, layer10_indices_topk.mem

  STOP HERE — report the above numbers to the user before writing Verilog.
""")
