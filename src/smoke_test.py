"""
smoke_test.py — Smoke test for KV-cache compression with SnapKVPress.

Workflow:
  1. Detect device (CUDA if available, else CPU).
  2. Load the model specified by --model (default: Qwen/Qwen2.5-0.5B-Instruct on CPU,
     Qwen/Qwen2.5-1.5B-Instruct on CUDA).
  3. Generate 50 new tokens from a >= 512-token prompt — no compression (baseline).
  4. Generate 50 new tokens from the same prompt — wrapped in SnapKVPress(0.5).
  5. Print both outputs side by side for visual comparison.

Usage:
    # CPU (laptop):
    python src/smoke_test.py --model Qwen/Qwen2.5-0.5B-Instruct

    # GPU:
    python src/smoke_test.py
"""

import argparse
import os
import textwrap

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── Argument / env parsing ───────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_default_model = os.environ.get(
    "MODEL_NAME",
    "Qwen/Qwen2.5-1.5B-Instruct" if DEVICE == "cuda" else "Qwen/Qwen2.5-0.5B-Instruct",
)

parser = argparse.ArgumentParser(description="Smoke test: baseline vs SnapKVPress")
parser.add_argument(
    "--model",
    default=_default_model,
    dest="model_name",
    help="HuggingFace model ID",
)
args = parser.parse_args()

MODEL_NAME = args.model_name
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MAX_NEW_TOKENS = 50
COMPRESSION_RATIO = 0.5
WRAP_WIDTH = 80

# ─── Prompt: must be >= 512 tokens to exercise SnapKV window ─────────────────
PROMPT = (
    "The key-value (KV) cache is a fundamental optimization technique in autoregressive "
    "transformer inference. During the prefill phase, every input token is processed "
    "in parallel through the full self-attention mechanism. The resulting key and value "
    "tensors for each layer are stored in GPU memory so they do not need to be recomputed "
    "during the subsequent token-by-token generation (decode) phase. "
    "Without caching, generating each new token would require O(n^2) attention over all "
    "previously seen tokens, making long-sequence generation prohibitively expensive. "
    "The KV cache reduces this to O(n) per new token at the cost of memory. "
    "For a model with H heads, head dimension d, L layers, and context length n, "
    "the cache uses 2 * H * d * L * n * sizeof(dtype) bytes. "
    "Token eviction methods (SnapKV, KnormPress, H2O, ScissorHands) score each cached "
    "token by its historical attention weight or key norm and permanently discard the "
    "lowest-scoring tokens when the cache exceeds a budget, reducing memory by 30-70%. "
    "Quantization methods (KIVI, WKVQuant, HQQ, KVQuant) represent stored keys and values "
    "in fewer bits — typically INT4 or INT8 — reducing memory by 2-4x. "
    "Hybrid systems (ZipCache, Gear) combine eviction and quantization. "
    "The kvpress library provides a unified API for experimenting with many strategies. "
    "Benchmarks such as LongBench and RULER are used to measure accuracy after compression. "
    "Explain in detail how the KV cache is allocated, when tokens are evicted, "
    "and what the key-norm eviction criterion is in KnormPress:"
)

print(f"\n{'─'*60}")
print(f"  Device : {DEVICE.upper()}")
print(f"  dtype  : {DTYPE}")
print(f"  Model  : {MODEL_NAME}")
print(f"{'─'*60}\n")

# ─── Load model & tokenizer ───────────────────────────────────────────────────
print("Loading tokenizer …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model …")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=DTYPE,
    device_map="auto" if DEVICE == "cuda" else None,
)
if DEVICE == "cpu":
    model = model.to(DEVICE)
model.eval()
print("Model loaded.\n")

# ─── Tokenize and assert prompt length ───────────────────────────────────────
inputs = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)
prompt_tokens = inputs["input_ids"].shape[-1]
print(f"Prompt tokens: {prompt_tokens}")
assert prompt_tokens >= 512, (
    f"Prompt too short ({prompt_tokens} tokens). SnapKV needs seq_len > window_size=64 "
    "and we require >= 512 for a meaningful benchmark."
)
print(f"  ✓ Prompt length {prompt_tokens} >= 512\n")


def decode_output(output_ids: torch.Tensor) -> str:
    new_ids = output_ids[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ─── Baseline generation ──────────────────────────────────────────────────────
print("Generating baseline (no compression) …")
with torch.no_grad():
    baseline_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )
baseline_text = decode_output(baseline_ids)
print("  Done.\n")

# ─── SnapKVPress generation ───────────────────────────────────────────────────
print(f"Generating with SnapKVPress(compression_ratio={COMPRESSION_RATIO}) …")
try:
    from kvpress import SnapKVPress
    press = SnapKVPress(compression_ratio=COMPRESSION_RATIO)
    with torch.no_grad(), press(model):
        snap_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    snap_text = decode_output(snap_ids)
    print("  Done.\n")
except Exception as exc:
    snap_text = f"[ERROR: {exc}]"
    print(f"  SnapKVPress generation failed: {exc}\n")

# ─── Side-by-side comparison ──────────────────────────────────────────────────
divider = "═" * WRAP_WIDTH
half = "─" * (WRAP_WIDTH // 2 - 1)

print(divider)
print("  SIDE-BY-SIDE COMPARISON (first 50 generated tokens)")
print(divider)
print(f"\n  Prompt tokens: {prompt_tokens}  (first 80 chars): {PROMPT[:80]!r}\n")
print(f"  {'BASELINE (no compression)':<38} │  {'SnapKVPress (ratio=0.5)'}")
print(f"  {half} │  {half}")

baseline_lines = textwrap.wrap(baseline_text, width=36) or ["(empty)"]
snap_lines = textwrap.wrap(snap_text, width=36) or ["(empty)"]

max_lines = max(len(baseline_lines), len(snap_lines))
for i in range(max_lines):
    b = baseline_lines[i] if i < len(baseline_lines) else ""
    s = snap_lines[i] if i < len(snap_lines) else ""
    print(f"  {b:<38} │  {s}")

print(f"\n{divider}\n")
