"""
smoke_test.py — Smoke test for KV-cache compression with SnapKVPress.

Workflow:
  1. Detect device (CUDA if available, else CPU).
  2. Load Qwen/Qwen2.5-1.5B-Instruct in float16 (CUDA) or float32 (CPU).
  3. Generate 50 new tokens from a short prompt — no compression (baseline).
  4. Generate 50 new tokens from the same prompt — wrapped in SnapKVPress(0.5).
  5. Print both outputs side by side for visual comparison.

Usage:
    python src/smoke_test.py
"""

import textwrap

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "Explain the key-value cache in transformer inference in simple terms:"
MAX_NEW_TOKENS = 50
COMPRESSION_RATIO = 0.5
WRAP_WIDTH = 80

# ─────────────────────────────────────────────────────────────────────────────
# Device & dtype
# ─────────────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"\n{'─'*60}")
print(f"  Device : {device.upper()}")
print(f"  dtype  : {dtype}")
print(f"  Model  : {MODEL_ID}")
print(f"{'─'*60}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Load model & tokenizer
# ─────────────────────────────────────────────────────────────────────────────
print("Loading tokenizer …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Loading model (this may take a moment on first run) …")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None,
)
if device == "cpu":
    model = model.to(device)
model.eval()
print("Model loaded.\n")

# ─────────────────────────────────────────────────────────────────────────────
# Tokenise prompt
# ─────────────────────────────────────────────────────────────────────────────
inputs = tokenizer(PROMPT, return_tensors="pt").to(device)


def decode_output(output_ids: torch.Tensor) -> str:
    """Decode only the newly generated tokens (strip the prompt prefix)."""
    new_ids = output_ids[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline generation — no compression
# ─────────────────────────────────────────────────────────────────────────────
print("Generating baseline (no compression) …")
with torch.no_grad():
    baseline_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )
baseline_text = decode_output(baseline_ids)
print("  Done.\n")

# ─────────────────────────────────────────────────────────────────────────────
# SnapKVPress generation
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# Side-by-side comparison
# ─────────────────────────────────────────────────────────────────────────────
divider = "═" * WRAP_WIDTH
half = "─" * (WRAP_WIDTH // 2 - 1)

print(divider)
print("  SIDE-BY-SIDE COMPARISON (first 50 generated tokens)")
print(divider)

print(f"\n  Prompt: {PROMPT!r}\n")

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
