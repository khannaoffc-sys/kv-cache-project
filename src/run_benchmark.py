"""
run_benchmark.py — KV-cache compression benchmark for edge LLM inference.

Configs evaluated
─────────────────
  1. Baseline            — no compression
  2. SnapKVPress(0.5)    — token eviction via SnapKV
  3. KnormPress(0.5)     — key-norm-based token eviction
  4. HQQ 4-bit cache     — QuantizedCache(backend="hqq", nbits=4)

Metrics
───────
  • Latency mean ± std (s) over 3 repetitions, one prior warm-up
  • Throughput (tokens/s)
  • KV cache size in bytes after compression
  • LongBench token-F1 accuracy on 5 examples
  • Power (W) and energy (J) — only when power_source != fake

Output
──────
  results/benchmark_results.csv

Usage
─────
    python src/run_benchmark.py [--model MODEL_NAME]

    # Laptop (CPU-only, 7.7 GB RAM):
    python src/run_benchmark.py --model Qwen/Qwen2.5-0.5B-Instruct

    # Cloud GPU (default):
    python src/run_benchmark.py

Environment variables
─────────────────────
    MODEL_NAME      — overridden by --model flag
    POWER_BACKEND   — fake | nvml | jtop  (default: fake)
"""

import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import QuantizedCache

# ─── Argument parsing ─────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(
    description="KV-cache compression benchmark",
    add_help=True,
)
_parser.add_argument(
    "--model",
    default=os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"),
    dest="model_name",
    help=(
        "HuggingFace model ID (default: Qwen/Qwen2.5-1.5B-Instruct). "
        "On a 7.7 GB CPU-only laptop use Qwen/Qwen2.5-0.5B-Instruct."
    ),
)
_args = _parser.parse_args()

MODEL_NAME: str = _args.model_name

# ─── Power backend ────────────────────────────────────────────────────────────
# POWER_BACKEND = fake | nvml | jtop
POWER_BACKEND: str = os.environ.get("POWER_BACKEND", "fake").lower()
if POWER_BACKEND not in ("fake", "nvml", "jtop"):
    sys.exit(f"[ERROR] POWER_BACKEND must be fake|nvml|jtop, got: {POWER_BACKEND!r}")

if POWER_BACKEND == "fake":
    from power_logger_fake import PowerLogger  # type: ignore[import]
elif POWER_BACKEND == "nvml":
    from power_logger_nvml import PowerLogger  # type: ignore[import]
else:  # jtop
    from power_logger import PowerLogger  # type: ignore[import]

# ─── Constants ────────────────────────────────────────────────────────────────
MAX_NEW_TOKENS = 50
LONGBENCH_TASK = "narrativeqa"
LONGBENCH_NUM_EXAMPLES = 5
NUM_REPS = 3  # timed repetitions per config
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_FILE = RESULTS_DIR / "benchmark_results.csv"

# ─── Device / dtype ───────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"\n{'═'*60}")
print(f"  Device  : {DEVICE.upper()}")
print(f"  dtype   : {DTYPE}")
print(f"  Model   : {MODEL_NAME}")
print(f"  Power   : {POWER_BACKEND}")
print(f"{'═'*60}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Model + tokeniser loading
# ─────────────────────────────────────────────────────────────────────────────

print("Loading tokeniser …")
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
print("Model ready.\n")

# ─────────────────────────────────────────────────────────────────────────────
# LATENCY_PROMPT — must be >= 512 tokens; must exceed SnapKV window_size=64
# ─────────────────────────────────────────────────────────────────────────────

LATENCY_PROMPT = (
    "The key-value (KV) cache is a fundamental optimization technique in autoregressive "
    "transformer inference. During the prefill phase, every input token is processed "
    "in parallel through the full self-attention mechanism. The resulting key and value "
    "tensors for each layer are stored in GPU memory so they do not need to be recomputed "
    "during the subsequent token-by-token generation (decode) phase. "
    "Without caching, generating each new token would require O(n²) attention over all "
    "previously seen tokens, making long-sequence generation prohibitively expensive. "
    "The KV cache reduces this to O(n) per new token. However, storing the cache for "
    "long contexts requires substantial memory: for a model with H heads, head dimension "
    "d, L layers, and context length n, the cache uses 2 × H × d × L × n × sizeof(dtype) "
    "bytes. For a 7B-parameter model at float16, a 32k-token context requires roughly "
    "16 GB of KV cache memory — more than the entire model parameters. "
    "To reduce this footprint, researchers have proposed several complementary strategies. "
    "Token eviction methods (H2O, SnapKV, KnormPress, ScissorHands) score each cached "
    "token by its historical attention weight, key norm, or observation frequency, then "
    "permanently discard the lowest-scoring tokens when the cache exceeds a budget. "
    "Quantization methods (KIVI, WKVQuant, HQQ, KVQuant) represent the stored keys and "
    "values in fewer bits — typically INT4 or INT8 — reducing memory by 2–4× at the cost "
    "of small accuracy degradation. Attention approximation methods (MagicPIG, Quest) "
    "avoid materializing the full cache entirely by using locality-sensitive hashing or "
    "page-level attention to find the relevant subset at decode time. "
    "Hybrid systems combine two or more of these techniques: for example, ZipCache applies "
    "both eviction and quantization, assigning more bits to tokens deemed important by an "
    "attention-based score and fewer bits to less important tokens. "
    "The trade-off space is multidimensional. Aggressive eviction or low-bit quantization "
    "can impair performance on tasks requiring access to distant context, such as multi-hop "
    "reasoning over long documents, needle-in-a-haystack retrieval, and code synthesis "
    "that references an API defined hundreds of lines earlier. Benchmarks such as LongBench "
    "and RULER are designed to stress-test these long-range dependencies. "
    "On edge and embedded hardware — Jetson Orin Nano, Raspberry Pi 5, mobile NPUs — the "
    "memory constraint is even more severe, making compression essential for deployment. "
    "The kvpress library provides a unified Python API for experimenting with a wide range "
    "of KV compression strategies, including SnapKVPress, KnormPress, ObservedAttentionPress, "
    "and several others, enabling systematic comparisons under identical generation settings."
)

# Assert tokenized length >= 512 AND > SnapKV window_size=64
_PROMPT_IDS = tokenizer(LATENCY_PROMPT, return_tensors="pt")["input_ids"]
_PROMPT_TOKENS = _PROMPT_IDS.shape[-1]
_SNAPKV_WINDOW = 64  # SnapKVPress default window_size
assert _PROMPT_TOKENS >= 512, (
    f"LATENCY_PROMPT is only {_PROMPT_TOKENS} tokens — must be >= 512. "
    "Lengthen the prompt."
)
assert _PROMPT_TOKENS > _SNAPKV_WINDOW, (
    f"LATENCY_PROMPT has {_PROMPT_TOKENS} tokens, must exceed SnapKV "
    f"window_size={_SNAPKV_WINDOW} to avoid eviction pass-through."
)
print(f"LATENCY_PROMPT: {_PROMPT_TOKENS} tokens  ✓  (>= 512 and > window_size={_SNAPKV_WINDOW})\n")


# ─────────────────────────────────────────────────────────────────────────────
# LongBench scoring (token-F1 fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _longbench_score(prediction: str, ground_truths: List[str]) -> float:
    try:
        from longbench_metrics import score as lb_score  # type: ignore[import]
        return max(lb_score(prediction, ref, LONGBENCH_TASK) for ref in ground_truths)
    except ImportError:
        pass

    def _token_f1(pred: str, ref: str) -> float:
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        if not pred_tokens or not ref_tokens:
            return 0.0
        common = set(pred_tokens) & set(ref_tokens)
        if not common:
            return 0.0
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    return max(_token_f1(prediction, ref) for ref in ground_truths)


# ─────────────────────────────────────────────────────────────────────────────
# LongBench dataset — small subset
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading LongBench/{LONGBENCH_TASK} ({LONGBENCH_NUM_EXAMPLES} examples) …")
lb_examples: List[Dict[str, Any]] = []
try:
    from datasets import load_dataset
    lb_dataset = load_dataset(
        "THUDM/LongBench",
        LONGBENCH_TASK,
        split="test",
        trust_remote_code=True,
    )
    lb_examples = list(lb_dataset.select(range(LONGBENCH_NUM_EXAMPLES)))
    print(f"  Loaded {len(lb_examples)} examples.\n")
except Exception as exc:
    print(f"  [WARN] Could not load LongBench: {exc}")
    print("  Accuracy column will be N/A for this run.\n")


def _build_lb_prompt(example: Dict[str, Any]) -> str:
    ctx = example.get("context", "")
    q = example.get("input", "")
    return (
        "Read the following passage and answer the question concisely.\n\n"
        f"Passage:\n{ctx[:3000]}\n\n"
        f"Question: {q}\nAnswer:"
    )


def _get_references(example: Dict[str, Any]) -> List[str]:
    answers = example.get("answers", [])
    if isinstance(answers, str):
        answers = [answers]
    return answers or [""]


# ─────────────────────────────────────────────────────────────────────────────
# KV cache size helper
# ─────────────────────────────────────────────────────────────────────────────

def _kv_cache_bytes(past_key_values) -> Optional[int]:
    """Return total bytes in a DynamicCache / tuple-of-tuples past_key_values."""
    if past_key_values is None:
        return None
    total = 0
    try:
        # DynamicCache (transformers >= 4.38)
        if hasattr(past_key_values, "key_cache"):
            for t in past_key_values.key_cache:
                total += t.numel() * t.element_size()
            for t in past_key_values.value_cache:
                total += t.numel() * t.element_size()
            return total
        # Legacy tuple-of-tuples
        for layer in past_key_values:
            for t in layer:
                total += t.numel() * t.element_size()
        return total
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Generation helper — returns (text, latency_s, prompt_tokens, new_tokens, kv_bytes)
# ─────────────────────────────────────────────────────────────────────────────

def generate(
    prompt: str,
    press_ctx=None,
    cache_factory=None,
) -> tuple:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    prompt_tok = enc["input_ids"].shape[-1]

    gen_kwargs: Dict[str, Any] = dict(
        **enc,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        return_dict_in_generate=True,
    )
    if cache_factory is not None:
        gen_kwargs["past_key_values"] = cache_factory()

    def _run():
        with torch.no_grad():
            return model.generate(**gen_kwargs)

    t0 = time.perf_counter()
    if press_ctx is not None:
        with press_ctx(model):
            out = _run()
    else:
        out = _run()
    latency = time.perf_counter() - t0

    # Handle both dict output (return_dict_in_generate=True) and plain tensor
    if hasattr(out, "sequences"):
        out_ids = out.sequences
        pkv = getattr(out, "past_key_values", None)
    else:
        out_ids = out
        pkv = None

    new_ids = out_ids[0, prompt_tok:]
    new_tok = new_ids.shape[-1]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    kv_bytes = _kv_cache_bytes(pkv)
    return text, latency, prompt_tok, new_tok, kv_bytes


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark configs
# ─────────────────────────────────────────────────────────────────────────────

def _build_configs():
    configs = []
    configs.append({"name": "baseline", "press": None, "cache_factory": None})

    try:
        from kvpress import SnapKVPress
        configs.append({
            "name": "SnapKVPress(0.5)",
            "press": SnapKVPress(compression_ratio=0.5),
            "cache_factory": None,
        })
    except Exception as exc:
        print(f"  [WARN] SnapKVPress unavailable: {exc}")

    try:
        from kvpress import KnormPress
        configs.append({
            "name": "KnormPress(0.5)",
            "press": KnormPress(compression_ratio=0.5),
            "cache_factory": None,
        })
    except Exception as exc:
        print(f"  [WARN] KnormPress unavailable: {exc}")

    try:
        _ = QuantizedCache
        configs.append({
            "name": "HQQ-4bit",
            "press": None,
            "cache_factory": lambda: QuantizedCache(
                backend="hqq",
                config=model.config,
                nbits=4,
            ),
        })
    except Exception as exc:
        print(f"  [WARN] HQQ QuantizedCache unavailable: {exc}")

    return configs


configs = _build_configs()

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Latency benchmark (fixed prompt, NUM_REPS reps, 1 warm-up)
# ─────────────────────────────────────────────────────────────────────────────

print("─" * 60)
print("  Phase 1 — Latency benchmark (fixed prompt)")
print(f"  {NUM_REPS} timed reps per config, 1 untimed warm-up per config")
print("─" * 60)

rows: List[Dict[str, Any]] = []

# Rotate config order across reps: rep-0 → [0,1,2,3], rep-1 → [1,2,3,0], ...
# This reduces systematic bias from thermal throttling and cache warm effects.
import random as _random

rotated_configs: List[Dict] = []
for rep in range(NUM_REPS):
    shift = rep % len(configs)
    rotated_configs.append(configs[shift:] + configs[:shift])

# Collect all timed results per config name
rep_latencies: Dict[str, List[float]] = {c["name"]: [] for c in configs}
rep_kv_bytes: Dict[str, List[Optional[int]]] = {c["name"]: [] for c in configs}
rep_new_tokens: Dict[str, List[int]] = {c["name"]: [] for c in configs}
prompt_tokens_per_cfg: Dict[str, int] = {}

# Warm-up: one untimed pass per config (in original order)
print("\n  --- Warm-up (untimed) ---")
for cfg in configs:
    name = cfg["name"]
    print(f"  [{name}] warm-up …", end=" ", flush=True)
    try:
        generate(
            LATENCY_PROMPT,
            press_ctx=cfg.get("press"),
            cache_factory=cfg.get("cache_factory"),
        )
        print("done")
    except Exception as exc:
        print(f"WARN: {exc}")

# Timed reps
print(f"\n  --- Timed reps ({NUM_REPS}) ---")
for rep_idx, rep_order in enumerate(rotated_configs):
    print(f"\n  Rep {rep_idx + 1}/{NUM_REPS}:")
    for cfg in rep_order:
        name = cfg["name"]
        print(f"    [{name}] …", end=" ", flush=True)
        try:
            _, lat, ptok, ntok, kv_b = generate(
                LATENCY_PROMPT,
                press_ctx=cfg.get("press"),
                cache_factory=cfg.get("cache_factory"),
            )
            rep_latencies[name].append(lat)
            rep_kv_bytes[name].append(kv_b)
            rep_new_tokens[name].append(ntok)
            prompt_tokens_per_cfg[name] = ptok
            kv_str = f"{kv_b//1024}KB" if kv_b else "?"
            print(f"{lat:.2f}s  kv={kv_str}")
        except Exception as exc:
            print(f"FAILED: {exc}")
            traceback.print_exc()
            rep_latencies[name].append(float("nan"))
            rep_kv_bytes[name].append(None)

# Summarise into rows
import math as _math

for cfg in configs:
    name = cfg["name"]
    lats = [l for l in rep_latencies[name] if not _math.isnan(l)]
    all_lats = rep_latencies[name]
    status = "ok" if lats and len(lats) == NUM_REPS else (
        "partial" if lats else "ERROR"
    )

    if lats:
        mean_lat = sum(lats) / len(lats)
        std_lat = (sum((l - mean_lat) ** 2 for l in lats) / len(lats)) ** 0.5 if len(lats) > 1 else 0.0
        mean_ntok = sum(rep_new_tokens[name]) / len(rep_new_tokens[name]) if rep_new_tokens[name] else MAX_NEW_TOKENS
        tps = mean_ntok / mean_lat if mean_lat > 0 else float("nan")
        kv_vals = [b for b in rep_kv_bytes[name] if b is not None]
        kv_bytes_mean = int(sum(kv_vals) / len(kv_vals)) if kv_vals else None
    else:
        mean_lat = std_lat = tps = float("nan")
        kv_bytes_mean = None

    # Power: only log if we have a real backend
    avg_w = dur_s = energy_j = float("nan")
    if status == "ok" and POWER_BACKEND != "fake":
        try:
            logger = PowerLogger(interval=0.1)
            logger.start()
            generate(
                LATENCY_PROMPT,
                press_ctx=cfg.get("press"),
                cache_factory=cfg.get("cache_factory"),
            )
            logger.stop()
            avg_w, dur_s, energy_j = logger.summary()
        except Exception as exc:
            print(f"  [WARN] Power logging failed for {name}: {exc}")

    rows.append({
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "config": name,
        "prompt_type": "latency_fixed",
        "prompt_tokens": prompt_tokens_per_cfg.get(name, _PROMPT_TOKENS),
        "new_tokens": int(mean_ntok) if not _math.isnan(tps) else "N/A",
        "kv_cache_bytes": kv_bytes_mean if kv_bytes_mean is not None else "N/A",
        "latency_s": round(mean_lat, 4) if not _math.isnan(mean_lat) else "ERROR",
        "latency_std_s": round(std_lat, 4) if not _math.isnan(std_lat) else "N/A",
        "tokens_per_sec": round(tps, 2) if not _math.isnan(tps) else "ERROR",
        "power_source": POWER_BACKEND,
        "avg_power_w": round(avg_w, 3) if POWER_BACKEND != "fake" and not _math.isnan(avg_w) else "",
        "energy_j": round(energy_j, 3) if POWER_BACKEND != "fake" and not _math.isnan(energy_j) else "",
        "accuracy": "N/A",
        "status": status,
        "output_preview": "",
    })

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — LongBench accuracy
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'─'*60}")
print("  Phase 2 — LongBench accuracy (baseline + SnapKVPress)")
print(f"{'─'*60}")

lb_configs = [c for c in configs if c["name"] in ("baseline", "SnapKVPress(0.5)")]

for cfg in lb_configs:
    name = cfg["name"]
    if not lb_examples:
        print(f"  [{name}] skipped — no LongBench examples loaded.")
        continue

    scores: List[float] = []
    total_latency = 0.0

    for i, ex in enumerate(lb_examples):
        prompt = _build_lb_prompt(ex)
        refs = _get_references(ex)
        print(f"  [{name}] example {i+1}/{len(lb_examples)} …", end=" ", flush=True)

        try:
            pred, lat, ptok, ntok, _ = generate(
                prompt,
                press_ctx=cfg.get("press"),
                cache_factory=cfg.get("cache_factory"),
            )
            score = _longbench_score(pred, refs)
            scores.append(score)
            total_latency += lat
            print(f"score={score:.3f}  lat={lat:.2f}s")
        except Exception:
            print("FAILED")
            traceback.print_exc()
            scores.append(float("nan"))

    valid_scores = [s for s in scores if not _math.isnan(s)]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else float("nan")
    avg_lat = total_latency / max(len(lb_examples), 1)
    tps = MAX_NEW_TOKENS / avg_lat if avg_lat > 0 else float("nan")

    rows.append({
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "config": name,
        "prompt_type": f"longbench_{LONGBENCH_TASK}",
        "prompt_tokens": "varies",
        "new_tokens": MAX_NEW_TOKENS,
        "kv_cache_bytes": "N/A",
        "latency_s": round(avg_lat, 4),
        "latency_std_s": "N/A",
        "tokens_per_sec": round(tps, 2) if not _math.isnan(tps) else "N/A",
        "power_source": POWER_BACKEND,
        "avg_power_w": "",
        "energy_j": "",
        "accuracy": round(avg_score, 4) if not _math.isnan(avg_score) else "NaN",
        "status": "ok" if valid_scores else "ERROR",
        "output_preview": f"{len(lb_examples)} examples",
    })
    print(f"  [{name}] avg score={avg_score:.4f}  avg lat={avg_lat:.2f}s\n")


# ─────────────────────────────────────────────────────────────────────────────
# Save CSV
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
fieldnames = [
    "model_name", "device", "dtype",
    "config", "prompt_type", "prompt_tokens", "new_tokens", "kv_cache_bytes",
    "latency_s", "latency_std_s", "tokens_per_sec",
    "power_source", "avg_power_w", "energy_j",
    "accuracy", "status", "output_preview",
]

with open(RESULTS_FILE, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\n{'═'*60}")
print(f"  Results saved → {RESULTS_FILE}")
print(f"{'═'*60}\n")

print(f"  {'Config':<22} {'Status':<8} {'Lat(s)'!s:>9} {'±Std':>7} {'Tok/s':>7} {'KV-bytes':>10} {'Acc':>7}")
print(f"  {'─'*22} {'─'*8} {'─'*9} {'─'*7} {'─'*7} {'─'*10} {'─'*7}")
for r in rows:
    print(
        f"  {r['config']:<22} {r['status']:<8} "
        f"{str(r['latency_s']):>9} {str(r['latency_std_s']):>7} "
        f"{str(r['tokens_per_sec']):>7} {str(r['kv_cache_bytes']):>10} "
        f"{str(r['accuracy']):>7}"
    )
print()
