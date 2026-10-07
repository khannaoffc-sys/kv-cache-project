"""
run_benchmark.py — KV-cache compression benchmark for edge LLM inference.

Configs evaluated
─────────────────
  1. Baseline            — no compression
  2. SnapKVPress(0.5)    — token eviction via SnapKV
  3. KnormPress(0.5)     — key-norm-based token eviction
  4. HQQ 4-bit cache     — HuggingFace QuantizedCacheConfig (nbits=4, backend="HQQ")

Metrics
───────
  • Latency (s) and throughput (tokens/s) via time.perf_counter()
  • LongBench accuracy on 5 examples from the "narrativeqa" task

Output
──────
  results/benchmark_results.csv

Usage
─────
    python src/run_benchmark.py
"""

import csv
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantizedCacheConfig

# ─── Project imports ──────────────────────────────────────────────────────────
# PowerLogger is imported for annotation purposes only; actual logging is
# deferred until Jetson hardware is available.
from power_logger import PowerLogger  # noqa: F401  (used in TODO below)

# ─── Constants ────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 50
LONGBENCH_TASK = "narrativeqa"
LONGBENCH_NUM_EXAMPLES = 5
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_FILE = RESULTS_DIR / "benchmark_results.csv"

# ─── Device / dtype ───────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"\n{'═'*60}")
print(f"  Device : {DEVICE.upper()}  |  dtype: {DTYPE}  |  Model: {MODEL_ID}")
print(f"{'═'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# LongBench scoring
# ─────────────────────────────────────────────────────────────────────────────

def _longbench_score(prediction: str, ground_truths: List[str]) -> float:
    """Score a single prediction against one or more reference answers.

    Tries to import LongBench's own scorer; falls back to a simple
    token-F1 if the scorer is not available in this environment.
    """
    # Attempt to use LongBench's official scorer
    try:
        # LongBench stores its metric under longbench.metrics
        from longbench.metrics import scorer as lb_scorer  # type: ignore[import]

        scores = [lb_scorer(prediction, ref, LONGBENCH_TASK) for ref in ground_truths]
        return max(scores)
    except ImportError:
        pass

    # Official scorer not available — use token-level F1 (standard NLP fallback)
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
# Model + tokeniser loading
# ─────────────────────────────────────────────────────────────────────────────

print("Loading tokeniser …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model …")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto" if DEVICE == "cuda" else None,
)
if DEVICE == "cpu":
    model = model.to(DEVICE)
model.eval()
print("Model ready.\n")


# ─────────────────────────────────────────────────────────────────────────────
# LongBench dataset — small subset
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading LongBench/{LONGBENCH_TASK} ({LONGBENCH_NUM_EXAMPLES} examples) …")
try:
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
    print("  Accuracy column will be NaN for this run.\n")
    lb_examples = []


def _build_lb_prompt(example: Dict[str, Any]) -> str:
    """Format a LongBench example into a model prompt."""
    ctx = example.get("context", "")
    q = example.get("input", "")
    return (
        f"Read the following passage and answer the question concisely.\n\n"
        f"Passage:\n{ctx[:3000]}\n\n"   # truncate to avoid OOM on CPU
        f"Question: {q}\nAnswer:"
    )


def _get_references(example: Dict[str, Any]) -> List[str]:
    answers = example.get("answers", [])
    if isinstance(answers, str):
        answers = [answers]
    return answers or [""]


# ─────────────────────────────────────────────────────────────────────────────
# Generation helper
# ─────────────────────────────────────────────────────────────────────────────

def generate(
    prompt: str,
    press_ctx=None,
    cache_config=None,
) -> tuple[str, float]:
    """Run model.generate() and return (decoded_text, latency_s).

    Parameters
    ----------
    prompt      : str   — input text
    press_ctx   : optional kvpress context manager (e.g. SnapKVPress)
    cache_config: optional QuantizedCacheConfig for HQQ quant
    """
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    gen_kwargs: Dict[str, Any] = dict(
        **enc,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )
    if cache_config is not None:
        gen_kwargs["cache_config"] = cache_config
        gen_kwargs["cache_implementation"] = "quantized"

    def _run():
        with torch.no_grad():
            return model.generate(**gen_kwargs)

    t0 = time.perf_counter()
    if press_ctx is not None:
        with press_ctx(model):
            out_ids = _run()
    else:
        out_ids = _run()
    latency = time.perf_counter() - t0

    new_ids = out_ids[0, enc["input_ids"].shape[-1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return text, latency


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark configs
# ─────────────────────────────────────────────────────────────────────────────

def _build_configs():
    configs = []

    # 1. Baseline
    configs.append({"name": "baseline", "press": None, "cache_config": None})

    # 2. SnapKVPress
    try:
        from kvpress import SnapKVPress
        configs.append({
            "name": "SnapKVPress(0.5)",
            "press": SnapKVPress(compression_ratio=0.5),
            "cache_config": None,
        })
    except Exception as exc:
        print(f"  [WARN] SnapKVPress unavailable: {exc}")

    # 3. KnormPress
    try:
        from kvpress import KnormPress
        configs.append({
            "name": "KnormPress(0.5)",
            "press": KnormPress(compression_ratio=0.5),
            "cache_config": None,
        })
    except Exception as exc:
        print(f"  [WARN] KnormPress unavailable: {exc}")

    # 4. HQQ 4-bit quantised cache (no eviction press)
    try:
        cache_cfg = QuantizedCacheConfig(nbits=4, backend="HQQ")
        configs.append({
            "name": "HQQ-4bit",
            "press": None,
            "cache_config": cache_cfg,
        })
    except Exception as exc:
        print(f"  [WARN] HQQ QuantizedCacheConfig unavailable: {exc}")

    return configs


configs = _build_configs()

# ─────────────────────────────────────────────────────────────────────────────
# Short latency benchmark (constant prompt, no LB)
# ─────────────────────────────────────────────────────────────────────────────

LATENCY_PROMPT = (
    "Explain the key-value cache mechanism in transformer inference:"
)

print("─" * 60)
print("  Phase 1 — Latency benchmark (fixed prompt)")
print("─" * 60)

rows: List[Dict[str, Any]] = []

for cfg in configs:
    name = cfg["name"]
    print(f"\n  [{name}] generating …", end=" ", flush=True)

    try:
        # TODO: wrap with PowerLogger once on Jetson:
        #   logger = PowerLogger(interval=0.1)
        #   logger.start()
        text, latency = generate(
            LATENCY_PROMPT,
            press_ctx=cfg["press"],
            cache_config=cfg["cache_config"],
        )
        #   logger.stop()
        #   avg_w, dur_s, energy_j = logger.summary()

        tps = MAX_NEW_TOKENS / latency if latency > 0 else float("nan")
        print(f"done  {latency:.2f}s  {tps:.1f} tok/s")
        rows.append({
            "config": name,
            "prompt_type": "latency_fixed",
            "latency_s": round(latency, 4),
            "tokens_per_sec": round(tps, 2),
            "avg_power_w": "N/A",   # Jetson only
            "energy_j": "N/A",
            "accuracy": "N/A",
            "output_preview": text[:120].replace("\n", " "),
        })
    except Exception:
        print("FAILED")
        traceback.print_exc()
        rows.append({
            "config": name,
            "prompt_type": "latency_fixed",
            "latency_s": "ERROR",
            "tokens_per_sec": "ERROR",
            "avg_power_w": "N/A",
            "energy_j": "N/A",
            "accuracy": "N/A",
            "output_preview": "ERROR",
        })

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — LongBench accuracy (baseline + SnapKVPress only)
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
            pred, latency = generate(
                prompt,
                press_ctx=cfg["press"],
                cache_config=cfg["cache_config"],
            )
            score = _longbench_score(pred, refs)
            scores.append(score)
            total_latency += latency
            print(f"score={score:.3f}  lat={latency:.2f}s")
        except Exception:
            print("FAILED")
            traceback.print_exc()
            scores.append(float("nan"))

    avg_score = sum(s for s in scores if s == s) / max(len(scores), 1)
    avg_lat = total_latency / max(len(lb_examples), 1)
    tps = MAX_NEW_TOKENS / avg_lat if avg_lat > 0 else float("nan")

    rows.append({
        "config": name,
        "prompt_type": f"longbench_{LONGBENCH_TASK}",
        "latency_s": round(avg_lat, 4),
        "tokens_per_sec": round(tps, 2),
        "avg_power_w": "N/A",
        "energy_j": "N/A",
        "accuracy": round(avg_score, 4),
        "output_preview": f"{len(lb_examples)} examples",
    })
    print(f"  [{name}] avg score={avg_score:.4f}  avg lat={avg_lat:.2f}s\n")


# ─────────────────────────────────────────────────────────────────────────────
# Save CSV
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
fieldnames = [
    "config", "prompt_type", "latency_s", "tokens_per_sec",
    "avg_power_w", "energy_j", "accuracy", "output_preview",
]

with open(RESULTS_FILE, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\n{'═'*60}")
print(f"  Results saved → {RESULTS_FILE}")
print(f"{'═'*60}\n")

# ─── Console summary table ─────────────────────────────────────────────────
print(f"  {'Config':<22} {'Type':<28} {'Lat(s)':>7} {'Tok/s':>7} {'Acc':>7}")
print(f"  {'─'*22} {'─'*28} {'─'*7} {'─'*7} {'─'*7}")
for r in rows:
    print(
        f"  {r['config']:<22} {r['prompt_type']:<28} "
        f"{str(r['latency_s']):>7} {str(r['tokens_per_sec']):>7} {str(r['accuracy']):>7}"
    )
print()
