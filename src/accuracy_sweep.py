"""
accuracy_sweep.py -- LongBench accuracy sweep across configs x compression ratios.

Evaluates:
  - Configs      : baseline, SnapKVPress, KnormPress, HQQ-4bit
  - Comp. ratios : 0.3, 0.5, 0.7  (applied to eviction-based presses only;
                   HQQ uses fixed nbits=4 and is run once at ratio=N/A)
  - Examples     : 30-50 from LongBench/narrativeqa

Output:
  results/accuracy_sweep.csv
  Columns: config, compression_ratio, task, num_examples,
           avg_accuracy, avg_latency_s, avg_tokens_per_sec

Usage:
    python src/accuracy_sweep.py [--num_examples 30]
"""

import argparse
import csv
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantizedCacheConfig

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="LongBench accuracy sweep")
parser.add_argument(
    "--num_examples", type=int, default=40,
    help="Number of LongBench examples to evaluate (default: 40)"
)
parser.add_argument(
    "--task", type=str, default="narrativeqa",
    help="LongBench task name (default: narrativeqa)"
)
args = parser.parse_args()

NUM_EXAMPLES: int = max(1, args.num_examples)
LONGBENCH_TASK: str = args.task
MAX_NEW_TOKENS: int = 50
CONTEXT_MAX_CHARS: int = 3000  # truncate to avoid OOM on CPU
MODEL_ID: str = "Qwen/Qwen2.5-1.5B-Instruct"

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_FILE = RESULTS_DIR / "accuracy_sweep.csv"

COMPRESSION_RATIOS: List[float] = [0.3, 0.5, 0.7]

# ---------------------------------------------------------------------------
# Device / dtype
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"\n{'='*65}")
print(f"  Accuracy Sweep  |  Device: {DEVICE.upper()}  |  Examples: {NUM_EXAMPLES}")
print(f"  Task: {LONGBENCH_TASK}  |  Ratios: {COMPRESSION_RATIOS}")
print(f"{'='*65}\n")

# ---------------------------------------------------------------------------
# Load model + tokeniser
# ---------------------------------------------------------------------------
print("Loading tokeniser ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto" if DEVICE == "cuda" else None,
)
if DEVICE == "cpu":
    model = model.to(DEVICE)
model.eval()
print("Model ready.\n")

# ---------------------------------------------------------------------------
# Load LongBench dataset
# ---------------------------------------------------------------------------
print(f"Loading LongBench/{LONGBENCH_TASK} ({NUM_EXAMPLES} examples) ...")
try:
    lb_dataset = load_dataset(
        "THUDM/LongBench",
        LONGBENCH_TASK,
        split="test",
        trust_remote_code=True,
    )
    n = min(NUM_EXAMPLES, len(lb_dataset))
    lb_examples = list(lb_dataset.select(range(n)))
    print(f"  Loaded {len(lb_examples)} examples.\n")
except Exception as exc:
    print(f"[FATAL] Could not load LongBench: {exc}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_prompt(example: Dict[str, Any]) -> str:
    ctx = example.get("context", "")[:CONTEXT_MAX_CHARS]
    q = example.get("input", "")
    return (
        "Read the following passage and answer the question concisely.\n\n"
        f"Passage:\n{ctx}\n\nQuestion: {q}\nAnswer:"
    )


def _get_refs(example: Dict[str, Any]) -> List[str]:
    answers = example.get("answers", [])
    if isinstance(answers, str):
        answers = [answers]
    return answers or [""]


def _token_f1(pred: str, ref: str) -> float:
    """Token-level F1 (standard NLP fallback when official scorer unavailable)."""
    pred_toks = pred.lower().split()
    ref_toks = ref.lower().split()
    if not pred_toks or not ref_toks:
        return 0.0
    common = set(pred_toks) & set(ref_toks)
    if not common:
        return 0.0
    p = len(common) / len(pred_toks)
    r = len(common) / len(ref_toks)
    return 2 * p * r / (p + r)


def _score(pred: str, refs: List[str]) -> float:
    try:
        from longbench.metrics import scorer as lb_scorer  # type: ignore[import]
        return max(lb_scorer(pred, ref, LONGBENCH_TASK) for ref in refs)
    except ImportError:
        pass
    return max(_token_f1(pred, ref) for ref in refs)


def generate(prompt: str, press_ctx=None, cache_config=None):
    """Return (text, latency_s)."""
    enc = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    ).to(DEVICE)
    gen_kwargs: Dict[str, Any] = dict(
        **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
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


# ---------------------------------------------------------------------------
# Build sweep plan:
#   (label, compression_ratio_or_NA, press_or_None, cache_config_or_None)
# ---------------------------------------------------------------------------
sweep: List[Dict[str, Any]] = []

# 1. Baseline (no compression) -- run once
sweep.append({
    "config": "baseline",
    "compression_ratio": "N/A",
    "press": None,
    "cache_config": None,
})

# 2. SnapKVPress x ratios
try:
    from kvpress import SnapKVPress
    for ratio in COMPRESSION_RATIOS:
        sweep.append({
            "config": "SnapKVPress",
            "compression_ratio": ratio,
            "press": SnapKVPress(compression_ratio=ratio),
            "cache_config": None,
        })
except Exception as exc:
    print(f"[WARN] SnapKVPress unavailable: {exc}")

# 3. KnormPress x ratios
try:
    from kvpress import KnormPress
    for ratio in COMPRESSION_RATIOS:
        sweep.append({
            "config": "KnormPress",
            "compression_ratio": ratio,
            "press": KnormPress(compression_ratio=ratio),
            "cache_config": None,
        })
except Exception as exc:
    print(f"[WARN] KnormPress unavailable: {exc}")

# 4. HQQ 4-bit quantised cache -- fixed precision, run once
try:
    hqq_cfg = QuantizedCacheConfig(nbits=4, backend="HQQ")
    sweep.append({
        "config": "HQQ-4bit",
        "compression_ratio": "N/A",
        "press": None,
        "cache_config": hqq_cfg,
    })
except Exception as exc:
    print(f"[WARN] HQQ QuantizedCacheConfig unavailable: {exc}")

# ---------------------------------------------------------------------------
# Run sweep
# ---------------------------------------------------------------------------
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
fieldnames = [
    "config", "compression_ratio", "task", "num_examples",
    "avg_accuracy", "avg_latency_s", "avg_tokens_per_sec",
]

rows: List[Dict[str, Any]] = []
total_runs = len(sweep)

for run_idx, entry in enumerate(sweep):
    cfg_label = entry["config"]
    ratio = entry["compression_ratio"]
    label_str = f"{cfg_label}(ratio={ratio})" if ratio != "N/A" else cfg_label

    print(f"\n[{run_idx+1}/{total_runs}] {label_str}")
    print("-" * 55)

    scores: List[float] = []
    latencies: List[float] = []

    for i, ex in enumerate(lb_examples):
        prompt = _build_prompt(ex)
        refs = _get_refs(ex)
        print(f"  ex {i+1:>3}/{len(lb_examples)}  ", end="", flush=True)

        try:
            pred, latency = generate(
                prompt,
                press_ctx=entry["press"],
                cache_config=entry["cache_config"],
            )
            sc = _score(pred, refs)
            scores.append(sc)
            latencies.append(latency)
            print(f"acc={sc:.3f}  lat={latency:.2f}s")
        except Exception:
            print("FAILED")
            traceback.print_exc()
            scores.append(float("nan"))

    valid_scores = [s for s in scores if not math.isnan(s)]
    valid_lats = [l for l in latencies if l > 0]

    avg_acc = sum(valid_scores) / len(valid_scores) if valid_scores else float("nan")
    avg_lat = sum(valid_lats) / len(valid_lats) if valid_lats else float("nan")
    avg_tps = MAX_NEW_TOKENS / avg_lat if avg_lat > 0 else float("nan")

    row = {
        "config": cfg_label,
        "compression_ratio": ratio,
        "task": LONGBENCH_TASK,
        "num_examples": len(lb_examples),
        "avg_accuracy": round(avg_acc, 4) if not math.isnan(avg_acc) else "NaN",
        "avg_latency_s": round(avg_lat, 4) if not math.isnan(avg_lat) else "NaN",
        "avg_tokens_per_sec": round(avg_tps, 2) if not math.isnan(avg_tps) else "NaN",
    }
    rows.append(row)

    print(
        f"  => avg_acc={avg_acc:.4f}  avg_lat={avg_lat:.2f}s  "
        f"tps={avg_tps:.1f}"
    )

    # Write incrementally so partial results are preserved on crash
    write_header = not RESULTS_FILE.exists() or run_idx == 0
    with open(RESULTS_FILE, "a" if not write_header else "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print(f"\n{'='*65}")
print(f"  Accuracy sweep complete. Results -> {RESULTS_FILE}")
print(f"{'='*65}")
print(f"\n  {'Config':<16} {'Ratio':>6}  {'AvgAcc':>8}  {'AvgLat(s)':>10}  {'Tok/s':>7}")
print(f"  {'-'*16} {'-'*6}  {'-'*8}  {'-'*10}  {'-'*7}")
for r in rows:
    print(
        f"  {r['config']:<16} {str(r['compression_ratio']):>6}  "
        f"{str(r['avg_accuracy']):>8}  {str(r['avg_latency_s']):>10}  "
        f"{str(r['avg_tokens_per_sec']):>7}"
    )
print()
