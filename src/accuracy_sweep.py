"""
accuracy_sweep.py — LongBench accuracy sweep across configs x compression ratios.

Evaluates:
  - Configs      : baseline, SnapKVPress, KnormPress, HQQ-4bit
  - Comp. ratios : 0.3, 0.5, 0.7 (eviction presses); N/A for HQQ-4bit and baseline
  - Examples     : N via --num_examples flag

Output:
  results/accuracy_sweep.csv  (append-mode — resumable; existing rows are skipped)
  Columns: model_name, device, dtype, config, compression_ratio, task,
           num_examples, avg_accuracy, avg_latency_s, avg_tokens_per_sec, status

LongBench loading (Phase 4):
  Does NOT use trust_remote_code. Loads Parquet files directly from HuggingFace Hub.
  Falls back to datasets load_dataset with trust_remote_code as a last resort.

Usage:
    # Laptop (0.5B) — 3 examples as a smoke test:
    python src/accuracy_sweep.py --model Qwen/Qwen2.5-0.5B-Instruct --num_examples 3

    # Cloud GPU (1.5B) — full sweep:
    python src/accuracy_sweep.py --num_examples 30

    # Resume after interruption (already-finished rows are skipped):
    python src/accuracy_sweep.py --num_examples 30
"""

import argparse
import csv
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import QuantizedCache

# ─── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="LongBench accuracy sweep")
parser.add_argument(
    "--model",
    default=os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"),
    dest="model_name",
    help="HuggingFace model ID. Laptop default: Qwen/Qwen2.5-0.5B-Instruct",
)
parser.add_argument(
    "--num_examples", type=int, default=40,
    help="Number of LongBench examples (default: 40)",
)
parser.add_argument(
    "--task", type=str, default="narrativeqa",
    help="LongBench task name (default: narrativeqa)",
)
args = parser.parse_args()

MODEL_NAME: str = args.model_name
NUM_EXAMPLES: int = max(1, args.num_examples)
LONGBENCH_TASK: str = args.task
MAX_NEW_TOKENS: int = 50
CONTEXT_MAX_CHARS: int = 3000
COMPRESSION_RATIOS: List[float] = [0.3, 0.5, 0.7]

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_FILE = RESULTS_DIR / "accuracy_sweep.csv"

# ─── Device / dtype ───────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"\n{'='*65}")
print(f"  Accuracy Sweep")
print(f"  Model: {MODEL_NAME}")
print(f"  Device: {DEVICE.upper()}  |  dtype: {DTYPE}  |  Examples: {NUM_EXAMPLES}")
print(f"  Task: {LONGBENCH_TASK}  |  Ratios: {COMPRESSION_RATIOS}")
print(f"{'='*65}\n")

# ─── Load model + tokeniser ───────────────────────────────────────────────────
print("Loading tokeniser ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=DTYPE,
    device_map="auto" if DEVICE == "cuda" else None,
)
if DEVICE == "cpu":
    model = model.to(DEVICE)
model.eval()
print("Model ready.\n")


# ─── LongBench loading — Phase 4: no trust_remote_code ───────────────────────

def _load_longbench_parquet(task: str, n: int) -> Optional[List[Dict[str, Any]]]:
    """Load LongBench examples directly from Parquet files, bypassing the
    custom loading script that datasets 5.1.0 rejects.

    Uses huggingface_hub.hf_hub_download to fetch the Parquet file for the
    given task, then reads it with pandas or pyarrow.

    Returns a list of example dicts, or None if loading fails.
    """
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import]
        import pyarrow.parquet as pq  # type: ignore[import]

        # THUDM/LongBench stores test splits as data/<task>/test-*.parquet
        # List repo files to find the right path.
        from huggingface_hub import list_repo_files  # type: ignore[import]
        repo_files = list(list_repo_files("THUDM/LongBench", repo_type="dataset"))
        # Find parquet file(s) for this task
        parquet_files = [
            f for f in repo_files
            if f.endswith(".parquet") and task in f and "test" in f
        ]
        if not parquet_files:
            print(f"  [WARN] No Parquet test file found for task={task!r}")
            return None

        examples = []
        for pf in parquet_files:
            local = hf_hub_download(
                repo_id="THUDM/LongBench",
                filename=pf,
                repo_type="dataset",
            )
            table = pq.read_table(local)
            for row in table.to_pylist():
                examples.append(row)
                if len(examples) >= n:
                    break
            if len(examples) >= n:
                break

        print(f"  Loaded {len(examples)} examples from Parquet (no trust_remote_code).")
        return examples[:n]

    except Exception as exc:
        print(f"  [WARN] Parquet loading failed: {exc}")
        return None


print(f"Loading LongBench/{LONGBENCH_TASK} ({NUM_EXAMPLES} examples) ...")

lb_examples: List[Dict[str, Any]] = []

# Try 1: Parquet without trust_remote_code
result = _load_longbench_parquet(LONGBENCH_TASK, NUM_EXAMPLES)
if result:
    lb_examples = result
else:
    # Try 2: datasets with trust_remote_code (may fail on datasets >= 5.1.0)
    try:
        from datasets import load_dataset
        print("  Attempting datasets load_dataset with trust_remote_code=True ...")
        lb_dataset = load_dataset(
            "THUDM/LongBench",
            LONGBENCH_TASK,
            split="test",
            trust_remote_code=True,
        )
        n = min(NUM_EXAMPLES, len(lb_dataset))
        lb_examples = list(lb_dataset.select(range(n)))
        print(f"  Loaded {len(lb_examples)} examples via datasets.\n")
    except Exception as exc:
        print(f"[FATAL] Could not load LongBench by any method: {exc}")
        sys.exit(1)


# ─── Helpers ──────────────────────────────────────────────────────────────────

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
        from longbench_metrics import score as lb_score  # type: ignore[import]
        return max(lb_score(pred, ref, LONGBENCH_TASK) for ref in refs)
    except ImportError:
        pass
    return max(_token_f1(pred, ref) for ref in refs)


def generate(prompt: str, press_ctx=None, cache_factory=None):
    """Return (text, latency_s)."""
    enc = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(DEVICE)
    gen_kwargs: Dict[str, Any] = dict(
        **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
    )
    if cache_factory is not None:
        gen_kwargs["past_key_values"] = cache_factory()

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


# ─── Build sweep plan ─────────────────────────────────────────────────────────
sweep: List[Dict[str, Any]] = []

sweep.append({
    "config": "baseline",
    "compression_ratio": "N/A",
    "press": None,
    "cache_factory": None,
})

try:
    from kvpress import SnapKVPress
    for ratio in COMPRESSION_RATIOS:
        sweep.append({
            "config": "SnapKVPress",
            "compression_ratio": ratio,
            "press": SnapKVPress(compression_ratio=ratio),
            "cache_factory": None,
        })
except Exception as exc:
    print(f"[WARN] SnapKVPress unavailable: {exc}")

try:
    from kvpress import KnormPress
    for ratio in COMPRESSION_RATIOS:
        sweep.append({
            "config": "KnormPress",
            "compression_ratio": ratio,
            "press": KnormPress(compression_ratio=ratio),
            "cache_factory": None,
        })
except Exception as exc:
    print(f"[WARN] KnormPress unavailable: {exc}")

try:
    _ = QuantizedCache
    sweep.append({
        "config": "HQQ-4bit",
        "compression_ratio": "N/A",
        "press": None,
        "cache_factory": lambda: QuantizedCache(
            backend="hqq",
            config=model.config,
            nbits=4,
        ),
    })
except Exception as exc:
    print(f"[WARN] HQQ QuantizedCache unavailable: {exc}")


# ─── Load already-finished rows (resumability) ─────────────────────────────────

fieldnames = [
    "model_name", "device", "dtype",
    "config", "compression_ratio", "task", "num_examples",
    "avg_accuracy", "avg_latency_s", "avg_tokens_per_sec", "status",
]

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

finished_keys: set = set()
if RESULTS_FILE.exists():
    with open(RESULTS_FILE, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = (row.get("model_name", ""), row.get("config", ""),
                   row.get("compression_ratio", ""), row.get("task", ""))
            finished_keys.add(key)
    print(f"  Resuming: {len(finished_keys)} rows already in CSV — will skip.\n")


# ─── Run sweep ────────────────────────────────────────────────────────────────

total_runs = len(sweep)

for run_idx, entry in enumerate(sweep):
    cfg_label = entry["config"]
    ratio = entry["compression_ratio"]
    label_str = f"{cfg_label}(ratio={ratio})" if ratio != "N/A" else cfg_label

    key = (MODEL_NAME, cfg_label, str(ratio), LONGBENCH_TASK)
    if key in finished_keys:
        print(f"[{run_idx+1}/{total_runs}] {label_str} — SKIPPED (already in CSV)")
        continue

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
                cache_factory=entry.get("cache_factory"),
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
    status = "ok" if valid_scores else "ERROR"

    row = {
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "config": cfg_label,
        "compression_ratio": ratio,
        "task": LONGBENCH_TASK,
        "num_examples": len(lb_examples),
        "avg_accuracy": round(avg_acc, 4) if not math.isnan(avg_acc) else "NaN",
        "avg_latency_s": round(avg_lat, 4) if not math.isnan(avg_lat) else "NaN",
        "avg_tokens_per_sec": round(avg_tps, 2) if not math.isnan(avg_tps) else "NaN",
        "status": status,
    }

    print(
        f"  => avg_acc={avg_acc:.4f}  avg_lat={avg_lat:.2f}s  "
        f"tps={avg_tps:.1f}  status={status}"
    )

    # Append incrementally — crash-safe resume
    write_header = not RESULTS_FILE.exists()
    with open(RESULTS_FILE, "a" if not write_header else "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  Accuracy sweep complete.  Results -> {RESULTS_FILE}")
print(f"{'='*65}\n")
