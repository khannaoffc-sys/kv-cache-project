#!/usr/bin/env python3
"""
cloud_runner.py — Kaggle / Colab runner for KV-cache compression benchmarks.

Usage (Kaggle notebook cell or Colab cell):
    !git clone https://github.com/<YOUR_REPO>/kv-cache-project.git
    %cd kv-cache-project
    !pip install torch --index-url https://download.pytorch.org/whl/cu121 -q
    !pip install -r requirements-gpu.txt -q
    !python src/cloud_runner.py

Kaggle setup notes:
  1. Settings → Accelerator → GPU T4 x2 (or P100)
  2. Settings → Internet → On
  3. Add your Hugging Face token as a Kaggle Secret named HF_TOKEN
     (Settings → Secrets → Add New Secret)
  4. The script reads the secret automatically via kaggle_secrets.

Colab setup notes:
  1. Runtime → Change runtime type → GPU
  2. Set HF_TOKEN: from google.colab import userdata; os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
  3. Mount Drive if you want persistent output: from google.colab import drive; drive.mount('/content/drive')

Output:
  results/benchmark_results.csv    — latency / power per config
  results/accuracy_sweep.csv       — LongBench accuracy
  results/power_nvml_<config>.csv  — raw NVML power samples (if available)
"""

import os
import platform
import subprocess
import sys
from pathlib import Path

# ─── 1. HuggingFace token ─────────────────────────────────────────────────────
def _setup_hf_token():
    if os.environ.get("HF_TOKEN"):
        print("[setup] HF_TOKEN already set.")
        return

    # Kaggle Secrets
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore[import]
        secrets = UserSecretsClient()
        token = secrets.get_secret("HF_TOKEN")
        os.environ["HF_TOKEN"] = token
        print("[setup] HF_TOKEN loaded from Kaggle Secrets.")
        return
    except Exception:
        pass

    # Colab userdata
    try:
        from google.colab import userdata  # type: ignore[import]
        token = userdata.get("HF_TOKEN")
        os.environ["HF_TOKEN"] = token
        print("[setup] HF_TOKEN loaded from Colab userdata.")
        return
    except Exception:
        pass

    print("[WARN] HF_TOKEN not found. LongBench download may be rate-limited.")


_setup_hf_token()


# ─── 2. Detect GPU and check NVML power ───────────────────────────────────────
import torch

def _detect_gpu():
    if not torch.cuda.is_available():
        print("[ERROR] No CUDA GPU detected. Set Accelerator = GPU in Kaggle/Colab settings.")
        sys.exit(1)
    gpu_name = torch.cuda.get_device_name(0)
    total_mb = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
    print(f"[setup] GPU: {gpu_name}  |  VRAM: {total_mb} MB")
    return gpu_name

gpu_name = _detect_gpu()


def _check_nvml_power():
    """Try to read GPU power via NVML and print a clear result."""
    try:
        import pynvml  # type: ignore[import]
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        try:
            pw_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            print(f"[setup] NVML power OK: current GPU power = {pw_mw / 1000:.1f} W")
            pynvml.nvmlShutdown()
            return "nvml"
        except pynvml.NVMLError as exc:
            print(
                f"[WARN] NVML power unavailable on this GPU ({exc}). "
                "This is expected on virtualised Kaggle/Colab T4 instances. "
                "Energy columns will be blank. Use POWER_BACKEND=fake or a bare-metal GPU."
            )
            pynvml.nvmlShutdown()
            return "fake"
    except ImportError:
        print("[WARN] pynvml not installed. Run: pip install nvidia-ml-py")
        return "fake"
    except Exception as exc:
        print(f"[WARN] NVML init failed: {exc}")
        return "fake"

power_backend = _check_nvml_power()
os.environ["POWER_BACKEND"] = power_backend
print(f"[setup] POWER_BACKEND = {power_backend}")


# ─── 3. Model selection ───────────────────────────────────────────────────────
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
os.environ["MODEL_NAME"] = MODEL_NAME
print(f"[setup] MODEL = {MODEL_NAME}")


# ─── 4. Output directory ──────────────────────────────────────────────────────
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "results"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[setup] Output dir: {OUTPUT_DIR.resolve()}")


# ─── 5. Run benchmark ─────────────────────────────────────────────────────────
def _run(cmd, **kwargs):
    print(f"\n{'='*60}")
    print(f"  Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(
        cmd,
        capture_output=False,   # stream to stdout
        text=True,
        **kwargs,
    )
    if result.returncode != 0:
        print(f"[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

SRC = Path(__file__).parent / "src"

# Latency + power benchmark
_run([
    sys.executable, str(SRC / "run_benchmark.py"),
    "--model", MODEL_NAME,
])

# Phase 4 accuracy sweep — 3 examples smoke test first
print("\n[Phase 4] Accuracy sweep — 3 examples smoke test first:")
_run([
    sys.executable, str(SRC / "accuracy_sweep.py"),
    "--model", MODEL_NAME,
    "--num_examples", "3",
    "--task", "narrativeqa",
])
print("\nSmoke test passed. Run the full sweep separately:")
print(f"  python src/accuracy_sweep.py --model {MODEL_NAME} --num_examples 30")


# ─── 6. Summary ───────────────────────────────────────────────────────────────
import csv

print(f"\n{'═'*60}")
print("  RESULTS SUMMARY")
print(f"{'═'*60}")

bench_csv = OUTPUT_DIR / "benchmark_results.csv"
if bench_csv.exists():
    print(f"\n  {bench_csv}:")
    with open(bench_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            lat = row.get("latency_s", "?")
            tps = row.get("tokens_per_sec", "?")
            kv  = row.get("kv_cache_bytes", "?")
            src = row.get("power_source", "?")
            eng = row.get("energy_j", "")
            st  = row.get("status", "?")
            eng_str = f"  energy={eng}J" if eng else f"  energy=N/A (power_source={src})"
            print(f"  {row['config']:<24} lat={lat}s  tps={tps}  kv={kv}B  {eng_str}  status={st}")

sweep_csv = OUTPUT_DIR / "accuracy_sweep.csv"
if sweep_csv.exists():
    print(f"\n  {sweep_csv}:")
    with open(sweep_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            print(
                f"  {row['config']:<16} ratio={row['compression_ratio']:<5} "
                f"acc={row['avg_accuracy']}  lat={row['avg_latency_s']}s  status={row['status']}"
            )

print(f"\n  Files in {OUTPUT_DIR}/:")
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"    {f.name}  ({f.stat().st_size} bytes)")
print()
