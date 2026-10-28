# Kaggle T4 results (saved copy)

These are the real result files from the Kaggle runs, copied out as text with `cat` and saved here because `results/*.csv` is git-ignored and the Kaggle session can reset. Saved 2026-10-25.

**Setup:** Qwen/Qwen2.5-1.5B-Instruct, float16, one Tesla T4, transformers 5.2.0, kvpress 0.5.5. Latency run used `POWER_BACKEND=nvml`.

## Files

| File | What it is |
|---|---|
| `kaggle_t4_benchmark_nvml.csv` | Latency and GPU power, 4 configs, 553-token prompt, 50 new tokens, 1 untimed warm-up + 3 timed repeats (rotated order). Last two rows are a small LongBench check (see caveats) |
| `kaggle_t4_accuracy_sweep_narrativeqa_30ex.csv` | LongBench narrativeqa accuracy, 30 examples, baseline / SnapKVPress / KnormPress at ratios 0.3, 0.5, 0.7 / HQQ-4bit |

## Caveats (read before quoting any number)

1. **`energy_j` is not `avg_power_w x latency_s`.** Baseline: 36.603 W x 2.7748 s = 101.6 J, but the log says 121.281 J. The logged energy corresponds to a power window about 0.45 to 0.54 s longer than the generation time in every config. Cause not yet confirmed. Quote both, or use power x latency as the generation-only estimate. The ranking of configs is the same either way.
2. **Cloud GPU power, not edge power.** This is a Kaggle T4. Do not call it edge energy.
3. **`kv_cache_bytes` is N/A.** KV-cache size was never measured.
4. **The two `longbench_narrativeqa` rows in the benchmark CSV use only 5 examples** (see `output_preview`), have no power values, and are **not comparable** to the 30-example sweep (baseline 0.1995 vs 0.1556). Use the sweep file for accuracy.
5. **Accuracy differences are inside the noise** (30 examples, one task, roughly plus or minus 0.03, rough estimate). Do not claim any method beats the baseline.
6. **Custom prompt and settings** (50 new tokens, passage cut to 24,000 characters, 8,192-token input cap). Not comparable with published LongBench numbers. Whether the official scorer was used is unverified.
7. **HQQ-4bit output quality was never inspected.**


> **Correction:** the energy_j column in these CSVs is overstated. The NVML logger window included a ~0.5 s idle-baseline phase (fixed in commit 07feb8a). Use avg_power_w x latency_s until the benchmark is rerun with the patched logger.
