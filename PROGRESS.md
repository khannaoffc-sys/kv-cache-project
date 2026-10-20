# PROGRESS.md — KV-Cache Compression Benchmarking

> **Instructions for upkeep:** Update this file after every task completes.
> Only mark something ✅ Completed if there is logged evidence it actually ran.
> Flag unverified claims explicitly with `[uncertain]`.

---

## 1 · Project Status

**One-line summary:** Phases 0-7 scaffold complete. All code committed. No valid benchmark numbers yet — those require the cloud GPU runner (Phase 5).

**Overall completion:** ~45 %

**Hardware scope change (PENDING ADVISOR APPROVAL):**
- **Laptop (CPU-only, 7.7 GB RAM):** scaffold, smoke tests, FPGA golden model. Model: Qwen/Qwen2.5-0.5B-Instruct only.
- **Cloud GPU (Kaggle/Colab):** real latency, NVML power, accuracy sweep with 1.5B model. Model: Qwen/Qwen2.5-1.5B-Instruct.
- **FPGA (Arty A7-100T / Artix-7):** KV eviction unit. Phase 6 software verified first, Verilog second.

---

## 2 · Completed

| # | Task | Date | Evidence |
|---|------|------|----------|
| 1 | Project scaffold | 2026-10-07 | git `3774cca` |
| 2 | `requirements.txt` | 2026-10-07 | git `2dbb948` |
| 3 | `check_env.py` | 2026-10-07 | task-78.log: torch 2.14.1+cpu, kvpress 0.5.5 all ok |
| 4 | `power_logger.py` (jtop/Jetson) | 2026-10-07 | git `f4823d1`; test_real_start_raises_without_jtop PASSED |
| 5 | `power_logger_fake.py` | 2026-10-07 | task-107.log: 10 interface-parity tests PASSED |
| 6 | `smoke_test.py` | 2026-10-07 | git `f4823d1` |
| 7-10 | run_benchmark, accuracy_sweep, plot_results, tests (initial) | 2026-10-07 | git `2dbb948` — numbers invalid, see §6 |
| 11 | Related Work draft | 2026-10-07 | Antigravity artifact only — not yet in repo |
| **12** | **Phase 0: PROGRESS.md root-cause corrections** | **2026-10-20** | **git `26883c3`** |
| **13** | **Phase 1: MODEL_NAME CLI/env flag** | **2026-10-20** | **git `9516047`** — run_benchmark.py, smoke_test.py, accuracy_sweep.py all updated; `--model` flag; dtype auto-selection (float16/CUDA, float32/CPU); `torch_dtype` deprecation fixed |
| **14** | **Phase 2: Valid benchmark** | **2026-10-20** | **git `9516047`** — LATENCY_PROMPT ≥512 tokens asserted; 1 warm-up + 3 timed reps with rotated order; mean±std; new CSV columns: model_name, device, dtype, status, power_source, prompt_tokens, new_tokens, kv_cache_bytes, latency_std_s; fake-power energy blank; plot_results.py guards fake energy |
| **15** | **Phase 2: Strengthened tests** | **2026-10-20** | **46 passed, 2 skipped** in 0.65 s (pytest-9.1.1, Python 3.10.11). Tests now fail on ERROR rows, NaN rows, fake-power energy, wrong columns, and invalid compression_ratio. LongBench metrics unit tests added. |
| **16** | **Phase 3: NVML power logger** | **2026-10-20** | **git `9516047`** — `src/power_logger_nvml.py`: deferred pynvml import, idle baseline, clear N/A warning; same interface as fake/jtop; `POWER_BACKEND` env var (fake|nvml|jtop); import-guard tested in `test_nvml_start_raises_without_pynvml` PASSED |
| **17** | **Phase 4: Real LongBench loading** | **2026-10-20** | **git `9516047`** — `longbench_metrics.py`: official scorers from THUDM/LongBench commit `2e00731` (2025-01-15), source URL in header; `load_longbench_jsonl()` reads `data.zip` without `trust_remote_code`; accuracy_sweep.py uses Parquet/zip path + resumable CSV |
| **18** | **Phase 4: LongBench repo inspection** | **2026-10-20** | Real output from `list_repo_files('THUDM/LongBench', repo_type='dataset')`: `.gitattributes`, `LongBench.py`, `README.md`, `data.zip`. Data is in `data.zip` → `data/<task>.jsonl`. No Parquet files. Loading script confirmed to download and extract `data.zip`. |
| **19** | **Phase 5: Cloud runner** | **2026-10-20** | **git `9516047`** — `src/cloud_runner.py`: auto-detects GPU, reads HF_TOKEN from Kaggle Secrets/Colab userdata, checks NVML power with N/A warning, runs benchmark + 3-example sweep; `requirements-gpu.txt`: no Windows pins |
| **20** | **Phase 6: FPGA golden model (software side — config confirmed)** | **2026-10-20** | **git `9516047`** — `src/fpga_golden_model.py`: inspects cache object, documents KnormPress scoring, INT8 golden model, Jaccard overlap, .mem hex files. **Model config confirmed** (real command output — see §6.2): layers=24, heads=14, kv_heads=2, hidden=896, head_dim=64. Forward-pass shapes and Jaccard overlap numbers must be run on cloud GPU (CPU load >5 min). |
| **21** | **Phase 7: Limitations + Related Work** | **2026-10-20** | **git `9516047`** — `docs/report_notes_phase7.md`: Limitations L1-L6; Related Work update with verified KnormPress scoring direction; [uncertain] tags retained |

---

## 3 · In Progress

Nothing is currently mid-execution.

---

## 4 · Blocked on Me

1. **Cloud GPU run** — Run `src/cloud_runner.py` on Kaggle or Colab to get real latency/power/accuracy.

   ```
   # Kaggle: turn on GPU + Internet, add HF_TOKEN secret, then in a notebook cell:
   !git clone https://github.com/<YOUR_REPO>/kv-cache-project.git
   %cd kv-cache-project
   !pip install torch --index-url https://download.pytorch.org/whl/cu121 -q
   !pip install -r requirements-gpu.txt -q
   !python src/cloud_runner.py
   ```

2. **Phase 6 shapes + overlap** — Run `fpga_golden_model.py` on cloud GPU (CPU load >5 min):

   ```
   python src/fpga_golden_model.py
   ```

   Report the printed shapes and Jaccard numbers back here. **Do not write Verilog until those numbers are confirmed.**

3. **Phase 4 accuracy sweep smoke test (3 examples)** — Run locally with 0.5B model:

   ```
   python src/accuracy_sweep.py --model Qwen/Qwen2.5-0.5B-Instruct --num_examples 3
   ```

   Report the actual output. Do not launch the full sweep locally.

4. **Advisor approval** — hardware scope change from Jetson to cloud GPU + FPGA.

5. **Copy Related Work draft** — `docs/report_notes_phase7.md` has the draft. Copy expanded version to `docs/related_work.md` and commit.

---

## 5 · Remaining (Ordered)

1. Cloud GPU run → real benchmark_results.csv (valid numbers)
2. Cloud GPU run → real accuracy_sweep.csv (narrativeqa, N=30)
3. Phase 6 on cloud GPU → shapes + Jaccard → report → write Verilog
4. Phase 4 smoke test locally (3 examples, 0.5B)
5. Full accuracy sweep on cloud GPU (N=30)
6. Copy Related Work draft to docs/related_work.md
7. Resolve all [uncertain] citations in Related Work
8. FPGA Verilog eviction unit (after Phase 6 numbers confirmed)
9. Validate FPGA output against golden model .mem files
10. Paper: experiment section, results section, figures, LaTeX/PDF

---

## 6 · Known Issues and Evidence

### 6.1 Phase 0 Root Causes (confirmed)

| Issue | Root Cause (confirmed) | Evidence |
|-------|------------------------|----------|
| SnapKVPress ERROR | 11-token prompt < window_size=64. NOT a kvpress API issue. | `SnapKVPress(compression_ratio=0.5).window_size = 64` — real command output (task-25) |
| LongBench failure | datasets 5.1.0 rejects trust_remote_code. NOT a missing HF_TOKEN. | LongBench.py confirmed via `hf_hub_download` — loading script exists; datasets 5.1.0 security policy |
| Benchmark numbers invalid | Short prompt (~11 tok), no warm-up, fake power | benchmark_results.csv deleted |
| Tests passed despite ERROR rows | Tests only checked column names, not row content | Fixed in Phase 2 — now 46 pass |
| HQQ-4bit degenerate output | Under investigation — repeated tokens suggest API mismatch | Phase 2 adds status column; will be visible after next run |

### 6.2 Phase 6 — Confirmed model config (real output)

```
Command: .\venv\Scripts\python.exe -c "from transformers import AutoConfig; ..."
Output:  layers: 24  heads: 14  kv_heads: 2  hidden: 896
```

**Derived facts (not yet confirmed from forward pass — pending cloud run):**
- head_dim = 896 / 14 = 64
- Key tensor shape (expected): `(1, 2, seq_len, 64)` — (batch, num_kv_heads, seq_len, head_dim)
- Layer 0 and layer 10 shapes will be identical for this model (uniform architecture)

**KnormPress scoring direction (confirmed from installed source):**
```python
# src: venv/lib/site-packages/kvpress/presses/knorm_press.py lines 45-46
score_dtype = torch.promote_types(keys.dtype, torch.float32)
return -torch.linalg.vector_norm(keys, dim=-1, dtype=score_dtype)
```
- Score = **negative L2 norm** per token, in float32+
- `topk(n_keep)` on negative scores = keep tokens with **smallest** key L2 norms
- **EVICTS** tokens with **large** key norms; **KEEPS** tokens with **small** key norms
- Does NOT use value norms

**Jaccard overlap (INT8 golden vs KnormPress float32):** PENDING cloud GPU run.

### 6.3 Fake Power

ALL energy/power in old `benchmark_results.csv` were fabricated (uniform 5-15 W random).
That CSV is deleted. New CSV has blank energy_j when power_source=fake.

---

## 7 · Hardware Status

| Component | Status | Notes |
|-----------|--------|-------|
| Laptop (Windows, CPU-only, 7.7 GB RAM) | ✅ Active | Qwen2.5-0.5B-Instruct only; 0.5B model config confirmed |
| Cloud GPU (Kaggle/Colab) | 🔲 Ready — needs run | cloud_runner.py committed; Kaggle setup: GPU+Internet on, HF_TOKEN secret |
| FPGA (Arty A7-100T / Artix-7) | 🔲 Software side done | fpga_golden_model.py committed; awaiting cloud run for shapes+overlap before Verilog |
| Jetson Orin Nano | ❌ Removed from scope | Pending advisor approval. jtop logger retained. |

---

## 8 · Test Suite Status

```
pytest tests/test_benchmark.py -v
46 passed, 2 skipped in 0.65 s
Python 3.10.11 / pytest-9.1.1
Date: 2026-10-20
```

Skipped: test_actual_benchmark_csv_if_exists, test_actual_sweep_csv_if_exists
(Both skip when CSVs absent — correct behaviour. Will PASS after cloud run.)

---

*Last updated: 2026-10-20 · Phases 0-7 scaffold complete. Awaiting cloud GPU run.*
