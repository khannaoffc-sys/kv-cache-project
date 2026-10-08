# PROGRESS.md — KV-Cache Compression Benchmarking

> **Instructions for upkeep:** Update this file after every task completes.  
> Only mark something ✅ Completed if there is logged evidence it actually ran.  
> Flag unverified claims explicitly with `[uncertain]`.

---

## 1 · Project Status

**One-line summary:** Laptop scaffold complete — 3 of 4 configs produce real numbers on CPU; Jetson hardware work not started.

**Overall completion:** ~25 %

---

## 2 · Completed

Evidence column cites the specific log/output that confirms the task ran.

| # | Task | Date | Files Touched | Verification Evidence |
|---|------|------|---------------|-----------------------|
| 1 | **Project scaffold** — `.gitignore`, `README.md`, `src/`, `results/`, `figures/` directories | 2026-10-07 | `README.md`, `.gitignore` | git commit `3774cca` |
| 2 | **`requirements.txt`** — generated via `pip freeze` in the venv | 2026-10-07 | `requirements.txt` (77 lines, frozen) | task-78/task-80 logs confirm `pip freeze | Out-File requirements.txt` ran successfully |
| 3 | **`check_env.py`** — environment sanity checker | 2026-10-07 | `src/check_env.py` | task-78.log: all three checks passed — torch 2.14.1+cpu ✓, transformers 5.2.0 ✓, kvpress 0.5.5 + KnormPress + SnapKVPress + ObservedAttentionPress all ✓ |
| 4 | **`power_logger.py`** — real jtop-based power logger (Jetson only) | 2026-10-07 | `src/power_logger.py` | git commit `f4823d1`. Interface verified by test suite; `test_real_start_raises_without_jtop` PASSED — confirms RuntimeError raised without jtop. |
| 5 | **`power_logger_fake.py`** — drop-in synthetic logger (laptop-safe) | 2026-10-07 | `src/power_logger_fake.py` | task-107.log: `test_fake_start_stop_summary_roundtrip` PASSED — avg_w in [5,15] W; `test_fake_save_raw` PASSED — two-column CSV produced; all 10 interface-parity tests PASSED |
| 6 | **`smoke_test.py`** — baseline vs. SnapKVPress side-by-side | 2026-10-07 | `src/smoke_test.py` | git commit `f4823d1` (committed alongside run_benchmark as "scaffold validated on laptop CPU") |
| 7 | **`run_benchmark.py`** — 4-config latency + LongBench Phase 2 | 2026-10-07 | `src/run_benchmark.py` | `results/benchmark_results.csv` exists with 5 data rows: baseline lat=695.2 s, KnormPress(0.5) lat=500.6 s, HQQ-4bit lat=526.9 s (CPU, fake power). SnapKVPress row = ERROR. |
| 8 | **`tests/test_benchmark.py`** — pytest suite | 2026-10-07 | `tests/test_benchmark.py` | task-107.log: **30 passed, 2 skipped** in 0.54 s on Python 3.10.11 / pytest 9.1.1 |
| 9 | **`plot_results.py`** — two-panel bar chart (energy/token + TPS) | 2026-10-07 | `src/plot_results.py` | `figures/tradeoff_summary.png` exists on disk (92,969 bytes); committed in `2dbb948` |
| 10 | **`accuracy_sweep.py`** — multi-ratio LongBench sweep script | 2026-10-07 | `src/accuracy_sweep.py` | Code committed in `2dbb948`. **Script has never been executed** — `results/accuracy_sweep.csv` does not exist. |
| 11 | **Related Work draft** | 2026-10-07 | artifact `brain/77591970-…/related_work.md` | Artifact file present (89 lines, 10,591 bytes). Not yet copied into the project repo. |

> **Note on run_benchmark.py:** task-180 log was truncated during model loading and does not show the summary table.
> The existence of `benchmark_results.csv` with numeric results for 3 of 4 configs is the primary evidence.
> SnapKVPress ERROR in that file indicates a runtime failure for that config only.

---

## 3 · In Progress

Nothing is currently mid-execution.

---

## 4 · Blocked on Me

> These are physical or manual steps only you can perform. Do them exactly as stated.

1. **Acquire a Jetson Orin Nano (or RPi 5)** — hardware not purchased. Without it, `power_logger.py` (real jtop path) cannot be validated and the energy-measurement research claim cannot be supported.

2. **Re-run `run_benchmark.py` and save the console output** — SnapKVPress errored and LongBench rows are absent from the current CSV. Run `python src/run_benchmark.py 2>&1 | Tee-Object results/run_benchmark_log.txt` so the summary table is captured.

3. **Run `accuracy_sweep.py`** — never executed. Run `python src/accuracy_sweep.py --num_examples 30` to produce `results/accuracy_sweep.csv`. Expected runtime: several hours on CPU.

4. **Copy Related Work draft into the repo** — draft lives only in the Antigravity artifact directory. Copy to `docs/related_work.md` and `git commit`.

5. **Set `HF_TOKEN`** before the next run — task-180.log shows `"unauthenticated requests to the HF Hub"`. LongBench examples may fail without it.

---

## 5 · Remaining (Ordered)

1. Diagnose and fix SnapKVPress ERROR in `run_benchmark.py` (check kvpress 0.5.5 API vs. transformers 5.2.0)
2. Re-run `run_benchmark.py` with fix; capture clean summary table; commit updated CSV
3. Run `accuracy_sweep.py` end-to-end; commit `results/accuracy_sweep.csv`
4. Validate `plot_results.py` with post-fix CSV (SnapKVPress bar should appear)
5. Copy Related Work draft into repo under `docs/related_work.md`
6. Resolve all `[uncertain]` citations in Related Work (see §6)
7. **Hardware phase — Jetson Orin Nano:**
   a. Flash JetPack 6.x
   b. Install `jetson-stats`; verify `jtop` works
   c. Set `USE_FAKE_POWER = False` in `run_benchmark.py`; re-run benchmark
   d. Record real watt-hour measurements; update CSV and figures
8. Extend benchmark to additional models (Phi-3-mini-4k, Gemma-2-2B, TinyLlama-1.1B)
9. Extend benchmark to additional methods (H2O, StreamingLLM, mixed-precision)
10. Write experiment section and results section of paper
11. Final figure polish; LaTeX/PDF compilation

---

## 6 · Known Issues and Uncertain Claims

### Code Issues

| Issue | Status | Detail |
|-------|--------|--------|
| **SnapKVPress ERROR in benchmark CSV** | ❌ Unresolved | `benchmark_results.csv` row 3: `latency_s=ERROR`. Root cause not diagnosed — likely kvpress 0.5.5 API incompatibility. |
| **LongBench rows absent from CSV** | ❌ Unresolved | Phase 2 in `run_benchmark.py` targets `baseline` and `SnapKVPress(0.5)`. SnapKVPress errored; baseline LB row should exist but is absent — possible run interruption or HF dataset fetch failure (no HF_TOKEN). |
| **`accuracy_sweep.csv` does not exist** | ❌ Not run | `accuracy_sweep.py` has never been executed. |
| **`torch_dtype` deprecation warning** | ⚠️ Minor | task-180.log: `torch_dtype is deprecated, use dtype instead`. Non-breaking. |
| **2 pytest tests permanently skipping** | ℹ️ Expected now | `test_actual_benchmark_csv_if_exists` + `test_actual_sweep_csv_if_exists` skip when CSVs are absent. After both scripts run they should PASS. |

### Uncertain Claims in Related Work Draft

Items marked `[uncertain]` in the draft (§2.6 note) requiring primary-source verification:

| Claim | Draft location | Action needed |
|-------|---------------|---------------|
| SnapKV clustering algorithm; whether value norms are used | §2.1 | Read Li et al. 2024 (arXiv 2404.14469) in full |
| AdaKV venue/year (believed 2024 preprint) | §2.1 | Find actual citation |
| WKVQuant venue and exact title | §2.2 | Find actual citation |
| HQQ venue (NeurIPS workshop 2023 — exact workshop unconfirmed) | §2.2 | Check Badri & Shaji proceedings |
| MagicPIG: compressed cache vs. on-the-fly approximation | §2.3 | Read He et al. 2024 (arXiv 2410.16179) |
| LCKV venue (ACL 2024 — unconfirmed) | §2.3 | Verify in ACL Anthology |
| CLA exact citation | §2.3 | Find paper |
| Gear exact title and venue (believed 2024 preprint) | §2.4 | Find paper |
| ZipCache exact citation | §2.4 | Find paper |
| Quest venue (ICML 2024 — unconfirmed) | §2.4 | Verify |
| kvpress library paper citation (Jobin et al. arXiv 2024) | §2.6 | Confirm arXiv ID |
| No concurrent joint eviction+quantisation energy study on Jetson-class hardware | §2.6 | Search MLSys 2025, ISCA 2025, MobiSys 2025 |
| "Energy consumption is rarely reported directly" | §2.6 | Verify with systematic literature search |

---

## 7 · Hardware Status

| Component | Status | Notes |
|-----------|--------|-------|
| Development laptop (Windows, CPU-only) | ✅ Active | torch 2.14.1+cpu; no CUDA. All benchmarks run here. |
| HuggingFace model cache (Qwen2.5-1.5B-Instruct) | ✅ Downloaded | 338 weight shards loaded during task-180 run; cached locally. |
| Jetson Orin Nano | ❌ Not acquired | Not purchased; no JetPack flash; no jtop; no real power measurements. |
| Raspberry Pi 5 | ❌ Not acquired | Listed as target in README; no work started. |
| Apple M-series Mac | ❌ Not acquired | Listed as target in README; no work started. |

---

*Last updated: 2026-10-08 · Next update due after: SnapKVPress fix, clean benchmark re-run, or hardware acquisition.*
