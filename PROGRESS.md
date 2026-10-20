# PROGRESS.md — KV-Cache Compression Benchmarking

> **Instructions for upkeep:** Update this file after every task completes.
> Only mark something ✅ Completed if there is logged evidence it actually ran.
> Flag unverified claims explicitly with `[uncertain]`.

---

## 1 · Project Status

**One-line summary:** Laptop scaffold complete — code structure valid but NO benchmark numbers are trustworthy yet (see §6 for reasons). Cloud GPU + FPGA work not started.

**Overall completion:** ~20 % (downgraded from 25% after invalidating fake-power results)

**Hardware scope change (PENDING ADVISOR APPROVAL):** The original proposal named a Jetson Orin Nano as the edge inference target. No Jetson is available and none is planned. The actual hardware platform is:
- **Laptop (CPU-only, 7.7 GB RAM):** scaffold, smoke tests, FPGA golden-model generation. Model: Qwen/Qwen2.5-0.5B-Instruct only.
- **Cloud GPU (Kaggle/Colab):** real latency numbers, NVML power, accuracy sweep with 1.5B model.
- **FPGA eviction unit (Arty A7-100T / Artix-7):** hardware KV eviction accelerator.

The Jetson-specific jtop power logger (`power_logger.py`) is retained but is not the primary power path. This scope change must be recorded in the paper Limitations section.

---

## 2 · Completed

Evidence column cites the specific log/output that confirms the task ran.

| # | Task | Date | Files Touched | Verification Evidence |
|---|------|------|---------------|-----------------------|
| 1 | **Project scaffold** | 2026-10-07 | `README.md`, `.gitignore` | git commit `3774cca` |
| 2 | **`requirements.txt`** | 2026-10-07 | `requirements.txt` | task-78/task-80 logs: pip freeze ran |
| 3 | **`check_env.py`** | 2026-10-07 | `src/check_env.py` | task-78.log: torch 2.14.1+cpu, transformers 5.2.0, kvpress 0.5.5 all ok |
| 4 | **`power_logger.py`** (jtop/Jetson) | 2026-10-07 | `src/power_logger.py` | git `f4823d1`; test_real_start_raises_without_jtop PASSED |
| 5 | **`power_logger_fake.py`** | 2026-10-07 | `src/power_logger_fake.py` | task-107.log: 10 interface-parity tests PASSED |
| 6 | **`smoke_test.py`** | 2026-10-07 | `src/smoke_test.py` | git `f4823d1` |
| 7 | **`run_benchmark.py`** code committed | 2026-10-07 | `src/run_benchmark.py` | CSV exists but numbers are **INVALID** — see §6 |
| 8 | **`tests/test_benchmark.py`** | 2026-10-07 | `tests/test_benchmark.py` | 30 passed, 2 skipped in 0.54 s — but tests had weaknesses, see §6 |
| 9 | **`plot_results.py`** | 2026-10-07 | `src/plot_results.py` | figure exists but based on invalid data — must regenerate |
| 10 | **`accuracy_sweep.py`** code committed | 2026-10-07 | `src/accuracy_sweep.py` | Never executed — no CSV |
| 11 | **Related Work draft** | 2026-10-07 | artifact only | Not yet in repo |
| 12 | **Phase 0 PROGRESS.md corrections** | 2026-10-20 | `PROGRESS.md` | This commit — see §6 for root-cause corrections |

---

## 3 · In Progress

Nothing is currently mid-execution.

---

## 4 · Blocked on Me

1. **Cloud GPU run** — All valid latency/power numbers need Kaggle/Colab GPU. Phase 5 runner provides exact command.
2. **Advisor approval** — hardware scope changed from Jetson to cloud GPU + FPGA. Must notify advisor.
3. **Copy Related Work draft into repo** — lives only in Antigravity artifact directory.

---

## 5 · Remaining (Ordered by Phase)

1. Phase 1 — MODEL_NAME CLI flag; model_name/device/dtype columns in CSV
2. Phase 2 — Prompt >= 512 tokens asserted; warm-up; 3-rep mean/std; status/power_source/kv_cache_bytes columns; fix tests
3. Phase 3 — NVML power logger; POWER_BACKEND env var
4. Phase 4 — Real LongBench loading (no trust_remote_code); official scorer; resumable sweep
5. Phase 5 — Kaggle/Colab runner + requirements-gpu.txt
6. Phase 6 — FPGA: cache inspection, KnormPress golden model, .mem test vectors
7. Phase 7 — Limitations + Related Work update
8. Hardware — Run on Kaggle/Colab; download CSV; commit real numbers
9. FPGA — Verilog eviction unit (after Phase 6 verified)
10. Paper — Experiment + results sections, final figures, LaTeX/PDF

---

## 6 · Known Issues and Uncertain Claims

### Phase 0 Root-Cause Corrections

| Issue | Status | Correct Root Cause |
|-------|--------|--------------------|
| **SnapKVPress ERROR** | ❌ Fixed in Phase 2 | Prompt had only ~11 tokens. SnapKVPress requires seq_len > window_size (default 64). **NOT a kvpress API issue.** Fix: assert prompt >= 512 tokens. |
| **LongBench load failure** | ❌ Fixed in Phase 4 | THUDM/LongBench uses a custom loading script. datasets 5.1.0 rejects trust_remote_code=True. **NOT a missing HF_TOKEN.** Fix: load Parquet files directly via huggingface_hub. |
| **Benchmark numbers INVALID** | ❌ Must discard | (a) ~11-token prompt — no real KV compression; (b) no warm-up before timing; (c) power is fake. Do not cite any number from benchmark_results.csv. |
| **Tests passed despite ERROR row** | ❌ Fixed in Phase 2 | Tests only checked column names, not row content. Phase 2 tests fail on any ERROR/NaN row. |
| **HQQ-4bit output degenerate** | ⚠️ Under investigation | Output had repeated tokens ("Provide provide a an example…"). Possible: QuantizedCache API changed in transformers 5.x, or nbits=4 on CPU is unstable. Investigate in Phase 2. |

### Fake Power — MUST EXCLUDE from all results

> ALL energy/power numbers in `benchmark_results.csv` are from `power_logger_fake.py`
> (uniform random 5-15 W). These are FABRICATED. Never put them in a table or figure.
> `plot_results.py` must refuse to plot fake-power energy.

### Uncertain Claims in Related Work Draft

| Claim | Action needed |
|-------|---------------|
| SnapKV clustering algorithm; value norms used? | Read Li et al. 2024 (arXiv 2404.14469) |
| AdaKV venue/year | Find actual citation |
| WKVQuant venue and exact title | Find actual citation |
| HQQ venue (NeurIPS workshop 2023?) | Check Badri & Shaji proceedings |
| MagicPIG: compressed vs. on-the-fly | Read He et al. 2024 (arXiv 2410.16179) |
| LCKV venue (ACL 2024?) | Verify in ACL Anthology |
| CLA exact citation | Find paper |
| Gear exact title and venue | Find paper |
| ZipCache exact citation | Find paper |
| Quest venue (ICML 2024?) | Verify |
| kvpress paper (Jobin et al. arXiv 2024) | Confirm arXiv ID |
| No joint eviction+quant energy study on Jetson | Now moot — hardware changed to cloud GPU + FPGA |
| "Energy rarely reported directly" | Systematic literature search needed |

---

## 7 · Hardware Status

| Component | Status | Notes |
|-----------|--------|-------|
| Laptop (Windows, CPU-only, 7.7 GB RAM) | ✅ Active | Qwen2.5-0.5B-Instruct only (0.5B fits; 1.5B causes swapping) |
| Cloud GPU (Kaggle/Colab) | 🔲 Planned | Qwen2.5-1.5B-Instruct; NVML power; Phase 5 runner |
| FPGA (Arty A7-100T / Artix-7) | 🔲 Planned | KV eviction unit; Phase 6 golden model + test vectors first |
| Jetson Orin Nano | ❌ Removed from scope | Pending advisor approval. jtop logger retained but not primary. |

---

*Last updated: 2026-10-20 · Phase 0 complete.*
