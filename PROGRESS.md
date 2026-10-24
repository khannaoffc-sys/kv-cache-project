# PROGRESS: KV-Cache Compression Benchmarking (BECE309L)

Status as of 2026-10-25. Everything below is tagged by evidence level:
**[measured]** = seen in real command output. **[calculated]** = arithmetic from known model dimensions. **[unverified]** = not checked yet.

## 1. One-paragraph status

The pipeline now runs end to end on a Kaggle Tesla T4 (latency, real GPU power through NVML, LongBench accuracy) for four configs: baseline, SnapKVPress(0.5), KnormPress(0.5) and HQQ-4bit. An FPGA token-eviction unit in Verilog matches a Python golden model bit-exactly in ModelSim (not yet synthesized). No Jetson numbers exist yet; the Orin Nano is available and its setup starts after the presentation. H2O, KIVI, the combined runtime policy and RULER are **not** implemented.

## 2. Latency and power [measured]

Setup: Qwen2.5-1.5B-Instruct, fp16, Tesla T4 (one of two), 553-token prompt, 50 new tokens, 1 untimed warm-up + 3 timed repeats, config order rotated, `POWER_BACKEND=nvml`.

| Config | Latency (s) | Std | Tok/s | Avg power (W) | energy_j (as logged) |
|---|---|---|---|---|---|
| baseline | 2.7748 | 0.0304 | 18.02 | 36.603 | 121.281 |
| SnapKVPress(0.5) | 2.9416 | 0.0215 | 17.00 | 35.570 | 123.292 |
| KnormPress(0.5) | 2.9557 | 0.0194 | 16.92 | 35.409 | 122.037 |
| HQQ-4bit | 3.6310 | 0.0209 | 13.77 | 34.748 | 141.858 |

Relative to baseline: SnapKV +6.0% latency, Knorm +6.5%, HQQ +30.9%. Idle GPU power was about 10 W in an earlier check.

### Known issue: `energy_j` does not equal avg_power x latency

| Config | avg_power x latency (J) | energy_j logged (J) | Implied power window (s) | Latency (s) |
|---|---|---|---|---|
| baseline | 101.6 | 121.3 | 3.31 | 2.77 |
| SnapKVPress(0.5) | 104.6 | 123.3 | 3.47 | 2.94 |
| KnormPress(0.5) | 104.7 | 122.0 | 3.45 | 2.96 |
| HQQ-4bit | 126.2 | 141.9 | 4.08 | 3.63 |

The logged energy corresponds to a power window about **0.45 to 0.54 s longer** than the measured generation time in every config. The cause has not been confirmed. Likely candidates (unverified): the logger's start/stop overhead or the sampling thread covering code outside `model.generate()`. Until this is checked:
- Report both columns and say which is which. `avg_power x latency` is the better estimate of generation-only energy; `energy_j` includes the extra window.
- Do not divide `energy_j` by 50 and call it "energy per token" without this caveat. (Reported: about 2.4 J per token baseline. P x t: about 2.0 J per token.)
- The **ordering** is the same either way: HQQ-4bit costs the most energy (+17% as logged, +24% by P x t); SnapKV and Knorm are within about 1 to 3% of baseline.

### Reading the result
At a 553-token context the KV cache is small (about 16 MB, calculated), so eviction saves little memory traffic while its scoring adds overhead. This is a plausible explanation, **not proven**. Compression is expected to matter at much longer contexts, which have not been tested. HQQ's cost is consistent with per-token dequantization overhead (same caveat).

## 3. Accuracy [measured]

LongBench `narrativeqa`, 30 examples, greedy decoding, 50 new tokens, passage cut to 24,000 characters, input cap 8192 tokens.

| Config | Ratio | Accuracy | Avg latency (s) |
|---|---|---|---|
| baseline | n/a | 0.1556 | 7.54 |
| SnapKVPress | 0.3 | 0.1616 | 8.23 |
| SnapKVPress | 0.5 | 0.1650 | 8.02 |
| SnapKVPress | 0.7 | 0.1620 | 7.89 |
| KnormPress | 0.3 | 0.1765 | 7.65 |
| KnormPress | 0.5 | 0.1897 | 7.45 |
| KnormPress | 0.7 | 0.1383 | 7.41 |
| HQQ-4bit | n/a | 0.1315 | 8.29 |

**Do not claim any method beats baseline.** With 30 examples the noise is roughly plus or minus 0.03 (rough estimate), larger than most differences. The two accuracy rows at the bottom of the latency run (baseline 0.1995, SnapKV 0.2036, power blank, std N/A, 1.3 to 1.6 s) come from a different, small subset and settings. They are **not comparable** to the table above.

Unverified: whether `accuracy_sweep.py` uses the official LongBench scorer in `longbench_metrics.py` or a simple token F1; the prompt template is custom, not the official one; the commit hash cited in that file's header was never checked. An earlier 3-example run (passages cut to 3000 characters) is invalid.

## 4. FPGA eviction unit [measured in ModelSim, not synthesized]

- Model facts (Qwen2.5-0.5B-Instruct): 24 layers, 14 attention heads, 2 KV heads, head_dim 64; key tensor (1, 2, 512, 64).
- `KnormPress` scores each token by the negative L2 norm of its key, so it keeps the **smallest** key norms.
- INT8 golden model vs float32 KnormPress, Jaccard overlap of kept sets over all 24 layers: ratio 0.3 mean 0.9686 (min 0.7900); ratio 0.5 mean 0.9546 (min 0.7099); ratio 0.7 mean 0.9351 (min 0.5976). Layer 0 is the weak spot (0.65 to 0.84) because its scores span only about 2%, so INT8 rounding reshuffles near-ties.
- Verilog: `kv_sumsq_unit.v` (MAC, LANES parameter, 21-bit score), `kv_evict_top.v` (rank-based selection, stable tie-break), `tb_kv_evict.v`.
- ModelSim (2026-10-21): layer 0 PASS, layer 10 PASS, LANES=1 PASS, zero score and index errors. LANES=4: 272,387 cycles (about 2.7 ms at an *assumed* 100 MHz; real Fmax unknown). Commit 49109d6.
- Not done: synthesis (no Fmax/LUT/BRAM/DSP numbers), second KV head, other layers, other ratios, running on a board. The FPGA block does not run the LLM and supports no energy-per-token claim.

## 5. Fixes that exist only in the Kaggle session (port into the repo)

1. `cloud_runner.py`: `SRC = Path(__file__).parent / "src"` points to `src/src`. Fix: `SRC = Path(__file__).parent`.
2. Kaggle's Python 3.13 has no `pipes` module; a shim `pipes.py` containing `from shlex import quote` on `PYTHONPATH` made SnapKV and Knorm load. The package that imports it was not identified.
3. LongBench loading in `accuracy_sweep.py` and `run_benchmark.py` must call `load_longbench_jsonl(...)` from `longbench_metrics` instead of the dataset-script loaders.
4. `accuracy_sweep.py`: `CONTEXT_MAX_CHARS` 3000 to 24000 and `max_length` 2048 to 8192.
5. `run_benchmark.py` defaults to the fake power backend unless `POWER_BACKEND=nvml` is set. Safer: refuse to run silently with fake power, or write `power_source` and blank energy (already done) and print a loud warning.

## 6. Do not claim

- Edge energy per token (Kaggle is a cloud GPU: say "cloud GPU board power"; the Jetson will supply edge numbers).
- Any number whose `power_source` is `fake`, any old laptop CPU timing, or the 3-example accuracy run.
- That compression speeds things up (it did not at this context length) or saves measured memory (`kv_cache_bytes` is still N/A).
- That any method beats baseline on accuracy.
- Any FPGA Fmax, resource or power figure, or that the FPGA block speeds up the Jetson.
- Anything about HQQ output quality (never inspected; an earlier laptop output looked degenerate) or any novelty (no literature check; COMET, EVICPRESS, MixKVQ, CXL-SpecKV are `[uncertain]`).

## 7. Next steps, in order

1. Check why `energy_j` covers a longer window than the latency and fix the logger's measurement window.
2. Port the Kaggle-session fixes (section 5) into the repo and commit. Make `kv_cache_bytes` a real measurement.
3. Jetson Orin Nano: run `cat /proc/device-tree/model`, `cat /etc/nv_tegra_release`, `free -h`, `python3 --version`; then install the JetPack-matched PyTorch, test `power_logger.py` with `POWER_BACKEND=jtop`, run the latency and energy benchmark. Model order: Qwen2.5-1.5B, Llama-3.2-1B, Llama-3.2-3B (only with 16 GB or more), Qwen2.5-0.5B as fallback. A 4 GB board is only safe for 0.5B.
4. Tell the advisor the Jetson is available and the FPGA block is an optional extra.
5. Verify the official scorer and prompt, raise N, add tasks, then RULER; implement H2O, KIVI and the combined policy; test longer contexts where eviction should matter.
6. FPGA: install Quartus Prime Lite with Cyclone V support, get the exact part, synthesize and report Fmax and resources.
7. Literature check on quantized-key eviction; resolve `[uncertain]` citations.
