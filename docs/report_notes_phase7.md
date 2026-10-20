# Report Notes — Phase 7

## 7.1 Limitations Section Draft

The following limitations must be stated clearly in the paper.

---

### L1 — Cloud GPU is not an edge device

All latency and throughput measurements are obtained on a cloud GPU (Kaggle/Colab
with an NVIDIA T4 or A100), not on an edge device such as a mobile SoC, Jetson
board, or embedded MCU. Cloud GPU measurements reflect server-class inference
performance and may not generalise to resource-constrained deployment targets.
The original proposal named a Jetson Orin Nano as the hardware target; this
scope change is pending advisor approval. Results labelled "edge performance"
should be reinterpreted as "upper-bound throughput on a low-cost cloud instance."

### L2 — Energy measurements excluded from CPU experiments

All CPU-side experiments (smoke tests, FPGA golden-model generation, and any
benchmarks run on the development laptop) use `power_logger_fake.py`, which
generates synthetic power samples drawn from Uniform(5, 15) W. These numbers
are **fabricated** and must not appear in any results table or figure. The
`plot_results.py` script is guarded to refuse plotting fake-power energy columns.
Genuine energy measurements are available only from the cloud GPU path using
`POWER_BACKEND=nvml`, and only on bare-metal GPU instances where NVML
`nvmlDeviceGetPowerUsage()` does not return N/A. On virtualised Kaggle T4
instances this call may return N/A; in that case energy columns remain blank.

### L3 — CPU timing excluded from valid results

CPU-side latency numbers from the development laptop (7.7 GB RAM, no GPU) are
**not valid research results** because:

1. The laptop has insufficient RAM for Qwen2.5-1.5B-Instruct (1.5B model causes
   swap); all local inference uses the smaller Qwen2.5-0.5B-Instruct model.
2. CPU inference is dominated by memory-bandwidth saturation, not by the KV-cache
   compression operations under study; compression ratios have little effect on
   CPU wall-clock time in this regime.
3. Prior runs used a ~11-token prompt, which is too short for SnapKVPress to
   exercise its window (window_size=64); those measurements are discarded.

Valid timing results come exclusively from the cloud GPU runner (Phase 5).

### L4 — FPGA area and timing are estimates, not measurements

The FPGA eviction unit (Arty A7-100T / Artix-7) is designed in this project
but resource utilisation and critical-path timing are reported as post-synthesis
estimates from Vivado, not from a deployed and verified bitstream. Until the
design is loaded onto real hardware and the output is verified against the
software golden model (Phase 6), FPGA figures must be labelled "synthesis
estimate" and treated as preliminary.

### L5 — LongBench evaluation uses a small subset

The accuracy sweep evaluates each configuration on N ≤ 30 examples from the
LongBench `narrativeqa` task (or another single task). This is insufficient to
estimate population accuracy reliably. The official LongBench evaluation uses
the full test set across 20 tasks. Our subset evaluation is a diagnostic tool,
not a replacement for full benchmark evaluation.

### L6 — Scoring for narrativeqa uses character-level QA-F1, not human eval

The official LongBench metric for `narrativeqa` is QA-F1 computed at the token
(whitespace-split word) level after normalising articles, punctuation, and
whitespace. This approximates but does not equal human-judged answer correctness.

---

## 7.2 Related Work Draft Update

> **Update policy:** Only add entries when the primary source has been verified.
> Retain all `[uncertain]` tags for unverified claims.

### 2.1 Token Eviction

**SnapKV** [Li et al. 2024, arXiv:2404.14469] identifies important KV positions
by computing attention patterns of the last `window_size` query tokens over the
context, applying average pooling across heads, and permanently evicting low-score
positions after prefill. The implementation uses `window_size=64` by default and
requires `seq_len > window_size` at prefill time. SnapKV does **not** use value
norms — only query-key attention scores. [uncertain: exact cluster algorithm
described in §3 of the paper — to be read in full]

**KnormPress** [Devoto et al. 2024, arXiv:2406.11430] scores each token by
the negative L2 norm of its key vector, computed in at least float32 to avoid
bfloat16 ties. Tokens with the **largest key norms** are evicted; tokens with the
**smallest key norms** are kept. This criterion is read directly from the installed
kvpress 0.5.5 source (`knorm_press.py`, lines 45-46) and verified: no value norms
are used.

**AdaKV** [uncertain: venue/year] [uncertain: citation incomplete]

### 2.2 Quantization

**KIVI** [Liu et al. 2024] stores keys in INT2 and values in INT4 with residual
full-precision components. [uncertain: full citation]

**HQQ** [Badri & Shaji 2023, NeurIPS workshop — workshop name unconfirmed]
uses half-quadratic quantization for weight and activation compression. Applied
to KV caches, it can achieve 4-bit storage with low reconstruction error.
[uncertain: exact venue]

**WKVQuant** [uncertain: full citation]

### 2.3 Attention Approximation

**MagicPIG** [He et al. 2024, arXiv:2410.16179] uses locality-sensitive hashing
to select a subset of cached keys without materialising the full cache. [uncertain:
whether this compresses the cache or only avoids full attention — read §3]

**LCKV** [uncertain: ACL 2024 — verify]

**CLA** [uncertain: full citation]

### 2.4 Hybrid Methods

**ZipCache** [uncertain: full citation — eviction + quantization]

**Quest** [uncertain: ICML 2024 — verify] uses page-level attention.

**Gear** [uncertain: 2024 preprint — find exact citation]

### 2.6 kvpress and Evaluation Infrastructure

The **kvpress** library (NVIDIA, 2024) [uncertain: Jobin et al. arXiv — confirm ID]
provides a unified hook-based API for applying KV eviction methods to any supported
transformers model (Llama, Qwen2, Mistral, Phi3, Gemma3). The compression hook is
applied only during prefill and is removed before decoding. The library includes
at least 40 press implementations as of version 0.5.5.

**LongBench** [Bai et al. 2023, arXiv:2308.14508] is a comprehensive bilingual
long-context benchmark covering 20 tasks including multi-document QA, single-document
QA, summarization, few-shot learning, synthetic tasks, and code completion.
The official evaluation uses task-specific metrics: QA-F1 for extraction tasks,
ROUGE-L for summarization tasks, exact-match variants for classification and
retrieval tasks. Source fetched from THUDM/LongBench commit 2e00731 (2025-01-15).

### 2.6 Research Gap

No concurrent work on joint eviction-plus-quantization energy measurement on
**cloud GPU + FPGA** platforms has been identified in MLSys 2025, ISCA 2025, or
MobiSys 2025 [uncertain: systematic search incomplete]. Earlier claim of a gap
on "Jetson-class hardware" is superseded by the hardware scope change to cloud
GPU + FPGA (pending advisor approval). The energy gap claim must be re-evaluated
against the actual hardware platform.

---

*Last updated: 2026-10-20 · Phase 7 draft — citations marked [uncertain] require primary-source verification.*
