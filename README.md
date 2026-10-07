# KV-Cache Compression Benchmarking for Edge LLM Inference

> A systematic benchmarking framework for evaluating KV-cache compression strategies on resource-constrained edge devices running large language models (LLMs).

---

## Overview

Key-Value (KV) cache is a critical memory bottleneck during autoregressive LLM inference. On edge devices — where DRAM capacity is measured in gigabytes rather than terabytes — the KV cache can account for the majority of memory usage, directly limiting the maximum sequence length and batch size achievable.

This project benchmarks a suite of KV-cache compression techniques to understand their **memory vs. quality vs. latency** trade-offs, with a focus on practical deployment on edge hardware (e.g., Raspberry Pi 5, Jetson Orin Nano, Apple M-series, mobile NPUs).

---

## Research Goals

1. **Quantify memory savings** of each compression method relative to a dense (uncompressed) baseline.
2. **Measure quality degradation** using standard NLP benchmarks (perplexity, ROUGE, downstream task accuracy).
3. **Profile inference latency** end-to-end on representative edge platforms.
4. **Identify Pareto-optimal** compression configurations that maximize quality retention under strict memory budgets.

---

## Compression Techniques Under Investigation

| Technique | Description |
|---|---|
| **Baseline (no compression)** | Full FP16/BF16 KV cache stored in DRAM |
| **KV Quantization** | Reduce KV cache to INT8 / INT4 / NF4 precision |
| **Token Eviction (H2O)** | Evict low-importance tokens based on attention scores |
| **StreamingLLM** | Retain only attention sinks + a sliding local window |
| **SnapKV** | Cluster-based KV compression preserving salient tokens |
| **MagicPIG / LCKV** | Learned / cross-layer KV sharing |
| **Mixed Precision KV** | Per-layer adaptive precision assignment |

---

## Project Structure

```
kv-cache-project/
├── src/                  # Source code for benchmarking scripts and utilities
├── results/              # Raw benchmark outputs (CSV files — git-ignored)
├── figures/              # Generated plots and visualizations (PNG/PDF — git-ignored)
├── .gitignore
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- A virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `requirements.txt` will be added as the project matures.

---

## Benchmarking Methodology

### Models
- Small open-weight models suitable for edge: `TinyLlama-1.1B`, `Phi-3-mini-4k`, `Gemma-2-2B`, `Qwen2.5-1.5B`

### Datasets
- **Perplexity**: WikiText-103
- **Long-context**: LongBench, SCROLLS
- **Downstream tasks**: HellaSwag, ARC-Easy, TriviaQA

### Metrics
| Metric | Tool |
|---|---|
| Memory (peak DRAM) | `psutil`, `torch.cuda.memory_stats()`, platform profilers |
| Latency (TTFT, TPS) | Custom timing harness |
| Perplexity | HuggingFace `evaluate` |
| Task accuracy | `lm-evaluation-harness` |

### Reproducibility
All experiment configurations will be stored as YAML files under `src/configs/`. Results are written to `results/` as CSV files (excluded from git to keep the repo lightweight).

---

## Results

Figures comparing memory-quality trade-offs will be generated in `figures/` after running the benchmark suite. Key plots will include:

- Memory footprint vs. compression ratio
- Perplexity vs. cache budget
- Latency breakdown by compression method
- Pareto frontier: quality vs. memory

---

## Contributing

1. Fork the repo and create a feature branch.
2. Add new compression strategies under `src/methods/`.
3. Ensure new methods implement the common `BaseKVCompressor` interface (TBD).
4. Open a pull request with benchmark results on at least one model.

---

## References

- [StreamingLLM — Xiao et al., 2023](https://arxiv.org/abs/2309.17453)
- [H2O — Zhang et al., 2023](https://arxiv.org/abs/2306.14048)
- [SnapKV — Li et al., 2024](https://arxiv.org/abs/2404.14469)
- [KVQuant — Hooper et al., 2024](https://arxiv.org/abs/2401.18079)
- [MagicPIG — He et al., 2024](https://arxiv.org/abs/2410.16179)

---

## License

MIT License — see `LICENSE` for details.
