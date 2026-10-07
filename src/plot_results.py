"""
plot_results.py -- Visualise benchmark results from results/benchmark_results.csv.

Produces a two-panel figure:
  Panel 1 (left)  -- Energy per token (J/tok) by config  [bar chart]
  Panel 2 (right) -- Tokens per second by config          [bar chart]

Saved to figures/tradeoff_summary.png at 150 dpi.

Usage
-----
    python src/plot_results.py

Requirements
------------
    pandas, matplotlib  (both installed in the project venv)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
CSV_PATH = ROOT / "results" / "benchmark_results.csv"
FIG_DIR = ROOT / "figures"
FIG_PATH = FIG_DIR / "tradeoff_summary.png"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
if not CSV_PATH.exists():
    print(f"[ERROR] Results file not found: {CSV_PATH}")
    print("        Run src/run_benchmark.py first to generate it.")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)

# Coerce numeric columns -- non-numeric values ("N/A", "ERROR") become NaN
for col in ("latency_s", "tokens_per_sec", "avg_power_w", "energy_j", "accuracy"):
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------------------------------------------------------------------------
# Derive energy-per-token where we have power data; otherwise estimate from
# fake-power run or leave NaN so the bar is just absent.
# ---------------------------------------------------------------------------
# energy_per_token = energy_j / tokens_generated
# If energy_j is NaN but avg_power_w and latency_s are both present, compute it.
mask_missing_energy = df["energy_j"].isna() & df["avg_power_w"].notna() & df["latency_s"].notna()
df.loc[mask_missing_energy, "energy_j"] = (
    df.loc[mask_missing_energy, "avg_power_w"]
    * df.loc[mask_missing_energy, "latency_s"]
)

MAX_NEW_TOKENS = 50  # must match run_benchmark.py
df["energy_per_token_j"] = df["energy_j"] / MAX_NEW_TOKENS

# ---------------------------------------------------------------------------
# Aggregate: one row per config (mean across prompt types / runs)
# For latency benchmarks we use the "latency_fixed" rows only to avoid mixing
# in LongBench rows which have different context lengths.
# ---------------------------------------------------------------------------
latency_df = df[df["prompt_type"] == "latency_fixed"].copy()

# If no latency_fixed rows exist (e.g. CSV only has LongBench rows), fall back
if latency_df.empty:
    latency_df = df.copy()

agg = (
    latency_df
    .groupby("config")[["tokens_per_sec", "energy_per_token_j"]]
    .mean()
    .reset_index()
)

if agg.empty:
    print("[ERROR] No plottable data found in CSV. Check that run_benchmark.py ran successfully.")
    sys.exit(1)

# Sort so baseline always appears first
order = ["baseline"] + [c for c in agg["config"] if c != "baseline"]
agg["config"] = pd.Categorical(agg["config"], categories=order, ordered=True)
agg = agg.sort_values("config").reset_index(drop=True)

configs = agg["config"].tolist()
x = range(len(configs))

# ---------------------------------------------------------------------------
# Color palette -- muted but distinct
# ---------------------------------------------------------------------------
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
colors = [PALETTE[i % len(PALETTE)] for i in range(len(configs))]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, (ax_energy, ax_tps) = plt.subplots(
    1, 2,
    figsize=(12, 5),
    constrained_layout=True,
)
fig.suptitle(
    "KV-Cache Compression: Efficiency Trade-offs\n"
    f"Model: Qwen2.5-1.5B-Instruct  |  {MAX_NEW_TOKENS} new tokens",
    fontsize=13,
    fontweight="bold",
)

# -- Panel 1: Energy per token -----------------------------------------------
energy_vals = agg["energy_per_token_j"].tolist()
bars1 = ax_energy.bar(x, energy_vals, color=colors, width=0.55, edgecolor="white", linewidth=0.8)

ax_energy.set_title("Energy per Token", fontsize=12, fontweight="semibold", pad=8)
ax_energy.set_xlabel("Compression Config", fontsize=10)
ax_energy.set_ylabel("Energy (J / token)", fontsize=10)
ax_energy.set_xticks(list(x))
ax_energy.set_xticklabels(configs, rotation=20, ha="right", fontsize=9)
ax_energy.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
ax_energy.set_ylim(bottom=0)
ax_energy.spines[["top", "right"]].set_visible(False)
ax_energy.grid(axis="y", linestyle="--", alpha=0.4)

# Annotate bars
for bar, val in zip(bars1, energy_vals):
    if pd.notna(val):
        ax_energy.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(energy_vals, default=1) * 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=8,
        )
    else:
        ax_energy.text(
            bar.get_x() + bar.get_width() / 2,
            0.002,
            "N/A",
            ha="center", va="bottom", fontsize=8, color="gray",
        )

# -- Panel 2: Tokens per second -----------------------------------------------
tps_vals = agg["tokens_per_sec"].tolist()
bars2 = ax_tps.bar(x, tps_vals, color=colors, width=0.55, edgecolor="white", linewidth=0.8)

ax_tps.set_title("Throughput (Tokens / Second)", fontsize=12, fontweight="semibold", pad=8)
ax_tps.set_xlabel("Compression Config", fontsize=10)
ax_tps.set_ylabel("Tokens / second", fontsize=10)
ax_tps.set_xticks(list(x))
ax_tps.set_xticklabels(configs, rotation=20, ha="right", fontsize=9)
ax_tps.set_ylim(bottom=0)
ax_tps.spines[["top", "right"]].set_visible(False)
ax_tps.grid(axis="y", linestyle="--", alpha=0.4)

for bar, val in zip(bars2, tps_vals):
    if pd.notna(val):
        ax_tps.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(tps_vals, default=1) * 0.01,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=8,
        )

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
print(f"Figure saved -> {FIG_PATH}")
plt.close(fig)
