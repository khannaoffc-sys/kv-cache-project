"""
plot_results.py -- Visualise benchmark results from results/benchmark_results.csv.

Produces a two-panel figure:
  Panel 1 (left)  -- Energy per token (J/tok) by config  [bar chart]
                     SKIPPED / blank if power_source == "fake"
  Panel 2 (right) -- Tokens per second by config          [bar chart]

Saved to figures/tradeoff_summary.png at 150 dpi.

Usage
-----
    python src/plot_results.py

Requirements
------------
    pandas, matplotlib
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

# ── Validate required columns ────────────────────────────────────────────────
REQUIRED_COLS = {
    "config", "prompt_type", "latency_s", "tokens_per_sec",
    "power_source", "energy_j", "status",
}
missing_cols = REQUIRED_COLS - set(df.columns)
if missing_cols:
    print(f"[ERROR] benchmark_results.csv is missing columns: {missing_cols}")
    print("        CSV was produced by an older run_benchmark.py.")
    print("        Re-run: python src/run_benchmark.py --model Qwen/Qwen2.5-0.5B-Instruct")
    sys.exit(1)

# ── Drop ERROR rows ───────────────────────────────────────────────────────────
error_rows = df[df["status"] == "ERROR"]["config"].tolist()
if error_rows:
    print(f"[WARN] Dropping ERROR rows: {error_rows}")
df = df[df["status"] != "ERROR"].copy()

# ── Coerce numeric columns ────────────────────────────────────────────────────
for col in ("latency_s", "tokens_per_sec", "energy_j"):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ── Guard: refuse to plot fake-power energy ───────────────────────────────────
# Fake power = uniform(5, 15) W random — these are fabricated numbers.
# Must NEVER appear in a results figure.
has_fake = "power_source" in df.columns and (df["power_source"] == "fake").any()
if has_fake:
    df.loc[df["power_source"] == "fake", "energy_j"] = float("nan")
    print(
        "[WARN] power_source=fake detected. Energy bars will be omitted.\n"
        "       Fake power numbers are fabricated and must not appear in figures.\n"
        "       Re-run with POWER_BACKEND=nvml on a real GPU."
    )

has_nvml = "power_source" in df.columns and (df["power_source"] == "nvml").any()

# ---------------------------------------------------------------------------
# Derive energy-per-token where genuine power data exists
# ---------------------------------------------------------------------------
MAX_NEW_TOKENS = 50
if "new_tokens" in df.columns:
    df["_ntok"] = pd.to_numeric(df["new_tokens"], errors="coerce").fillna(MAX_NEW_TOKENS)
else:
    df["_ntok"] = MAX_NEW_TOKENS

df["energy_per_token_j"] = df["energy_j"] / df["_ntok"]

# ---------------------------------------------------------------------------
# Aggregate: latency_fixed rows, one row per config
# ---------------------------------------------------------------------------
latency_df = df[df["prompt_type"] == "latency_fixed"].copy()
if latency_df.empty:
    latency_df = df.copy()

agg = (
    latency_df
    .groupby("config")[["tokens_per_sec", "energy_per_token_j"]]
    .mean()
    .reset_index()
)

if agg.empty:
    print("[ERROR] No plottable data found in CSV.")
    sys.exit(1)

order = ["baseline"] + [c for c in agg["config"] if c != "baseline"]
agg["config"] = pd.Categorical(agg["config"], categories=order, ordered=True)
agg = agg.sort_values("config").reset_index(drop=True)

configs = agg["config"].tolist()
x = range(len(configs))

# ---------------------------------------------------------------------------
# Color palette
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

model_name = df["model_name"].iloc[0] if "model_name" in df.columns else "unknown"
device = df["device"].iloc[0] if "device" in df.columns else "unknown"
power_note = "NVML power" if has_nvml else "energy N/A (fake power excluded)"

fig.suptitle(
    f"KV-Cache Compression: Efficiency Trade-offs\n"
    f"Model: {model_name}  |  Device: {device}  |  {MAX_NEW_TOKENS} new tokens  |  {power_note}",
    fontsize=11,
    fontweight="bold",
)

# -- Panel 1: Energy per token ------------------------------------------------
energy_vals = agg["energy_per_token_j"].tolist()
bars1 = ax_energy.bar(x, energy_vals, color=colors, width=0.55, edgecolor="white", linewidth=0.8)

energy_title = "Energy per Token" if has_nvml else "Energy per Token (N/A — fake power excluded)"
ax_energy.set_title(energy_title, fontsize=11, fontweight="semibold", pad=8)
ax_energy.set_xlabel("Compression Config", fontsize=10)
ax_energy.set_ylabel("Energy (J / token)", fontsize=10)
ax_energy.set_xticks(list(x))
ax_energy.set_xticklabels(configs, rotation=20, ha="right", fontsize=9)
ax_energy.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
ax_energy.set_ylim(bottom=0)
ax_energy.spines[["top", "right"]].set_visible(False)
ax_energy.grid(axis="y", linestyle="--", alpha=0.4)

non_nan_energy = [v for v in energy_vals if pd.notna(v) and v > 0]
for bar, val in zip(bars1, energy_vals):
    if pd.notna(val) and val > 0:
        ax_energy.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(non_nan_energy, default=1) * 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=8,
        )
    else:
        ax_energy.text(
            bar.get_x() + bar.get_width() / 2,
            0.001,
            "N/A",
            ha="center", va="bottom", fontsize=7, color="gray",
        )

if not has_nvml:
    ax_energy.text(
        0.5, 0.5,
        "No valid energy data.\nRe-run with POWER_BACKEND=nvml\non a real GPU.",
        transform=ax_energy.transAxes,
        ha="center", va="center",
        fontsize=10, color="red", alpha=0.7,
    )

# -- Panel 2: Tokens per second -----------------------------------------------
tps_vals = agg["tokens_per_sec"].tolist()
bars2 = ax_tps.bar(x, tps_vals, color=colors, width=0.55, edgecolor="white", linewidth=0.8)

ax_tps.set_title("Throughput (Tokens / Second)", fontsize=11, fontweight="semibold", pad=8)
ax_tps.set_xlabel("Compression Config", fontsize=10)
ax_tps.set_ylabel("Tokens / second", fontsize=10)
ax_tps.set_xticks(list(x))
ax_tps.set_xticklabels(configs, rotation=20, ha="right", fontsize=9)
ax_tps.set_ylim(bottom=0)
ax_tps.spines[["top", "right"]].set_visible(False)
ax_tps.grid(axis="y", linestyle="--", alpha=0.4)

non_nan_tps = [v for v in tps_vals if pd.notna(v)]
for bar, val in zip(bars2, tps_vals):
    if pd.notna(val):
        ax_tps.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(non_nan_tps, default=1) * 0.01,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=8,
        )

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
print(f"Figure saved -> {FIG_PATH}")
if not has_nvml:
    print("[NOTE] Energy panel blank — power_source=fake. Run on GPU with POWER_BACKEND=nvml.")
plt.close(fig)
