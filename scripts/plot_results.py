#!/usr/bin/env python3
"""
Generate performance comparison charts from benchmark results.

Usage:
    python scripts/plot_results.py

Outputs:
    docs/figures/performance_lines.png   (line chart: GFLOPS vs matrix size)
    docs/figures/performance_bars.png    (bar chart: GFLOPS at N=4096)
    docs/figures/speedup_bars.png        (bar chart: speedup vs naive at N=4096)

Numbers below come from the most recent full-power benchmark run on
RTX 4060 Laptop (sm_89), CUDA 12.4, alpha=1, beta=0, 10 timed iterations.
Update this dict when you re-run benchmarks.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Benchmark results (GFLOPS) — full-power run on RTX 4060 Laptop, sm_89
# Update this when you re-run ./benchmark
# ──────────────────────────────────────────────────────────────────────
SIZES = [512, 1024, 2048, 4096]

RESULTS = {
    "v0 Naive":          [664.9,  706.2,  859.5,  754.5],
    "v1 Tiling":         [846.7,  740.4, 1062.3,  990.6],
    "v2 RegTile":        [2386.8, 3316.6, 5640.7, 4953.0],
    "v3 + float4":       [2746.3, 4084.7, 7863.3, 6755.9],
    "v4 +1 padding (-)": [2680.9, 3444.1, 7372.2, 6554.6],
    "v5 DoubleBuf (-)":  [2832.5, 3265.1, 5835.6, 4273.1],
    "cuBLAS":            [4920.6, 4855.9, 9618.4, 8659.9],
}

# Color scheme: positive optimizations bright, negative results muted
COLORS = {
    "v0 Naive":          "#888888",
    "v1 Tiling":         "#4477aa",
    "v2 RegTile":        "#228833",
    "v3 + float4":       "#ee6677",   # headline
    "v4 +1 padding (-)": "#cc99cc",   # muted (negative)
    "v5 DoubleBuf (-)":  "#cc99cc",   # muted (negative)
    "cuBLAS":            "#000000",
}

LINESTYLES = {
    "v0 Naive":          "-",
    "v1 Tiling":         "-",
    "v2 RegTile":        "-",
    "v3 + float4":       "-",
    "v4 +1 padding (-)": "--",
    "v5 DoubleBuf (-)":  "--",
    "cuBLAS":            ":",
}

OUT_DIR = Path(__file__).parent.parent / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# Plot 1: Line chart - GFLOPS vs matrix size
# ──────────────────────────────────────────────────────────────────────
def plot_lines():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, gflops in RESULTS.items():
        ax.plot(
            SIZES, gflops,
            marker="o", markersize=6,
            linewidth=2.0,
            color=COLORS[name],
            linestyle=LINESTYLES[name],
            label=name,
        )
    ax.set_xlabel("Matrix size (M = N = K)", fontsize=11)
    ax.set_ylabel("GFLOPS", fontsize=11)
    ax.set_title(
        "SGEMM performance on RTX 4060 Laptop (sm_89, FP32)",
        fontsize=12, pad=12
    )
    ax.set_xticks(SIZES)
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.grid(True, alpha=0.3, linestyle="-")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    out = OUT_DIR / "performance_lines.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")


# ──────────────────────────────────────────────────────────────────────
# Plot 2: Bar chart - GFLOPS at N=4096
# ──────────────────────────────────────────────────────────────────────
def plot_bars_at_4096():
    names = list(RESULTS.keys())
    values = [RESULTS[n][-1] for n in names]   # last column = N=4096

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        range(len(names)), values,
        color=[COLORS[n] for n in names],
        edgecolor="black", linewidth=0.5,
    )
    # value labels on top
    for i, v in enumerate(values):
        ax.text(
            i, v + max(values) * 0.01,
            f"{v:,.0f}",
            ha="center", va="bottom", fontsize=9
        )
    # cuBLAS reference line
    cublas_v = RESULTS["cuBLAS"][-1]
    ax.axhline(cublas_v, color="black", linestyle=":", alpha=0.4, linewidth=1)
    ax.text(
        len(names) - 0.5, cublas_v * 1.02,
        "cuBLAS ceiling",
        ha="right", va="bottom", fontsize=9, color="gray", style="italic"
    )

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("GFLOPS", fontsize=11)
    ax.set_title(
        "SGEMM @ M=N=K=4096 on RTX 4060 Laptop",
        fontsize=12, pad=12
    )
    ax.grid(True, axis="y", alpha=0.3, linestyle="-")
    ax.set_ylim(0, max(values) * 1.12)
    plt.tight_layout()
    out = OUT_DIR / "performance_bars.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")


# ──────────────────────────────────────────────────────────────────────
# Plot 3: Speedup vs naive at N=4096
# ──────────────────────────────────────────────────────────────────────
def plot_speedup_at_4096():
    naive_4096 = RESULTS["v0 Naive"][-1]
    names = [n for n in RESULTS.keys() if n != "v0 Naive"]
    speedups = [RESULTS[n][-1] / naive_4096 for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        range(len(names)), speedups,
        color=[COLORS[n] for n in names],
        edgecolor="black", linewidth=0.5,
    )
    for i, sp in enumerate(speedups):
        ax.text(
            i, sp + 0.15,
            f"{sp:.2f}×",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(-0.4, 1.05, "naive baseline", fontsize=9, color="gray", style="italic")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Speedup vs Naive", fontsize=11)
    ax.set_title(
        "Speedup over Naive @ M=N=K=4096",
        fontsize=12, pad=12
    )
    ax.grid(True, axis="y", alpha=0.3, linestyle="-")
    ax.set_ylim(0, max(speedups) * 1.18)
    plt.tight_layout()
    out = OUT_DIR / "speedup_bars.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")


if __name__ == "__main__":
    print("Generating performance figures...")
    plot_lines()
    plot_bars_at_4096()
    plot_speedup_at_4096()
    print("Done.")
