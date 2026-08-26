"""
Publication-Grade Visualizations: CSC vs. MEM1 and Multi-Turn Baselines.
Generates 3 key figures:
1. Performance vs Multi-Objective Sequence Horizon (N = 2, 4, 8, 16)
2. Average Context Token Footprint vs Horizon
3. Paradigm Trade-off: Accuracy vs Memory vs Training Compute
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11

def main():
    csv_path = Path("results/mem1_vs_csc_literature_comparison.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run scripts/compare_with_mem1_literature.py first.")
        return

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    # Styling dictionary for each method
    styles = {
        "Full-Context (Unbounded)": {"color": "#1f77b4", "marker": "o", "label": "Full-Context (Unbounded)", "linestyle": "-"},
        "Sliding Window (FIFO)": {"color": "#d62728", "marker": "s", "label": "Sliding Window (FIFO, B=512)", "linestyle": "--"},
        "MEM1 (Published Reference)": {"color": "#9467bd", "marker": "^", "label": "MEM1 (RL Recurrent, B=512)", "linestyle": "-."},
        "CSC (Ours, Budget=512)": {"color": "#2ca02c", "marker": "D", "label": "CSC (Ours, Knapsack, B=512)", "linestyle": "-"},
    }

    horizons = [2, 4, 8, 16]

    # -------------------------------------------------------------
    # Plot 1: State Coverage / Retention vs. Horizon (N)
    # -------------------------------------------------------------
    ax1 = axes[0]
    for method, style in styles.items():
        sub = df[df["method"] == method].sort_values(by="horizon_N")
        if not sub.empty:
            ax1.plot(
                sub["horizon_N"],
                sub["avg_state_coverage"] * 100,
                color=style["color"],
                marker=style["marker"],
                label=style["label"],
                linewidth=2.4,
                markersize=8,
                linestyle=style["linestyle"],
            )

    ax1.set_title("A. State Retention vs. Sequence Horizon ($N$)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Chained Objectives Horizon ($N$ turns)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("State Coverage / Retention (%)", fontsize=11, fontweight="bold")
    ax1.set_xticks(horizons)
    ax1.set_ylim(40, 105)
    ax1.legend(loc="lower left", frameon=True, fontsize=9.5)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # -------------------------------------------------------------
    # Plot 2: Context Token Footprint vs. Horizon (N)
    # -------------------------------------------------------------
    ax2 = axes[1]
    for method, style in styles.items():
        sub = df[df["method"] == method].sort_values(by="horizon_N")
        if not sub.empty:
            ax2.plot(
                sub["horizon_N"],
                sub["avg_token_cost"],
                color=style["color"],
                marker=style["marker"],
                label=style["label"],
                linewidth=2.4,
                markersize=8,
                linestyle=style["linestyle"],
            )

    # Add constant budget line
    ax2.axhline(512, color="black", linestyle=":", alpha=0.7, label="Budget Limit (512 tokens)")

    ax2.set_title("B. Context Memory Footprint vs. Horizon ($N$)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Chained Objectives Horizon ($N$ turns)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Average Tokens in Prompt", fontsize=11, fontweight="bold")
    ax2.set_xticks(horizons)
    ax2.legend(loc="upper left", frameon=True, fontsize=9.5)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # -------------------------------------------------------------
    # Plot 3: Paradigm Comparison: Efficiency vs Training Overhead
    # -------------------------------------------------------------
    ax3 = axes[2]
    
    methods = ["Full-Context", "Sliding Window", "MEM1 (RL)", "CSC (Ours)"]
    n16_coverage = []
    for m_key in ["Full-Context (Unbounded)", "Sliding Window (FIFO)", "MEM1 (Published Reference)", "CSC (Ours, Budget=512)"]:
        val = df[(df["method"] == m_key) & (df["horizon_N"] == 16)]["avg_state_coverage"].values
        n16_coverage.append(val[0] * 100 if len(val) > 0 else 0)

    bar_colors = ["#1f77b4", "#d62728", "#9467bd", "#2ca02c"]
    bars = ax3.bar(methods, n16_coverage, color=bar_colors, width=0.55, edgecolor="black", alpha=0.85)

    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f"{height:.1f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4),
                     textcoords="offset points",
                     ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax3.set_title("C. 16-Turn Long-Horizon Retention ($N=16$)", fontsize=13, fontweight="bold", pad=12)
    ax3.set_ylabel("Retention at N=16 (%)", fontsize=11, fontweight="bold")
    ax3.set_ylim(0, 115)
    ax3.grid(True, linestyle="--", alpha=0.6, axis="y")

    # Add text box for zero-shot advantage
    props = dict(boxstyle='round,pad=0.5', facecolor='#e8f5e9', edgecolor='#2ca02c', alpha=0.9)
    ax3.text(0.5, 0.15, "CSC Advantage:\n• 100% Zero-Shot (0 hrs RL)\n• Exact Knapsack Budget (B <= 512)\n• Heterogeneous Pools {K, E, C}",
             transform=ax3.transAxes, fontsize=9.5, verticalalignment='bottom', horizontalalignment='center', bbox=props)

    plt.tight_layout()
    out_file = Path("results/mem1_vs_csc_comparison.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Generated comparison figure saved to: {out_file}")

if __name__ == "__main__":
    main()
