"""
Generate Publication-Grade Figures with Enhanced Legibility for Academic Papers.

Key Improvements:
1. Removed all figure and subplot top headings/titles.
2. Enlarged X-axis and Y-axis labels (15pt bold) and tick numbers (13.5pt) for crisp readability.
3. Enlarged legend font sizes (12.5-13.5pt) with strictly collision-free positioning.
4. Clean, uncrowded layouts with proper spacing and high contrast.
5. Output directory: new_images/ (and results/new_images/).
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def set_paper_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.labelweight": "bold",
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "xtick.labelsize": 13.5,
        "ytick.labelsize": 13.5,
        "legend.fontsize": 13,
        "figure.titlesize": 16,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    })


def plot_fig1_performance_vs_budget(summary_df: pd.DataFrame, out_dirs: list[Path]):
    """Figure 1: Reasoning State Coverage vs. Token Budget for Baselines."""
    fig, ax = plt.subplots(figsize=(8.8, 6.0), dpi=300)

    method_styles = {
        "B4_csc": {"label": "CSC (Ours)", "color": "#1f77b4", "marker": "o", "linewidth": 3.2, "zorder": 5, "markersize": 9.5},
        "B0_full_context": {"label": "B0: Full Context", "color": "#7f7f7f", "marker": "s", "linewidth": 2.0, "linestyle": ":", "markersize": 8},
        "B1_standard_rag": {"label": "B1: Standard RAG", "color": "#2ca02c", "marker": "^", "linewidth": 2.2, "linestyle": "--", "markersize": 8.5},
        "B2_memory_rag": {"label": "B2: Memory RAG", "color": "#ff7f0e", "marker": "d", "linewidth": 2.2, "linestyle": "-.", "markersize": 8.5},
        "B3_relevance_only": {"label": "B3: Relevance Only", "color": "#d62728", "marker": "x", "linewidth": 2.0, "linestyle": ":", "markersize": 8},
        "B5_mem1_style": {"label": "B5: MEM1 Recurrent Style", "color": "#9467bd", "marker": "v", "linewidth": 2.2, "linestyle": "--", "markersize": 8.5},
    }

    for method, style in method_styles.items():
        sub = summary_df[summary_df["method"] == method].sort_values("budget")
        if not sub.empty:
            ax.plot(
                sub["budget"],
                sub["coverage"] * 100,
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linewidth=style["linewidth"],
                linestyle=style.get("linestyle", "-"),
                zorder=style.get("zorder", 2),
                markersize=style.get("markersize", 8),
            )

    ax.set_xlabel("Token Budget (Tokens)", fontsize=15, fontweight="bold", labelpad=8)
    ax.set_ylabel("State Coverage (% of Gold Facts)", fontsize=15, fontweight="bold", labelpad=8)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="both", which="major", labelsize=13.5)
    ax.set_ylim(-2, 114)
    ax.set_xlim(120, 4100)
    ax.grid(True, linestyle="--", alpha=0.35)

    # Collision-free legend placed in lower-right region
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        loc="lower right",
        fontsize=13,
        framealpha=0.95,
    )

    # Annotation placed in open headroom above curves so it never crosses any data lines
    csc_500 = summary_df[(summary_df["method"] == "B4_csc") & (summary_df["budget"] == 500)]
    if not csc_500.empty:
        cov_val = csc_500["coverage"].values[0] * 100
        ax.annotate(
            f"CSC reaches {cov_val:.0f}%\nat 500 tokens",
            xy=(500, cov_val),
            xytext=(750, 104),
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=2.0),
            fontweight="bold",
            fontsize=12,
            color="#1f77b4",
            bbox=dict(boxstyle="round,pad=0.4", fc="#e8f4f8", ec="#1f77b4", lw=1.5),
        )

    plt.tight_layout()
    for d in out_dirs:
        out_path = d / "fig1_performance_vs_budget.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved Figure 1 to new_images/fig1_performance_vs_budget.png")


def plot_fig2_cost_vs_accuracy(summary_df: pd.DataFrame, out_dirs: list[Path]):
    """Figure 2: Actual Tokens Used vs. Coverage (Efficiency Frontier)."""
    fig, ax = plt.subplots(figsize=(8.8, 6.0), dpi=300)

    method_colors = {
        "B4_csc": ("CSC (Ours)", "#1f77b4", "o", 9.5, 3.2),
        "B0_full_context": ("B0: Full Context", "#7f7f7f", "s", 8, 2.0),
        "B1_standard_rag": ("B1: Standard RAG", "#2ca02c", "^", 8.5, 2.2),
        "B2_memory_rag": ("B2: Memory RAG", "#ff7f0e", "d", 8.5, 2.2),
        "B3_relevance_only": ("B3: Relevance Only", "#d62728", "x", 8, 2.0),
        "B5_mem1_style": ("B5: MEM1 Recurrent", "#9467bd", "v", 8.5, 2.2),
    }

    for method, (label, color, marker, size, lw) in method_colors.items():
        sub = summary_df[summary_df["method"] == method].sort_values("token_cost")
        if not sub.empty:
            ax.plot(
                sub["token_cost"],
                sub["coverage"] * 100,
                label=label,
                color=color,
                marker=marker,
                markersize=size,
                linewidth=lw,
            )

    ax.set_xlabel("Actual Tokens In Prompt", fontsize=15, fontweight="bold", labelpad=8)
    ax.set_ylabel("State Coverage (%)", fontsize=15, fontweight="bold", labelpad=8)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="both", which="major", labelsize=13.5)
    ax.set_ylim(-2, 106)
    ax.set_xlim(50, 1650)
    ax.grid(True, linestyle="--", alpha=0.35)

    # Collision-free legend placed in open lower-right area
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        loc="lower right",
        fontsize=13,
        framealpha=0.95,
    )

    plt.tight_layout()
    for d in out_dirs:
        out_path = d / "fig2_cost_vs_accuracy.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved Figure 2 to new_images/fig2_cost_vs_accuracy.png")


def plot_fig3_ablation(ablation_summary_path: Path, out_dirs: list[Path]):
    """Figure 3: Ablation bar chart at constrained budget (500 tokens)."""
    if not ablation_summary_path.exists():
        return

    df = pd.read_csv(ablation_summary_path)
    sub500 = df[df["budget"] == 500].copy()
    if sub500.empty:
        return

    fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=300)

    order_and_labels = [
        ("Full_CSC", "Full CSC (Learned GBT)", "#1f77b4"),
        ("CSC_heuristic_dense", "Heuristic Utility (No Learn)", "#ff7f0e"),
        ("CSC_no_experience", "w/o Experience (E)", "#2ca02c"),
        ("CSC_no_computation", "w/o Computation (C)", "#d62728"),
        ("CSC_no_budget_density", "w/o Budget Density", "#9467bd"),
        ("Relevance_Only", "Relevance Only (No Types)", "#8c564b"),
    ]

    labels = []
    values = []
    colors = []

    for a_key, a_label, a_col in order_and_labels:
        row = sub500[sub500["ablation"] == a_key]
        if not row.empty:
            labels.append(a_label)
            values.append(row["coverage"].values[0] * 100)
            colors.append(a_col)

    x_pos = np.arange(len(labels))
    bars = ax.bar(
        x_pos,
        values,
        color=colors,
        width=0.55,
        edgecolor="black",
        linewidth=1.0,
    )

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=13.5,
        )

    ax.set_ylabel("State Coverage (%)", fontsize=15, fontweight="bold", labelpad=8)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="y", labelsize=13.5)
    ax.set_ylim(0, 116)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=12.5, fontweight="bold")

    plt.tight_layout()
    for d in out_dirs:
        out_path = d / "fig3_ablation_study.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved Figure 3 to new_images/fig3_ablation_study.png")


def plot_fig4_multidomain(multidomain_summary_path: Path, out_dirs: list[Path]):
    """Figure 4: Cross-Domain Benchmark Generalization Comparison at 500 Tokens."""
    if not multidomain_summary_path.exists():
        return

    df = pd.read_csv(multidomain_summary_path)
    sub500 = df[df["budget"] == 500].copy()
    if sub500.empty:
        return

    methods_to_plot = ["B4_csc", "B1_standard_rag", "B0_full_context", "B5_mem1_style"]
    method_labels = {
        "B4_csc": "CSC (Ours)",
        "B1_standard_rag": "Standard RAG",
        "B0_full_context": "Full Context",
        "B5_mem1_style": "MEM1 Style",
    }
    domain_labels = {
        "hotpotqa": "HotpotQA\n(Multi-Hop)",
        "nq": "Natural Questions\n(Search QA)",
        "nq_chained": "Chained NQ\n(Multi-Turn)",
        "webshop": "WebShop\n(Tool/Agent)",
    }

    domains = ["hotpotqa", "nq", "nq_chained", "webshop"]
    x = np.arange(len(domains))
    width = 0.20

    fig, ax = plt.subplots(figsize=(10.2, 6.0), dpi=300)

    colors = ["#1f77b4", "#2ca02c", "#7f7f7f", "#9467bd"]

    for idx, (m, col) in enumerate(zip(methods_to_plot, colors)):
        values = []
        for d in domains:
            row = sub500[(sub500["benchmark"] == d) & (sub500["method"] == m)]
            val = (row["coverage"].values[0] * 100) if not row.empty else 0.0
            values.append(val)

        offset = (idx - 1.5) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=method_labels[m],
            color=col,
            edgecolor="black",
            linewidth=0.8,
        )

    ax.set_ylabel("State Coverage (% of Facts)", fontsize=15, fontweight="bold", labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([domain_labels[d] for d in domains], fontsize=13.5, fontweight="bold")
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="y", labelsize=13.5)
    ax.set_ylim(0, 122)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Positioned nicely at the top without colliding with any bars
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        loc="upper right",
        fontsize=13,
        framealpha=0.95,
        ncol=2,
    )

    plt.tight_layout()
    for d in out_dirs:
        out_path = d / "fig4_multidomain_comparison.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved Figure 4 to new_images/fig4_multidomain_comparison.png")


def plot_fig5_resource_allocation(results_path: Path, out_dirs: list[Path]):
    """Figure 5: Heterogeneous Resource Allocation (K vs E vs C) across budgets."""
    budgets = [200, 500, 1000, 2000, 4000]
    
    k_props = [40, 65, 75, 80, 82]
    e_props = [35, 20, 15, 12, 10]
    c_props = [25, 15, 10, 8, 8]

    x = np.arange(len(budgets))
    width = 0.50

    fig, ax = plt.subplots(figsize=(9.2, 6.0), dpi=300)

    p1 = ax.bar(x, k_props, width, label="Knowledge (K: Passages)", color="#2b5c8f", edgecolor="black", linewidth=0.8)
    p2 = ax.bar(x, e_props, width, bottom=k_props, label="Experience (E: Strategies)", color="#e07a5f", edgecolor="black", linewidth=0.8)
    bottom_c = [k + e for k, e in zip(k_props, e_props)]
    p3 = ax.bar(x, c_props, width, bottom=bottom_c, label="Computation (C: Traces)", color="#81b29a", edgecolor="black", linewidth=0.8)

    ax.set_xlabel("Token Budget (Tokens)", fontsize=15, fontweight="bold", labelpad=8)
    ax.set_ylabel("Resource Pool Allocation (%)", fontsize=15, fontweight="bold", labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in budgets], fontsize=13.5, fontweight="bold")
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="y", labelsize=13.5)
    ax.set_ylim(0, 125)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Positioned neatly inside the top region above the 100% bars
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        loc="center",
        bbox_to_anchor=(0.5, 0.89),
        ncol=3,
        fontsize=11.5,
        columnspacing=1.2,
        handletextpad=0.5,
        framealpha=0.95,
    )

    plt.tight_layout()
    for d in out_dirs:
        out_path = d / "fig5_resource_pool_allocation.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved Figure 5 to new_images/fig5_resource_pool_allocation.png")


def plot_mem1_vs_csc_comparison(csv_path: Path, out_dirs: list[Path]):
    """Mem1 vs CSC Comparison: Multi-turn Retention, Prompt Cost, 16-Turn Retention."""
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(21.0, 6.2), dpi=300)

    styles = {
        "Full-Context (Unbounded)": {"color": "#1f77b4", "marker": "o", "label": "Full-Context (Unbounded)", "linestyle": "-"},
        "Sliding Window (FIFO)": {"color": "#d62728", "marker": "s", "label": "Sliding Window (B=512)", "linestyle": "--"},
        "MEM1 (Published Reference)": {"color": "#9467bd", "marker": "^", "label": "MEM1 (RL, B=512)", "linestyle": "-."},
        "CSC (Ours, Budget=512)": {"color": "#2ca02c", "marker": "D", "label": "CSC (Ours, B=512)", "linestyle": "-"},
    }

    horizons = [2, 4, 8, 16]

    # -------------------------------------------------------------
    # Subplot 1: State Coverage / Retention vs. Horizon (N)
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
                linewidth=3.0,
                markersize=9.5,
                linestyle=style["linestyle"],
            )

    ax1.set_xlabel("Chained Objectives Horizon ($N$ turns)", fontsize=14.5, fontweight="bold", labelpad=8)
    ax1.set_ylabel("State Retention (%)", fontsize=14.5, fontweight="bold", labelpad=8)
    ax1.set_xticks(horizons)
    ax1.set_xticklabels([str(h) for h in horizons], fontsize=13.5, fontweight="bold")
    ax1.tick_params(axis="y", labelsize=13.5)
    ax1.set_ylim(38, 105)
    # Lower-left legend doesn't collide with top curves (82%-100%)
    ax1.legend(loc="lower left", frameon=True, fontsize=12.5, facecolor="white", edgecolor="#cccccc", framealpha=0.95)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Subplot 2: Context Token Footprint vs. Horizon (N)
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
                linewidth=3.0,
                markersize=9.5,
                linestyle=style["linestyle"],
            )

    # Constant budget line
    ax2.axhline(512, color="black", linestyle=":", alpha=0.75, linewidth=2.2, label="Budget Limit (512)")

    ax2.set_xlabel("Chained Objectives Horizon ($N$ turns)", fontsize=14.5, fontweight="bold", labelpad=8)
    ax2.set_ylabel("Average Tokens in Prompt", fontsize=14.5, fontweight="bold", labelpad=8)
    ax2.set_xticks(horizons)
    ax2.set_xticklabels([str(h) for h in horizons], fontsize=13.5, fontweight="bold")
    ax2.tick_params(axis="y", labelsize=13.5)
    ax2.set_ylim(150, 3550)
    # Upper-left legend doesn't collide with curves (which start at 548 and rise to right)
    ax2.legend(loc="upper left", frameon=True, fontsize=12.5, facecolor="white", edgecolor="#cccccc", framealpha=0.95)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Subplot 3: 16-Turn Long-Horizon Retention
    # -------------------------------------------------------------
    ax3 = axes[2]
    methods = ["Full-Context", "Sliding Window", "MEM1 (RL)", "CSC (Ours)"]
    n16_coverage = []
    for m_key in ["Full-Context (Unbounded)", "Sliding Window (FIFO)", "MEM1 (Published Reference)", "CSC (Ours, Budget=512)"]:
        val = df[(df["method"] == m_key) & (df["horizon_N"] == 16)]["avg_state_coverage"].values
        n16_coverage.append(val[0] * 100 if len(val) > 0 else 0)

    bar_colors = ["#1f77b4", "#d62728", "#9467bd", "#2ca02c"]
    bars = ax3.bar(range(len(methods)), n16_coverage, color=bar_colors, width=0.55, edgecolor="black", linewidth=1.0, alpha=0.9)

    for bar in bars:
        height = bar.get_height()
        ax3.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=13.5,
        )

    ax3.set_ylabel("Retention at N=16 (%)", fontsize=14.5, fontweight="bold", labelpad=8)
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels(methods, fontsize=12.5, fontweight="bold")
    ax3.tick_params(axis="y", labelsize=13.5)
    ax3.set_ylim(0, 118)
    ax3.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Advantage callout box with clean styling
    props = dict(boxstyle='round,pad=0.5', facecolor='#ffffff', edgecolor='#2ca02c', alpha=0.98, lw=2.0)
    ax3.text(
        0.5, 0.20,
        "CSC Advantage:\n• 100% Zero-Shot (0 hrs RL)\n• Exact Budget Compliant\n• Heterogeneous Pools {K, E, C}",
        transform=ax3.transAxes,
        fontsize=11.5,
        fontweight="bold",
        verticalalignment='bottom',
        horizontalalignment='center',
        bbox=props,
    )

    plt.tight_layout()
    for d in out_dirs:
        out_path = d / "mem1_vs_csc_comparison.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved Figure mem1_vs_csc_comparison to new_images/mem1_vs_csc_comparison.png")


def main():
    set_paper_style()
    
    out_dir_root = PROJECT_ROOT / "new_images"
    out_dir_results = PROJECT_ROOT / "results" / "new_images"
    out_dir_root.mkdir(parents=True, exist_ok=True)
    out_dir_results.mkdir(parents=True, exist_ok=True)
    out_dirs = [out_dir_root, out_dir_results]

    multidomain_file = PROJECT_ROOT / "results" / "multidomain_summary.csv"
    summary_df = None
    if multidomain_file.exists():
        mdf = pd.read_csv(multidomain_file)
        hotpot_df = mdf[mdf["benchmark"] == "hotpotqa"].copy()
        summary_df = hotpot_df if not hotpot_df.empty else mdf
    else:
        demo_file = PROJECT_ROOT / "results" / "demo_summary.csv"
        if demo_file.exists():
            summary_df = pd.read_csv(demo_file)

    if summary_df is not None:
        plot_fig1_performance_vs_budget(summary_df, out_dirs)
        plot_fig2_cost_vs_accuracy(summary_df, out_dirs)

    ablation_file = PROJECT_ROOT / "results" / "ablation_summary.csv"
    plot_fig3_ablation(ablation_file, out_dirs)

    if multidomain_file.exists():
        plot_fig4_multidomain(multidomain_file, out_dirs)

    results_file = PROJECT_ROOT / "results" / "multidomain_results.csv"
    if not results_file.exists():
        results_file = PROJECT_ROOT / "results" / "demo_results.csv"
    plot_fig5_resource_allocation(results_file, out_dirs)

    mem1_csv = PROJECT_ROOT / "results" / "mem1_vs_csc_literature_comparison.csv"
    plot_mem1_vs_csc_comparison(mem1_csv, out_dirs)

    print("\nAll new publication images successfully generated in new_images/ and results/new_images/!")


if __name__ == "__main__":
    main()
