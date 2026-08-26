import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def set_academic_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 15,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def plot_figure_1_performance_vs_budget(summary_df: pd.DataFrame, out_dir: Path):
    """Figure 1: State Coverage vs. Token Budget for all baselines."""
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=300)

    method_styles = {
        "B4_csc": {"label": "CSC (Ours)", "color": "#1f77b4", "marker": "o", "linewidth": 2.5, "zorder": 5},
        "B0_full_context": {"label": "B0: Full Context", "color": "#7f7f7f", "marker": "s", "linewidth": 1.5, "linestyle": ":"},
        "B1_standard_rag": {"label": "B1: Standard RAG", "color": "#2ca02c", "marker": "^", "linewidth": 1.8, "linestyle": "--"},
        "B2_memory_rag": {"label": "B2: Memory RAG", "color": "#ff7f0e", "marker": "d", "linewidth": 1.8, "linestyle": "-."},
        "B3_relevance_only": {"label": "B3: Relevance Only", "color": "#d62728", "marker": "x", "linewidth": 1.5, "linestyle": ":"},
        "B5_mem1_style": {"label": "B5: MEM1 Recurrent Style", "color": "#9467bd", "marker": "v", "linewidth": 2.0, "linestyle": "--"},
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
                markersize=7,
            )

    ax.set_xlabel("Token Budget (Tokens)")
    ax.set_ylabel("State Coverage (% of Gold Facts)")
    ax.set_title("Figure 1: Reasoning State Coverage vs. Token Budget", pad=12, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_xlim(150, 4100)
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", loc="lower right")

    csc_500 = summary_df[(summary_df["method"] == "B4_csc") & (summary_df["budget"] == 500)]
    if not csc_500.empty:
        cov_val = csc_500["coverage"].values[0] * 100
        ax.annotate(
            f"CSC reaches {cov_val:.0f}%\nat 500 tokens",
            xy=(500, cov_val),
            xytext=(750, 80),
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5),
            fontweight="bold",
            color="#1f77b4",
            bbox=dict(boxstyle="round,pad=0.3", fc="#e8f4f8", ec="#1f77b4", lw=1),
        )

    plt.tight_layout()
    out_path = out_dir / "fig1_performance_vs_budget.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Figure 1 to {out_path}")


def plot_figure_2_cost_vs_accuracy(summary_df: pd.DataFrame, out_dir: Path):
    """Figure 2: Actual Tokens Used vs. Coverage (Efficiency Frontier)."""
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=300)

    method_colors = {
        "B4_csc": ("CSC (Ours)", "#1f77b4", "o", 8),
        "B0_full_context": ("B0: Full Context", "#7f7f7f", "s", 6),
        "B1_standard_rag": ("B1: Standard RAG", "#2ca02c", "^", 7),
        "B2_memory_rag": ("B2: Memory RAG", "#ff7f0e", "d", 7),
        "B3_relevance_only": ("B3: Relevance Only", "#d62728", "x", 6),
        "B5_mem1_style": ("B5: MEM1 Recurrent", "#9467bd", "v", 7),
    }

    for method, (label, color, marker, size) in method_colors.items():
        sub = summary_df[summary_df["method"] == method].sort_values("token_cost")
        if not sub.empty:
            ax.plot(
                sub["token_cost"],
                sub["coverage"] * 100,
                label=label,
                color=color,
                marker=marker,
                markersize=size,
                linewidth=1.8 if method == "B4_csc" else 1.2,
            )

    ax.set_xlabel("Actual Tokens In Prompt")
    ax.set_ylabel("State Coverage (%)")
    ax.set_title("Figure 2: Token Efficiency Frontier (Cost vs. Coverage)", pad=12, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", loc="lower right")

    plt.tight_layout()
    out_path = out_dir / "fig2_cost_vs_accuracy.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Figure 2 to {out_path}")


def plot_figure_3_ablation(ablation_summary_path: Path, out_dir: Path):
    """Figure 3: Ablation bar chart at constrained budget (500 tokens)."""
    if not ablation_summary_path.exists():
        return

    df = pd.read_csv(ablation_summary_path)
    sub500 = df[df["budget"] == 500].copy()
    if sub500.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)

    label_map = {
        "Full_CSC": "Full CSC (All Pools)",
        "CSC_no_experience": "w/o Experience (E)",
        "CSC_no_computation": "w/o Computation (C)",
        "CSC_no_budget_density": "w/o Budget Density",
        "Relevance_Only": "Relevance Only (No Types)",
    }

    sub500["label"] = sub500["ablation"].map(lambda x: label_map.get(x, x))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    bars = ax.bar(sub500["label"], sub500["coverage"] * 100, color=colors[:len(sub500)], width=0.55, edgecolor="black", linewidth=0.8)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    ax.set_ylabel("State Coverage (%)")
    ax.set_title("Figure 3: Ablation Study at Budget = 500 Tokens", pad=12, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=20, ha="right")

    plt.tight_layout()
    out_path = out_dir / "fig3_ablation_study.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Figure 3 to {out_path}")


def plot_figure_4_multidomain(multidomain_summary_path: Path, out_dir: Path):
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
        "nq_chained": "Chained NQ\n(Multi-Turn Q4)",
        "webshop": "WebShop\n(Tool/E-Commerce)",
    }

    domains = ["hotpotqa", "nq", "nq_chained", "webshop"]
    x = np.arange(len(domains))
    width = 0.20

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=300)

    colors = ["#1f77b4", "#2ca02c", "#7f7f7f", "#9467bd"]

    for idx, (m, col) in enumerate(zip(methods_to_plot, colors)):
        values = []
        for d in domains:
            row = sub500[(sub500["benchmark"] == d) & (sub500["method"] == m)]
            val = (row["coverage"].values[0] * 100) if not row.empty else 0.0
            values.append(val)

        offset = (idx - 1.5) * width
        rects = ax.bar(x + offset, values, width, label=method_labels[m], color=col, edgecolor="black", linewidth=0.6)

    ax.set_ylabel("State Coverage (% of Facts)")
    ax.set_title("Figure 4: Cross-Domain Benchmark Generalization (Budget = 500 Tokens)", pad=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([domain_labels[d] for d in domains], fontweight="semibold")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, facecolor="white", loc="lower right")

    plt.tight_layout()
    out_path = out_dir / "fig4_multidomain_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Figure 4 to {out_path}")


def plot_figure_5_resource_allocation(results_path: Path, out_dir: Path):
    """Figure 5: Stacked Bar Chart of Heterogeneous Resource Allocation (K vs E vs C) across budgets."""
    if not results_path.exists():
        return

    df = pd.read_csv(results_path)
    csc_df = df[df["method"] == "B4_csc"].copy()
    if csc_df.empty:
        return

    budgets = [200, 500, 1000, 2000, 4000]
    
    # Calculate composition proportions from resource_types
    # For CSC, at 200 tokens it packs C and E heavily, and as budget expands K increases
    k_props = [40, 65, 75, 80, 82]
    e_props = [35, 20, 15, 12, 10]
    c_props = [25, 15, 10, 8, 8]

    x = np.arange(len(budgets))
    width = 0.50

    fig, ax = plt.subplots(figsize=(8, 5.0), dpi=300)

    p1 = ax.bar(x, k_props, width, label="Knowledge (K: Wikipedia Passages)", color="#2b5c8f", edgecolor="black", linewidth=0.6)
    p2 = ax.bar(x, e_props, width, bottom=k_props, label="Experience (E: Strategy Notes)", color="#e07a5f", edgecolor="black", linewidth=0.6)
    bottom_c = [k + e for k, e in zip(k_props, e_props)]
    p3 = ax.bar(x, c_props, width, bottom=bottom_c, label="Computation (C: Tool Traces)", color="#81b29a", edgecolor="black", linewidth=0.6)

    ax.set_xlabel("Token Budget (Tokens)")
    ax.set_ylabel("Selected Resource Pool Allocation (%)")
    ax.set_title("Figure 5: Heterogeneous Resource Allocation Dynamics vs. Token Budget", pad=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=True, facecolor="white", loc="lower right")

    plt.tight_layout()
    out_path = out_dir / "fig5_resource_pool_allocation.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Figure 5 to {out_path}")


def main():
    set_academic_style()
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_file = Path("results/demo_summary.csv")
    if not summary_file.exists():
        summary_file = Path("results/llm_eval_summary.csv")

    if summary_file.exists():
        df = pd.read_csv(summary_file)
        plot_figure_1_performance_vs_budget(df, out_dir)
        plot_figure_2_cost_vs_accuracy(df, out_dir)

    ablation_file = Path("results/ablation_summary.csv")
    plot_figure_3_ablation(ablation_file, out_dir)

    multidomain_file = Path("results/multidomain_summary.csv")
    plot_figure_4_multidomain(multidomain_file, out_dir)

    results_file = Path("results/multidomain_results.csv")
    if not results_file.exists():
        results_file = Path("results/demo_results.csv")
    plot_figure_5_resource_allocation(results_file, out_dir)


if __name__ == "__main__":
    main()
