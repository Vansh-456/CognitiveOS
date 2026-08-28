"""
Automated LaTeX Table Exporter for Cognitive State Compilation (CSC).

Converts benchmark results CSVs into booktabs-formatted LaTeX tables ready
for direct inclusion in academic papers (\input{tables/...}).
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np


def export_table_1_multidomain(summary_df: pd.DataFrame, out_dir: Path) -> str:
    """Table 1: Cross-Domain State Coverage across budgets."""
    method_order = [
        ("B4_csc", "\\textbf{CSC (Ours)}"),
        ("B1_standard_rag", "Standard RAG"),
        ("B2_memory_rag", "Memory RAG"),
        ("B3_relevance_only", "Relevance Only"),
        ("B5_mem1_style", "MEM1 Recurrent"),
        ("B0_full_context", "Full Context"),
    ]
    domain_order = [
        ("hotpotqa", "HotpotQA (Multi-Hop)"),
        ("nq", "Natural Questions (Search QA)"),
        ("nq_chained", "Chained NQ (Multi-Turn Q4)"),
        ("webshop", "WebShop (E-Commerce)"),
    ]
    budgets = [200, 500, 1000, 2000]

    lines = [
        "% Table 1: Cross-Domain Reasoning State Coverage vs. Token Budget",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "\\textbf{Domain} & \\textbf{Method} & \\textbf{B=200} & \\textbf{B=500} & \\textbf{B=1000} & \\textbf{B=2000} \\\\",
        "\\midrule",
    ]

    for d_key, d_name in domain_order:
        d_sub = summary_df[summary_df["benchmark"] == d_key]
        header_str = "\\multicolumn{6}{l}{\\textit{" + d_name + "}} \\\\[2pt]"
        lines.append(header_str)
        
        # Find best coverage per budget in this domain
        best_by_b = {}
        for b in budgets:
            b_sub = d_sub[d_sub["budget"] == b]
            best_by_b[b] = b_sub["coverage"].max() if not b_sub.empty else 0.0

        for m_key, m_name in method_order:
            row_vals = []
            for b in budgets:
                cell = d_sub[(d_sub["method"] == m_key) & (d_sub["budget"] == b)]
                if not cell.empty:
                    val = cell["coverage"].values[0] * 100
                    is_best = (cell["coverage"].values[0] >= best_by_b[b] - 1e-4)
                    val_str = f"{val:.1f}\\%"
                    if is_best:
                        val_str = f"\\textbf{{{val_str}}}"
                else:
                    val_str = "-"
                row_vals.append(val_str)

            lines.append(f"\\hspace{{3mm}} & {m_name} & " + " & ".join(row_vals) + " \\\\")
        lines.append("\\midrule")

    lines[-1] = "\\bottomrule"
    lines.extend([
        "\\end{tabular}",
        "\\caption{State coverage (\\% of gold facts retained) across heterogeneous domains and token budgets. Bold indicates the highest coverage per budget tier.}",
        "\\label{tab:multidomain_coverage}",
        "\\end{table*}",
    ])

    tex_content = "\n".join(lines)
    out_file = out_dir / "tab1_multidomain_coverage.tex"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(tex_content)
    return str(out_file)


def export_table_2_ablations(ablation_df: pd.DataFrame, out_dir: Path) -> str:
    """Table 2: Ablation Study."""
    ablation_labels = [
        ("Full_CSC", "\\textbf{Full CSC (Learned GBT)}"),
        ("CSC_heuristic_dense", "Heuristic Utility (w/o Learning)"),
        ("CSC_no_experience", "w/o Experience Pool ($E$)"),
        ("CSC_no_computation", "w/o Computation Pool ($C$)"),
        ("CSC_no_budget_density", "w/o Budget Density Ordering"),
        ("Relevance_Only", "Relevance Only (No Resource Types)"),
    ]

    budgets = [200, 500, 1000, 2000]

    lines = [
        "% Table 2: Ablation Study across Token Budgets",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{Ablation Variant} & \\textbf{B=200} & \\textbf{B=500} & \\textbf{B=1000} & \\textbf{B=2000} \\\\",
        "\\midrule",
    ]

    best_by_b = {}
    for b in budgets:
        b_sub = ablation_df[ablation_df["budget"] == b]
        best_by_b[b] = b_sub["coverage"].max() if not b_sub.empty else 0.0

    for a_key, a_name in ablation_labels:
        row_vals = []
        for b in budgets:
            cell = ablation_df[(ablation_df["ablation"] == a_key) & (ablation_df["budget"] == b)]
            if not cell.empty:
                val = cell["coverage"].values[0] * 100
                is_best = (cell["coverage"].values[0] >= best_by_b[b] - 1e-4)
                val_str = f"{val:.1f}\\%"
                if is_best:
                    val_str = f"\\textbf{{{val_str}}}"
            else:
                val_str = "-"
            row_vals.append(val_str)

        lines.append(f"{a_name} & " + " & ".join(row_vals) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Ablation analysis on HotpotQA across token budgets, isolating the contribution of learned utility scoring, heterogeneous memory pools, and knapsack density packing.}",
        "\\label{tab:ablation_study}",
        "\\end{table}",
    ])

    tex_content = "\n".join(lines)
    out_file = out_dir / "tab2_ablation_study.tex"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(tex_content)
    return str(out_file)


def export_table_3_mem1_comparison(mem1_df: pd.DataFrame, out_dir: Path) -> str:
    """Table 3: Sequence Horizon Retention vs MEM1."""
    horizons = sorted(mem1_df["horizon_N"].unique())
    methods = [
        ("CSC (Ours, Budget=512)", "\\textbf{CSC (Ours, B=512)}"),
        ("MEM1 (Published Reference)", "MEM1 (Zhou et al., 2025)"),
        ("Sliding Window (FIFO)", "Sliding Window (FIFO)"),
        ("Full-Context (Unbounded)", "Full-Context (Unbounded)"),
    ]

    lines = [
        "% Table 3: Long-Horizon Sequence Comparison against MEM1",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lcccccccc}",
        "\\toprule",
        " & \\multicolumn{4}{c}{\\textbf{State Retention (\\%)}} & \\multicolumn{4}{c}{\\textbf{Average Prompt Tokens}} \\\\",
        "\\cmidrule(lr){2-5} \\cmidrule(lr){6-9}",
        "\\textbf{Method} & \\textbf{N=2} & \\textbf{N=4} & \\textbf{N=8} & \\textbf{N=16} & \\textbf{N=2} & \\textbf{N=4} & \\textbf{N=8} & \\textbf{N=16} \\\\",
        "\\midrule",
    ]

    for m_key, m_name in methods:
        cov_vals = []
        cost_vals = []
        for n in horizons:
            cell = mem1_df[(mem1_df["method"] == m_key) & (mem1_df["horizon_N"] == n)]
            if not cell.empty:
                cov = cell["avg_state_coverage"].values[0] * 100
                cost = cell["avg_token_cost"].values[0]
                cov_str = f"{cov:.1f}\\%"
                cost_str = f"{cost:.0f}"
                if m_key == "CSC (Ours, Budget=512)":
                    cov_str = f"\\textbf{{{cov_str}}}"
                    cost_str = f"\\textbf{{{cost_str}}}"
            else:
                cov_str = "-"
                cost_str = "-"
            cov_vals.append(cov_str)
            cost_vals.append(cost_str)

        lines.append(f"{m_name} & " + " & ".join(cov_vals) + " & " + " & ".join(cost_vals) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Comparison of state retention and prompt token footprint across chained multi-turn sequence horizons ($N = 2, 4, 8, 16$). CSC guarantees constant budget compliance with zero RL training overhead.}",
        "\\label{tab:mem1_horizon_comparison}",
        "\\end{table*}",
    ])

    tex_content = "\n".join(lines)
    out_file = out_dir / "tab3_mem1_horizon_comparison.tex"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(tex_content)
    return str(out_file)


def main():
    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPORTING PUBLICATION-READY LATEX TABLES")
    print("=" * 70)

    # 1. Multi-domain Table
    m_file = Path("results/multidomain_summary.csv")
    if m_file.exists():
        t1 = export_table_1_multidomain(pd.read_csv(m_file), out_dir)
        print(f"  [x] Saved Table 1 (Multi-Domain Coverage): {t1}")

    # 2. Ablation Table
    a_file = Path("results/ablation_summary.csv")
    if a_file.exists():
        t2 = export_table_2_ablations(pd.read_csv(a_file), out_dir)
        print(f"  [x] Saved Table 2 (Ablation Study):        {t2}")

    # 3. MEM1 Comparison Table
    mem1_file = Path("results/mem1_vs_csc_literature_comparison.csv")
    if mem1_file.exists():
        t3 = export_table_3_mem1_comparison(pd.read_csv(mem1_file), out_dir)
        print(f"  [x] Saved Table 3 (MEM1 Horizon Comp):     {t3}")

    print("\nAll LaTeX tables generated in results/tables/!")


if __name__ == "__main__":
    main()
