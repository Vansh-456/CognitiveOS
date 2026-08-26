import argparse
import sys
from pathlib import Path
from typing import List

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from csc.dataset import load_benchmark
from csc.baselines import (
    full_context,
    standard_rag,
    memory_augmented_rag,
    relevance_only,
    mem1_style_recurrent,
)
from csc.compiler import CognitiveStateCompiler
from csc.scorers import DenseUtilityScorer
from csc.evaluation import evaluate_state


def run_benchmark_eval(benchmark_name: str, limit: int = 15) -> pd.DataFrame:
    print(f"\n{'='*70}\nEVALUATING BENCHMARK DOMAIN: {benchmark_name.upper()}\n{'='*70}")
    tasks, resource_map = load_benchmark(name=benchmark_name, limit=limit)
    budgets = [200, 500, 1000, 2000, 4000]

    rows = []
    for budget in budgets:
        for task in tasks:
            resources = resource_map[task.task_id]

            methods = {
                "B0_full_context": lambda: full_context(task, resources, budget),
                "B1_standard_rag": lambda: standard_rag(task, resources, budget),
                "B2_memory_rag": lambda: memory_augmented_rag(task, resources, budget),
                "B3_relevance_only": lambda: relevance_only(task, resources, budget),
                "B4_csc": lambda: CognitiveStateCompiler(
                    DenseUtilityScorer(
                        type_priors={"knowledge": 0.05, "experience": 0.15, "computation": 0.20},
                        entity_weight=0.25,
                        hint_weight=0.35,
                    ),
                    budget,
                ).compile(task, resources),
                "B5_mem1_style": lambda: mem1_style_recurrent(task, resources, budget),
            }

            for method, fn in methods.items():
                state = fn()
                m = evaluate_state(task, state)
                rows.append({
                    "benchmark": benchmark_name,
                    "task_id": task.task_id,
                    "budget": budget,
                    "method": method,
                    **m,
                    "resource_types": ",".join(sorted({r.resource_type for r in state.resources})),
                })

    df = pd.DataFrame(rows)
    return df


def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluation sweep across benchmark domains.")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="hotpotqa",
        choices=["hotpotqa", "nq", "nq_chained", "webshop", "all"],
        help="Benchmark domain to evaluate: 'hotpotqa', 'nq', 'nq_chained', 'webshop', or 'all'",
    )
    parser.add_argument("--limit", type=int, default=15, help="Number of benchmark tasks per domain")
    args = parser.parse_args()

    benchmarks_to_run = (
        ["hotpotqa", "nq", "nq_chained", "webshop"] if args.benchmark == "all" else [args.benchmark]
    )

    all_dfs = []
    for b_name in benchmarks_to_run:
        df_b = run_benchmark_eval(b_name, limit=args.limit)
        all_dfs.append(df_b)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    out = Path("results")
    out.mkdir(exist_ok=True)
    combined_df.to_csv(out / "multidomain_results.csv", index=False)

    summary = (
        combined_df.groupby(["benchmark", "method", "budget"])
        .agg(
            coverage=("coverage", "mean"),
            state_precision=("state_precision", "mean"),
            token_cost=("token_cost", "mean"),
            selected_resources=("selected_resources", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out / "multidomain_summary.csv", index=False)

    print("\n" + "="*80)
    print("MULTI-DOMAIN BENCHMARK EVALUATION SUMMARY (500-Token Constrained Budget)")
    print("="*80)
    sub500 = summary[summary["budget"] == 500].sort_values(["benchmark", "coverage"], ascending=[True, False])
    print(sub500.to_string(index=False))
    print("\nSaved results/multidomain_results.csv and results/multidomain_summary.csv")


if __name__ == "__main__":
    main()
