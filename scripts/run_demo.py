import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from csc.dataset import build_demo_dataset, load_mem1_benchmark
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


def run(use_mem1: bool = False, limit: int = 20):
    mem1_file = Path("data/mem1_hotpotqa_benchmark.json")

    if use_mem1 or mem1_file.exists():
        print(f"Loading MEM1 HotpotQA Benchmark ({mem1_file})...")
        tasks, resource_map = load_mem1_benchmark(file_path=str(mem1_file), limit=limit)
        budgets = [200, 500, 1000, 2000, 4000]
    else:
        print("Using default toy dataset (3 synthetic tasks)...")
        tasks, resource_map = build_demo_dataset()
        budgets = [20, 40, 80, 120]

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
                    "task_id": task.task_id,
                    "budget": budget,
                    "method": method,
                    **m,
                    "resource_types": ",".join(sorted({r.resource_type for r in state.resources})),
                })

    df = pd.DataFrame(rows)
    out = Path("results")
    out.mkdir(exist_ok=True)
    df.to_csv(out / "demo_results.csv", index=False)

    summary = (
        df.groupby(["method", "budget"])
        .agg(
            coverage=("coverage", "mean"),
            state_precision=("state_precision", "mean"),
            token_cost=("token_cost", "mean"),
            selected_resources=("selected_resources", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out / "demo_summary.csv", index=False)

    print(summary.to_string(index=False))
    print("\nSaved results/demo_results.csv and results/demo_summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline evaluation sweep.")
    parser.add_argument("--use_mem1", action="store_true", help="Force using MEM1 benchmark dataset")
    parser.add_argument("--limit", type=int, default=20, help="Number of benchmark tasks to evaluate")
    args = parser.parse_args()

    run(use_mem1=args.use_mem1, limit=args.limit)
