import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from csc.dataset import load_mem1_benchmark, build_demo_dataset
from csc.compiler import CognitiveStateCompiler
from csc.scorers import DenseUtilityScorer, RelevanceOnlyScorer
from csc.selector import BudgetedSelector
from csc.composer import StateComposer
from csc.evaluation import evaluate_state


def run_ablations(limit: int = 15):
    mem1_file = Path("data/mem1_hotpotqa_benchmark.json")

    if mem1_file.exists():
        print(f"Loading MEM1 HotpotQA Benchmark ({mem1_file})...")
        tasks, resource_map = load_mem1_benchmark(file_path=str(mem1_file), limit=limit)
        budgets = [200, 500, 1000, 2000, 4000]
    else:
        print("Using default toy dataset...")
        tasks, resource_map = build_demo_dataset()
        budgets = [20, 40, 80, 120]

    rows = []
    scorer = DenseUtilityScorer(
        type_priors={"knowledge": 0.05, "experience": 0.15, "computation": 0.20},
        entity_weight=0.25,
        hint_weight=0.35,
    )

    print(f"Running ablation study across {len(tasks)} tasks and {len(budgets)} budgets...")

    for budget in budgets:
        for task in tasks:
            resources = resource_map[task.task_id]

            # 1. Full CSC (Knowledge + Experience + Computation)
            full_csc_state = CognitiveStateCompiler(scorer, budget).compile(task, resources)

            # 2. CSC without Experience (K + C only)
            no_exp_resources = [r for r in resources if r.resource_type != "experience"]
            no_exp_state = CognitiveStateCompiler(scorer, budget).compile(task, no_exp_resources)

            # 3. CSC without Computation (K + E only)
            no_comp_resources = [r for r in resources if r.resource_type != "computation"]
            no_comp_state = CognitiveStateCompiler(scorer, budget).compile(task, no_comp_resources)

            # 4. CSC without Budget Constraint (Greedy by score, ignoring token cost density)
            scored = scorer.score(task, resources)
            unconstrained_selected = []
            rem = budget
            for r, s in scored:
                if r.token_cost <= rem:
                    unconstrained_selected.append((r, s))
                    rem -= r.token_cost
            no_budget_state = StateComposer().compose(unconstrained_selected, task.query, budget)

            # 5. Relevance-Only (No resource-type priors)
            rel_scorer = RelevanceOnlyScorer()
            rel_scored = rel_scorer.score(task, resources)
            rel_selected = BudgetedSelector(budget).select(rel_scored)
            rel_state = StateComposer().compose(rel_selected, task.query, budget)

            ablations = {
                "Full_CSC": full_csc_state,
                "CSC_no_experience": no_exp_state,
                "CSC_no_computation": no_comp_state,
                "CSC_no_budget_density": no_budget_state,
                "Relevance_Only": rel_state,
            }

            for name, state in ablations.items():
                m = evaluate_state(task, state)
                rows.append({
                    "task_id": task.task_id,
                    "budget": budget,
                    "ablation": name,
                    **m,
                })

    df = pd.DataFrame(rows)
    out = Path("results")
    out.mkdir(exist_ok=True)
    df.to_csv(out / "ablation_results.csv", index=False)

    summary = (
        df.groupby(["ablation", "budget"])
        .agg(
            coverage=("coverage", "mean"),
            state_precision=("state_precision", "mean"),
            token_cost=("token_cost", "mean"),
            selected_resources=("selected_resources", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out / "ablation_summary.csv", index=False)

    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    print(summary.to_string(index=False))
    print("\nSaved results/ablation_results.csv and results/ablation_summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CSC ablation study.")
    parser.add_argument("--limit", type=int, default=15, help="Number of benchmark tasks")
    args = parser.parse_args()

    run_ablations(limit=args.limit)
