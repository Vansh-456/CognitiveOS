"""
CSC Ablation Study Suite.

Directly addresses Reviewer rTKQ (W3, Q3), Reviewer LsqK, and Reviewer vD7Q (Q3):
1. Evaluates ablations on both:
   (A) Standard Multi-hop QA (HotpotQA): Demonstrating budget behavior on textual retrieval.
   (B) Dedicated Heterogeneous Cross-Resource Benchmark: Demonstrating that removing
       Experience or Computation pools leads to severe, quantifiable coverage degradation.
2. Evaluates:
   - Full CSC (Learned GBT across {K, E, C})
   - CSC without Experience (K + C only)
   - CSC without Computation (K + E only)
   - Standard RAG (Knowledge-only baseline)
   - CSC without Budget Density (Raw score greedy selection)
   - Controlled No-Type Ablation model
   - Relevance-Only baseline
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from csc.dataset import load_benchmark
from csc.compiler import CognitiveStateCompiler
from csc.scorers import DenseUtilityScorer, RelevanceOnlyScorer
from csc.learned_scorer import TrainedUtilityScorer, MODEL_DIR
from csc.selector import BudgetedSelector
from csc.composer import StateComposer
from csc.evaluation import evaluate_state


def run_ablation_experiment(
    benchmark_name: str = "cross_resource",
    limit: int = 25,
    output_prefix: str = "ablation_cross_resource",
):
    print("=" * 80)
    print(f"RUNNING ABLATION STUDY: {benchmark_name.upper()} (Limit: {limit})")
    print("=" * 80)

    tasks, resource_map = load_benchmark(name=benchmark_name, limit=limit)
    print(f"Loaded {len(tasks)} tasks.")

    if benchmark_name == "cross_resource":
        budgets = [50, 100, 150, 250]
    else:
        budgets = [200, 500, 1000, 2000, 4000]

    learned_scorer = TrainedUtilityScorer()
    notype_scorer = TrainedUtilityScorer(
        model_path=MODEL_DIR / "learned_scorer_notype.joblib",
        feature_mode="no_type",
    )
    heuristic_scorer = DenseUtilityScorer(
        type_priors={"knowledge": 0.05, "experience": 0.15, "computation": 0.20},
        entity_weight=0.25,
        hint_weight=0.35,
    )
    rel_scorer = RelevanceOnlyScorer()

    rows = []

    for budget in budgets:
        for task in tasks:
            resources = resource_map[task.task_id]

            # 1. Full CSC (Learned GBT over all pools {K, E, C})
            full_csc_state = CognitiveStateCompiler(learned_scorer, budget).compile(task, resources)

            # 2. CSC without Experience (K + C only)
            no_exp_resources = [r for r in resources if r.resource_type != "experience"]
            no_exp_state = CognitiveStateCompiler(learned_scorer, budget).compile(task, no_exp_resources)

            # 3. CSC without Computation (K + E only)
            no_comp_resources = [r for r in resources if r.resource_type != "computation"]
            no_comp_state = CognitiveStateCompiler(learned_scorer, budget).compile(task, no_comp_resources)

            # 4. Standard RAG (Knowledge-only baseline)
            rag_resources = [r for r in resources if r.resource_type == "knowledge"]
            rag_state = CognitiveStateCompiler(learned_scorer, budget).compile(task, rag_resources)

            # 5. CSC without Budget Density (greedy by utility score, ignoring token cost density)
            scored = learned_scorer.score(task, resources)
            unconstrained_selected = []
            rem = budget
            for r, s in scored:
                if r.token_cost <= rem:
                    unconstrained_selected.append((r, s))
                    rem -= r.token_cost
            no_density_state = StateComposer().compose(unconstrained_selected, task.query, budget)

            # 6. Controlled Model Ablation: No-Type Model
            notype_state = CognitiveStateCompiler(notype_scorer, budget).compile(task, resources)

            # 7. Relevance-Only (unweighted TF-IDF matching)
            rel_scored = rel_scorer.score(task, resources)
            rel_selected = BudgetedSelector(budget).select(rel_scored)
            rel_state = StateComposer().compose(rel_selected, task.query, budget)

            ablations = {
                "Full_CSC": full_csc_state,
                "CSC_no_experience": no_exp_state,
                "CSC_no_computation": no_comp_state,
                "Standard_RAG_Knowledge_Only": rag_state,
                "CSC_no_density_ranking": no_density_state,
                "CSC_controlled_no_type": notype_state,
                "Relevance_Only": rel_state,
            }

            for name, state in ablations.items():
                m = evaluate_state(task, state)
                rows.append({
                    "benchmark": benchmark_name,
                    "task_id": task.task_id,
                    "budget": budget,
                    "ablation": name,
                    **m,
                })

    df = pd.DataFrame(rows)
    out = Path("results")
    out.mkdir(exist_ok=True)
    df.to_csv(out / f"{output_prefix}_results.csv", index=False)

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
    summary.to_csv(out / f"{output_prefix}_summary.csv", index=False)

    print("\n" + "=" * 80)
    print(f"ABLATION STUDY SUMMARY: {benchmark_name.upper()}")
    print("=" * 80)
    print(summary.to_string(index=False))
    print(f"\nSaved results to results/{output_prefix}_results.csv and results/{output_prefix}_summary.csv")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run CSC ablation study.")
    parser.add_argument("--dataset", type=str, default="cross_resource", choices=["cross_resource", "hotpotqa", "all"], help="Dataset to evaluate on")
    parser.add_argument("--limit", type=int, default=25, help="Number of benchmark tasks")
    args = parser.parse_args()

    if args.dataset in ("cross_resource", "all"):
        run_ablation_experiment(
            benchmark_name="cross_resource",
            limit=args.limit,
            output_prefix="ablation_cross_resource",
        )

    if args.dataset in ("hotpotqa", "all"):
        run_ablation_experiment(
            benchmark_name="hotpotqa",
            limit=args.limit,
            output_prefix="ablation_hotpotqa",
        )


if __name__ == "__main__":
    main()
