from pathlib import Path
import pandas as pd

from csc.dataset import build_demo_dataset
from csc.baselines import (
    full_context,
    standard_rag,
    memory_augmented_rag,
    relevance_only,
    mem1_style_recurrent,
)
from csc.compiler import CognitiveStateCompiler
from csc.scorers import TfidfUtilityScorer
from csc.evaluation import evaluate_state


def run():
    tasks, resource_map = build_demo_dataset()

    # Small synthetic budgets for the offline smoke test.
    # Replace with real token budgets after switching to a real tokenizer.
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
                    TfidfUtilityScorer(
                        {"knowledge": 0.02, "experience": 0.02, "computation": 0.02}
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
    run()
