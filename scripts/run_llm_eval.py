import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from csc.dataset import load_mem1_benchmark, build_demo_dataset
from csc.baselines import (
    full_context,
    standard_rag,
    memory_augmented_rag,
    relevance_only,
    mem1_style_recurrent,
)
from csc.compiler import CognitiveStateCompiler
from csc.scorers import DenseUtilityScorer
from csc.evaluation import evaluate_state, evaluate_prediction
from csc.llm import OpenAICompatibleLLM, MockLLM


def run_llm_evaluation(
    limit: int = 10,
    mock: bool = False,
    api_key: str = None,
    base_url: str = None,
    model: str = None,
):
    mem1_file = Path("data/mem1_hotpotqa_benchmark.json")

    if mem1_file.exists():
        print(f"Loading MEM1 HotpotQA Benchmark ({mem1_file})...")
        tasks, resource_map = load_mem1_benchmark(file_path=str(mem1_file), limit=limit)
        budgets = [200, 500, 1000, 2000, 4000]
    else:
        print("Using default toy dataset...")
        tasks, resource_map = build_demo_dataset()
        budgets = [20, 40, 80, 120]

    # Initialize LLM
    if mock:
        print("Initializing MockLLM (offline dry-run)...")
        answer_map = {t.query: t.answer for t in tasks}
        llm = MockLLM(answer_map=answer_map)
    else:
        print(f"Initializing Live LLM Client ({model or os.getenv('LLM_MODEL', 'gpt-4o-mini')})...")
        try:
            llm = OpenAICompatibleLLM(api_key=api_key, base_url=base_url, model=model)
        except ValueError as e:
            print(f"Warning: {e}")
            print("Falling back to MockLLM. Use --api_key or set LLM_API_KEY to evaluate live models.")
            answer_map = {t.query: t.answer for t in tasks}
            llm = MockLLM(answer_map=answer_map)

    rows = []
    total_runs = len(budgets) * len(tasks) * 6
    run_idx = 0

    print(f"Running LLM evaluation across {len(tasks)} tasks and {len(budgets)} budgets ({total_runs} total evaluations)...")

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
                run_idx += 1
                state = fn()
                state_metrics = evaluate_state(task, state)

                # Generate answer with LLM and measure latency
                t0 = time.time()
                try:
                    prediction = llm.generate(state.text)
                except Exception as e:
                    prediction = f"Error: {e}"
                latency = time.time() - t0

                qa_metrics = evaluate_prediction(task, prediction)

                rows.append({
                    "task_id": task.task_id,
                    "budget": budget,
                    "method": method,
                    "prediction": prediction,
                    "ground_truth": task.answer,
                    "latency_sec": round(latency, 3),
                    **state_metrics,
                    **qa_metrics,
                })

    df = pd.DataFrame(rows)
    out = Path("results")
    out.mkdir(exist_ok=True)
    df.to_csv(out / "llm_eval_results.csv", index=False)

    summary = (
        df.groupby(["method", "budget"])
        .agg(
            coverage=("coverage", "mean"),
            exact_match=("exact_match", "mean"),
            f1=("f1", "mean"),
            substring_match=("substring_match", "mean"),
            token_cost=("token_cost", "mean"),
            latency_sec=("latency_sec", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out / "llm_eval_summary.csv", index=False)

    print("\n" + "="*80)
    print("LLM EVALUATION SUMMARY")
    print("="*80)
    print(summary.to_string(index=False))
    print("\nSaved detailed logs to results/llm_eval_results.csv and results/llm_eval_summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run live or mock LLM evaluation across baselines.")
    parser.add_argument("--limit", type=int, default=10, help="Number of benchmark tasks to evaluate")
    parser.add_argument("--mock", action="store_true", help="Run with MockLLM (no API calls)")
    parser.add_argument("--api_key", type=str, default=None, help="LLM API Key")
    parser.add_argument("--base_url", type=str, default=None, help="LLM API Base URL")
    parser.add_argument("--model", type=str, default=None, help="LLM Model name")
    args = parser.parse_args()

    run_llm_evaluation(
        limit=args.limit,
        mock=args.mock,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )
