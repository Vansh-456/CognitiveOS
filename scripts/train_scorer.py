"""
Training script for the Learned Utility Scorer.

Loads all benchmark datasets, generates (task, resource) training pairs with
ground-truth labels from required_facts coverage, trains a GBT classifier
with 5-fold stratified cross-validation, and saves the model.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from csc.dataset import load_benchmark
from csc.learned_scorer import TrainedUtilityScorer, FEATURE_NAMES


def main():
    print("=" * 70)
    print("TRAINING LEARNED UTILITY SCORER")
    print("=" * 70)

    # Load all benchmark datasets
    all_tasks = []
    all_resource_map = {}

    for benchmark_name in ["hotpotqa", "nq", "nq_chained", "webshop"]:
        try:
            tasks, resource_map = load_benchmark(name=benchmark_name)
            print(f"  Loaded {benchmark_name}: {len(tasks)} tasks, "
                  f"{sum(len(v) for v in resource_map.values())} resources")
            all_tasks.extend(tasks)
            all_resource_map.update(resource_map)
        except Exception as e:
            print(f"  Warning: Could not load {benchmark_name}: {e}")

    print(f"\nTotal: {len(all_tasks)} tasks, "
          f"{sum(len(v) for v in all_resource_map.values())} resources")

    # Train
    print("\n" + "=" * 70)
    print("TRAINING WITH 5-FOLD STRATIFIED CROSS-VALIDATION (Option B)")
    print("=" * 70 + "\n")

    scorer = TrainedUtilityScorer()
    model, metrics, importances = scorer.fit(all_tasks, all_resource_map, save=True)

    # Report metrics
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 70)
    print(f"  Accuracy:  {metrics['accuracy']:.4f} (+/- {metrics['accuracy_std']:.4f})")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f} (+/- {metrics['f1_std']:.4f})")

    # Report feature importances
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCES (Learned Weights)")
    print("=" * 70)
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    for name, imp in sorted_imp:
        bar = "#" * int(imp * 50)
        print(f"  {name:25s}  {imp:.4f}  {bar}")

    print("\nDone! Model saved to models/learned_scorer.joblib")


if __name__ == "__main__":
    main()
