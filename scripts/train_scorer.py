"""
Training and Validation Script for the Learned Utility Scorer.

Directly addresses Reviewer rTKQ (W1/Q1, Q3), Reviewer LsqK, and Reviewer vD7Q (Q1):
1. Trains with GroupKFold grouped strictly by task_id across all benchmark domains.
2. Evaluates the Leak-Free model (no utility_hint or utility_density label leakage).
3. Evaluates the Controlled No-Type Ablation (retraining the same GBT without type features).
4. Evaluates the Legacy Hint Model to document the exact historical delta.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from csc.dataset import load_benchmark
from csc.learned_scorer import TrainedUtilityScorer, MODEL_DIR


def main():
    print("=" * 80)
    print("COGNITIVE STATE COMPILATION — LEARNED SCORER RESEARCH SUITE")
    print("Strict Task-Disjoint GroupKFold Cross-Validation (No Label Leakage)")
    print("=" * 80)

    # 1. Load all benchmark datasets
    all_tasks = []
    all_resource_map = {}

    for benchmark_name in ["hotpotqa", "nq", "nq_chained", "webshop"]:
        try:
            tasks, resource_map = load_benchmark(name=benchmark_name)
            print(f"  Loaded {benchmark_name:12s}: {len(tasks):4d} tasks, "
                  f"{sum(len(v) for v in resource_map.values()):5d} resources")
            all_tasks.extend(tasks)
            all_resource_map.update(resource_map)
        except Exception as e:
            print(f"  Warning: Could not load {benchmark_name}: {e}")

    total_candidates = sum(len(v) for v in all_resource_map.values())
    print(f"\nAggregated Benchmark Corpus: {len(all_tasks)} tasks, {total_candidates} candidates")

    # 2. Train Primary Model: Unbiased Leak-Free (GroupKFold by task_id, 6 Observable Features)
    print("\n" + "=" * 80)
    print("[ARM 1] PRIMARY MODEL: Unbiased Leak-Free GBDT (GroupKFold by task_id, 6 Features)")
    print("=" * 80)
    primary_scorer = TrainedUtilityScorer(
        model_path=MODEL_DIR / "learned_scorer.joblib",
        feature_mode="no_type",
    )
    _, m_primary, imp_primary = primary_scorer.fit(
        all_tasks, all_resource_map, feature_mode="no_type", save=True
    )

    # 3. Train Controlled Ablation: Type-Conditioned Model
    print("\n" + "=" * 80)
    print("[ARM 2] CONTROLLED ABLATION: Type-Conditioned Model (9 Observable Features)")
    print("=" * 80)
    type_scorer = TrainedUtilityScorer(
        model_path=MODEL_DIR / "learned_scorer_type.joblib",
        feature_mode="leak_free",
    )
    _, m_type, imp_type = type_scorer.fit(
        all_tasks, all_resource_map, feature_mode="leak_free", save=True
    )


    # 4. Train Historical Reference: Legacy Hint Model
    print("\n" + "=" * 80)
    print("[ARM 3] HISTORICAL REFERENCE: Legacy Hint Model (Stratified/Hints)")
    print("=" * 80)
    legacy_scorer = TrainedUtilityScorer(
        model_path=MODEL_DIR / "learned_scorer_legacy.joblib",
        feature_mode="legacy_hint",
    )
    _, m_legacy, imp_legacy = legacy_scorer.fit(
        all_tasks, all_resource_map, feature_mode="legacy_hint", save=True
    )

    # 5. Comparative Validation Summary Table
    print("\n" + "=" * 85)
    print(f"{'EVALUATION ARM':<24} | {'ACCURACY':<16} | {'PRECISION':<10} | {'RECALL':<8} | {'F1 SCORE':<16} | {'ROC-AUC':<8}")
    print("=" * 85)

    arms = [
        ("Leak-Free Primary", m_primary),
        ("Type-Conditioned", m_type),
        ("Legacy (With Hint)", m_legacy),
    ]
    for name, m in arms:
        acc_str = f"{m['accuracy']:.4f} (+/-{m['accuracy_std']:.3f})"
        f1_str = f"{m['f1']:.4f} (+/-{m['f1_std']:.3f})"
        auc_str = f"{m['roc_auc']:.4f}"
        print(f"{name:<24} | {acc_str:<16} | {m['precision']:<10.4f} | {m['recall']:<8.4f} | {f1_str:<16} | {auc_str:<8}")
    print("=" * 85)

    # 6. Feature Importance Breakdown for Primary Leak-Free Model
    print("\n" + "=" * 80)
    print("PRIMARY LEAK-FREE MODEL FEATURE IMPORTANCES")
    print("=" * 80)
    for name, imp in sorted(imp_primary.items(), key=lambda x: x[1], reverse=True):
        bar = "#" * int(imp * 45)
        print(f"  {name:25s}  {imp:.4f}  {bar}")

    print("\nTraining completed successfully! Primary model written to models/learned_scorer.joblib")


if __name__ == "__main__":
    main()
