"""
Learned Utility Scorer for Cognitive State Compilation (CSC).

Trains a Gradient Boosted Tree classifier on (task, resource) feature vectors
to predict P(resource is useful | task). The predicted probability serves as
the utility score for budgeted resource selection.

Addressed Reviewer Critiques (rTKQ W1/Q1, LsqK, vD7Q Q1):
1. Task-Disjoint Cross-Validation: Enforces GroupKFold grouped strictly by task_id,
   guaranteeing test tasks are never seen during training.
2. Leak-Free Feature Mode: Strictly removes utility_hint and utility_density,
   eliminating any label leakage and forcing the scorer to rely entirely on
   observable textual, semantic, lexical, and structural features.
3. Controlled Ablations: Supports 'no_type' feature mode to isolate the exact
   contribution of resource-type awareness.
"""

import re
import os
from abc import ABC
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .types import Resource, Task
from .scorers import UtilityScorer, DenseUtilityScorer


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "learned_scorer.joblib"


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def _extract_keywords(text: str) -> set:
    words = re.findall(r"\b[a-zA-Z0-9]{3,}\b", text.lower())
    stopwords = {
        "the", "and", "was", "were", "what", "which", "who",
        "when", "where", "with", "from", "for", "that", "this",
    }
    return {w for w in words if w not in stopwords}


FEATURE_NAMES_LEAK_FREE = [
    "tfidf_word_sim",
    "tfidf_char_sim",
    "keyword_overlap",
    "title_match",
    "relative_sim_rank",
    "is_knowledge",
    "is_experience",
    "is_computation",
    "normalized_token_cost",
]

FEATURE_NAMES_NO_TYPE = [
    "tfidf_word_sim",
    "tfidf_char_sim",
    "keyword_overlap",
    "title_match",
    "relative_sim_rank",
    "normalized_token_cost",
]

FEATURE_NAMES_LEGACY = [
    "tfidf_unigram_sim",
    "tfidf_bigram_sim",
    "keyword_overlap",
    "is_knowledge",
    "is_experience",
    "is_computation",
    "utility_hint",
    "normalized_token_cost",
    "utility_density",
]

# Default export for backward compatibility
FEATURE_NAMES = FEATURE_NAMES_LEAK_FREE


def extract_batch_features(
    task: Task,
    resources: List[Resource],
    feature_mode: str = "leak_free",
) -> np.ndarray:
    """
    Extracts fixed-size feature vectors for a batch of resources under a given task.
    
    Modes:
      - 'leak_free' (default, 9-dim): observable lexical, subword, keyword, title, rank, type, and cost features.
      - 'no_type' (6-dim): leak-free features without resource-type indicators (for controlled ablation).
      - 'legacy_hint' (9-dim): historical features including utility_hint and utility_density.
    """
    if not resources:
        dim = 6 if feature_mode == "no_type" else 9
        return np.empty((0, dim), dtype=np.float64)

    texts = [task.query] + [r.text for r in resources]

    if feature_mode == "legacy_hint":
        try:
            vec_uni = TfidfVectorizer(stop_words="english", ngram_range=(1, 1))
            X_uni = vec_uni.fit_transform(texts)
            sims_uni = cosine_similarity(X_uni[0:1], X_uni[1:]).ravel()
        except Exception:
            sims_uni = np.zeros(len(resources))

        try:
            vec_bi = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            X_bi = vec_bi.fit_transform(texts)
            sims_bi = cosine_similarity(X_bi[0:1], X_bi[1:]).ravel()
        except Exception:
            sims_bi = np.zeros(len(resources))

        query_kw = _extract_keywords(task.query)
        max_cost = max((r.token_cost for r in resources), default=1)

        rows = []
        for i, r in enumerate(resources):
            res_kw = _extract_keywords(r.text)
            overlap = len(query_kw & res_kw) / max(1, len(query_kw))
            rows.append([
                sims_uni[i],
                sims_bi[i],
                overlap,
                float(r.resource_type == "knowledge"),
                float(r.resource_type == "experience"),
                float(r.resource_type == "computation"),
                float(r.utility_hint),
                r.token_cost / max(1, max_cost),
                float(r.utility_hint) / max(1, r.token_cost),
            ])
        return np.array(rows, dtype=np.float64)

    # Leak-free feature extraction (observable only)
    try:
        vec_word = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
        X_w = vec_word.fit_transform(texts)
        sims_w = cosine_similarity(X_w[0:1], X_w[1:]).ravel()
    except Exception:
        sims_w = np.zeros(len(resources))

    try:
        vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), max_features=3000)
        X_c = vec_char.fit_transform(texts)
        sims_c = cosine_similarity(X_c[0:1], X_c[1:]).ravel()
    except Exception:
        sims_c = np.zeros(len(resources))

    query_kw = _extract_keywords(task.query)
    max_cost = max((r.token_cost for r in resources), default=1)
    ranks = np.argsort(np.argsort(-sims_w)) / max(1, len(resources) - 1)

    rows = []
    for i, r in enumerate(resources):
        res_kw = _extract_keywords(r.text)
        overlap = len(query_kw & res_kw) / max(1, len(query_kw))

        title_match = 0.0
        if ":" in r.text:
            title = r.text.split(":", 1)[0].lower()
            t_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", title))
            if t_words and (t_words & query_kw):
                title_match = len(t_words & query_kw) / max(1, len(t_words))

        norm_cost = r.token_cost / max(1, max_cost)

        if feature_mode == "no_type":
            rows.append([
                sims_w[i],
                sims_c[i],
                overlap,
                title_match,
                ranks[i],
                norm_cost,
            ])
        else:  # 'leak_free' (9-dim)
            rows.append([
                sims_w[i],
                sims_c[i],
                overlap,
                title_match,
                ranks[i],
                float(r.resource_type == "knowledge"),
                float(r.resource_type == "experience"),
                float(r.resource_type == "computation"),
                norm_cost,
            ])

    return np.array(rows, dtype=np.float64)


def extract_features(
    task: Task,
    resource: Resource,
    all_resources: List[Resource],
    feature_mode: str = "leak_free",
) -> np.ndarray:
    """Extracts features for a single (task, resource) pair."""
    return extract_batch_features(task, [resource], feature_mode=feature_mode)[0]


def build_training_data(
    tasks: List[Task],
    resource_map: Dict[str, List[Resource]],
    feature_mode: str = "leak_free",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds feature matrix X, binary label vector y, and task-group array for GroupKFold.

    Returns:
        X: (N_samples, N_features) float array
        y: (N_samples,) float array where 1.0 indicates resource contains a gold required fact
        groups: (N_samples,) string array containing task_ids for strict task-disjoint splitting
    """
    X_rows = []
    y_rows = []
    groups = []

    for task in tasks:
        resources = resource_map.get(task.task_id, [])
        if not resources:
            continue
        feats = extract_batch_features(task, resources, feature_mode=feature_mode)
        X_rows.append(feats)

        for resource in resources:
            groups.append(task.task_id)
            rtext = _normalize(resource.text)
            is_useful = any(_normalize(fact) in rtext for fact in task.required_facts)
            y_rows.append(float(is_useful))

    X = np.vstack(X_rows) if X_rows else np.empty((0, 9))
    y = np.array(y_rows, dtype=np.float64)
    groups = np.array(groups)
    return X, y, groups


class TrainedUtilityScorer(UtilityScorer):
    """
    Learned utility scorer using a trained GradientBoostingClassifier.
    Supports task-disjoint GroupKFold training, leak-free feature representations,
    and controlled ablation modes.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        feature_mode: str = "leak_free",
    ):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.feature_mode = feature_mode
        self._model = None
        self._model_metadata = {}
        self._fallback = DenseUtilityScorer(
            type_priors={"knowledge": 0.05, "experience": 0.15, "computation": 0.20},
            entity_weight=0.25,
            hint_weight=0.35,
        )

    def _load_model(self) -> bool:
        if self._model is not None:
            return True

        if not self.model_path.exists():
            return False

        try:
            import joblib
            loaded = joblib.load(self.model_path)
            if isinstance(loaded, dict) and "model" in loaded:
                self._model = loaded["model"]
                self.feature_mode = loaded.get("feature_mode", self.feature_mode)
                self._model_metadata = loaded
            else:
                self._model = loaded
            return True
        except Exception as e:
            print(f"Warning: Failed to load learned scorer model: {e}")
            return False

    def score(self, task: Task, resources: List[Resource]) -> List[Tuple[Resource, float]]:
        if not resources:
            return []

        if not self._load_model():
            return self._fallback.score(task, resources)

        features = extract_batch_features(task, resources, feature_mode=self.feature_mode)
        probabilities = self._model.predict_proba(features)[:, 1]

        scored = list(zip(resources, [float(p) for p in probabilities]))
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def fit(
        self,
        tasks: List[Task],
        resource_map: Dict[str, List[Resource]],
        feature_mode: Optional[str] = None,
        save: bool = True,
        n_splits: int = 5,
        random_state: int = 42,
    ):
        """
        Train the scorer model with task-disjoint GroupKFold cross-validation.
        
        Guarantees that test tasks in every cross-validation fold were never seen
        during training, completely preventing label and task leakage.
        """
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import GroupKFold, cross_validate
        import joblib

        mode = feature_mode or self.feature_mode
        self.feature_mode = mode

        X, y, groups = build_training_data(tasks, resource_map, feature_mode=mode)
        feature_names = (
            FEATURE_NAMES_NO_TYPE if mode == "no_type"
            else FEATURE_NAMES_LEGACY if mode == "legacy_hint"
            else FEATURE_NAMES_LEAK_FREE
        )

        n_groups = len(set(groups))
        effective_splits = min(n_splits, n_groups)

        model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=random_state,
        )

        # Strict task-disjoint GroupKFold
        gkf = GroupKFold(n_splits=effective_splits)
        cv_results = cross_validate(
            model, X, y, groups=groups, cv=gkf,
            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
            return_train_score=False,
        )

        metrics = {
            "accuracy": float(cv_results["test_accuracy"].mean()),
            "accuracy_std": float(cv_results["test_accuracy"].std()),
            "precision": float(cv_results["test_precision"].mean()),
            "recall": float(cv_results["test_recall"].mean()),
            "f1": float(cv_results["test_f1"].mean()),
            "f1_std": float(cv_results["test_f1"].std()),
            "roc_auc": float(cv_results["test_roc_auc"].mean()),
            "roc_auc_std": float(cv_results["test_roc_auc"].std()),
            "n_samples": int(X.shape[0]),
            "n_tasks": n_groups,
            "feature_mode": mode,
        }

        # Fit final model on all data
        model.fit(X, y)
        self._model = model

        importances = dict(zip(feature_names, [float(v) for v in model.feature_importances_]))
        self._model_metadata = {
            "model": model,
            "feature_mode": mode,
            "feature_names": feature_names,
            "metrics": metrics,
            "importances": importances,
        }

        if save:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self._model_metadata, self.model_path)
            print(f"Saved trained model ({mode}) to {self.model_path}")

        return model, metrics, importances
