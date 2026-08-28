"""
Learned Utility Scorer for Cognitive State Compilation (CSC).

Trains a Gradient Boosted Tree classifier on (task, resource) feature vectors
to predict P(resource is useful | task). The predicted probability serves as
the utility score for budgeted resource selection.

Training labels are derived from required_facts coverage: a resource is
labeled useful=1 if its text contains any of the task's required_facts.
"""

import re
import os
from abc import ABC
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def extract_features(task: Task, resource: Resource, all_resources: List[Resource]) -> np.ndarray:
    """
    Extracts a fixed-size feature vector for a (task, resource) pair.

    Features (9-dim):
        0: TF-IDF cosine similarity (unigram)
        1: TF-IDF cosine similarity (bigram)
        2: Keyword/entity overlap ratio
        3: Resource type == knowledge (binary)
        4: Resource type == experience (binary)
        5: Resource type == computation (binary)
        6: Utility hint value
        7: Normalized token cost (relative to max in pool)
        8: Utility density (hint / token_cost)
    """
    # TF-IDF unigram similarity
    texts_uni = [task.query, resource.text]
    try:
        vec_uni = TfidfVectorizer(stop_words="english", ngram_range=(1, 1))
        X_uni = vec_uni.fit_transform(texts_uni)
        sim_uni = float(cosine_similarity(X_uni[0:1], X_uni[1:]).ravel()[0])
    except ValueError:
        sim_uni = 0.0

    # TF-IDF bigram similarity
    try:
        vec_bi = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        X_bi = vec_bi.fit_transform(texts_uni)
        sim_bi = float(cosine_similarity(X_bi[0:1], X_bi[1:]).ravel()[0])
    except ValueError:
        sim_bi = 0.0

    # Keyword overlap
    query_kw = _extract_keywords(task.query)
    res_kw = _extract_keywords(resource.text)
    overlap = len(query_kw & res_kw) / max(1, len(query_kw))

    # Resource type one-hot
    is_knowledge = float(resource.resource_type == "knowledge")
    is_experience = float(resource.resource_type == "experience")
    is_computation = float(resource.resource_type == "computation")

    # Utility hint
    hint = float(resource.utility_hint)

    # Normalized token cost
    max_cost = max(r.token_cost for r in all_resources) if all_resources else 1
    norm_cost = resource.token_cost / max(1, max_cost)

    # Utility density
    density = hint / max(1, resource.token_cost)

    return np.array([
        sim_uni, sim_bi, overlap,
        is_knowledge, is_experience, is_computation,
        hint, norm_cost, density,
    ], dtype=np.float64)


FEATURE_NAMES = [
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


def build_training_data(
    tasks: List[Task],
    resource_map: Dict[str, List[Resource]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds feature matrix X and binary label vector y from benchmark data.

    A resource is labeled useful=1 if its text contains any of the task's
    required_facts (same logic as evaluate_state in evaluation.py).
    """
    X_rows = []
    y_rows = []

    for task in tasks:
        resources = resource_map.get(task.task_id, [])
        for resource in resources:
            # Extract features
            feats = extract_features(task, resource, resources)
            X_rows.append(feats)

            # Derive label: does this resource cover any required fact?
            rtext = _normalize(resource.text)
            is_useful = any(_normalize(fact) in rtext for fact in task.required_facts)
            y_rows.append(float(is_useful))

    return np.array(X_rows), np.array(y_rows)


class TrainedUtilityScorer(UtilityScorer):
    """
    Learned utility scorer using a trained GradientBoostingClassifier.

    Predicts P(resource is useful | task, resource) from a 9-dim feature vector.
    Falls back to DenseUtilityScorer if no trained model is available.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self._model = None
        self._fallback = DenseUtilityScorer(
            type_priors={"knowledge": 0.05, "experience": 0.15, "computation": 0.20},
            entity_weight=0.25,
            hint_weight=0.35,
        )

    def _load_model(self):
        if self._model is not None:
            return True

        if not self.model_path.exists():
            return False

        try:
            import joblib
            self._model = joblib.load(self.model_path)
            return True
        except Exception as e:
            print(f"Warning: Failed to load learned scorer model: {e}")
            return False

    def score(self, task: Task, resources: List[Resource]) -> List[Tuple[Resource, float]]:
        if not resources:
            return []

        # Fall back to heuristic if model not available
        if not self._load_model():
            print("Warning: No trained model found, falling back to DenseUtilityScorer.")
            return self._fallback.score(task, resources)

        # Extract features for all resources
        features = np.array([
            extract_features(task, r, resources) for r in resources
        ])

        # Predict P(useful)
        probabilities = self._model.predict_proba(features)[:, 1]

        scored = list(zip(resources, [float(p) for p in probabilities]))
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def fit(self, tasks: List[Task], resource_map: Dict[str, List[Resource]], save: bool = True):
        """
        Train the scorer model on benchmark data.

        Returns the trained model and cross-validation metrics dict.
        """
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import StratifiedKFold, cross_validate
        import joblib

        X, y = build_training_data(tasks, resource_map)

        print(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"  Positive (useful): {int(y.sum())} ({y.mean()*100:.1f}%)")
        print(f"  Negative (not useful): {int(len(y) - y.sum())} ({(1-y.mean())*100:.1f}%)")

        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
        )

        # 5-fold stratified cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_validate(
            model, X, y, cv=cv,
            scoring=["accuracy", "precision", "recall", "f1"],
            return_train_score=False,
        )

        metrics = {
            "accuracy": cv_results["test_accuracy"].mean(),
            "precision": cv_results["test_precision"].mean(),
            "recall": cv_results["test_recall"].mean(),
            "f1": cv_results["test_f1"].mean(),
            "accuracy_std": cv_results["test_accuracy"].std(),
            "f1_std": cv_results["test_f1"].std(),
        }

        # Train final model on all data
        model.fit(X, y)
        self._model = model

        # Feature importances
        importances = dict(zip(FEATURE_NAMES, model.feature_importances_))

        if save:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, self.model_path)
            print(f"Saved trained model to {self.model_path}")

        return model, metrics, importances
