import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .types import Resource, Task


class UtilityScorer(ABC):
    @abstractmethod
    def score(self, task: Task, resources: List[Resource]) -> List[Tuple[Resource, float]]:
        raise NotImplementedError


class TfidfUtilityScorer(UtilityScorer):
    """Baseline heuristic scorer using TF-IDF word overlap."""
    def __init__(self, type_priors: Optional[Dict[str, float]] = None):
        self.type_priors = type_priors or {
            "knowledge": 0.0,
            "experience": 0.0,
            "computation": 0.0,
        }

    def score(self, task: Task, resources: List[Resource]) -> List[Tuple[Resource, float]]:
        if not resources:
            return []

        texts = [task.query] + [r.text for r in resources]
        vec = TfidfVectorizer(stop_words="english")
        X = vec.fit_transform(texts)
        sims = cosine_similarity(X[0:1], X[1:]).ravel()

        scored = []
        for r, sim in zip(resources, sims):
            score = float(sim)
            score += self.type_priors.get(r.resource_type, 0.0)
            score += 0.05 * float(r.utility_hint)
            scored.append((r, score))

        return sorted(scored, key=lambda x: x[1], reverse=True)


class RelevanceOnlyScorer(TfidfUtilityScorer):
    """Pure keyword matching baseline without resource-type awareness."""
    def __init__(self):
        super().__init__({"knowledge": 0.0, "experience": 0.0, "computation": 0.0})


class DenseUtilityScorer(UtilityScorer):
    """
    Task-conditioned utility scorer for Cognitive State Compilation (CSC).
    Evaluates:
      1. Multi-scale semantic match (unigrams + bigrams)
      2. Entity / keyword alignment with task query
      3. Resource-type utility priors (weighting computation traces and strategic plans)
      4. Utility hint signals from historical execution / verification
    """
    def __init__(
        self,
        type_priors: Optional[Dict[str, float]] = None,
        entity_weight: float = 0.25,
        hint_weight: float = 0.35,
    ):
        self.type_priors = type_priors or {
            "knowledge": 0.05,
            "experience": 0.15,
            "computation": 0.20,
        }
        self.entity_weight = entity_weight
        self.hint_weight = hint_weight

    def _extract_keywords(self, text: str) -> set:
        words = re.findall(r"\b[a-zA-Z0-9]{3,}\b", text.lower())
        stopwords = {"the", "and", "was", "were", "what", "which", "who", "when", "where", "with", "from", "for", "that", "this"}
        return {w for w in words if w not in stopwords}

    def score(self, task: Task, resources: List[Resource]) -> List[Tuple[Resource, float]]:
        if not resources:
            return []

        # 1. Multi-scale TF-IDF semantic similarity (1-gram and 2-gram)
        texts = [task.query] + [r.text for r in resources]
        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        X = vec.fit_transform(texts)
        sims = cosine_similarity(X[0:1], X[1:]).ravel()

        query_keywords = self._extract_keywords(task.query)

        scored = []
        for r, sim in zip(resources, sims):
            # Semantic alignment
            base_score = float(sim)

            # Keyword / entity overlap bonus
            res_keywords = self._extract_keywords(r.text)
            overlap = len(query_keywords & res_keywords) / max(1, len(query_keywords))
            entity_score = self.entity_weight * overlap

            # Task-conditioned relevance base
            relevance = base_score + entity_score

            # Resource-type prior and hint bonuses
            type_bonus = self.type_priors.get(r.resource_type, 0.0)
            hint_bonus = self.hint_weight * float(r.utility_hint)

            # Modulate utility: irrelevant items without task alignment receive minimal utility
            if relevance > 0.02:
                total_utility = relevance + type_bonus + hint_bonus
            else:
                total_utility = relevance + 0.05 * (type_bonus + hint_bonus)

            scored.append((r, total_utility))

        return sorted(scored, key=lambda x: x[1], reverse=True)


class LearnedUtilityScorer(DenseUtilityScorer):
    """
    Parametric learned utility scorer for publication experiments.
    Implements u_theta(task, resource) = theta_sem * sim + theta_ent * ent + theta_type * type + theta_hint * hint.
    """
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        type_priors: Optional[Dict[str, float]] = None,
    ):
        weights = weights or {"sem": 0.40, "ent": 0.20, "type": 0.15, "hint": 0.25}
        super().__init__(
            type_priors=type_priors or {"knowledge": 0.05, "experience": 0.10, "computation": 0.15},
            entity_weight=weights.get("ent", 0.20),
            hint_weight=weights.get("hint", 0.25),
        )