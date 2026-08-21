from abc import ABC, abstractmethod
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .types import Resource, Task


class UtilityScorer(ABC):
    @abstractmethod
    def score(self, task: Task, resources: List[Resource]) -> List[Tuple[Resource, float]]:
        raise NotImplementedError


class TfidfUtilityScorer(UtilityScorer):
    # Prototype scorer. Replace with a learned scorer for the paper.
    def __init__(self, type_priors=None):
        self.type_priors = type_priors or {
            "knowledge": 0.0,
            "experience": 0.0,
            "computation": 0.0,
        }

    def score(self, task: Task, resources: List[Resource]):
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
    def __init__(self):
        super().__init__({"knowledge": 0.0, "experience": 0.0, "computation": 0.0})
