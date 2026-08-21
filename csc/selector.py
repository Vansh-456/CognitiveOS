from typing import List, Tuple
from .types import Resource


class BudgetedSelector:
    # Greedy utility-per-token budgeted selector.
    def __init__(self, budget: int, type_caps=None):
        self.budget = budget
        self.type_caps = type_caps or {}

    def select(self, scored_resources: List[Tuple[Resource, float]]):
        remaining = self.budget
        counts = {}
        selected = []

        candidates = sorted(
            scored_resources,
            key=lambda x: x[1] / max(1, x[0].token_cost),
            reverse=True,
        )

        for resource, score in candidates:
            if resource.token_cost > remaining:
                continue

            cap = self.type_caps.get(resource.resource_type)
            if cap is not None and counts.get(resource.resource_type, 0) >= cap:
                continue

            selected.append((resource, score))
            remaining -= resource.token_cost
            counts[resource.resource_type] = counts.get(resource.resource_type, 0) + 1

        return selected
