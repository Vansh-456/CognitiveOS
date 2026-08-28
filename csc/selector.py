from typing import Dict, List, Optional, Tuple
from .types import Resource


class GreedyKnapsackSelector:
    """
    Greedy utility-per-token density budgeted selector.
    Sorts candidate items by utility / max(1, token_cost) and greedily packs within budget.
    Complexity: O(N log N).
    """
    def __init__(self, budget: int, type_caps: Optional[Dict[str, int]] = None):
        self.budget = budget
        self.type_caps = type_caps or {}

    def select(self, scored_resources: List[Tuple[Resource, float]]) -> List[Tuple[Resource, float]]:
        remaining = self.budget
        counts: Dict[str, int] = {}
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


# Alias for backward compatibility
BudgetedSelector = GreedyKnapsackSelector


class ExactKnapsackSelector:
    """
    Exact 0-1 Integer Knapsack solver via Dynamic Programming / Branch & Bound.
    Finds the provably optimal subset of resources maximizing total utility subject to
    token budget constraint and resource type caps.
    """
    def __init__(self, budget: int, type_caps: Optional[Dict[str, int]] = None):
        self.budget = budget
        self.type_caps = type_caps or {}

    def select(self, scored_resources: List[Tuple[Resource, float]]) -> List[Tuple[Resource, float]]:
        if not scored_resources:
            return []

        # Filter out items exceeding budget
        items = [(r, s) for r, s in scored_resources if r.token_cost <= self.budget]
        n = len(items)
        if n == 0:
            return []

        memo = {}

        def solve(idx: int, rem_budget: int, counts: Tuple[Tuple[str, int], ...]) -> Tuple[float, List[int]]:
            if idx >= n or rem_budget <= 0:
                return 0.0, []

            state_key = (idx, rem_budget, counts)
            if state_key in memo:
                return memo[state_key]

            # Option 1: Skip current item
            best_val, best_indices = solve(idx + 1, rem_budget, counts)

            # Option 2: Include current item if budget and type cap allow
            r, score = items[idx]
            counts_dict = dict(counts)
            current_type_count = counts_dict.get(r.resource_type, 0)
            type_cap = self.type_caps.get(r.resource_type)

            if r.token_cost <= rem_budget and (type_cap is None or current_type_count < type_cap):
                new_counts_dict = dict(counts_dict)
                new_counts_dict[r.resource_type] = current_type_count + 1
                sorted_counts = tuple(sorted(new_counts_dict.items()))

                inc_val, inc_indices = solve(idx + 1, rem_budget - r.token_cost, sorted_counts)
                total_val = inc_val + score

                if total_val > best_val:
                    best_val = total_val
                    best_indices = [idx] + inc_indices

            memo[state_key] = (best_val, best_indices)
            return best_val, best_indices

        init_counts = tuple(sorted((k, 0) for k in self.type_caps.keys()))
        _, optimal_indices = solve(0, self.budget, init_counts)

        return [items[i] for i in optimal_indices]
