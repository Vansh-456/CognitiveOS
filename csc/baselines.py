from typing import List
from .composer import StateComposer
from .scorers import RelevanceOnlyScorer
from .selector import BudgetedSelector
from .types import CompiledState, Resource, Task


def full_context(task: Task, resources: List[Resource], budget: int) -> CompiledState:
    selected = []
    remaining = budget
    for r in resources:
        if r.token_cost <= remaining:
            selected.append((r, 0.0))
            remaining -= r.token_cost
    return StateComposer().compose(selected, task.query, budget)


def standard_rag(task: Task, resources: List[Resource], budget: int, k: int = 5):
    knowledge = [r for r in resources if r.resource_type == "knowledge"]
    scored = RelevanceOnlyScorer().score(task, knowledge)[:k]
    selected = BudgetedSelector(budget).select(scored)
    return StateComposer().compose(selected, task.query, budget)


def memory_augmented_rag(task: Task, resources: List[Resource], budget: int):
    candidates = [r for r in resources if r.resource_type in ("knowledge", "experience")]
    scored = RelevanceOnlyScorer().score(task, candidates)
    selected = BudgetedSelector(budget).select(scored)
    return StateComposer().compose(selected, task.query, budget)


def relevance_only(task: Task, resources: List[Resource], budget: int):
    scored = RelevanceOnlyScorer().score(task, resources)
    selected = BudgetedSelector(budget).select(scored)
    return StateComposer().compose(selected, task.query, budget)


def mem1_style_recurrent(task: Task, resources: List[Resource], budget: int):
    # Controlled approximation: undifferentiated compact-state selection.
    # This is not a reproduction of MEM1's RL training.
    scored = RelevanceOnlyScorer().score(task, resources)
    selected = BudgetedSelector(budget).select(scored)
    return StateComposer().compose(selected, task.query, budget)
