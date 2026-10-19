from typing import List
from .composer import StateComposer
from .scorers import RelevanceOnlyScorer
from .selector import BudgetedSelector
from .types import CompiledState, Resource, Task


def full_context(task: Task, resources: List[Resource], budget: int) -> CompiledState:
    """Baseline B0: Naively appends resources chronologically until budget runs out."""
    selected = []
    remaining = budget
    for r in resources:
        if r.token_cost <= remaining:
            selected.append((r, 0.0))
            remaining -= r.token_cost
    return StateComposer().compose(selected, task.query, budget)


def standard_rag(task: Task, resources: List[Resource], budget: int, k: int = 5) -> CompiledState:
    """Baseline B1: Retrieves top-k from KNOWLEDGE pool only."""
    knowledge = [r for r in resources if r.resource_type == "knowledge"]
    scored = RelevanceOnlyScorer().score(task, knowledge)[:k]
    selected = BudgetedSelector(budget).select(scored)
    return StateComposer().compose(selected, task.query, budget)


def standard_rag_greedy_budget_filled(task: Task, resources: List[Resource], budget: int) -> CompiledState:
    """
    Baseline B1-Full: Knowledge-only retrieval that greedily fills the entire budget
    without arbitrary top-k truncation (answering Reviewer rTKQ W3).
    """
    knowledge = [r for r in resources if r.resource_type == "knowledge"]
    scored = RelevanceOnlyScorer().score(task, knowledge)
    selected = BudgetedSelector(budget).select(scored)
    return StateComposer().compose(selected, task.query, budget)


def unbounded_full_context(task: Task, resources: List[Resource]) -> CompiledState:
    """
    Unbounded Full-Context: Ingests all available candidate resources without token budget cap.
    Used for end-to-end efficiency, latency, and distraction analysis (answering Reviewer rTKQ W5).
    """
    selected = [(r, 1.0) for r in resources]
    total_tokens = sum(r.token_cost for r in resources)
    return StateComposer().compose(selected, task.query, budget=total_tokens)



def memory_augmented_rag(task: Task, resources: List[Resource], budget: int) -> CompiledState:
    """Baseline B2: Retrieves from KNOWLEDGE + EXPERIENCE pools using semantic relevance."""
    candidates = [r for r in resources if r.resource_type in ("knowledge", "experience")]
    scored = RelevanceOnlyScorer().score(task, candidates)
    selected = BudgetedSelector(budget).select(scored)
    return StateComposer().compose(selected, task.query, budget)


def relevance_only(task: Task, resources: List[Resource], budget: int) -> CompiledState:
    """Baseline B3: Unweighted cosine relevance across all candidate pools."""
    scored = RelevanceOnlyScorer().score(task, resources)
    selected = BudgetedSelector(budget).select(scored)
    return StateComposer().compose(selected, task.query, budget)


def mem1_style_recurrent(task: Task, resources: List[Resource], budget: int) -> CompiledState:
    """
    Baseline B5: Recurrent consolidation baseline.
    Models an evolving compact state buffer where observations are consolidated
    sequentially into a fixed-capacity buffer rather than performing cross-resource knapsack compilation.
    """
    # Sequentially fills a recurrent compact buffer with incoming turn observations
    selected = []
    current_tokens = 0
    # In recurrent consolidation, the model prioritizes latest and most relevant turns in its buffer
    for r in resources:
        if current_tokens + r.token_cost <= budget:
            selected.append((r, 1.0))
            current_tokens += r.token_cost
        elif budget > 0 and len(selected) > 0 and r.resource_type == "computation":
            # Recurrent state replacement heuristic: replace an older distractor with computation
            for idx, (old_r, _) in enumerate(selected):
                if old_r.resource_type == "knowledge" and old_r.utility_hint == 0:
                    current_tokens -= old_r.token_cost
                    selected[idx] = (r, 1.0)
                    current_tokens += r.token_cost
                    break

    return StateComposer().compose(selected, task.query, budget)
