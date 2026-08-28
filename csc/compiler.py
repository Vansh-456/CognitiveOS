from typing import Any, List, Optional
from .composer import StateComposer
from .scorers import UtilityScorer
from .selector import BudgetedSelector, GreedyKnapsackSelector, ExactKnapsackSelector
from .types import CompiledState, Resource, Task


class CognitiveStateCompiler:
    def __init__(
        self,
        scorer: UtilityScorer,
        budget: int,
        type_caps: Optional[dict] = None,
        selector_cls: Any = GreedyKnapsackSelector,
    ):
        self.scorer = scorer
        self.budget = budget
        self.type_caps = type_caps
        self.selector_cls = selector_cls
        self.composer = StateComposer()

    def compile(self, task: Task, resources: List[Resource]) -> CompiledState:
        scored = self.scorer.score(task, resources)
        selected = self.selector_cls(
            budget=self.budget,
            type_caps=self.type_caps,
        ).select(scored)

        return self.composer.compose(
            selected=selected,
            task_query=task.query,
            budget=self.budget,
        )
