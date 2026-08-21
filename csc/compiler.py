from typing import List, Optional
from .composer import StateComposer
from .scorers import UtilityScorer
from .selector import BudgetedSelector
from .types import CompiledState, Resource, Task


class CognitiveStateCompiler:
    def __init__(
        self,
        scorer: UtilityScorer,
        budget: int,
        type_caps: Optional[dict] = None,
    ):
        self.scorer = scorer
        self.budget = budget
        self.type_caps = type_caps
        self.composer = StateComposer()

    def compile(self, task: Task, resources: List[Resource]) -> CompiledState:
        scored = self.scorer.score(task, resources)
        selected = BudgetedSelector(
            budget=self.budget,
            type_caps=self.type_caps,
        ).select(scored)

        return self.composer.compose(
            selected=selected,
            task_query=task.query,
            budget=self.budget,
        )
