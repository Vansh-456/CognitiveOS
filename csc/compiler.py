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
        budget_mode: str = "payload",
    ):
        self.scorer = scorer
        self.budget = budget
        self.type_caps = type_caps
        self.selector_cls = selector_cls
        self.budget_mode = budget_mode  # "payload" (default) or "strict_total"
        self.composer = StateComposer()

    def compile(self, task: Task, resources: List[Resource]) -> CompiledState:
        if self.budget_mode == "strict_total":
            base_overhead = self.composer.base_overhead(task.query)
            # Create framed candidate copies where token_cost includes framing header tokens
            framed_resources = []
            for r in resources:
                from .tokenizer import count_tokens
                hdr_tokens = count_tokens(f"[{r.resource_type.upper()} | {r.resource_id}]\n\n")
                framed_r = Resource(
                    resource_id=r.resource_id,
                    resource_type=r.resource_type,
                    text=r.text,
                    token_cost=r.token_cost + hdr_tokens,
                    utility_hint=r.utility_hint,
                    metadata=r.metadata,
                )
                framed_resources.append(framed_r)

            scored = self.scorer.score(task, framed_resources)
            effective_budget = max(0, self.budget - base_overhead)
            selected_framed = self.selector_cls(
                budget=effective_budget,
                type_caps=self.type_caps,
            ).select(scored)

            orig_map = {r.resource_id: r for r in resources}
            selected_orig = [(orig_map[r.resource_id], score) for r, score in selected_framed]
            return self.composer.compose(
                selected=selected_orig,
                task_query=task.query,
                budget=self.budget,
            )

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


