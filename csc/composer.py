from typing import List, Tuple
from .types import Resource, CompiledState


class StateComposer:
    def compose(self, selected: List[Tuple[Resource, float]], task_query: str, budget: int):
        blocks = []
        resources = []

        for resource, score in selected:
            blocks.append(
                f"[{resource.resource_type.upper()} | {resource.resource_id}]\\n"
                f"{resource.text}"
            )
            resources.append(resource)

        text = (
            "TASK:\\n"
            f"{task_query}\\n\\n"
            "COMPILED COGNITIVE STATE:\\n"
            + "\\n\\n".join(blocks)
        )

        total = sum(r.token_cost for r in resources)
        return CompiledState(resources=resources, text=text, token_cost=total)
