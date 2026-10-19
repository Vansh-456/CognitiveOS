from typing import List, Optional, Tuple
from .types import Resource, CompiledState
from .tokenizer import count_tokens


class StateComposer:
    def base_overhead(self, task_query: str) -> int:
        """Calculate the baseline token overhead of the prompt framing (task query and headers)."""
        empty_template = (
            "TASK:\n"
            f"{task_query}\n\n"
            "COMPILED COGNITIVE STATE:\n"
        )
        return count_tokens(empty_template)

    def compose(
        self,
        selected: List[Tuple[Resource, float]],
        task_query: str,
        budget: int,
    ) -> CompiledState:
        blocks = []
        resources = []

        for resource, score in selected:
            blocks.append(
                f"[{resource.resource_type.upper()} | {resource.resource_id}]\n"
                f"{resource.text}"
            )
            resources.append(resource)

        text = (
            "TASK:\n"
            f"{task_query}\n\n"
            "COMPILED COGNITIVE STATE:\n"
            + "\n\n".join(blocks)
        )

        payload_tokens = sum(r.token_cost for r in resources)
        prompt_tokens = count_tokens(text)
        overhead_tokens = max(0, prompt_tokens - payload_tokens)

        return CompiledState(
            resources=resources,
            text=text,
            token_cost=payload_tokens,
            payload_tokens=payload_tokens,
            prompt_tokens=prompt_tokens,
            overhead_tokens=overhead_tokens,
        )
