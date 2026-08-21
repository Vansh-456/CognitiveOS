from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Task:
    task_id: str
    query: str
    answer: str
    required_facts: List[str]
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class Resource:
    resource_id: str
    resource_type: str  # knowledge | experience | computation
    text: str
    token_cost: int
    utility_hint: float = 0.0
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class CompiledState:
    resources: List[Resource]
    text: str
    token_cost: int
    coverage: float = 0.0
    relevant_count: int = 0
    unnecessary_count: int = 0

    @property
    def resource_ids(self):
        return [r.resource_id for r in self.resources]
