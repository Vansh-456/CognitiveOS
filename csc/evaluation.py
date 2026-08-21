import re
from .types import CompiledState, Task


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def evaluate_state(task: Task, state: CompiledState):
    text = normalize(state.text)
    hits = sum(1 for fact in task.required_facts if normalize(fact) in text)
    coverage = hits / max(1, len(task.required_facts))

    useful_resources = 0
    for r in state.resources:
        rtext = normalize(r.text)
        if any(normalize(f) in rtext for f in task.required_facts):
            useful_resources += 1

    unnecessary = len(state.resources) - useful_resources
    precision = useful_resources / max(1, len(state.resources))

    state.coverage = coverage
    state.relevant_count = useful_resources
    state.unnecessary_count = unnecessary

    return {
        "coverage": coverage,
        "state_precision": precision,
        "selected_resources": len(state.resources),
        "token_cost": state.token_cost,
        "unnecessary_resources": unnecessary,
    }


def answer_accuracy(task: Task, answer: str) -> float:
    return float(normalize(task.answer) in normalize(answer))
