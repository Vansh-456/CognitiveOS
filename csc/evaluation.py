import collections
import re
import string
from typing import Dict
from .types import CompiledState, Task


def normalize(s: str) -> str:
    """Basic whitespace normalization."""
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def normalize_answer(s: str) -> str:
    """
    Standard SQuAD/HotpotQA answer normalization:
    Lower text, remove punctuation, remove articles (a, an, the), and fix whitespace.
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Computes exact match score (1.0 if match, 0.0 otherwise)."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Computes token-level F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def evaluate_prediction(task: Task, prediction: str) -> Dict[str, float]:
    """Evaluates an LLM generated prediction against the task answer."""
    em = exact_match_score(prediction, task.answer)
    f1 = f1_score(prediction, task.answer)
    sub = float(normalize_answer(task.answer) in normalize_answer(prediction))

    return {
        "exact_match": em,
        "f1": f1,
        "substring_match": sub,
    }


def evaluate_state(task: Task, state: CompiledState):
    """Evaluates the compiled state against the required facts."""
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
