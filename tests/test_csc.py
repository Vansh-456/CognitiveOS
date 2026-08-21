from csc.types import Task, Resource
from csc.scorers import TfidfUtilityScorer
from csc.compiler import CognitiveStateCompiler


def test_budget_is_respected():
    task = Task("t", "choose PostgreSQL for strong consistency", "PostgreSQL",
                ["strong consistency", "PostgreSQL"])
    resources = [
        Resource("a", "knowledge", "PostgreSQL supports strong consistency.", 10),
        Resource("b", "experience", "A previous system used PostgreSQL.", 10),
        Resource("c", "knowledge", "Unrelated weather information.", 10),
    ]
    state = CognitiveStateCompiler(TfidfUtilityScorer(), 15).compile(task, resources)
    assert state.token_cost <= 15


def test_compiler_returns_state():
    task = Task("t", "choose PostgreSQL", "PostgreSQL", ["PostgreSQL"])
    resources = [Resource("a", "knowledge", "PostgreSQL is used.", 5)]
    state = CognitiveStateCompiler(TfidfUtilityScorer(), 10).compile(task, resources)
    assert state.text
    assert state.resource_ids == ["a"]
