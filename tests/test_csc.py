import unittest
from csc.types import Task, Resource
from csc.scorers import TfidfUtilityScorer, DenseUtilityScorer, RelevanceOnlyScorer
from csc.learned_scorer import TrainedUtilityScorer
from csc.compiler import CognitiveStateCompiler
from csc.tokenizer import count_tokens, truncate_to_tokens


class TestCSC(unittest.TestCase):
    def test_budget_is_respected(self):
        task = Task("t", "choose PostgreSQL for strong consistency", "PostgreSQL",
                    ["strong consistency", "PostgreSQL"])
        resources = [
            Resource("a", "knowledge", "PostgreSQL supports strong consistency.", 10),
            Resource("b", "experience", "A previous system used PostgreSQL.", 10),
            Resource("c", "knowledge", "Unrelated weather information.", 10),
        ]
        state = CognitiveStateCompiler(TfidfUtilityScorer(), 15).compile(task, resources)
        self.assertLessEqual(state.token_cost, 15)

    def test_compiler_returns_state(self):
        task = Task("t", "choose PostgreSQL", "PostgreSQL", ["PostgreSQL"])
        resources = [Resource("a", "knowledge", "PostgreSQL is used.", 5)]
        state = CognitiveStateCompiler(TfidfUtilityScorer(), 10).compile(task, resources)
        self.assertTrue(state.text)
        self.assertEqual(state.resource_ids, ["a"])

    def test_tokenizer_tiktoken(self):
        text = "Cognitive state compilation under fixed resource budgets."
        tokens = count_tokens(text)
        self.assertGreater(tokens, 0)
        truncated = truncate_to_tokens(text, budget=3)
        self.assertLessEqual(count_tokens(truncated), 3)

    def test_dense_utility_scorer(self):
        task = Task("t1", "latency benchmark in Delhi", "28 ms", ["Delhi latency"])
        resources = [
            Resource("c1", "computation", "Latency benchmark result: Delhi 28 ms.", 10, utility_hint=0.8),
            Resource("k1", "knowledge", "Unrelated weather report for Singapore.", 10, utility_hint=0.0),
        ]
        scorer = DenseUtilityScorer()
        scored = scorer.score(task, resources)
        self.assertEqual(len(scored), 2)
        # The relevant computation trace should be ranked #1
        self.assertEqual(scored[0][0].resource_id, "c1")
        self.assertGreater(scored[0][1], scored[1][1])

    def test_trained_utility_scorer(self):
        task = Task("t1", "latency benchmark in Delhi", "28 ms", ["Delhi latency"])
        resources = [
            Resource("c1", "computation", "Latency benchmark result: Delhi 28 ms.", 10, utility_hint=0.8),
            Resource("k1", "knowledge", "Unrelated weather report for Singapore.", 10, utility_hint=0.0),
        ]
        scorer = TrainedUtilityScorer()
        scored = scorer.score(task, resources)
        self.assertEqual(len(scored), 2)
        self.assertEqual(scored[0][0].resource_id, "c1")
        self.assertGreater(scored[0][1], scored[1][1])


if __name__ == "__main__":
    unittest.main()
