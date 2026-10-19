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
    def test_exact_knapsack_selector(self):
        from csc.selector import ExactKnapsackSelector, GreedyKnapsackSelector
        resources = [
            Resource("r1", "knowledge", "Short vital fact", token_cost=3),
            Resource("r2", "knowledge", "Another short vital fact", token_cost=3),
            Resource("r3", "experience", "Long medium fact", token_cost=5),
        ]
        # Scored pairs: (r1: 0.5), (r2: 0.5), (r3: 0.8) with Budget = 6
        # Greedy packs r3 first (density 0.8/5 = 0.16) leaving rem=1 (total utility 0.8)
        # Exact 0-1 DP packs r1 + r2 (cost 3+3=6, total utility 0.5+0.5 = 1.0)
        scored = [(resources[0], 0.5), (resources[1], 0.5), (resources[2], 0.8)]
        
        exact_sel = ExactKnapsackSelector(budget=6).select(scored)
        exact_ids = [r.resource_id for r, _ in exact_sel]
        self.assertEqual(set(exact_ids), {"r1", "r2"})
        self.assertAlmostEqual(sum(s for _, s in exact_sel), 1.0)

    def test_tokenizer_backends(self):
        from csc.tokenizer import count_tokens, truncate_to_tokens, get_tokenizer_info, set_default_backend
        info = get_tokenizer_info()
        self.assertIn("backend", info)
        text = "Cognitive State Compilation frames reasoning state construction as a 0/1 knapsack."
        # Qwen backend
        qwen_count = count_tokens(text, backend="qwen")
        self.assertGreater(qwen_count, 0)
        qwen_trunc = truncate_to_tokens(text, budget=5, backend="qwen")
        self.assertLessEqual(count_tokens(qwen_trunc, backend="qwen"), 5)
        # cl100k backend
        cl_count = count_tokens(text, backend="cl100k")
        self.assertGreater(cl_count, 0)

    def test_prompt_envelope_accounting(self):
        task = Task("t", "Find the capital of France", "Paris", ["Paris"])
        resources = [
            Resource("r1", "knowledge", "Paris is the capital of France.", 8),
            Resource("r2", "knowledge", "France is in Western Europe.", 6),
        ]
        compiler = CognitiveStateCompiler(TfidfUtilityScorer(), budget=20, budget_mode="payload")
        state = compiler.compile(task, resources)
        self.assertGreater(state.payload_tokens, 0)
        self.assertGreater(state.prompt_tokens, state.payload_tokens)
        self.assertEqual(state.overhead_tokens, state.prompt_tokens - state.payload_tokens)
        self.assertEqual(state.token_cost, state.payload_tokens)

    def test_strict_total_budget_mode(self):
        task = Task("t", "Query for test", "ans", ["ans"])
        resources = [
            Resource("r1", "knowledge", "Short fact one", 5),
            Resource("r2", "knowledge", "Short fact two", 5),
            Resource("r3", "knowledge", "Short fact three", 5),
        ]
        compiler = CognitiveStateCompiler(TfidfUtilityScorer(), budget=35, budget_mode="strict_total")
        state = compiler.compile(task, resources)
        self.assertLessEqual(state.prompt_tokens, 35)

    def test_baselines_extended(self):
        from csc.baselines import standard_rag_greedy_budget_filled, unbounded_full_context
        task = Task("t", "Query", "ans", ["ans"])
        resources = [
            Resource("k1", "knowledge", "Knowledge fact 1", 10),
            Resource("k2", "knowledge", "Knowledge fact 2", 10),
            Resource("k3", "knowledge", "Knowledge fact 3", 10),
            Resource("e1", "experience", "Experience trace", 10),
        ]
        # Greedy budget filled should take as many knowledge items as budget allows
        rag_filled = standard_rag_greedy_budget_filled(task, resources, budget=25)
        self.assertEqual(len(rag_filled.resources), 2)
        # Unbounded full context should take all 4 resources
        full = unbounded_full_context(task, resources)
        self.assertEqual(len(full.resources), 4)


if __name__ == "__main__":
    unittest.main()

