import unittest
from csc.types import Task
from csc.evaluation import exact_match_score, f1_score, evaluate_prediction
from csc.llm import MockLLM


class TestLLMEvaluation(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(exact_match_score("Doctor Strange", "Doctor Strange"), 1.0)
        self.assertEqual(exact_match_score("doctor strange", "Doctor Strange"), 1.0)
        self.assertEqual(exact_match_score("The Doctor Strange", "Doctor Strange"), 1.0)
        self.assertEqual(exact_match_score("Iron Man", "Doctor Strange"), 0.0)

    def test_f1_score(self):
        self.assertAlmostEqual(f1_score("United States of America", "United States"), 2/3, places=2)
        self.assertAlmostEqual(f1_score("Paris, France", "Paris"), 2/3, places=2)
        self.assertEqual(f1_score("London", "Tokyo"), 0.0)

    def test_evaluate_prediction(self):
        task = Task("t1", "Where was X born?", "United States", ["born in US"])
        res = evaluate_prediction(task, "the united states")
        self.assertEqual(res["exact_match"], 1.0)
        self.assertEqual(res["f1"], 1.0)
        self.assertEqual(res["substring_match"], 1.0)

    def test_mock_llm(self):
        answer_map = {"What is the capital?": "Paris"}
        llm = MockLLM(answer_map=answer_map)
        prompt = "TASK:\nWhat is the capital?\n\nCOMPILED COGNITIVE STATE:\nParis is the capital."
        pred = llm.generate(prompt)
        self.assertEqual(pred, "Paris")


if __name__ == "__main__":
    unittest.main()
