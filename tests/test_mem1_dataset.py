import json
import tempfile
import unittest
from pathlib import Path
from csc.dataset import load_mem1_benchmark
from scripts.prepare_mem1_hotpotqa import convert_to_csc_format


class TestMem1Dataset(unittest.TestCase):
    def test_convert_to_csc_format(self):
        sample_raw = [
            {
                "_id": "hotpot_test_01",
                "question": "Which company founded Android and who was the CEO?",
                "answer": "Google",
                "supporting_facts": [["Android Inc.", 0], ["Google", 1]],
                "context": [
                    ["Android Inc.", ["Android Inc. was founded in Palo Alto, California in 2003.", "It was later acquired by Google."]],
                    ["Google", ["Google is an American multinational technology company.", "Sundar Pichai is the CEO."]],
                    ["Apple Inc.", ["Apple Inc. is an American multinational technology company headquartered in Cupertino."]],
                ],
            }
        ]

        benchmark = convert_to_csc_format(sample_raw, num_samples=1)
        self.assertEqual(len(benchmark), 1)

        item = benchmark[0]
        self.assertEqual(item["task_id"], "hotpot_test_01")
        self.assertEqual(item["query"], "Which company founded Android and who was the CEO?")
        self.assertEqual(item["answer"], "Google")
        self.assertTrue(len(item["required_facts"]) > 0)

        # Verify resource types present
        resource_types = {r["resource_type"] for r in item["resources"]}
        self.assertIn("knowledge", resource_types)
        self.assertIn("experience", resource_types)
        self.assertIn("computation", resource_types)

    def test_load_mem1_benchmark_synthetic(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_file = Path(tmp_dir) / "test_benchmark.json"
            dummy_data = [
                {
                    "task_id": "dummy_1",
                    "query": "Test query?",
                    "answer": "Test answer",
                    "required_facts": ["fact 1"],
                    "resources": [
                        {
                            "resource_id": "k1",
                            "resource_type": "knowledge",
                            "text": "Fact 1 content",
                            "token_cost": 5,
                            "utility_hint": 1.0,
                        }
                    ],
                }
            ]
            with open(test_file, "w", encoding="utf-8") as f:
                json.dump(dummy_data, f)

            tasks, resource_map = load_mem1_benchmark(file_path=str(test_file))
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].task_id, "dummy_1")
            self.assertIn("dummy_1", resource_map)
            self.assertEqual(len(resource_map["dummy_1"]), 1)
            self.assertEqual(resource_map["dummy_1"][0].resource_type, "knowledge")


if __name__ == "__main__":
    unittest.main()
