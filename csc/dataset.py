import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .types import Resource, Task


def build_demo_dataset() -> Tuple[List[Task], Dict[str, List[Resource]]]:
    tasks = [
        Task(
            "t1",
            "Which deployment region should the agent choose if the service must be close to Delhi users and Mumbai has a latency issue?",
            "Delhi",
            ["Delhi users", "Mumbai region has a latency issue", "Delhi region has the lower latency"],
        ),
        Task(
            "t2",
            "Which database should the agent use when transactions require strong consistency and Redis is only for reads?",
            "PostgreSQL",
            ["strong consistency", "Redis cache is only used for reads", "PostgreSQL supports transactional consistency"],
        ),
        Task(
            "t3",
            "Which model should the agent select when memory is limited to 4 GB and latency is more important than maximum accuracy?",
            "Small model",
            ["memory is limited to 4 GB", "latency is more important", "Small model has lower memory usage"],
        ),
    ]

    resources = {
        "t1": [
            Resource("k1", "knowledge", "Delhi users are geographically closest to the Delhi deployment region.", 14),
            Resource("k2", "knowledge", "Mumbai region has a latency issue for the current service.", 13),
            Resource("k3", "knowledge", "Delhi region has the lower latency for Delhi users.", 12),
            Resource("k4", "knowledge", "Singapore is a common deployment region for international users.", 11),
            Resource("e1", "experience", "A previous deployment selected Delhi after observing lower latency for Delhi users.", 15),
            Resource("e2", "experience", "A previous deployment selected Mumbai for a different customer.", 12),
            Resource("c1", "computation", "Latency benchmark result: Delhi 28 ms, Mumbai 71 ms.", 11),
            Resource("c2", "computation", "A failed deployment trace showed repeated timeouts in Mumbai.", 12),
            Resource("c3", "computation", "Unrelated CPU benchmark: 18,000 operations per second.", 10),
        ],
        "t2": [
            Resource("k1", "knowledge", "Strong consistency is required for the transaction database.", 12),
            Resource("k2", "knowledge", "Redis cache is only used for reads and is not the system of record.", 14),
            Resource("k3", "knowledge", "PostgreSQL supports transactional consistency and durable writes.", 13),
            Resource("k4", "knowledge", "MongoDB is a document-oriented database.", 10),
            Resource("e1", "experience", "A previous transaction system used PostgreSQL as the system of record.", 13),
            Resource("e2", "experience", "A read-heavy analytics service used Redis.", 11),
            Resource("c1", "computation", "Transaction test: PostgreSQL preserved atomic commits under concurrent writes.", 14),
            Resource("c2", "computation", "Redis benchmark showed very high read throughput.", 11),
            Resource("c3", "computation", "Unrelated cache hit-rate experiment.", 9),
        ],
        "t3": [
            Resource("k1", "knowledge", "The deployment has a 4 GB memory limit.", 11),
            Resource("k2", "knowledge", "Latency is more important than maximum benchmark accuracy.", 12),
            Resource("k3", "knowledge", "The Small model has lower memory usage and lower latency.", 13),
            Resource("k4", "knowledge", "The Large model generally has higher benchmark accuracy.", 12),
            Resource("e1", "experience", "A previous 4 GB deployment used the Small model successfully.", 13),
            Resource("e2", "experience", "A high-memory offline experiment used the Large model.", 11),
            Resource("c1", "computation", "Latency benchmark: Small model 45 ms, Large model 180 ms.", 12),
            Resource("c2", "computation", "Memory benchmark: Small model 2.8 GB, Large model 8.4 GB.", 12),
            Resource("c3", "computation", "Unrelated GPU utilization benchmark.", 9),
        ],
    }

    return tasks, resources


def load_mem1_benchmark(
    file_path: str = "data/mem1_hotpotqa_benchmark.json",
    limit: Optional[int] = None,
) -> Tuple[List[Task], Dict[str, List[Resource]]]:
    """
    Loads the real MEM1 HotpotQA benchmark dataset.
    Returns:
      tasks: List[Task]
      resource_map: Dict[task_id, List[Resource]]
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark file '{file_path}' not found. "
            "Please run 'python scripts/prepare_mem1_hotpotqa.py' to download and format the dataset."
        )

    with open(path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    if limit is not None:
        raw_items = raw_items[:limit]

    tasks = []
    resource_map = {}

    for item in raw_items:
        task = Task(
            task_id=item["task_id"],
            query=item["query"],
            answer=item["answer"],
            required_facts=item["required_facts"],
        )
        tasks.append(task)

        resources = [
            Resource(
                resource_id=r["resource_id"],
                resource_type=r["resource_type"],
                text=r["text"],
                token_cost=r["token_cost"],
                utility_hint=r.get("utility_hint", 0.0),
            )
            for r in item["resources"]
        ]
        resource_map[task.task_id] = resources

    return tasks, resource_map
