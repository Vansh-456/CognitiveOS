from .types import Task, Resource


def build_demo_dataset():
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
