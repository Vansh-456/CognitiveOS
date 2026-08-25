import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from csc.tokenizer import count_tokens


def fetch_hotpotqa_from_hf(num_samples: int = 100) -> list:
    """Fetches HotpotQA distractor validation rows from Hugging Face Datasets API."""
    print(f"Fetching {num_samples} samples from Hugging Face HotpotQA dataset API...")
    base_url = "https://datasets-server.huggingface.co/rows"
    rows = []
    offset = 0
    limit_per_call = min(100, num_samples)

    while len(rows) < num_samples:
        params = {
            "dataset": "hotpotqa/hotpot_qa",
            "config": "distractor",
            "split": "validation",
            "offset": offset,
            "limit": min(limit_per_call, num_samples - len(rows)),
        }
        resp = requests.get(base_url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        batch = [item["row"] for item in data.get("rows", [])]
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        print(f"Fetched {len(rows)}/{num_samples} samples...")

    return rows


def convert_to_csc_format(raw_rows, num_samples: int = None):
    """
    Transforms HotpotQA rows into CSC Task & Resource objects.
    Each sample contains:
      - Task: query, answer, gold required facts
      - Knowledge (K): Gold Wikipedia paragraphs + distractor passages
      - Experience (E): Generated sub-goal decomposition & prior strategy traces
      - Computation (C): Intermediate entity extraction & tool verification traces
    """
    benchmark = []
    items_to_process = raw_rows[:num_samples] if num_samples is not None else raw_rows

    for item in items_to_process:
        task_id = item.get("id") or item.get("_id") or f"task_{len(benchmark)}"
        query = item.get("question", "")
        answer = item.get("answer", "")

        # 1. Extract gold supporting titles and facts
        supporting_facts_dict = item.get("supporting_facts", {})
        if isinstance(supporting_facts_dict, dict):
            supporting_titles = set(supporting_facts_dict.get("title", []))
        elif isinstance(supporting_facts_dict, list):
            supporting_titles = {sf[0] for sf in supporting_facts_dict}
        else:
            supporting_titles = set()

        supporting_sentences = []

        # 2. Build Knowledge Pool (K) from context paragraphs
        resources = []
        context_data = item.get("context", {})

        if isinstance(context_data, dict):
            titles = context_data.get("title", [])
            sentences_list = context_data.get("sentences", [])
            context_pairs = list(zip(titles, sentences_list))
        elif isinstance(context_data, list):
            context_pairs = context_data
        else:
            context_pairs = []

        for i, (title, sentences) in enumerate(context_pairs):
            sentences_text = " ".join(sentences) if isinstance(sentences, list) else str(sentences)
            paragraph_text = f"{title}: {sentences_text}"
            is_gold = title in supporting_titles

            if is_gold:
                if isinstance(sentences, list):
                    supporting_sentences.extend(sentences)
                else:
                    supporting_sentences.append(str(sentences))

            resources.append({
                "resource_id": f"k_{i+1}",
                "resource_type": "knowledge",
                "text": paragraph_text,
                "token_cost": count_tokens(paragraph_text),
                "utility_hint": 1.0 if is_gold else 0.0,
            })

        # 3. Build Experience Pool (E)
        resources.append({
            "resource_id": "e_1",
            "resource_type": "experience",
            "text": (
                f"Prior strategy note: To answer multi-hop query '{query}', first identify "
                "the key subject entities from the primary supporting context before bridging."
            ),
            "token_cost": count_tokens(query) + 20,
            "utility_hint": 0.5,
        })
        resources.append({
            "resource_id": "e_2",
            "resource_type": "experience",
            "text": (
                "Prior failure trace: Attempting to answer directly without validating intermediate "
                "supporting entities led to hallucinated bridge entities."
            ),
            "token_cost": 20,
            "utility_hint": 0.3,
        })

        # 4. Build Computation Pool (C)
        entities_str = ", ".join(list(supporting_titles)) if supporting_titles else "None"
        resources.append({
            "resource_id": "c_1",
            "resource_type": "computation",
            "text": f"Entity extraction tool trace: Subject entities identified: {entities_str}.",
            "token_cost": 15,
            "utility_hint": 0.8,
        })
        resources.append({
            "resource_id": "c_2",
            "resource_type": "computation",
            "text": "Intermediate verification check: Fact alignment confirmed across candidate passages.",
            "token_cost": 12,
            "utility_hint": 0.4,
        })

        benchmark.append({
            "task_id": task_id,
            "query": query,
            "answer": answer,
            "required_facts": supporting_sentences[:3] if supporting_sentences else [answer],
            "resources": resources,
        })

    return benchmark


def main():
    parser = argparse.ArgumentParser(description="Download and format MEM1 HotpotQA dataset.")
    parser.add_argument("--samples", type=int, default=100, help="Number of benchmark samples to format")
    parser.add_argument("--output", type=str, default="data/mem1_hotpotqa_benchmark.json", help="Output path")
    args = parser.parse_args()

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    raw_cache_file = data_dir / "hotpot_hf_raw.json"
    output_file = Path(args.output)

    # Step 1: Fetch raw samples (using local cache if available and sufficient)
    if raw_cache_file.exists():
        with open(raw_cache_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        if len(raw_data) < args.samples:
            raw_data = fetch_hotpotqa_from_hf(num_samples=args.samples)
            with open(raw_cache_file, "w", encoding="utf-8") as f:
                json.dump(raw_data, f)
    else:
        raw_data = fetch_hotpotqa_from_hf(num_samples=args.samples)
        with open(raw_cache_file, "w", encoding="utf-8") as f:
            json.dump(raw_data, f)

    # Step 2: Convert to CSC format
    print(f"Formatting {args.samples} samples into CSC benchmark format...")
    csc_benchmark = convert_to_csc_format(raw_data[:args.samples])

    # Step 3: Save processed benchmark
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(csc_benchmark, f, indent=2)

    print(f"Successfully saved {len(csc_benchmark)} benchmark tasks to {output_file}!")


if __name__ == "__main__":
    main()
