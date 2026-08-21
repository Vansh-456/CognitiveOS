# Cognitive State Compilation (CSC)

Resource-Bounded Construction of Reasoning States for Long-Horizon LLM Agents.

This repository implements a minimal research prototype of:

Task -> Task Encoder -> Candidate Resource Pool -> Utility Scorer
-> Budgeted Selector -> State Composer -> Compiled Cognitive State -> LLM

Resource types:
- knowledge
- experience
- computation

## Install

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Offline demo

```bash
python scripts/run_demo.py
```

The demo uses TF-IDF and a deterministic state evaluator, so no API key is
required. It is an architecture smoke test, NOT a publication-quality LLM
benchmark.

Results are written to `results/demo_results.csv` and
`results/demo_summary.csv`.

## Tests

```bash
pytest -q
```

## Real LLM mode

Set:
- `LLM_API_KEY`
- `LLM_BASE_URL`
- `LLM_MODEL`

The provided OpenAI-compatible adapter can then be connected to the experiment
runner.

## Important research note

The included B5 baseline is a controlled approximation of MEM1-style recurrent
compact-state consolidation. It is NOT a reproduction of MEM1's original RL
training procedure. For the final paper, label it accordingly unless you
implement the original method faithfully.

## Next research stages

1. Replace TF-IDF with a learned utility scorer.
2. Use a real long-horizon benchmark.
3. Use the exact target-model tokenizer.
4. Add real LLM evaluation.
5. Add noise/conflict/stale-memory robustness.
6. Add computational-memory traces.
7. Run the full budget sweep: 500/1000/2000/4000/8000 tokens.
