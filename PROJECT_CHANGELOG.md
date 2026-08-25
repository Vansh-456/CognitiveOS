# Cognitive State Compilation (CSC) — Project Changelog & Architecture Updates

This document provides a comprehensive record of the original project state versus all newly implemented modules, benchmark datasets, empirical results, and research tools.

---

## 1. Executive Summary of Upgrades

```mermaid
flowchart LR
    subgraph Old State
        O1[3 Synthetic Toy Tasks] --> O2[Word-Count Estimation]
        O2 --> O3[Heuristic TF-IDF Scorer]
        O3 --> O4[Identical B4 vs B5 Output]
    end

    subgraph New State (Publication Grade)
        N1[Real MEM1 HotpotQA Benchmark] --> N2[Exact subword tiktoken BPE]
        N2 --> N3[Dense & Learned Utility Scorer]
        N3 --> N4[Live LLM Evaluation EM & F1]
        N4 --> N5[Ablation Suite & 300 DPI Figures]
    end
```

---

## 2. Component-by-Component Breakdown (Old vs. New)

### A. Dataset Layer
* **Old State:**
  * Only contained 3 synthetic toy tasks (`t1`, `t2`, `t3`) with artificial short sentences in [`csc/dataset.py`](csc/dataset.py).
  * Tested artificial token budgets (`[20, 40, 80, 120]`).
* **New State:**
  * Created [`scripts/prepare_mem1_hotpotqa.py`](scripts/prepare_mem1_hotpotqa.py) to download and parse the official **HotpotQA Distractor Evaluation Dataset** (the core benchmark used in the **MEM1** research paper).
  * Partitioned every task into heterogeneous candidate pools:
    * **Knowledge ($K$):** Gold Wikipedia paragraphs + 8 distractor passages.
    * **Experience ($E$):** Prior sub-goal decompositions & strategy traces.
    * **Computation ($C$):** Entity extraction tool outputs & verification checks.
  * Formatted and saved to [`data/mem1_hotpotqa_benchmark.json`](data/mem1_hotpotqa_benchmark.json).
  * Added `load_mem1_benchmark()` in [`csc/dataset.py`](csc/dataset.py) supporting standard research budgets (`[200, 500, 1000, 2000, 4000]` tokens).

---

### B. Tokenization Layer
* **Old State:**
  * Used naive regex whitespace splitting (`len(re.findall(r"\S+", text))`) in [`csc/tokenizer.py`](csc/tokenizer.py).
* **New State:**
  * Integrated **`tiktoken`** (`cl100k_base` BPE subwords) for exact model-level token counting and prompt truncation.
  * Added `count_tokens(text)` and `truncate_to_tokens(text, budget)` with graceful offline fallback.

---

### C. Utility Scoring & Baselines Layer
* **Old State:**
  * Only had a basic TF-IDF scorer with equal priors (`0.02, 0.02, 0.02`), causing `B4_csc` and `B5_mem1_style` to produce identical rankings.
* **New State:**
  * Implemented **`DenseUtilityScorer`** in [`csc/scorers.py`](csc/scorers.py):
    * Multi-scale n-gram semantic similarity (1-gram + 2-gram).
    * Task-query entity alignment boost.
    * Heterogeneous resource-type priors ($K, E, C$).
    * Verification hint weights.
  * Implemented parametric **`LearnedUtilityScorer`**.
  * Refined [`csc/baselines.py`](csc/baselines.py) to faithfully model:
    * `B0_full_context`: Chronological loading.
    * `B1_standard_rag`: Top-$k$ Knowledge only.
    * `B2_memory_rag`: Knowledge + Experience only.
    * `B3_relevance_only`: Unweighted semantic relevance.
    * `B4_csc`: Task-conditioned heterogeneous knapsack compilation.
    * `B5_mem1_style`: Sequential turn consolidation in a compact state buffer.

---

### D. LLM Evaluation & Metrics Layer
* **Old State:**
  * No live LLM evaluation; only basic deterministic string matching.
* **New State:**
  * Implemented standard SQuAD / HotpotQA metrics in [`csc/evaluation.py`](csc/evaluation.py):
    * `exact_match_score(pred, gold)`
    * `f1_score(pred, gold)` (token-level precision/recall/F1)
    * `evaluate_prediction(task, pred)`
  * Enhanced [`csc/llm.py`](csc/llm.py):
    * `OpenAICompatibleLLM` with automatic rate-limit backoff (HTTP 429), timeouts, and `<think>` block stripping.
    * `MockLLM` for offline dry-run testing without API keys.
  * Created [`scripts/run_llm_eval.py`](scripts/run_llm_eval.py) to run live and offline evaluations across all 6 baselines and 5 budgets.
  * Created [`scripts/test_llm_connection.py`](scripts/test_llm_connection.py) to verify API connectivity.

---

### E. Ablations & Publication Figures Layer
* **Old State:**
  * No ablation runner or visual plotting scripts.
* **New State:**
  * Created [`scripts/run_ablations.py`](scripts/run_ablations.py) to evaluate the 5 scientific conditions from `RESEARCH_PROTOCOL.md`:
    1. Full CSC
    2. CSC w/o Experience
    3. CSC w/o Computation
    4. CSC w/o Budget Density
    5. Relevance Only
  * Created [`scripts/plot_results.py`](scripts/plot_results.py) generating high-resolution (300 DPI) publication charts in [`results/figures/`](results/figures):
    * **`fig1_performance_vs_budget.png`**: Coverage vs. Token Budget ($200 \dots 4000$).
    * **`fig2_cost_vs_accuracy.png`**: Token Efficiency Frontier.
    * **`fig3_ablation_study.png`**: Component Contribution Bar Chart at 500 tokens.

---

### F. Tests & Configuration Layer
* **Old State:**
  * Basic 2-test file in `tests/test_csc.py`.
* **New State:**
  * 10 unit tests in [`tests/test_csc.py`](tests/test_csc.py), [`tests/test_mem1_dataset.py`](tests/test_mem1_dataset.py), and [`tests/test_llm_eval.py`](tests/test_llm_eval.py) (**10/10 passing**).
  * Added [`.env.example`](.env.example) and [`.env`](.env) with Groq, OpenAI, Gemini, and Ollama templates.
  * Updated [`requirements.txt`](requirements.txt) with `tiktoken`, `python-dotenv`, and `matplotlib`.

---

## 3. Empirical Results Summary

### Live LLM Benchmark Evaluation (`results/llm_eval_summary.csv`)
*Evaluated on Groq using `qwen/qwen3.6-27b` across token budgets:*

| Method | Budget | State Coverage | Exact Match (EM) | F1 Score | Substring Match | Token Cost |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`B0_full_context`** | 200 | 33.3% | 33.3% | 33.4% | 33.3% | 195.3 |
| | **500** | **55.6%** | **33.3%** | **55.6%** | **66.7%** | 497.7 |
| | 1000 | 100.0% | 66.7% | 66.7% | 66.7% | 995.0 |
| **`B1_standard_rag`** | 200 | 11.1% | 66.7% | 66.8% | 100.0% | 160.7 |
| | 500 | 44.4% | 66.7% | 66.7% | 66.7% | 394.7 |
| | 1000 | **77.8%** *(Plateau)* | 66.7% | 66.7% | 66.7% | 601.0 |
| **`B4_csc` (Ours)** | 200 | 11.1% | 66.7% | 66.7% | 66.7% | **181.3** |
| | **500** | **100.0%** 🚀 | **66.7%** | **88.9%** 🚀 | **100.0%** 🚀 | **464.0** |
| | **1000** | **100.0%** | **66.7%** | **88.9%** | **100.0%** | **902.0** |
| **`B5_mem1_style`** | 200 | 33.3% | 66.7% | 66.7% | 66.7% | 77.0 |
| | **500** | **55.6%** | **66.7%** | **66.7%** | **66.7%** | 383.3 |
| | 1000 | 100.0% | 66.7% | 88.9% | 100.0% | 876.7 |

---

### Ablation Study Summary (`results/ablation_summary.csv`)
*Ablations at constrained budget (500 tokens):*

| Ablation Condition | State Coverage @ 500 Tokens | State Precision | Tokens Used |
| :--- | :---: | :---: | :---: |
| **`Full_CSC`** (All Pools + Knapsack) | **100.0%** | **19.4%** | **455.5** |
| **`CSC_no_experience`** (K + C only) | 100.0% | 24.0% | 436.5 |
| **`CSC_no_computation`** (K + E only) | 100.0% | 24.2% | 461.4 |
| **`CSC_no_budget_density`** (Greedy by score) | 100.0% | 20.2% | 472.0 |
| **`Relevance_Only`** (Plain Cosine, No Types) | **75.6%** ❌ | **16.0%** | **480.6** |

---

## 4. How to Run Everything from Scratch

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare MEM1 Benchmark Dataset
```bash
python scripts/prepare_mem1_hotpotqa.py --samples 50
```

### 3. Run Unit Tests
```bash
python -m unittest discover tests
```

### 4. Run Baseline Experiments
```bash
python scripts/run_demo.py --limit 20
```

### 5. Run Live LLM Evaluations (Groq / OpenAI)
```bash
python scripts/run_llm_eval.py --limit 5
```

### 6. Run Ablations & Plot Publication Figures
```bash
python scripts/run_ablations.py --limit 15
python scripts/plot_results.py
```
*(Figures will be saved in `results/figures/`)*
