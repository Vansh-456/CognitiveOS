# Cognitive State Compilation (CSC) — Project Changelog & Multi-Domain Benchmark

This document provides a comprehensive record of the architecture, datasets, empirical evaluation results, and research tools for Cognitive State Compilation (CSC).

---

## 1. Executive Summary of Architecture

```mermaid
flowchart TD
    CSC["Cognitive State Compilation (CSC) Engine\n(Dense Utility Scorer + Knapsack Budget Optimizer)"]
    
    subgraph Multi-Domain Evaluation Suite (4 Benchmarks)
        D1["1. HotpotQA\n(Multi-Hop Wikipedia Reasoning + Distractors)"]
        D2["2. Natural Questions (NQ)\n(Open-Domain Search Queries)"]
        D3["3. Chained Multi-Objective (Q2, Q4, Q8)\n(Long-Horizon Multi-Turn Dialogues)"]
        D4["4. WebShop\n(Interactive E-Commerce Tool Traces & Product Matching)"]
    end
    
    CSC --> D1
    CSC --> D2
    CSC --> D3
    CSC --> D4
```

---

## 2. Multi-Domain Benchmark Suite

The framework includes all 4 benchmark domains evaluated in the literature (e.g. MEM1 paper, Zhou et al., ICLR 2026):

| # | Benchmark Dataset | File Location | Task Count | Domain & Evaluation Goal |
| :-: | :--- | :--- | :-: | :--- |
| **1** | **HotpotQA** | [`data/mem1_hotpotqa_benchmark.json`](data/mem1_hotpotqa_benchmark.json) | 50 tasks | Multi-hop Wikipedia QA with 8 distractor articles per task. |
| **2** | **Natural Questions (NQ)** | [`data/mem1_nq_benchmark.json`](data/mem1_nq_benchmark.json) | 100 tasks | Real-world open-domain user search queries. |
| **3** | **Chained Multi-Objective** | [`data/mem1_nq_chained_benchmark.json`](data/mem1_nq_chained_benchmark.json) | 100 tasks | Sequential multi-turn dialogue episodes ($Q2, Q4, Q8$). |
| **4** | **WebShop** | [`data/webshop_benchmark.json`](data/webshop_benchmark.json) | 5 tasks | Interactive tool use, price filters, attribute matching, and purchase trajectories. |

---

## 3. Empirical Results Across All 4 Domains

### Multi-Domain Evaluation Summary @ 500-Token Constrained Budget (`results/multidomain_summary.csv`):

| Benchmark Domain | Method | State Coverage | State Precision | Actual Tokens Used |
| :--- | :--- | :---: | :---: | :---: |
| **HotpotQA** (Multi-Hop) | **`B4_csc` (Ours)** | **100.0%** 🚀 | **19.1%** | **459.6** |
| | `B1_standard_rag` | 83.3% | 34.7% | 388.0 |
| | `B2_memory_rag` | 83.3% | 22.0% | 469.3 |
| | `B3_relevance_only` | 83.3% | 16.4% | 486.0 |
| | `B0_full_context` | 60.0% | 17.9% | 496.9 |
| | `B5_mem1_style` | 60.0% | 15.2% | 346.8 |
| **Natural Questions (NQ)** | **`B4_csc`** | **100.0%** | **14.3%** | **373.1** |
| | `B1_standard_rag` | 100.0% | 20.0% | 312.0 |
| | `B5_mem1_style` | 100.0% | 14.3% | 373.1 |
| **Chained Multi-Turn ($Q4$)** | **`B4_csc`** | **100.0%** | **11.0%** | **402.3** |
| | `B1_standard_rag` | 100.0% | 20.0% | 312.0 |
| | `B5_mem1_style` | 100.0% | 11.0% | 402.3 |
| **WebShop** (Tool/E-Commerce) | **`B4_csc`** | **100.0%** | **14.3%** | **244.8** |
| | `B1_standard_rag` | 100.0% | 20.0% | 158.0 |
| | `B5_mem1_style` | 100.0% | 14.3% | 244.8 |

---

### Live LLM Model Evaluation on Groq (`qwen/qwen3.6-27b`):

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

## 4. How to Run Any Benchmark Domain

### 1. Run Baseline Experiments (Single Domain or All 4)
```bash
# Evaluate all 4 benchmark domains together:
python scripts/run_demo.py --benchmark all --limit 10

# Or evaluate individual domains:
python scripts/run_demo.py --benchmark hotpotqa --limit 20
python scripts/run_demo.py --benchmark nq --limit 20
python scripts/run_demo.py --benchmark nq_chained --limit 20
python scripts/run_demo.py --benchmark webshop --limit 10
```

### 2. Run Live LLM Evaluations
```bash
# Live question-answering evaluation on any domain:
python scripts/run_llm_eval.py --benchmark hotpotqa --limit 5
python scripts/run_llm_eval.py --benchmark nq --limit 5
python scripts/run_llm_eval.py --benchmark webshop --limit 5
```

### 3. Run Ablation Suite & Plot Publication Figures
```bash
python scripts/run_ablations.py --limit 15
python scripts/plot_results.py
```
*(Publication figures saved to `results/figures/`)*
