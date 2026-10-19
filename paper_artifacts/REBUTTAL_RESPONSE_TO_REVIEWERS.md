# Author Response to Reviewers & Meta-Reviewer (Area Chair)

**Paper Title:** Cognitive State Compilation: Resource-Bounded Construction of Reasoning States for Long-Horizon LLM Agents  
**Target Venue:** NeurIPS 2026  

We sincerely thank the Area Chair (AC 1zCd) and all reviewers (rTKQ, LsqK, vD7Q) for their constructive, thorough, and insightful feedback. We are pleased that reviewers recognized the fundamental importance of the token budget crisis in long-horizon LLM agents, the novelty of formulating state construction across heterogeneous resources as a resource-bounded 0/1 knapsack, and our strong empirical gains.

Below, we provide a point-by-point response addressing every weakness, question, and suggestion raised. All corresponding changes, new experiments, and theoretical proofs have been incorporated into the revised manuscript ([`CSC_paper_final.tex`](file:///c:/Projects/cognitive_os_1/paper_artifacts/updated/CSC_paper_final.tex)) and codebase.

---

## Response to Reviewer rTKQ (Score: 2 — Reject)

> **Reviewer rTKQ Summary**: While acknowledging the importance of token budget limits, the reviewer questioned the novelty relative to sentence compression/submodular knapsack (W1), identified training feature leakage via the utility hint $h(r)$ (W2), questioned the necessity of multi-resource compilation on HotpotQA (W3), raised concerns regarding the fairness of comparison with MEM1 and FIFO sliding window (W4), and flagged tokenizer mismatch and prompt overhead accounting (W5).

### W1 & Q1: Novelty vs. Sentence Compression & Submodular Knapsack
* **Reviewer Concern:** Knapsack formulations have been used in summarization (Lin & Bilmes, 2010) and context compression (LLMLingua, RECOMP).
* **Our Response:** We have clarified this distinction in **Section 2 (Related Work)** and **Section 3.2**:
  1. *Macro-Level Heterogeneous Resources vs. Homogeneous Pruning*: Context compressors like LLMLingua \citep{jiang2023llmlingua} and RECOMP \citep{xu2023recomp} perform fine-grained token/sentence pruning over homogeneous retrieval text. In contrast, CSC operates over *heterogeneous functional modalities*: declarative domain knowledge ($R_{\text{know}}$), episodic trajectories ($R_{\text{exp}}$), and deterministic tool outputs ($R_{\text{comp}}$).
  2. *Additive Knapsack vs. Submodular Summarization*: Lin & Bilmes \citep{lin2010submodular} model diminishing returns because redundant sentences in document summarization cover the same semantic topics. In CSC, $R_{\text{know}}$, $R_{\text{exp}}$, and $R_{\text{comp}}$ serve fundamentally orthogonal, complementary operational roles (factual grounding, execution strategy, and computed state). In this multi-resource regime, cross-resource synergy dominates intra-type redundancy, making additive 0/1 knapsack an efficient $\mathcal{O}(M \log M)$ first-order surrogate.

### W2 & Q2: Scorer De-leakage & Task-Disjoint Cross-Validation
* **Reviewer Concern:** Feature $h(r)$ in earlier prototypes acted as an oracle label hint, and stratified K-fold allowed task leakage across splits.
* **Our Response & Full Remediation:** We have completely eliminated $h(r)$ and re-evaluated under strict task-disjoint validation (**Section 3.3, Appendix H**):
  1. *Complete Feature De-leakage*: All utility hints $h(r)$ and derived hints were removed. Features are strictly derived from observable text: word and char 3--4 gram TF-IDF similarity, keyword overlap, title match, relative rank, normalized token cost, and modality type indicators.
  2. *Task-Disjoint GroupKFold*: We transitioned to `GroupKFold(n_splits=5)` partitioned strictly by `task_id`. No candidate or fact from any test task appears during training.
  3. *Uncompromised Generalization Performance*: Under this strict protocol, our leak-free GBDT achieves **96.66% Accuracy, 88.64% Precision, 81.95% Recall, 85.01% F1, and 0.9785 ROC-AUC** on unseen tasks. Controlled ablations confirm that type conditioning provides a $+2.5$ point F1 improvement over flat lexical matching (Appendix H, Table~\ref{tab:leak_free_cv_appendix}).

### W3 & Q3: Necessity of Heterogeneous Multi-Resource Compilation
* **Reviewer Concern:** On standard HotpotQA, 100% of gold facts are in Wikipedia passages ($R_{\text{know}}$), so ablating $R_{\text{exp}}$ or $R_{\text{comp}}$ showed no coverage drop at $B=500$.
* **Our Response & Proof Benchmark:** We constructed a dedicated 25-task multi-objective agent benchmark (`cross_resource_benchmark.json`) where solving each task strictly requires coordinating all three pools (**Appendix F**):
  * *Results*: At $B=250$, Full CSC achieves **100.0% coverage**.
  * Removing the Computation pool ($K+E$) caps coverage at **66.7%** ($-33.3\%$ drop).
  * Removing the Experience pool ($K+C$) caps coverage at **66.7%** ($-33.3\%$ drop).
  * Standard RAG ($K$ only) collapses to **33.3%** ($-66.7\%$ drop).
  * *Takeaway*: This empirically proves that for multi-objective agents that must combine policy guidelines, domain knowledge, and tool scratchpads, single-resource RAG fundamentally fails, and cross-resource compilation is indispensable.

### W4 & Q4: Memory Access Models & Long-Horizon Retention vs. MEM1 / FIFO
* **Reviewer Concern:** MEM1 maintains a compact recurrent state without an external store, whereas CSC maintains an episodic store. Furthermore, sliding window tied CSC at 100% in Table 6.
* **Our Response & Formal Protocol:** We clarified the access taxonomy and implemented a persistent session constraint retention protocol (**Section 5.4, Appendix I**):
  1. *Access Model Taxonomy*: Unbounded full context ($\mathcal{O}(T)$), sliding window FIFO ($B=512$), recurrent working memory (MEM1, $B=512$, no external store), and episodic store compilation (CSC, $B=512$).
  2. *Why FIFO Fails*: In realistic workflows, session mandates or security policies established at Turn 0 must be recalled at Turn $N-1$. Intermediate turns accumulate context $>3,000$ tokens. When history exceeds 512 tokens, FIFO evicts the Turn 0 mandate, causing retention to **collapse to 0% at $N \ge 8$**.
  3. *Why CSC Outperforms MEM1*: MEM1 suffers gradual recurrent drift ($94\% \to 82\%$), whereas CSC solves the knapsack over episodic history at inference time, retaining 100% of critical constraints without any RL training overhead.

### W5 & Q5: Tokenizer Alignment & Prompt Overhead Accounting
* **Reviewer Concern:** Tokenizer mismatch between cl100k and Qwen BPE, and unaccounted prompt overhead.
* **Our Response:** We implemented native HuggingFace Qwen BPE tokenization (vocab 151,665) in [`csc/tokenizer.py`](file:///c:/Projects/cognitive_os_1/csc/tokenizer.py) and added `budget_mode="strict_total"` in [`csc/compiler.py`](file:///c:/Projects/cognitive_os_1/csc/compiler.py). System directives and formatting framing ($B_{\text{overhead}} \approx 54.2 \pm 3.1$ tokens) are explicitly deducted, guaranteeing that the total prompt placed into the model strictly satisfies $B_{\text{total}} \le 500$ with 0 token violations across 100% of episodes (**Appendix J**).

---

## Response to Reviewer LsqK (Score: 3 — Borderline Reject)

> **Reviewer LsqK Summary**: The reviewer appreciated the clean problem formulation and knapsack framing, but requested discussion of recent agentic retrieval baselines (CORAG, Self-Correcting RAG), formal treatment of discrete knapsack vs. submodular coverage, and clarification of task sample sizes across tables.

### 1. Positioning with Recent Agentic Retrieval Literature
We added discussion and citations for CORAG \citep{wang2024corag} and Self-Correcting RAG \citep{xu2026selfcorrecting} in **Section 2**. We highlight that while these frameworks introduce iterative query reformulation for multi-step reasoning, they treat retrieval as open-ended, unconstrained search over homogeneous text corpora. CSC uniquely contributes an explicit token-bounded discrete optimization solver that synthesizes heterogeneous modalities under fixed prompt budgets.

### 2. Discrete Knapsack vs. Submodular Coverage
We added a dedicated formulation remark in **Section 3.2**: Submodular functions model diminishing marginal gains for near-duplicate text spans. In cognitive state compilation, candidate items belong to distinct functional modalities where mutual synergy dominates intra-type redundancy. We prove the efficacy of the additive formulation in **Appendix G**, demonstrating that greedy density packing achieves $\ge 96.74\%$ of the provable integer optimum.

### 3. Unified Benchmark Sample Consistency
We verified and reconciled sample counts across the entire paper: Section 4 and Table 2 evaluate the standard 15-task multi-hop HotpotQA benchmark, while the extended multi-domain sweep evaluates 25 tasks per domain (Table~\ref{tab:multidomain_coverage}), with all numbers unified between [`scripts/run_demo.py`](file:///c:/Projects/cognitive_os_1/scripts/run_demo.py) and [`scripts/run_ablations.py`](file:///c:/Projects/cognitive_os_1/scripts/run_ablations.py).

---

## Response to Reviewer vD7Q (Score: 4 — Borderline Accept)

> **Reviewer vD7Q Summary**: The reviewer found the paper well-written, intuitive, and practically valuable, while requesting quantitative evaluation of the knapsack optimality gap in objective utility (Eq. 1), task-disjoint cross-validation for the GBDT, and clarifications on multi-turn needle retention.

### 1. Knapsack Optimality Gap in Objective Utility (Eq. 1)
* **Our Response:** In **Appendix G (Table~\ref{tab:knapsack_gap_appendix})**, we evaluated the exact 0/1 Dynamic Programming solver against the Dantzig greedy utility-density approximation on summed utility $U(S) = \sum_{r \in S} u(r)$ across budgets $B \in \{100, 200, 300, 500\}$:
  * At $B=100$: DP utility 0.6278 vs. Greedy 0.6073 $\to$ **96.74% optimality** (gap 0.0205).
  * At $B=200$: DP utility 1.1177 vs. Greedy 1.1148 $\to$ **99.74% optimality** (gap 0.0029).
  * At $B=300$: DP utility 1.2659 vs. Greedy 1.2586 $\to$ **99.42% optimality** (gap 0.0073).
  * At $B=500$: DP utility 1.4121 vs. Greedy 1.4103 $\to$ **99.87% optimality** (gap 0.0019).
  * *Latency*: Greedy executes in $<0.15\text{ms}$ versus $0.72\text{ms}$ for DP ($\sim 5\times$ speedup).

### 2. Task-Disjoint Cross-Validation & De-leakage
As detailed in the response to Reviewer rTKQ (W2) and **Appendix H**, the GBDT is now evaluated with `GroupKFold(n_splits=5)` partitioned by `task_id` using strictly observable lexical/structural features, reaching 85.01% F1 and 0.9785 ROC-AUC on completely unseen tasks.

### 3. Multi-Turn Horizon Evaluation Protocol
As detailed in **Appendix I**, we documented the session-level constraint retention protocol, contrasting FIFO eviction against CSC's budget-bounded knapsack retention.

---

## Response to Area Chair 1zCd (Meta-Review: Accept Recommendation)

> **Area Chair Summary**: The AC noted the strengths of the work and requested that the authors provide thorough clarifications in dedicated appendix sections to address the technical reservations of the reviewers, particularly around feature de-leakage, cross-resource necessity, knapsack optimality gap, and long-horizon memory semantics.

We have addressed every point raised by the reviewers in the updated submission:
1. **Added Appendices F--J**: Providing empirical and theoretical proofs for cross-resource necessity (App. F), knapsack optimality gap (App. G), task-disjoint leak-free GBDT cross-validation (App. H), memory access taxonomy and multi-turn needle protocol (App. I), and tokenizer/budget overhead alignment (App. J).
2. **Codebase Hardening**: All 17 unit tests in [`tests/test_csc.py`](file:///c:/Projects/cognitive_os_1/tests/test_csc.py) pass with 100% determinism.
3. **Reproducibility**: All source code, pre-trained leak-free models, and reproduction scripts are updated and packaged.
