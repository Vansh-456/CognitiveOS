# Cognitive State Compilation: Resource-Bounded Construction of Reasoning States for Long-Horizon LLM Agents

**Target venue:** Resource-Aware Agentic AI @ NeurIPS 2026 (8 pages excl. refs/appendix, deadline Aug 29 2026 AoE)
**Secondary target:** Evaluation of Interactive Agents @ NeurIPS 2026 (4p short / 9p full, same deadline)
**Status:** Day 1 of 9 — literature lock in progress

---

## 1. Problem Statement

Long-horizon LLM agents have access to heterogeneous information and prior computational
artifacts (retrieved knowledge, past experience, previous plans, tool outputs), but existing
approaches optimize storage, retrieval, or memory management **independently** and largely
**per-component**. There is no unified, resource-bounded formulation of how these heterogeneous
resources should be jointly transformed into the transient reasoning state actually handed to
the model for a specific task.

## 2. Research Question

**RQ:** Can a learned cognitive state compiler construct a smaller and more effective reasoning
state from heterogeneous resources than conventional retrieval/context construction, under an
equivalent resource budget?

Sub-questions for the Experiments section:
- **RQ1:** Does CSC improve task performance under fixed resource budgets?
- **RQ2:** Does CSC reduce context/inference cost at comparable performance?
- **RQ3:** Which resource types (knowledge / experience / computation) contribute most to the compiled state?
- **RQ4:** Does CSC generalize across task distributions?
- **RQ5:** How robust is CSC when memory contains noisy or conflicting information?

## 3. Hypothesis

Under a fixed resource budget, task-conditioned cognitive state compilation achieves higher
task performance and/or better performance-per-resource than conventional retrieval and
memory-augmented baselines that are given the same candidate pool but select from it heuristically
or by relevance alone.

## 4. Literature Matrix

| System | Year | Store | Retrieve | Update/Manage | Compute reuse | Budget-aware | Cross-resource state construction | Main limitation (relative to CSC) |
|---|---|---|---|---|---|---|---|---|
| **MemGPT** (Packer et al.) | 2023 | ✓ (tiered: main/recall/archival) | ✓ (agent-issued paging) | partial (LLM decides what to page) | ✗ | ✗ (no explicit budget objective) | ✗ | Treats memory as OS-style virtual memory paged by the LLM itself; optimizes *what's in context*, not a joint budgeted construction over heterogeneous resource types. |
| **A-MEM** (Xu et al.) | 2025 | ✓ (Zettelkasten-style linked notes) | ✓ (top-k + link traversal) | ✓ (memory evolution: retroactive note updates) | ✗ | partial (reports large token reductions, but not budget-constrained by design) | ✗ (single memory type) | Dynamic organization/linking of *one* memory type (episodic notes); doesn't unify knowledge + experience + computation under one budget. |
| **Memory-R1** (Yan et al.) | 2025 | ✓ | ✓ (Answer Agent distills up to 60 candidates) | ✓ (RL-learned ADD/UPDATE/DELETE/NOOP) | ✗ | ✗ | ✗ (memory bank only, no computational-artifact resource type) | Learns *memory editing* policy via RL; doesn't formulate a resource-budgeted compiler across knowledge/experience/compute jointly. |
| **AgeMem** (Yu et al.) | 2026 | ✓ | ✓ | ✓ (unified LTM+STM as tool-based actions, 3-stage RL) | ✗ | partial (token-efficiency reported as a side effect, not the objective) | partial (unifies LTM+STM, not knowledge+experience+computation) | Unifies *two* memory tiers (long/short-term) into one policy; still doesn't treat computational reuse or an explicit budget constraint as first-class. |
| **MemOS** (Li et al.) | 2025 | ✓ (parametric, activation, plaintext via MemCube) | ✓ | ✓ (full lifecycle: scheduling, governance) | ✓ (activation/KV memory as a managed type) | ✗ (systems-level scheduling, not a learned budget-utility tradeoff) | partial (unifies memory *types*, not task-conditioned selection under budget) | Closest in spirit ("OS for memory") but is a systems/infrastructure contribution — no learned, task-conditioned, budget-constrained selection objective; this is also why we must NOT frame our paper as "an OS for agents." |
| **MEM1** (Zhou et al.) | 2025 | ✓ (compact internal state, constant size) | n/a (state carried forward, not retrieved) | ✓ (RL-trained consolidation each turn) | partial (reasoning state doubles as compute trace) | ✓ (implicitly — constant-size state is the mechanism) | ✓✓ (closest prior work — jointly consolidates memory + reasoning into one state) | **Most important comparison.** MEM1 learns end-to-end *when to discard* via RL over a single evolving state; it does not explicitly select/budget *across distinct heterogeneous resource pools* (knowledge vs. experience vs. computation) with an interpretable utility+cost objective — it's recurrent consolidation, not compilation from a candidate pool. This distinction is where CSC's novelty must live. |
| **CSC (ours)** | 2026 | ✓ | ✓ | ✓ | ✓ (explicit computational-artifact resource type) | ✓ (explicit constrained optimization) | ✓ (explicit, the core contribution) | — |

## 5. Novelty Statement (draft, 3 sentences)

Existing agent-memory systems focus primarily on deciding what to store, retrieve, update, or
discard, while agentic retrieval systems optimize the acquisition of external information and
computational-memory systems optimize reuse of prior execution state. These approaches generally
do not formulate the construction of the *transient reasoning state* presented to the model as a
unified resource-allocation problem over heterogeneous representations. We introduce Cognitive
State Compilation (CSC), a task-conditioned mechanism that selects, transforms, and budgets
knowledge and prior computation into a minimum-sufficient reasoning state under fixed resource
constraints.

**Positioning line for Related Work's closing paragraph (vs. MemOS and MEM1 specifically):**
"We study the state-construction problem at the interface between persistent resources and
transient reasoning" — not "an OS for agents" (too close to the AgenticOS workshop and to MemOS),
and not "a better compact recurrent state" (too close to MEM1) — but the explicit, budgeted,
cross-resource-type selection problem that neither treats as the primary object of study.

## 6. Six Questions Per Paper (fill in as you read primary sources, not just abstracts)

| Question | MemGPT | A-MEM | Memory-R1 | AgeMem | MemOS | MEM1 |
|---|---|---|---|---|---|---|
| What problem does it solve? | | | | | | |
| What is its representation? | | | | | | |
| What decision does the system learn? | | | | | | |
| What resource does it optimize? | | | | | | |
| What does it *not* solve? | | | | | | |

(Table 4 above already answers most of this at a summary level — use this table only if you need
finer-grained notes per paper before drafting Related Work §2.)

## 7. Immediate Next Steps (per the Aug 20–29 sprint plan)

- [ ] Aug 20–21: Finish literature lock — read MEM1 and MemOS primary papers in full (not just abstracts); confirm the novelty distinction holds under scrutiny
- [ ] Aug 21: Formalize CSC — definitions, objective, budget, algorithm, pseudocode (Section 9 of the planning doc)
- [ ] Aug 22–24: Build minimal prototype (task encoder → candidate pool → utility scorer → budgeted selector → state composer → LLM)
- [ ] Aug 24–26: Run experiments — baselines B0–B4, 1 benchmark, budgets {500,1000,2000,4000,8000} tokens
- [ ] Aug 26–27: Ablations + figures (Performance vs Budget, Performance vs Cost, Ablation, Qualitative example)
- [ ] Aug 27–28: Write
- [ ] Aug 28–29: Hostile self-review, fix top 3 weaknesses, submit
