TITLE
Cognitive State Compilation: Resource-Bounded Construction of
Reasoning States for Long-Horizon LLM Agents

ABSTRACT
## Abstract

Long-horizon LLM agents increasingly rely on retrieval, persistent memory, and external tools to maintain information across extended interactions. However, existing approaches largely treat memory retrieval, experience reuse, and computational artifacts as separate mechanisms, often constructing the model's context through relevance-based retrieval or memory-specific policies. This can lead to redundant computation, excessive context consumption, and inefficient use of heterogeneous information.

We introduce **Cognitive State Compilation (CSC)**, a task-conditioned framework for constructing resource-bounded reasoning states for long-horizon LLM agents. CSC treats previously available knowledge, episodic experiences, and reusable computational artifacts as heterogeneous cognitive resources and learns to select, transform, and compose them into a task-specific state presented to the reasoning model. Rather than maximizing the amount of retrieved context, CSC formulates state construction as a constrained optimization problem that seeks to preserve task-relevant information while respecting a specified resource budget.

We propose an experimental framework comparing CSC with conventional retrieval, memory-augmented retrieval, and relevance-based state selection under controlled context and computation budgets. We evaluate task performance, context consumption, inference cost, retrieval overhead, and robustness to irrelevant or conflicting memory. Our central hypothesis is that explicit cognitive state compilation can improve **performance per unit of computational resource** by reducing unnecessary information and redundant reasoning while retaining information critical to long-horizon task completion.

This work presents cognitive state construction as a distinct research problem at the intersection of agent memory, retrieval-augmented generation, and resource-aware inference, providing a foundation for more efficient long-horizon autonomous agents.

