# CSC Research Protocol

## Main hypothesis

Under the same resource budget, CSC should achieve better task performance
and/or better performance-per-resource than conventional retrieval and
memory-augmented context construction.

## Required baselines

B0 Full Context
B1 Standard RAG
B2 Memory-Augmented RAG
B3 Relevance-only Selection
B4 CSC
B5 MEM1-style recurrent consolidation approximation

## Fairness rules

1. Same base LLM across LLM baselines.
2. Same candidate resource pool where applicable.
3. Same token budget.
4. Same task distribution.
5. Temperature 0 for deterministic evaluation.
6. Count prompt tokens with the exact target-model tokenizer in final runs.
7. Report retrieval/tool latency separately from generation latency.
8. Do not call the offline rule-based demo an LLM benchmark.
9. Do not call B5 a MEM1 reproduction.

## Primary metrics

- task accuracy / success rate
- total input tokens
- total inference cost
- end-to-end latency
- state coverage
- state precision
- compression ratio

## Robustness

Inject irrelevant, stale, conflicting, and misleading resources.

## Ablations

- CSC without experience
- CSC without computation
- CSC without explicit budget
- CSC without resource-type information
- random selection
- relevance-only selection

## Publication-quality extension

The current prototype uses TF-IDF utility scoring. Replace it with a learned
scorer:

    u_theta(task, resource, context) -> utility

A simple pairwise objective is:

    L = -log(sigmoid(u_positive - u_negative))

where positive/negative resources can be derived from successful trajectories.

Then evaluate the learned scorer against relevance-only selection.

## Novelty boundary

CSC should be presented as a formulation and mechanism for task-conditioned
construction of a transient reasoning state under an explicit resource budget.

Do not claim that memory management, adaptive retrieval, persistent KV memory,
or OS-style memory management are individually new.
