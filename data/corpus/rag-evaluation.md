# Evaluating Retrieval-Augmented Generation

Evaluating a RAG system means measuring two stages separately: retrieval quality
and generation quality. Treating the pipeline as a single black box hides whether
a bad answer came from retrieving the wrong context or from the model misusing
good context.

Retrieval is scored with classic information-retrieval metrics. Recall@k measures
whether the relevant passages appear in the top k results, precision@k measures
how many of the top k are relevant, and ranking-aware metrics such as Mean
Reciprocal Rank (MRR) and Normalised Discounted Cumulative Gain (NDCG) reward
placing the right passage near the top. Strong retrieval is a prerequisite,
because the generator cannot cite evidence it never received.

Generation is scored on faithfulness and relevance. Faithfulness (also called
groundedness) asks whether every claim in the answer is supported by the retrieved
context; answer relevance asks whether the answer actually addresses the question.
Frameworks such as RAGAS and TruLens popularised metrics including faithfulness,
answer relevance, context precision, and context recall, frequently computed with
an LLM acting as a judge.

LLM-as-a-judge evaluation is scalable and correlates reasonably with human
judgment, but it has known biases, including position bias, verbosity bias, and
self-preference for a model's own outputs. Best practice is to combine automated
metrics with a small set of human-labeled golden examples, and to report which
numbers are deterministic plumbing checks versus which depend on a judge model.
Honest evaluation clearly separates what is validated cheaply from what requires a
real model.
