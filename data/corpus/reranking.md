# Reranking with Cross-Encoders

Reranking is a second-stage step that reorders an initial set of retrieved
candidates to put the most relevant passages first. The first stage (a bi-encoder
or BM25) is optimised for speed and recall over a large collection; the reranker
is optimised for precision over a small candidate set, usually the top 20 to 100
results.

The key architectural difference is the cross-encoder. A bi-encoder embeds the
query and each document separately and compares the resulting vectors, which is
fast because document vectors can be precomputed. A cross-encoder instead feeds
the query and a candidate document together through the model, letting every query
token attend to every document token. This joint attention produces much more
accurate relevance judgments, at the cost of being far slower because nothing can
be precomputed and every query-document pair must be scored at request time.

This speed-accuracy contrast is exactly why the two are layered: retrieve cheaply
and broadly with a bi-encoder, then rerank a short list expensively and
accurately with a cross-encoder. The reranker typically improves the precision of
the top results substantially, which matters because the language model only sees
the few passages that survive to the top of the list.

The trade-off is added latency and compute per query. Practitioners tune how many
candidates to rerank to balance quality against response time, and some use
distilled or smaller cross-encoders to make reranking affordable in production.
Reranking is one of the highest-leverage upgrades to a basic RAG pipeline.
