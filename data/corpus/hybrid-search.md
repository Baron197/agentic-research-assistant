# Hybrid Search and Score Fusion

Hybrid search combines the results of dense (semantic) and sparse (keyword)
retrieval into a single ranked list. Its motivation is that the two methods make
different mistakes: dense retrieval captures meaning but can miss exact terms,
while sparse retrieval nails exact terms but misses paraphrases. Fusing them
recovers relevant results that either method alone would drop.

The hard part of hybrid search is combining scores that live on different scales.
Cosine similarities and BM25 scores are not directly comparable, so naive addition
is biased toward whichever score has the larger range. Two robust solutions are
common. The first is score normalisation, such as min-max scaling each score list
before a weighted sum. The second, and often more stable, is Reciprocal Rank
Fusion (RRF), which ignores raw scores entirely and combines results using their
ranks with the formula one divided by a constant k plus the rank. RRF is popular
precisely because it is simple, parameter-light, and resilient to score-scale
mismatches.

A typical hybrid pipeline retrieves a candidate set from each method, fuses the
lists with RRF or normalised weighting, and then optionally reranks the top
candidates with a cross-encoder. The weighting between dense and sparse can be
tuned per domain; keyword-heavy domains such as legal or code search benefit from
more sparse weight, while conversational domains benefit from more dense weight.

The trade-off hybrid search makes is improved recall and robustness in exchange
for the operational cost of running and maintaining two retrieval systems plus a
fusion step.
