# Dense, Sparse, and Hybrid Retrieval

Retrieval methods fall into two broad families. Sparse retrieval represents text
as high-dimensional, mostly-zero vectors over a vocabulary and matches on exact
terms; the classic algorithm is BM25, a bag-of-words ranking function that weights
terms by frequency and rarity. Dense retrieval represents text as compact learned
embeddings and matches on semantic similarity rather than exact words.

Each family has complementary strengths and weaknesses. Sparse BM25 excels at
exact keyword matches, rare terms, codes, and names, and it needs no training, but
it fails when the query and document use different words for the same idea (the
vocabulary mismatch problem). Dense retrieval handles synonyms and paraphrases
well because it matches meaning, but it can miss exact identifiers and may
underperform on domains far from its training data.

Because the two methods fail in different ways, combining them usually beats
either alone. This is the central trade-off behind hybrid search: dense retrieval
buys recall on semantically related content, while sparse retrieval guarantees
that exact-term matches are not lost. The cost is added system complexity, since
two indexes must be maintained and their scores combined.

Choosing a retrieval method is therefore a trade-off between precision on exact
terms, recall on paraphrased meaning, infrastructure complexity, and latency. A
common production pattern is to retrieve a broad candidate set cheaply with both
sparse and dense methods, then apply a more expensive reranking model to the
combined candidates for final ordering.
