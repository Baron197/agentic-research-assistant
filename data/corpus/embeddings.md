# Text Embeddings

Text embeddings are dense numerical vectors that represent the meaning of a
piece of text in a continuous high-dimensional space. Models map words,
sentences, or whole documents to vectors so that semantically similar texts land
close together, while unrelated texts land far apart. This property is what makes
embeddings the backbone of semantic search and retrieval-augmented generation.

Similarity between two embeddings is most commonly measured with cosine
similarity, which compares the angle between vectors and ignores their magnitude.
Euclidean (L2) distance and dot product are also used; many modern embedding
models are trained so that dot product and cosine similarity rank results
identically because the vectors are normalised to unit length.

Embedding dimensionality typically ranges from a few hundred to a few thousand.
Larger dimensions can capture more nuance but cost more memory and make indexes
slower, so dimensionality is a practical trade-off. Some models support
Matryoshka representation learning, which lets a single model produce vectors
that can be safely truncated to a shorter length with graceful quality loss.

Embeddings are produced by encoder models trained with contrastive objectives on
large collections of paired texts. A key limitation is that a single dense vector
compresses an entire passage into one point, so fine-grained or rare details can
be lost; this is one reason hybrid search and reranking are often added on top.
Embeddings should also be generated with the same model at indexing time and
query time, since vectors from different models are not comparable.
