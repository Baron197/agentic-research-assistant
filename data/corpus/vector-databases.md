# Vector Databases and Approximate Nearest Neighbor Search

A vector database stores embeddings and retrieves the ones most similar to a
query vector. Because comparing a query against every stored vector (exact or
brute-force search) becomes too slow at scale, vector databases rely on
approximate nearest neighbor (ANN) algorithms that trade a small amount of recall
for very large speed gains.

The most widely used ANN index is HNSW (Hierarchical Navigable Small World), a
graph-based structure that navigates from coarse to fine layers to find close
neighbors in logarithmic time. Other common approaches include IVF (inverted file
indexes that cluster vectors into cells) and product quantization (PQ), which
compresses vectors to reduce memory at some accuracy cost. IVF and PQ are often
combined for billion-scale collections.

Key tuning parameters trade recall against latency. For HNSW, a larger
``ef_search`` value explores more of the graph and improves recall at the cost of
speed, while ``M`` controls how many neighbors each node keeps. Choosing these
values is an empirical balance specific to a dataset and a latency target.

Production vector databases such as FAISS, Pinecone, Weaviate, Milvus, and qdrant
add metadata filtering, persistence, sharding, and hybrid search on top of the raw
index. Metadata filtering is important because real systems usually need to
restrict results by attributes like date, source, or access permissions in
addition to semantic similarity. The central trade-off of every vector database is
recall versus latency versus memory, and good systems make that trade-off
explicit and tunable.
