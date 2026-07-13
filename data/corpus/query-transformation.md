# Query Transformation and Routing

Query transformation improves retrieval by rewriting or expanding the user's
original question before it reaches the index. Users phrase questions in ways that
do not always match how source documents are written, so transforming the query
can dramatically raise recall without changing the underlying corpus.

Common techniques address different problems. Query expansion adds synonyms or
related terms so that keyword retrieval catches more relevant documents.
Multi-query retrieval generates several paraphrases of the question, retrieves for
each, and merges the results, which smooths over any single bad phrasing. HyDE,
short for Hypothetical Document Embeddings, asks a model to draft a hypothetical
answer and then retrieves documents similar to that answer rather than to the
question; this works because an answer often resembles the target passage more
than the question does. Step-back prompting asks a more general version of the
question first to retrieve broad context.

Decomposition is essential for complex questions. A multi-part question is split
into focused sub-questions that are each researched independently, and the partial
findings are then synthesised. This is the same planning step that agentic systems
rely on, and it pairs naturally with a per-sub-question evidence-gathering loop.

Routing chooses where a query should go. A router might send a question to a
specific index, to a SQL database, or to a web search depending on its type. The
trade-off of all these techniques is extra model calls and latency in exchange for
better recall and more relevant context, so they should be applied where
evaluation shows they actually help.
