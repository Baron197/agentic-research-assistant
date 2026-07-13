# Chunking Strategies for Retrieval

Chunking is the process of splitting source documents into smaller passages
before they are embedded and indexed. Chunk size is one of the most influential
and underrated parameters in a retrieval-augmented generation system, because it
determines both what can be retrieved and how much irrelevant text surrounds each
relevant fact.

Small chunks produce precise, focused matches but risk losing the surrounding
context needed to understand a passage. Large chunks preserve context but dilute
the embedding with multiple topics, which lowers retrieval precision and wastes
the model's context window with irrelevant text. A common practical range is 200
to 500 tokens per chunk with a small overlap of 10 to 20 percent so that facts
spanning a boundary are not cut in half.

Fixed-size chunking is simplest but ignores document structure. Recursive
character splitting respects natural separators such as paragraphs and sentences.
Semantic chunking groups sentences by embedding similarity so each chunk covers a
single coherent idea. Structure-aware chunking uses headings, tables, or code
boundaries from the original document.

A powerful refinement is to decouple the unit you search from the unit you return.
Small-to-big retrieval, sometimes called parent-document retrieval, indexes small
chunks for precise matching but returns the larger parent passage to the language
model for context. Sentence-window retrieval is a similar idea that expands a
matched sentence with its neighbors. The right strategy depends on the documents
and the questions, so chunking should be evaluated empirically rather than guessed.
