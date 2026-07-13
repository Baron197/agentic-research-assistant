# Hallucination and Grounding

A hallucination is a confident statement produced by a language model that is not
supported by its input or by facts. In retrieval-augmented generation,
hallucinations usually take one of two forms: the model contradicts the retrieved
context, or it adds details that the context never mentioned. Both undermine trust,
which is why grounding is the core promise of RAG.

Grounding means tying every generated claim to specific evidence. The most direct
defenses are to instruct the model to answer only from the provided context, to
require inline citations for each claim, and to verify after generation that each
claim's cited source actually supports it. Retrieval quality matters too: if the
relevant passage was never retrieved, even a perfectly obedient model cannot ground
its answer and is tempted to fill the gap.

A robust safeguard is to allow and even reward abstention. A system that can say
"the provided sources do not contain this information" is more trustworthy than one
that always produces an answer. Enforcing citations in code makes this concrete: if
a claim cites a source that was not actually gathered, the claim should be removed
rather than shown, so it is structurally impossible to present a fabricated source.

A verifier or critic step is an increasingly common pattern. After drafting, a
separate pass checks each claim against its cited evidence and removes or flags any
claim that is unsupported, optionally looping back to gather more evidence. This
reflection loop measurably reduces unsupported claims compared with single-pass
generation, at the cost of extra model calls and latency.
