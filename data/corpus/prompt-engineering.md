# Prompt Engineering for Grounded Generation

Prompt engineering is the practice of structuring the instructions and context
given to a language model to make its outputs more accurate, reliable, and useful.
In retrieval-augmented generation the prompt is where retrieved evidence, task
instructions, and output-format requirements come together, so small changes here
have outsized effects on faithfulness.

Several techniques are well established. Clear role and task instructions reduce
ambiguity. Few-shot examples show the model the desired format and style. Chain-of-
thought prompting, which asks the model to reason step by step, improves
performance on multi-step reasoning at the cost of extra tokens. Explicitly
instructing the model to answer only from the provided context, and to say when the
context is insufficient, is one of the most effective ways to reduce hallucination.

Requesting structured output is especially valuable in engineered systems. Asking
for JSON that conforms to a schema, and then validating it, turns a free-text model
into a reliable component; if validation fails, the system can retry with the error
message appended. This validate-and-retry loop is a practical guardrail that keeps
downstream code simple.

Citation discipline belongs in the prompt too. Instructing the model to attach an
explicit source identifier to every claim makes after-the-fact verification
possible, because a critic step can then check each claim against the source it
names. Prompts should also be treated as versioned artifacts and evaluated against
a golden set, because a prompt that helps one model or task can hurt another. Like
chunking, prompt design should be measured, not assumed.
