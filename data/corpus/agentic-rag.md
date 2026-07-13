# Agentic RAG and Multi-Agent Patterns

Agentic RAG extends basic retrieval-augmented generation by letting a language
model plan, use tools, and iterate instead of retrieving once and answering in a
single pass. Rather than a fixed retrieve-then-generate chain, an agent decides
what to search for, evaluates what it found, and can search again until it has
enough evidence. This adaptivity helps with complex, multi-part questions that no
single query can answer.

Several patterns recur. Planning decomposes a hard question into focused
sub-questions that can be researched independently. Tool use lets the agent call
search, fetch a page, run code, or query a database through well-defined
interfaces. Reflection, sometimes implemented as a critic or verifier, reviews a
draft and decides whether it is good enough or needs another iteration. These ideas
were popularised by frameworks for reasoning-and-acting agents and by self-critique
methods.

Multi-agent systems assign these roles to separate cooperating agents: a planner,
one or more researchers, a writer, and a critic, each with a narrow
responsibility. Splitting roles keeps each prompt focused and makes the system
easier to test and reason about than a single monolithic prompt. Orchestration
frameworks such as LangGraph model this as a graph of nodes with explicit edges
and shared state, which makes loops, branches, and human-in-the-loop pauses first
class.

The trade-offs of agentic approaches are real. Extra iterations cost more tokens,
add latency, and can loop indefinitely without guardrails, so production systems
impose iteration caps, token budgets, and schema validation. The benefit is higher
answer quality and reliability on questions that single-pass RAG handles poorly.
