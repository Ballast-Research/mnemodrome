# The Epistemic Governance Thesis

> AI memory is not a retrieval problem. It's a trust problem.

## Where Many Memory Systems Fall Short

Many of the AI agent memory offerings in production today primarily treat memory as a search index. They store vectors. They retrieve by similarity. They return the top-k results. This is simple, and reliable, but not sophisticated. Certainly not sophisticated enough for today's users and today's agents. 

This works until it doesn't. And when it fails, it fails silently.

Your AI remembers that your server runs Ubuntu 24.04. You migrated to NixOS three weeks ago. The vector embedding for "what OS does the server run?" still retrieves Ubuntu with high confidence. No conflict flag. No temporal check. No indication that the retrieved fact might be stale.

The problem isn't retrieval. The problem is that no one built the layer that asks: 
**"Can I/Should I/Do I trust this memory?"**

Mnemodrome, like the others, also stores vectors, but as one component layer of a multi-layer memory system.

## The Six Layers

Most systems use one or two storage backends. Mnemodrome uses six, each optimized for a different cognitive function:

| Layer | Analogy | Function |
|-------|---------|----------|
| Graph (Neo4j) | Semantic memory | Who, what, where, when, and how they relate |
| Vector (Qdrant) | Associative memory | "What's similar to this?": fast semantic search |
| File (Markdown) | Declarative memory | Rich, structured knowledge, human-readable |
| Event Log (SQLite) | Episodic memory | What happened, in what order, in which session |
| Procedural | Motor memory | How to do things: diagnostic procedures, patterns |
| Cache (Redis) | Working memory | What's hot right now: sub-millisecond access |

This isn't over-engineering. Each layer serves queries that the others handle poorly. A graph traversal ("who worked on Project X?") is the wrong tool for semantic search ("find similar debugging patterns"). An event log ("what happened last Tuesday?") is the wrong tool for entity resolution. The layers compose into a retrieval cascade that returns the right type of memory for each query.

## Temporal Validity: A Critical Primitive

In memory systems without temporal validity, facts are treated as either eternal or deleted. There is no middle ground.

Mnemodrome treats temporal validity as a first-class primitive. Every fact carries:

- `created_at`: when it was first stored
- `valid_from`: when it became true (may differ from creation)
- `expired_at`: when it stopped being true (null if still valid)
- `superseded_by`: pointer to the fact that replaced it

This means you can ask: **"What did the system believe about X on date Y?"**, a query that is impossible in systems that destructively update or delete facts.

When you correct a fact, the old version isn't destroyed. It's marked as superseded with full provenance. This creates an audit trail of belief evolution that is essential for debugging, compliance, and understanding how the system's knowledge changed over time.

## Cognitive Science, Not Infrastructure

Mnemodrome is built to leverage three theoretical foundations from cognitive science, not ad-hoc engineering.

### ACT-R Cognitive Activation (Anderson, 1993)

In John Anderson's Adaptive Control of Thought-Rational theory, human memory retrieval is governed by *activation levels*. A memory's activation depends on:

- **Base-level activation**: How frequently and recently the memory has been accessed
- **Associative activation**: How strongly the memory is connected to the current context

Mnemodrome implements this directly. Every entity in the graph carries activation metadata. Retrieval isn't just "most similar": it's "most similar × most relevant right now." A fact retrieved daily has higher activation than one retrieved once six months ago, even if the vector similarity is identical.

```
Activation(i) = ln(Σ t_j^(-d)) + Σ W_j × S_ji
                 ↑ base-level     ↑ associative
```

This means Mnemodrome's retrieval naturally adapts to your working context without explicit configuration.

### Hebbian Strengthening (Hebb, 1949)

"Neurons that fire together wire together."

When two facts are consistently retrieved in the same context, Mnemodrome strengthens their association in the graph. Over time, the system learns which facts belong together (not from explicit labeling, but from observed co-occurrence patterns).

A developer who always retrieves "Redis configuration" alongside "cache invalidation patterns" will find these facts increasingly co-activated. The system builds implicit topic clusters through use, not taxonomy.

### Bayesian Confidence Modulation

Every fact in Mnemodrome carries a confidence score. This score is not static: it's modulated by:

- **Corroboration**: Multiple independent sources confirming the same fact increase confidence
- **Conflict history**: Facts that have been contradicted carry lower confidence
- **Source reliability**: Facts from verified sources score higher than casual assertions
- **Temporal decay**: Confidence decreases as facts age without reconfirmation

When a new fact contradicts an existing one, the system doesn't simply overwrite. It compares confidence scores, checks temporal validity, and either supersedes (with full provenance tracking) or flags for human review. This bidirectional conflict detection ensures coherence in the factual canon, preserving historical information alongside its contemporary successor.

## Portable Memory

Mnemodrome exposes all six of its layers through the Model Context Protocol (MCP) as stateless RPC endpoints. This means:

- Your memory isn't locked to one AI provider
- Multiple AI clients and multiple team members can share the same memory simultaneously
- Move freely between Claude, ChatGPT, and Cursor or VS Code, and whether you use API calls, subscriptions, or local models, data migration is never required
- The memory layer is orthogonal to the agent framework

This is the "portable memory" thesis: your AI's knowledge belongs to you, not to your AI vendor. Mnemodrome makes this practical by implementing an open protocol (MCP) over a self-hosted stack with no cloud dependencies required.

## Design Principles

**Never delete, only supersede.** Destructive updates destroy provenance. Every state change is recorded.

**Confidence is earned, not assumed.** New facts start with moderate confidence. Corroboration increases it. Contradiction triggers review.

**Silence is a failure mode.** If a retrieved fact might be wrong, the system must say so. Silent retrieval of stale data is worse than no retrieval at all.

**Local by default.** Embeddings via Ollama. All storage local. Cloud providers optional, never required.

**Layers compose, not compete.** Each storage layer handles what it's best at. The cascade orchestrates.

## Acknowledgments

This work is inspired by:

- **Mem0**: First agentic memory system I encountered
- **Zep/Graphiti**: Neo4j temporal graph patterns
- **Anthropic**: Model Context Protocol
- **Daniel Miessler**: Personal AI Infrastructure methodology
- **Nate Jones**: The 6-layer cognitive framework concept
- **MJ Bonanno / Scrypster, [MuninnDB](https://github.com/scrypster/muninndb)**: an independently developed cognitive database that applies ACT-R activation, Hebbian association, and Bayesian confidence as first-class memory primitives. Mnemodrome drew conceptual inspiration from this approach and reimplemented it independently, sharing no code.
- **John R. Anderson**: ACT-R: The Atomic Components of Thought (1993)
- **Donald O. Hebb**: The Organization of Behavior (1949)

---

*Mnemodrome is open source under the Apache 2.0 license. The epistemic governance thesis is our contribution to the AI memory landscape: we believe trust, not retrieval, is the fundamental problem to solve.*
