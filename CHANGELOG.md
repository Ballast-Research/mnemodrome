# Changelog

All notable changes to Mnemodrome will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-06-15

Initial public release. Mnemodrome is a multi-layer cognitive memory system exposed
as an MCP server, giving agents persistent, conflict-aware, temporally-grounded
memory that lives entirely on your own infrastructure.

### Memory architecture
- Six complementary memory layers behind a single MCP interface:
  - **Graph** (Neo4j): entities, relationships, and temporal context.
  - **Vector** (Qdrant): semantic similarity search over embeddings.
  - **File** (Markdown): structured learnings and session syntheses, readable on disk.
  - **Event log**: episodic, session-aware audit trail. SQLite by default;
    PostgreSQL available opt-in (the `postgres` Docker Compose profile) for concurrent
    multi-agent or team deployments.
  - **Procedural**: reusable diagnostic and problem-solving procedures.
  - **Cache** (Redis): smart TTL caching with graceful fallback.
- Full retrieval cascade across layers: Vector -> Redis -> Neo4j -> Markdown.

### Knowledge integrity
- Temporal validity system: automatic expiration, supersession chains, and
  non-destructive updates, so memory reflects what is true *now* without losing history.
- Conflict detection and resolution with complexity scoring and progressive automation.
- Bidirectional conflict detection validates both inbound user claims and outbound
  model assertions against stored facts. It is exposed as MCP tools (`validate_prompt`,
  `validate_reasoning`) and as HTTP routes, with reference Claude Code hooks in
  `examples/hooks/` for automatic per-turn validation (portable to any harness).
- Deduplication via a semantic-similarity and fuzzy-matching cascade; relationship
  deduplication is baked directly into `capture_fact`, merging duplicate graph edges
  rather than accumulating them.

### Capture and retrieval
- Rich multi-entity context capture from natural language with automatic subgraph
  construction (`capture_context`).
- Factual relationship capture (`capture_fact`) and structured learning capture
  (`capture_learning`).
- End-of-session synthesis and knowledge consolidation with entity linking
  (`capture_session`).
- Cross-session pattern recognition that learns from history across conversations.
- `semantic_search` with a `format='report'` option that returns a Curated Evidence
  Report of ranked, cited excerpts from the vector layer.
- MCP server exposing 20+ tools across Capture, Query, Conflict Management, and
  Administration categories.

### Models and providers
- Local-first via Ollama; embeddings are GPU-accelerated and run comfortably on
  cards with as little as 8 GB of VRAM.
- Additional LLM providers: Anthropic, OpenAI, and any OpenAI-compatible endpoint
  (OpenRouter, Grok, and similar) via `LLM_BASE_URL`.
- Two-stage entity pipeline (extraction then classification) with configurable models.
- `LLM_DISABLE_THINKING` flag to suppress hidden reasoning tokens on reasoning-style
  models that would otherwise return empty extractions.
- `EMBEDDING_DIM` configuration with collection-dimension mismatch detection (768 for
  nomic-embed-text, 1024 for mxbai-embed-large, 384 for all-minilm); the installer
  probes the live endpoint and writes the correct value.

### Deployment and developer experience
- Docker Compose stack: Neo4j, Qdrant, Redis, and the MCP server, with an opt-in
  PostgreSQL event-log profile. All image tags are pinned to specific versions.
- One-command interactive installer (`install.sh`) with three questions: install
  destination (auto-detected from the git clone), bulk data location (`DATA_ROOT`),
  and LLM endpoint (auto-detects Ollama, lists available models, writes correct
  embedding and chat defaults). A `--defaults` flag covers non-interactive installs,
  with `--data-root`, `--llm-endpoint`, `--llm-provider`, `--chat-model`, and
  `--embedding-model` flags for scripted deployments. The installer re-runs safely,
  reusing an existing `NEO4J_PASSWORD` so reinstallation never orphans a sealed database.
- `DATA_ROOT` bind-mount layout for all persistent volumes (Neo4j, Qdrant, Redis,
  Markdown), making data portable and host-inspectable.
- Administration web UI for real-time monitoring and control of all services.

### License
- Released under the **Apache License 2.0**.
