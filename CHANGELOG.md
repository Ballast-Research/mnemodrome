# Changelog

All notable changes to Mnemodrome will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Embedding providers and fingerprints** (#134). New environment variables,
  all `experimental`: `EMBED_PROVIDER=ollama|local-onnx|openai-compatible`
  (default `ollama`, unchanged behaviour); for `local-onnx`: `EMBED_MODEL_DIR`,
  `EMBED_MODEL_URL`, `EMBED_MODEL_DOWNLOAD`, `EMBED_MAX_TOKENS` (2048),
  `EMBED_THREADS` (2); for `openai-compatible`: `EMBED_BASE_URL`, `EMBED_MODEL`,
  `EMBED_API_KEY`. `local-onnx` runs the pinned `nomic-embed-text-v1.5` fp32
  model in-process (new optional extra `onnx`: `onnxruntime==1.31.0`; included
  in the Docker image). The model is downloaded once and verified by sha256;
  a wrong hash fails closed. Offline path in INSTALLATION.md.
- `python -m src.cli.reindex status|build|catch-up|swap|rollback`
  (`experimental`; `build --rebuild-on-drift` re-embeds a same-fingerprint
  collection whose canary drifted into a new `_rN` generation): rebuilds collections with the configured embedder from the
  stored payload text, resumable, then moves the alias; the old collection is
  kept for rollback.
- Stored data (additive, no manual step): Qdrant alias
  `mnemodrome_<type>_live` per memory type; a `mnemodrome_embed_registry`
  collection holding each collection's fingerprint, profile and canary
  vectors; payload keys `embed_fp` and `embed_template` on every new point
  (`embed_*` keys are reserved). New collections are named
  `mnemodrome_<type>__<fingerprint8>`.
- `vector_health` and the vector health check report the embed provider,
  fingerprint and per-collection state (`ok`, `adopted`, `mismatch`,
  `canary_drift`, `untagged`). `semantic_search` adds a warning line when a
  searched collection does not match the active embedder.
- `python -m src.services.embedding.model_store [--fetch DIR | --verify DIR]
  [--url URL]` (`experimental`): lists the pinned model files (URL, size,
  sha256), downloads them into a folder, or verifies a folder (offline path).
  Downloads stop at the pinned size; partial files from a killed download are
  removed at start.
- `scripts/embedding_parity.py`: read-only P1/P2 parity check of the local
  model against Ollama and stored vectors.
- The local tokenizer matches `tokenizers` on the golden set and common text.
  Known limit: for a few hundred rare code points it follows Python's
  `unicodedata` tables, which `tokenizers` does not; the Unicode version is
  stored per collection and logged at start.
- `scripts/release_check.py`: verifies a release tag (format `vX.Y.Z` or
  `vX.Y.ZrcN`, equals the `pyproject.toml` version, CHANGELOG heading, `uv.lock`
  version, legal next step after the existing tags; the next patch of the last
  final tag stays allowed while an rc is open). The release workflow runs it first.
- **Per-agent API keys** for the MCP server (Feature 040). One key per agent,
  stored as SHA-256 digests in a JSON file (`MNEMODROME_AGENT_KEYS_FILE`) and
  compared in constant time. Covers `/mcp`, the SSE endpoints, and every hook
  route. Clients send `Authorization: Bearer <key>` or `X-Mnemodrome-Key: <key>`.
  Rollout switch `MNEMODROME_AUTH_MODE=off|warn|enforce` (default `off`): `warn`
  serves and logs unauthenticated requests and counts them in `/health`;
  `enforce` answers 401. `/health` stays open by default
  (`MNEMODROME_AUTH_EXEMPT_PATHS`). The keys file is re-read on change, no restart.
- `python -m src.cli.agent_keys add|list|rotate|revoke`: prints a new key once,
  stores only its digest, never lists keys or digests.
- Provenance: captures by an authenticated agent carry a `captured_by` property
  (facts, learnings, sessions, procedures; property only, no new labels or
  relationship types), and event-log entries record the agent as `actor` and in
  `metadata.agent`.
- **Memory Browser** in the Admin UI (`/admin/memory`): a read-only, human-readable
  view of every memory layer. Overview with per-layer counts, paths, and health;
  paginated Facts view from Neo4j with entity / relation / validity / date filters
  and a detail panel (triple, timestamps, confidence, supersession chain, linked
  Markdown file); Entity view with relationship table and inline SVG graph;
  semantic Search through Qdrant; newest-first Learnings, Sessions, Procedures,
  Updates, and Graphs views rendered from the Markdown file layer. Server-side
  pagination everywhere and a TTL-cached file index with manual refresh, so
  folders with 28,000+ files stay fast. No new store, no build step, no CDN.
- `MNEMODROME_STORAGE_PATH` is now the umbrella for the file layer: each memory
  type defaults to `<umbrella>/<type>`; a per-type `MNEMODROME_*_PATH` overrides
  only its own folder. Installs that set all six per-type variables are unchanged.
- Admin UI sign-in page with a signed, HttpOnly, SameSite=Strict session
  cookie (Secure behind TLS) and a Log out link. Works with password managers.
  HTTP Basic auth still works for scripts. Failed sign-ins are throttled per
  client. New `ADMIN_SESSION_SECRET` and `ADMIN_SESSION_TTL_HOURS`.
- `ADMIN_WEB_HOST` (default `127.0.0.1`) controls the Admin UI bind address.
- `MEMORY_BROWSER_FILE_INDEX_TTL` (default 300 s).
- Explicit dependency pin `markdown-it-py==4.0.0` (server-side Markdown rendering).
- Documentation: release docs in `docs/release/` (overview, versioning policy with
  the public API surface and stability marks, cut-a-release procedure) and
  decision records in `docs/decisions/` (ADR-001 to ADR-003, proposed).

### Fixed
- Admin UI endpoints that touch Neo4j no longer fail with "attached to a
  different loop": `Neo4jService` keeps one driver per asyncio event loop (the
  MCP server and the Admin UI run on separate loops in one process). The Memory
  Browser owns its own vector-service instance for the same reason.
- `validate_reasoning` (MCP tool and `/validate-reasoning` HTTP route) no longer
  reports a false `clean` result when claim extraction times out, errors, or
  otherwise fails to run. Those cases now return the distinct `degraded`
  status with the failure reason, so a validation that never actually ran is
  never indistinguishable from one that ran and found nothing. Legitimate
  zero-claim results (extraction succeeded, nothing to check) keep returning
  `clean` with `claims_analyzed=0` made explicit in the summary text.
- `load_context_cascade` no longer returns hits with a similarity score but
  an empty body. The formatter was reading the wrong payload key for
  vector-search prompt hits (`text` instead of the `matched_text` key
  `VectorService` actually sets), so every "Previous Similar Questions" hit
  rendered with no text. Each result layer (prompts, learnings, facts,
  entities) now maps its own fields explicitly, and hits with no recoverable
  text — e.g. a stale vector pointing at a record that no longer exists —
  are dropped instead of rendered empty.

### Changed
- Qdrant is now read and written through the `mnemodrome_<type>_live`
  aliases. On the first start after upgrade each alias is pointed at the
  existing `mnemodrome_<type>` collection; data is not changed (#134).
- Deduplication of learnings and procedures is skipped (the capture is
  stored) when the collection's embedding fingerprint or canary does not match
  the active embedder, or could not be checked, instead of comparing scores
  across embedders (#134).
- Vectors are not written into a collection whose fingerprint does not match
  the active embedder, nor while the check cannot run (re-checked at most every
  30 s on the next write, including a collection setup that failed at start).
  The capture is still stored in the graph and files; `vector_health` counts
  blocked writes and mixed points (#134).
- Canary drift with a matching fingerprint (for example after an Ollama
  upgrade on the default `ollama` provider) only warns: search shows a
  warning, dedup is skipped, `vector_health` reports it; writes continue.
  Repair: `python -m src.cli.reindex build --rebuild-on-drift`, then `swap`
  (#134).
- Upgrade note: a fresh install on this version creates
  `mnemodrome_<type>__<fingerprint8>` collections, which earlier versions do
  not read. Upgraded installs keep their `mnemodrome_<type>` collections, so a
  downgrade works until you run `reindex swap` (#134).
- An Ollama fallback host (`OLLAMA_FALLBACK_URL`) that embeds the canary
  texts differently from the primary is disabled at startup (#134).
- Release workflow: the `latest` image tag is pushed only for final tags, never
  for an `rc` tag, and the GitHub release of an `rc` tag is marked as a
  pre-release.
- `ADMIN_WEB_ENABLED` now defaults to `false` in code as well as in Docker Compose.
  Docker Compose publishes the Admin UI port on the host loopback only.
  Upgrade note: set ADMIN_WEB_ENABLED=true (and ADMIN_PASSWORD) to keep the Admin UI on.
- Documentation: README and INSTALLATION.md describe the umbrella storage path,
  how to enable the Admin UI, and a reverse-proxy example.
- The Admin UI and its `/admin/health-check` route report the installed package
  version (from `pyproject.toml`) instead of a fixed string. The README version
  badge shows the latest GitHub release.
- The MCP HTTP server is now started through `run_mcp_transport()` so the agent-key
  middleware wraps the FastMCP app; the stdio transport is unchanged.
- Documentation: README and INSTALLATION.md describe agent keys, rollout, and
  client configuration.

### Deprecated
- `MNEMODROME_HOOK_TOKEN`. It keeps working on the hook routes; in `warn` and
  `enforce` modes it is accepted on the hook routes only, as the key `legacy-hook-token`.
  Move hook clients to per-agent keys.

### Removed
- Librarian admin tools. They were added and removed between
  releases and were never part of a tagged release, so no published interface
  changes.

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
