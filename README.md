# Mnemodrome

**Cognitive memory for AI agents.** Six layers. Epistemic governance. Portable via MCP.

[![Version](https://img.shields.io/badge/version-1.0.0-brightgreen.svg)](https://github.com/Ballast-Research/mnemodrome/releases)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io)

---

![Mnemodrome: clone to first memory in under a minute](docs/assets/demo.gif)

*Playback is time-compressed (5× speed, idle trimmed). A real first install runs a few minutes — most of it the one-time image pull and server build — recorded in a clean Docker-in-Docker environment.*

---

## The Problem

Every AI memory system solves retrieval. None of them solve **trust**.

Your AI remembers that your server runs Ubuntu. Then last month you migrated to NixOS. It confidently retrieves the outdated fact. No conflict detection. No temporal awareness. No governance over what it believes.

Mnemodrome fixes this. It is not a vector database with a wrapper. It is a six-layer cognitive architecture with bidirectional conflict detection, temporal validity, and Bayesian confidence scoring. Your AI does not just remember things: it **knows when to doubt them**.

---

## Quick Start

### What Mnemodrome needs from you

Mnemodrome uses two local models. Neither is for chatting with you.

**Embedding model** (`nomic-embed-text`, required): Converts text to vectors so memories are retrieved by meaning, not keyword match. When you store a fact it is embedded; retrieval uses cosine similarity to surface what is semantically relevant. `nomic-embed-text` is the standard local choice (768 dimensions, about 274 MB).

**Extraction/classification model** (required for LLM-assisted features): Reads prose and pulls out entities and relationships in a two-stage pipeline: extract, then classify. This turns "I moved my server to NixOS last week" into a structured, queryable graph fact. A memory system needs a small model of its own because turning free-form text into structured knowledge requires natural-language understanding, not keyword parsing. A narrow, well-defined task like entity extraction does not need a large model: a well-tuned 4B model is faster and costs nothing at inference time when running locally.

Recommended extraction models:
- `gemma3:4b-it-qat` (about 4.0 GB): verified default, good accuracy/speed balance. `gemma4:e4b-it-qat` (Google's newer Gemma 4) is a good alternative for users on a recent Ollama version, but it requires a newer Ollama than some Linux distributions package, which is why `gemma3:4b-it-qat` is the verified default.
- `gemma4:e2b-it-qat` (about 4.3 GB): lighter option, fits in 8 GB VRAM

Qwen models (`qwen3`, `qwen2.5`) also work well as an alternative.

> **Note on fine-tuning:** Mnemodrome performs best with a model fine-tuned specifically for entity extraction. The maintainers run a Gemma-4-E2B extraction fine-tune and a Qwen2.5-1.5B classifier internally and may publish tuned variants later. Any capable small instruct model works to get started.

### Choosing your extraction model

> [!WARNING]
> **Avoid reasoning models.** The Qwen3 family, DeepSeek-R1, and similar "thinking" models emit hidden reasoning tokens that can exhaust the token budget before producing any JSON. When that happens, entity extraction and classification return nothing: no error, no warning, just silence. The graph, conflict detection, and temporal validity all degrade invisibly. This is Mnemodrome's most damaging failure mode precisely because it gives no signal.

Use a non-reasoning instruction-tuned model. The default `gemma3:4b-it-qat` is a safe choice. Qwen2.5, Gemma, and Llama instruct models all work.

**If you must use a reasoning model:** set `LLM_DISABLE_THINKING=true` in your `.env` (already the default). This flag suppresses thinking-token output for Ollama and OpenAI-compatible providers. Caveat: Ollama versions 0.20.3 and earlier ignore the no-think hint on the OpenAI-compatible endpoint and only honor it on the native `/api/chat` API. On those versions, either upgrade Ollama, raise `LLM_MAX_TOKENS` substantially so the model can finish thinking and still emit the JSON, or switch to a non-reasoning model.

Mnemodrome currently uses a single model for both extraction and classification.

### Hardware

Mnemodrome runs wherever your Ollama runs: NVIDIA, AMD, or Apple Silicon. The extraction model fits in as little as 8 GB VRAM (`gemma4:e2b-it-qat` is about 4.3 GB). Concrete data point: the maintainers' extraction fine-tune runs about 55 tokens/sec on an AMD RX 6600.

### Prerequisites

- Docker 24.0+ and Docker Compose v2
- Ollama (or another LLM provider) running and reachable. Mnemodrome does **not** install Ollama.

If using Ollama, pull the required models first:

```bash
ollama pull nomic-embed-text    # embedding model (required)
ollama pull gemma3:4b-it-qat    # extraction model (verified default, ~4.0 GB)
# or the lighter option:
ollama pull gemma4:e2b-it-qat   # (~4.3 GB, fits in ~8 GB VRAM)
```

### Install (agent / CI / scripted)

Most people will let an agent run this. `./install.sh --defaults` accepts all defaults non-interactively:

```bash
git clone https://github.com/Ballast-Research/mnemodrome.git
cd mnemodrome
./install.sh --defaults
```

Override specific settings with flags:

```bash
./install.sh --defaults --data-root /mnt/fast-nvme/mnemodrome --chat-model gemma3:4b-it-qat
./install.sh --dry-run    # full flow without docker compose up
```

### Install (interactive)

For first-time human setup, run without flags. The installer asks **three questions**, each with a smart default:

```bash
./install.sh
```

1. **Install destination**: auto-detected from the git clone; press Enter to confirm.
2. **Bulk data location**: where Neo4j, Qdrant, Redis, and Markdown files are stored. Default: `<install-dir>/data`. Disk space is shown.
3. **LLM endpoint**: auto-detects a running Ollama at `http://localhost:11434`. If found, lists available models and defaults to `nomic-embed-text` + `gemma3:4b-it-qat`. If not found, prompts for provider (ollama/openai/anthropic/openai-compatible), endpoint URL, model names, and API key (hidden input for cloud providers).

Then connect your MCP client:

**Claude Code** (`~/.claude/.mcp.json`):
```json
{
  "mcpServers": {
    "mnemodrome": {
      "type": "http",
      "url": "http://localhost:8765/mcp"
    }
  }
}
```

**Claude Desktop** (`claude_desktop_config.json`) — the desktop app runs local MCP servers as **stdio subprocesses**, so it connects through the [`mcp-remote`](https://www.npmjs.com/package/mcp-remote) bridge rather than a raw URL:
```json
{
  "mcpServers": {
    "mnemodrome": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8765/mcp", "--transport", "http-only"]
    }
  }
}
```
- **Mnemodrome on another host (LAN):** replace `localhost` with its IP **and add `--allow-http`** to the args — `mcp-remote` refuses plain HTTP to a non-localhost URL otherwise.
- **macOS PATH gotcha:** GUI apps launch with a minimal `PATH`. If the app can't find `npx`, use an absolute path (e.g. `/usr/local/bin/npx`) and add `"env": { "PATH": "/usr/local/bin:/usr/bin:/bin" }`.

Mnemodrome defaults to the **Streamable HTTP** transport (the current MCP standard, spec 2025-03-26+), served at `http://<host>:8765/mcp`.

**Legacy SSE clients:** If a client only supports the deprecated HTTP+SSE transport, set `MCP_TRANSPORT=sse` and connect at `http://<host>:8765/sse` with `"type": "sse"`. SSE is provided for backwards compatibility only and may be removed in a future release.

Any MCP-compatible client works, and that includes automation agents, not only interactive assistants. A scheduled or background agent (Hermes, OpenClaw, or any MCP-capable runner) connects to the same server with the same read and write access to the memory substrate as Claude Code, Codex, or Claude Desktop. Your automations and your interactive sessions share one memory. Mnemodrome exposes all six layers as stateless RPC endpoints.

**See [INSTALLATION.md](INSTALLATION.md) for manual setup and flag reference.**

---

## Client compatibility & limitations

Mnemodrome is **self-hosted**: it runs on your machine or your LAN and answers at `http://<host>:8765/mcp`. It works from any client that **executes locally** and can open a connection to that address:

- **Claude Code CLI** (terminal) — direct HTTP.
- **A regular Claude Desktop chat** — via the `mcp-remote` stdio bridge (see above).
- **Cursor, and any MCP-capable runner** on the same host or LAN — including background/automation agents.

> **⚠️ Cloud/hosted Claude sessions cannot reach a self-hosted Mnemodrome.**
>
> Some Claude surfaces run in **Anthropic's cloud**, not on your machine — notably **claude.ai on the web** and the **"Claude Code" / cowork mode inside the Claude Desktop app**. These execute in a remote container and can only see account-level remote connectors; they **cannot** connect to a server on your `localhost` or LAN. If one of them reports that it "can't see" Mnemodrome's tools, nothing is broken — switch to a locally-executing client (the Claude Code CLI, or a regular Desktop chat). Reaching Mnemodrome from a cloud session would require exposing it on a public HTTPS URL behind OAuth, which the self-hosted edition does not do by design.

**Coming later: a hosted Mnemodrome.** A hosted edition — reachable from *every* Claude surface, cloud sessions included, without you running the stack yourself — is on the roadmap. No timeline yet. If that would be useful to you, we want to hear it: tell us how you'd want to use it in **[GitHub Discussions](https://github.com/Ballast-Research/mnemodrome/discussions)**. Demand is what moves it up the list.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Mnemodrome Cognitive Architecture                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  │
│  │  Graph        │  │  Vector       │  │  File         │  │  Event Log    │  │
│  │  (Neo4j)      │  │  (Qdrant)     │  │  (Markdown)   │  │  (SQLite/PG)  │  │
│  │               │  │               │  │               │  │               │  │
│  │  Entities     │  │  Semantic     │  │  Learnings    │  │  Sessions     │  │
│  │  Relations    │  │  Search       │  │  Insights     │  │  Episodes     │  │
│  │  Temporal     │  │  Embeddings   │  │  Procedures   │  │  Audit Trail  │  │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘  │
│          │                  │                  │                  │          │
│          └──────────────────┬──────────────────┘                  │          │
│                             ▼                                     │          │
│  ┌────────────────────────────────────────────────┐               │          │
│  │              Epistemic Governance              │◄──────────────┘          │
│  │                                                │                          │
│  │  ACT-R Cognitive Activation                    │                          │
│  │  Hebbian Co-occurrence Strengthening           │                          │
│  │  Bayesian Confidence Scoring                   │                          │
│  │  Bidirectional Conflict Detection (see hooks)  │                          │
│  │  Temporal Validity Checking                    │                          │
│  └───────────────────────┬────────────────────────┘                          │
│                          ▼                                                   │
│  ┌───────────────────────▼──────────────────────┐  ┌──────────────────────┐  │
│  │  Cache (Redis)                               │  │  Procedural Memory   │  │
│  │  Hot path optimization                       │  │  Diagnostic patterns │  │
│  │  Entity resolution cache                     │  │  Problem-solving     │  │
│  └──────────────────────────────────────────────┘  └──────────────────────┘  │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                    MCP Server (Streamable HTTP Transport)                    │
│            20+ tools · Any MCP client · Stateless RPC                        │
└──────────────────────────────────────────────────────────────────────────────┘
```

Six layers, each optimized for a different cognitive function:

| Layer | Store | Purpose |
|-------|-------|---------|
| **Graph** | Neo4j | Entity relationships, temporal context, graph traversal |
| **Vector** | Qdrant | Semantic similarity search via local Ollama embeddings |
| **File** | Markdown | Human-readable learnings, detailed analysis, Obsidian-compatible |
| **Event Log** | SQLite | Session tracking, episodic memory, audit trail (embedded, no extra container) |
| **Procedural** | Structured | Diagnostic procedures, step-by-step problem-solving patterns |
| **Cache** | Redis | Hot path optimization, entity resolution (<1ms lookups) |
| **Embeddings** | Ollama (user-provided) | Local vector embeddings, dimension probed at install time |

---

## What Makes Mnemodrome Different

### Epistemic Governance

Most memory systems are retrieval engines. Mnemodrome is a **trust engine**.

- **Bidirectional Conflict Detection**: the `validate_prompt` and `validate_reasoning` MCP tools (plus matching HTTP routes) check inbound prompts and outbound responses against stored facts. The system ships ready-to-use example hooks in [`examples/hooks/`](examples/hooks/README.md) that wire these tools to Claude Code's `UserPromptSubmit` and `Stop` events so validation runs automatically on every prompt and response. The same two-interceptor pattern ports to any agent harness or MCP client; registering the hooks (or equivalent middleware) is required for automatic coverage, and the [hooks guide](examples/hooks/README.md) explains how. Without them, you can call the tools explicitly. Catches hallucinations *and* user misconceptions.
- **Temporal Validity**: facts carry `valid_from` and `expired_at`. When you migrate from Ubuntu to NixOS, the old fact is superseded, not deleted. You can query what the system believed on any date.
- **Bayesian Confidence**: each fact carries a confidence score modulated by corroboration, conflict history, and source reliability.
- **ACT-R Cognitive Activation**: memory retrieval weighted by frequency x recency, inspired by Anderson's cognitive architecture (1993). Frequently accessed, recently confirmed facts surface first.
- **Hebbian Strengthening**: facts retrieved together strengthen their association. "Neurons that fire together wire together."

### Portable Memory via MCP

Mnemodrome exposes all 6 layers through the [Model Context Protocol](https://modelcontextprotocol.io) as stateless RPC endpoints. Your memory moves with you:

- **Claude Code**: full MCP integration
- **Claude Desktop**: supported via remote MCP endpoint
- **Any MCP client**: standard protocol, no vendor lock-in
- **Multiple clients simultaneously**: the HTTP transport supports concurrent connections

One `docker compose up` gives any AI agent a complete cognitive layer.

### Privacy-First

- **Local embeddings** via Ollama (nomic-embed-text, 768-dim). No embedding data leaves your machine.
- **All backing stores local**: Neo4j, Qdrant, and Redis run in your Docker stack; the event log is embedded SQLite. No cloud services required.
- **Optional cloud LLM**: use Ollama for fully offline operation, or Anthropic/OpenAI for LLM-assisted features.

### Supply chain & security

- **No npm, no Node.** Mnemodrome is pure Python — nothing is fetched from the npm registry, so the npm-ecosystem worm attacks structurally do not apply.
- **Reproducible, hash-verified builds** from a committed `uv.lock` (`uv pip install --require-hashes`), version-pinned service images, and hash-pinned CI.
- Full posture and private vulnerability reporting: [SECURITY.md](SECURITY.md).

---

## Comparison

| Capability | Mem0 | Zep | Letta | **Mnemodrome** |
|------------|------|-----|-------|-------------|
| Architecture | Single-layer | Graph + vector | Agent framework | **6-layer cognitive** |
| Conflict detection | None | None | None | **Bidirectional** |
| Temporal queries | Weak (58.1% LoCoMo) | Basic | None | **Native validity tracking** |
| Confidence scoring | Proprietary | None | None | **Bayesian + ACT-R** |
| Pattern learning | None | None | None | **Cross-session extraction** |
| Privacy | Cloud default | Cloud default | Cloud default | **Local-first (Ollama)** |
| Transport | REST API | REST API | REST API | **MCP (portable)** |
| License | Proprietary core | Open source | Open source | **Apache 2.0** |

> **Benchmarks:** Independent LoCoMo / LongMemEval evaluations are in progress; scores will be published as a fast-follow after v1.0.

---

## MCP Tools

20+ tools organized by function:

**Capture**
- `capture_fact`: store entity relationships in the graph
- `capture_context`: multi-entity context from natural language
- `capture_learning`: structured insights with deduplication
- `capture_session`: session synthesis and knowledge consolidation

**Retrieve**
- `load_context`: temporal context with entity-aware filtering
- `load_context_cascade`: multi-layer retrieval across all 6 layers
- `semantic_search`: vector similarity search
- `query_patterns`: cross-session pattern intelligence

**Govern**
- `validate_prompt`: inbound conflict detection
- `validate_reasoning`: outbound response verification
- `auto_correct_response`: automatic hallucination correction
- `update_fact`: atomic supersession (old fact invalidated, new fact created)
- `invalidate_fact`: temporal invalidation with reason tracking

**Observe**
- `health`: system-wide health check
- `vector_health`: embedding service status
- `get_event_log_health`: event log statistics
- `query_events`: event log queries with filtering

Each MCP tool is self-describing: any MCP client can enumerate the full tool catalog, parameters, and descriptions at runtime through the standard MCP protocol.

---

## Configuration

Mnemodrome works with four LLM provider types. Configure via `.env`.

**Ollama (local, recommended default):** Simple, private, zero API cost. Embeddings always run through a local Ollama endpoint regardless of which chat provider you choose.

```bash
LLM_PROVIDER=ollama
CHAT_MODEL=gemma3:4b-it-qat
LLM_BASE_URL=http://localhost:11434
```

**Anthropic:**

```bash
LLM_PROVIDER=anthropic
CHAT_MODEL=claude-sonnet-4-6
LLM_API_KEY=sk-ant-your-key
```

**OpenAI:**

```bash
LLM_PROVIDER=openai
CHAT_MODEL=gpt-4o
LLM_API_KEY=sk-your-key
```

**OpenAI-compatible** (covers OpenRouter, xAI/Grok, Groq, Together AI, DeepSeek, Mistral, and self-hosted engines such as vLLM, SGLang, and LM Studio):

```bash
LLM_PROVIDER=openai-compatible
CHAT_MODEL=your-model-name
LLM_BASE_URL=https://openrouter.ai/api/v1   # or your engine's base URL
LLM_API_KEY=your-key
```

Note: even when using a cloud chat provider, the embedding model still needs a local (or reachable) embedding endpoint. The default is Ollama running `nomic-embed-text`. Configure with `OLLAMA_EMBEDDING_HOST` if Ollama is on a different host.

Each backing store can be swapped independently:

| Layer | Default | Alternatives |
|-------|---------|-------------|
| Graph | Neo4j | not swappable (Cypher queries are deeply integrated) |
| Vector | Qdrant | pgvector, Weaviate, Milvus |
| Cache | Redis | KeyDB, Valkey, Memcached |
| Event Log | SQLite (embedded) | PostgreSQL (opt-in via `EVENT_LOG_BACKEND=postgres`, recommended for concurrent/multi-agent) |
| File | Local Markdown | S3, any filesystem |
| Embeddings | Ollama (nomic-embed-text) | OpenAI, Cohere |

SQLite serializes writes and is ideal for a single user or a few agents. For simultaneous multi-agent or team deployments (multiple MCP clients writing events at once) switch to PostgreSQL for parallel writes without write-lock contention: set `EVENT_LOG_BACKEND=postgres` and start with `docker compose --profile postgres up`.

**See [.env.example](.env.example) for all configuration options.**

### Where the Markdown files live

One variable sets the root for the file layer; each memory type gets a folder under it:

```bash
MNEMODROME_STORAGE_PATH=./data/mnemodrome   # -> facts/, learnings/, sessions/, graphs/, updates/, procedures/
```

To move a single type somewhere else (for example learnings into an Obsidian vault), set its per-type variable; the others keep using the root:

```bash
MNEMODROME_LEARNINGS_PATH=/home/you/Vault/Inbox/Learnings
```

Installs that set all six `MNEMODROME_*_PATH` variables keep working without the root.

---

## Memory Browser (Admin UI)

Mnemodrome ships a read-only web view of everything it remembers, so you can audit memory without Obsidian or a database client and at any size (tested with 28,000+ fact files and 50,000 graph facts):

- **Overview**: counts per layer, storage paths, backend health.
- **Facts**: paginated list from Neo4j; filter by entity, relation, validity (valid / invalidated / superseded), and date; click for the full triple, timestamps, confidence, supersession chain, and the fact's Markdown file.
- **Entities**: pick an entity, see its relationships as a table and an inline graph.
- **Search**: one box, semantic search through Qdrant, results link to the matching fact, learning, or session.
- **Learnings, Sessions, Procedures, Updates, Graphs**: newest first from the file layer, rendered Markdown with the header block on top.

Every page reads the layer where it lives. Nothing is copied or re-indexed; large folders are listed once into a cached index with a manual refresh.

**Enable it** (off by default):

```bash
ADMIN_WEB_ENABLED=true
ADMIN_PASSWORD=<a long random password>     # openssl rand -base64 32
```

Then open `http://localhost:8083/admin/memory` (HTTP Basic auth, user `admin`). With Docker Compose the port is published on the host's loopback only. On a bare-metal install it binds to `127.0.0.1` unless you set `ADMIN_WEB_HOST`. To reach it from another machine put a TLS reverse proxy in front; see [INSTALLATION.md](INSTALLATION.md#admin-ui-and-memory-browser) for an nginx example.

The Admin UI also hosts the configuration editor, health dashboard, event viewer, and service controls.

---

---

## Documentation

- **[Installation Guide](INSTALLATION.md)**: Docker Compose quickstart, manual setup, Admin UI and Memory Browser
- **[Contributing](CONTRIBUTING.md)**: How to contribute
- **[Security Policy](SECURITY.md)**: Vulnerability reporting

---

## Contributing

```bash
git clone https://github.com/Ballast-Research/mnemodrome.git
cd mnemodrome
uv sync                      # install dependencies
cp .env.example .env         # configure
docker compose up -d --build  # start backing stores
uv run python -m pytest      # run tests
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Standing on Shoulders

Mnemodrome applies known techniques from prior art. It does not claim to have invented bi-temporal validity, conflict detection, or semantic recall by meaning. Its contribution is integrating these techniques into a single governed, six-layer cognitive system with MCP portability.

- **[Graphiti](https://github.com/getzep/graphiti)** (Zep): The bi-temporal validity model that Mnemodrome implements -- `valid_from`, `expired_at`, and supersession rather than deletion -- is borrowed directly from Graphiti's temporal graph design. The idea that stored facts carry explicit time bounds, allowing point-in-time queries of what the system believed on any given date, originates there. Graphiti also shaped the Neo4j-backed entity and relationship layer.
- **[mem0](https://github.com/mem0ai/mem0)**: Semantic memory retrieval -- surfacing relevant memories by meaning rather than keyword -- was a key reference for Mnemodrome's vector layer design and multi-layer retrieval philosophy.
- **[MuninnDB](https://github.com/scrypster/muninndb)** (MJ Bonanno / Scrypster): an independently developed cognitive database that applies ACT-R activation, Hebbian association, and Bayesian confidence as first-class memory primitives. Mnemodrome drew conceptual inspiration from this approach and reimplemented it independently, sharing no code.
- **[ACT-R](https://en.wikipedia.org/wiki/ACT-R)** (John R. Anderson, 1993): Cognitive activation theory underlying Mnemodrome's retrieval weighting model.
- **[Donald O. Hebb](https://en.wikipedia.org/wiki/Donald_O._Hebb)** (1949): Hebbian learning ("neurons that fire together wire together") underpinning Mnemodrome's co-occurrence strengthening mechanism.
- **[Nate Jones](https://www.natebjones.com)**: The six-layer cognitive framework concept that informed Mnemodrome's architectural thinking.
- **[Daniel Miessler](https://danielmiessler.com)**: PAI methodology inspiration
- **[Anthropic MCP](https://modelcontextprotocol.io)**: Model Context Protocol and FastMCP
- **[Ollama](https://ollama.ai)**: Privacy-preserving local embeddings
- **[Neo4j](https://neo4j.com)**, **[Qdrant](https://qdrant.tech)**, **[Redis](https://redis.io)**: The battle-tested open-source stores backing the graph, vector, and cache layers.
- **[Nomic](https://www.nomic.ai)**: `nomic-embed-text`, the open-weights embedding model behind every semantic recall.
- **[Google Gemma](https://ai.google.dev/gemma)**: The open-weights model family powering entity extraction.

---

## License

Apache License 2.0. See [LICENSE](LICENSE).

---

## Citation

```bibtex
@software{mnemodrome2026,
  author = {Carlos Mendez},
  title = {Mnemodrome: Cognitive Memory for AI Agents},
  year = {2024--2026},
  url = {https://github.com/Ballast-Research/mnemodrome},
  license = {Apache-2.0}
}
```
