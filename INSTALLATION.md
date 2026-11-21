# Mnemodrome Installation Guide

Complete installation guide for Mnemodrome. The recommended path is the
one-command installer (`./install.sh`). Manual installation is documented
as an alternative.

Mnemodrome does **not** bundle Ollama or any cloud LLM provider. You need
to bring your own LLM backend. Ollama is the recommended local option.

---

## Table of Contents

1. [Quick Start: one-command installer](#quick-start-one-command-installer)
2. [Installer flags reference](#installer-flags-reference)
3. [Manual Installation](#manual-installation)
4. [LLM Provider Configuration](#llm-provider-configuration)
5. [Embedding Provider (experimental)](#embedding-provider-experimental)
6. [MCP Client Setup](#mcp-client-setup)
7. [Verification](#verification)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start: one-command installer

### Prerequisites

- Docker 24.0+ and Docker Compose v2 plugin
- 8 GB RAM minimum (16 GB recommended), 20 GB disk
- Your own LLM backend. Ollama is the most common self-hosted option.
  Mnemodrome does NOT install Ollama for you.

### If you use Ollama (recommended for local/private deploys)

```bash
# Install Ollama from https://ollama.ai, then pull models:
ollama pull nomic-embed-text    # embedding model (required)
ollama pull gemma3:4b-it-qat    # extraction model (verified default, ~4.0 GB)
# lighter option:
ollama pull gemma4:e2b-it-qat   # (~4.3 GB, fits in ~8 GB VRAM)
```

The installer auto-detects a running Ollama at `http://localhost:11434`
and defaults to these models. You can override at the prompt.

### Run the installer

```bash
git clone https://github.com/Ballast-Research/mnemodrome.git
cd mnemodrome
./install.sh
```

**Three questions, each with a smart default:**

```
Q1: Install directory
    Installing here: /path/to/mnemodrome - press Enter to confirm, or give another path:

Q2: Bulk data location  (Neo4j, Qdrant, Redis, Markdown files)
    Default: /path/to/mnemodrome/data   Available disk: 42GB
    Enter path [/path/to/mnemodrome/data]:

Q3: LLM endpoint
    Ollama detected at http://localhost:11434
    Available models: gemma3:4b-it-qat nomic-embed-text ...
    Embedding model [nomic-embed-text]:
    Chat model      [gemma3:4b-it-qat]:
```

The installer then:
- Generates a secure `.env` (random passwords, never printed)
- Probes the embedding endpoint to write `EMBEDDING_DIM` automatically
- Translates `localhost` → `host.docker.internal` for the container-side URLs
- Runs `docker compose up -d --build`
- Polls `http://localhost:8765/health` until healthy (180 s timeout)
- Prints MCP client config snippets

### Connect your client

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

**Claude Desktop** (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "mnemodrome": {
      "url": "http://localhost:8765/mcp"
    }
  }
}
```

Restart your client after updating the config.

---

## Installer flags reference

```
./install.sh --defaults              # non-interactive; accept all defaults (CI mode)
./install.sh --dry-run               # full flow without docker compose up
./install.sh --data-root PATH        # override Q2 data location
./install.sh --llm-endpoint URL      # override Ollama/LLM base URL
./install.sh --llm-provider X        # ollama | openai | anthropic | openai-compatible
./install.sh --chat-model X          # chat model name
./install.sh --embedding-model X     # embedding model name
```

Flags can be combined:
```bash
# Fully scripted CI install
./install.sh --defaults \
  --data-root /mnt/fast-nvme/mnemodrome \
  --llm-endpoint http://gpu-server:11434 \
  --chat-model mistral:7b \
  --embedding-model nomic-embed-text

# Prove the flow without touching Docker
./install.sh --defaults --dry-run
```

**Re-running the installer** is safe: it backs up the existing `.env` to
`.env.bak.<timestamp>` before writing a new one.

**Changing the embedding model** after install: update `EMBEDDING_DIM` in
`.env`, delete the affected Qdrant collections, restart Mnemodrome, then
re-run `scripts/bootstrap_vector_embeddings.py` to re-embed your data.
Mnemodrome logs a clear error at startup when it detects a dimension mismatch.

---

## Manual Installation

For custom deployments or development environments.

### Prerequisites

- **Python 3.11+** (`python --version`)
- **Neo4j 5.0+** ([Install Neo4j](https://neo4j.com/download/))
- **Ollama** ([Install Ollama](https://ollama.ai)) - For embeddings (pull models as shown above)
- **Redis 7.0+** (optional, recommended) ([Install Redis](https://redis.io/download/))

### Step 1: Install System Dependencies

**Ubuntu/Debian**:
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip redis-server
```

**macOS** (via Homebrew):
```bash
brew install python@3.11 redis
brew install --cask neo4j
```

**Install Ollama**:
```bash
# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# macOS
brew install ollama

# Windows - Download from https://ollama.ai
```

### Step 2: Clone Repository

```bash
git clone https://github.com/Ballast-Research/mnemodrome.git
cd mnemodrome
```

### Step 3: Create Python Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# OR
.venv\Scripts\activate     # Windows
```

### Step 4: Install Python Dependencies

```bash
# Install dependencies using uv
uv sync
```

**Note**: If you don't have `uv` installed, install it first:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 5: Start Neo4j

**Using Neo4j Desktop**:
1. Download [Neo4j Desktop](https://neo4j.com/download/)
2. Create new project
3. Add local database
4. Set password
5. Start database

**Using Docker**:
```bash
docker run -d \
  --name mnemodrome-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your-password \
  -v $HOME/neo4j/data:/data \
  neo4j:5-community
```

**Using Package Manager**:
```bash
# Ubuntu/Debian
sudo systemctl start neo4j
sudo systemctl enable neo4j

# macOS
brew services start neo4j
```

### Step 6: Start Redis (Optional but Recommended)

```bash
# Ubuntu/Debian
sudo systemctl start redis
sudo systemctl enable redis

# macOS
brew services start redis

# Docker
docker run -d --name mnemodrome-redis -p 6379:6379 redis:7-alpine
```

### Step 7: Start Ollama and Pull Models

```bash
# Start Ollama service
ollama serve  # Run in background or separate terminal

# Pull embedding model (required)
ollama pull nomic-embed-text

# Pull LLM model (if using Ollama as provider)
ollama pull gemma3:4b-it-qat  # or mistral, qwen2.5, etc.
```

### Step 8: Configure Environment

```bash
# Copy example configuration
cp .env.example .env

# Edit configuration
nano .env  # See LLM Provider Configuration section below
```

**Manual installation .env example**:
```bash
# Data root (used by docker-compose.yml bind mounts; not needed for non-Docker manual installs)
DATA_ROOT=$HOME/.mnemodrome

# Embedding dimension: set to match your model's actual output
# nomic-embed-text → 768  |  mxbai-embed-large → 1024  |  all-minilm → 384
EMBEDDING_DIM=768

# Neo4j (adjust if different)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-neo4j-password
NEO4J_DATABASE=neo4j

# Redis (adjust if different)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# LLM Provider (Ollama is user-provided; Mnemodrome does not install it)
LLM_PROVIDER=ollama
CHAT_MODEL=gemma3:4b-it-qat
LLM_BASE_URL=http://localhost:11434

# Ollama embedding endpoint (same as LLM_BASE_URL for single-host setups)
OLLAMA_EMBEDDING_HOST=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Storage root for Markdown memory files. Each type lives in
# $MNEMODROME_STORAGE_PATH/<type> (facts, learnings, sessions, graphs,
# updates, procedures).
MNEMODROME_STORAGE_PATH=$HOME/.mnemodrome

# Optional: move one type somewhere else (e.g. into an Obsidian vault).
# A per-type var overrides only that folder.
# MNEMODROME_LEARNINGS_PATH=$HOME/Documents/MyVault/Inbox/Learnings

# MCP Server
MCP_PORT=8765
MCP_HOST=0.0.0.0
```

### Step 9: Create Storage Directories

```bash
mkdir -p ~/.mnemodrome/{facts,learnings,sessions,graphs,updates,procedures,events}
```

(The server also creates missing type folders under `MNEMODROME_STORAGE_PATH` on startup.)

### Step 10: Start Mnemodrome MCP Server

```bash
# Activate virtual environment (if not already active)
source .venv/bin/activate

# Start MCP server
python -m src.mcp_server
```

Server will start on `http://localhost:8765`

**For production deployment, see [Running as Service](#running-as-service).**

---

## LLM Provider Configuration

Mnemodrome supports three LLM providers. Configure in `.env` file.

> [!WARNING]
> **Avoid reasoning models for extraction.** Qwen3, DeepSeek-R1, and similar "thinking" models can silently produce empty extraction results by exhausting the token budget on hidden reasoning before emitting JSON. Use a non-reasoning instruct model regardless of which provider you choose. See [Choosing your extraction model](README.md#choosing-your-extraction-model) in the README for full details, including the `LLM_DISABLE_THINKING` flag and Ollama version caveats.

### Option 1: Anthropic (Claude)

**Best for**: Quality, reasoning capability
**Requires**: API key from [console.anthropic.com](https://console.anthropic.com/)

```bash
LLM_PROVIDER=anthropic
CHAT_MODEL=claude-sonnet-4-6
LLM_API_KEY=sk-ant-your-api-key-here
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=1000
```

**Available models**:
- `claude-sonnet-4-6` - Recommended (best quality/speed balance)
- `claude-haiku-4-5-20251001` - Fast and cost-effective
- `claude-opus-4-6` - Maximum capability

### Option 2: OpenAI (GPT-4)

**Best for**: Broad model selection
**Requires**: API key from [platform.openai.com](https://platform.openai.com/api-keys)

```bash
LLM_PROVIDER=openai
CHAT_MODEL=gpt-4
LLM_API_KEY=sk-your-api-key-here
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=1000
```

**Available models**:
- `gpt-4` - Recommended
- `gpt-4-turbo` - Faster, newer
- `gpt-3.5-turbo` - Cost-effective

### Option 3: Ollama (Local Models - Privacy First)

**Best for**: Privacy, zero cost, offline operation
**Requires**: Ollama installed and running

```bash
LLM_PROVIDER=ollama
CHAT_MODEL=gemma3:4b-it-qat
LLM_BASE_URL=http://localhost:11434
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=1000
# No API key required
```

**Available models** (install via `ollama pull <model>`):
- `gemma3:4b-it-qat` - Recommended default (verified working, ~4.0 GB). `gemma4:e4b-it-qat` is a good alternative on a recent Ollama version, but requires a newer Ollama than some Linux distributions package.
- `mistral` - Fast and capable (7B)
- `qwen2.5` - Multilingual support (7B/14B/72B)
- `llama3.1` - Meta's latest (8B/70B variants)

**Check available models**:
```bash
ollama list
```

**Pull new models**:
```bash
ollama pull gemma3:4b-it-qat  # Quantized, fast, good quality
ollama pull mistral:7b  # Fast and capable
```

---

## Embedding Provider (experimental)

Mnemodrome turns memories into vectors with an embedding model. Which engine
runs the model is set by `EMBED_PROVIDER`:

| Value | What runs | Notes |
|---|---|---|
| `ollama` (default) | `nomic-embed-text` in your Ollama | Vectors depend on the Ollama version. Ollama 0.30.0 changed how this model handles upper case, so an Ollama upgrade can silently change search results. |
| `local-onnx` | pinned `nomic-embed-text-v1.5` (fp32) inside Mnemodrome, on CPU | No Ollama needed for memory. Upgrading Ollama never changes stored vectors. Memory: about 0.9 GB resident once loaded, up to about 2.2 GB peak while embedding 2048-token texts (measured; lower `EMBED_MAX_TOKENS` lowers the peak). Needs the `onnx` extra (included in the Docker image). |
| `openai-compatible` | any `/v1/embeddings` endpoint (`EMBED_BASE_URL`, `EMBED_MODEL`, `EMBED_API_KEY`) | Mnemodrome cannot pin a remote engine; the startup canary detects changes. |

Every Qdrant collection records the fingerprint of the embedder that built it
(model files, tokenizer, prefixes, max length, template version) and a set of
canary vectors. At startup Mnemodrome compares both. On a mismatch, search
results carry a warning, deduplication is skipped, new vectors are not
written into that collection (the capture itself is still stored in the
graph and files; only its vector is held back), and `vector_health` says
which collections to rebuild and how many writes were blocked.

### Switching provider on existing data

Changing `EMBED_PROVIDER` does not convert stored vectors. Rebuild them with
the new provider; the old collections stay untouched until you delete them:

```bash
# 1. Build new collections with the provider you switch TO (resumable).
EMBED_PROVIDER=local-onnx uv run --extra onnx python -m src.cli.reindex build
# 2. Move the aliases (runs a catch-up first), then restart with the new provider.
EMBED_PROVIDER=local-onnx uv run --extra onnx python -m src.cli.reindex swap
# 3. After the restart, re-embed anything written in between.
EMBED_PROVIDER=local-onnx uv run --extra onnx python -m src.cli.reindex catch-up
# Undo at any time: point the aliases back and restart with the old provider.
EMBED_PROVIDER=local-onnx uv run --extra onnx python -m src.cli.reindex rollback
```

Between `swap` and the restart, a server still running the old provider writes
old-provider vectors into the new collection; `vector_health` counts them as
mixed points and step 3 re-embeds them.

Downgrade note: versions before this one read `mnemodrome_<type>` directly.
An upgraded install keeps those collections (the aliases point at them until
you swap), so a downgrade works. A fresh install on this version creates only
`mnemodrome_<type>__<fp8>`, which older versions do not read.

`python -m src.cli.reindex status` shows, per memory type, the live and target
collections and their point counts. Use `build --dry-run` to count first.
Thresholds (`QDRANT_SIMILARITY_THRESHOLD`, `DEDUP_*`) were tuned for the
Ollama vectors; check them after a switch.

### Offline embedding model

With `EMBED_PROVIDER=local-onnx`, the first start downloads two files into
`EMBED_MODEL_DIR` (default `~/.cache/mnemodrome/models`; Docker:
`$DATA_ROOT/mnemodrome/models`) and checks their sha256. A file with another
hash is never loaded: startup fails and says which file is wrong.

For a host without internet access, place the files yourself and set
`EMBED_MODEL_DOWNLOAD=false`. The two files are `model.onnx` (fp32, about
547 MB) and `vocab.txt` from Hugging Face `nomic-ai/nomic-embed-text-v1.5`
at a pinned revision. The pinned URLs, sizes and sha256 values live in
`src/services/embedding/model_store.py`; print, fetch and verify them on any
machine with a checkout:

```bash
uv run python -m src.services.embedding.model_store                 # URL, size, sha256 per file
uv run python -m src.services.embedding.model_store --fetch ./models # download and verify
uv run python -m src.services.embedding.model_store --verify ./models
```

Copy the folder to the offline host by any means. To download from your own
mirror instead of Hugging Face, set `EMBED_MODEL_URL` to a base URL that serves
`model.onnx` and `vocab.txt` (flat names); the same hashes apply. The model is
Apache-2.0 licensed.

---

## MCP Client Setup

Configure your MCP client to connect to Mnemodrome.

### Claude Code (Anthropic)

**File**: `~/.claude/.mcp.json`

```json
{
  "mcpServers": {
    "mnemodrome": {
      "type": "http",
      "url": "http://localhost:8765/mcp",
      "description": "Mnemodrome AI memory system"
    }
  }
}
```

**Restart Claude Code** after configuration.

### Continue.dev (VS Code)

**File**: `~/.continue/config.json`

Add to `mcpServers` section:

```json
{
  "mcpServers": {
    "mnemodrome": {
      "type": "http",
      "url": "http://localhost:8765/mcp",
      "description": "Mnemodrome AI memory system"
    }
  }
}
```

### Cody (Sourcegraph)

MCP support coming soon. Check [Cody documentation](https://sourcegraph.com/docs/cody) for updates.

### Custom MCP Client

Connect to the MCP (Streamable HTTP) endpoint:
```
http://localhost:8765/mcp
```

MCP protocol documentation: [modelcontextprotocol.io](https://modelcontextprotocol.io)

---

## Verification

### Step 1: Check MCP Server Health

```bash
curl http://localhost:8765/health
```

Expected response:
```json
{
  "status": "healthy",
  "services": {
    "neo4j": "connected",
    "redis": "connected",
    "event_log": "connected",
    "qdrant": "connected"
  },
  "timestamp": 1781634903.09,
  "version": "1.0.0",
  "llm_extraction_degraded_count": 0
}
```

`status` will be `"healthy"`, `"degraded"`, or `"unhealthy"`. A `llm_extraction_degraded_count` greater than 0 means LLM extraction has been silently failing (embeddings still work, but entity extraction from text is impaired).

### Step 2: Test Neo4j Connection

```bash
# Via browser
open http://localhost:7474

# Via CLI
cypher-shell -u neo4j -p your-password
```

### Step 3: Test Ollama Embeddings

```bash
curl http://localhost:11434/api/embeddings -d '{"model":"nomic-embed-text","prompt":"test"}'
```

A correct response contains an `"embedding"` array of 768 floats. If you get an error or an empty array, the model is not loaded or Ollama is not running.

### Step 4: Test MCP Tool

From your MCP client (e.g., Claude Code):

```
Use the mnemodrome echo tool to test: "Hello Mnemodrome"
```

Expected response:
```
ECHO: Hello Mnemodrome
```

---

## Agent keys (authentication)

Out of the box the MCP server answers anyone who can reach port 8765. Per-agent keys let you decide which clients may read and write memory, rotate or revoke one client without touching the others, and see which agent wrote each capture. The Admin UI keeps its own login and is not affected.

### 1. Choose a keys file

```bash
# .env
MNEMODROME_AGENT_KEYS_FILE=./data/mnemodrome/agent-keys.json
MNEMODROME_AUTH_MODE=off        # off (default) | warn | enforce
```

Docker Compose already sets `MNEMODROME_AGENT_KEYS_FILE=/data/mnemodrome/agent-keys.json` inside the container, which lives on the `${DATA_ROOT}/mnemodrome` bind mount, and passes `MNEMODROME_AUTH_MODE` through from `.env`.

The file is JSON. It holds, per agent, the SHA-256 digest of the key, the creation and rotation dates, and a `scopes` list. The plaintext key is never written to disk. The CLI writes the file atomically with mode `0600`. `scopes` is reserved for a later read-only option; today every valid key has full access.

### 2. Create one key per agent

```bash
# bare metal (from the repo, with the venv active)
python -m src.cli.agent_keys add claude-code

# Docker Compose
docker compose exec mcp-server python -m src.cli.agent_keys add claude-code
```

The key (it starts with `mnd_`) is printed **once** on stdout. Put it in your secret manager or the client's environment right away; it cannot be shown again. Other commands:

| Command | Effect |
|---|---|
| `list` | Agent names, created and rotated dates, scopes. Never keys or digests. |
| `rotate <name>` | New key for that agent, printed once. The old key stops working. |
| `revoke <name>` | Removes that agent. Its key stops working. |

Add `--file PATH` to work on a file other than `$MNEMODROME_AGENT_KEYS_FILE`. The running server re-reads the file within about a second of a change; no restart is needed. Writes by the CLI are serialized with a file lock and replaced atomically (mode 0600). If the file becomes unreadable (for example invalid JSON): in `enforce` mode the server **fails closed**: it drops all file-based keys, logs a critical error, and rejects those keys until the file is fixed, so a revoked key can never stay valid because of a bad edit. In `warn` mode it keeps the last good set of keys and logs an error.

### 3. Configure the clients

Send the key in the `Authorization` header:

```
Authorization: Bearer mnd_...
```

Clients that cannot set `Authorization` can use `X-Mnemodrome-Key: mnd_...` instead. If both are present, `Authorization` wins.

**Claude Code** (`.mcp.json`; `${MNEMODROME_KEY}` is expanded from the environment, so the key stays out of the file):

```json
{
  "mcpServers": {
    "mnemodrome": {
      "type": "http",
      "url": "http://localhost:8765/mcp",
      "headers": { "Authorization": "Bearer ${MNEMODROME_KEY}" }
    }
  }
}
```

Or from the command line:

```bash
claude mcp add --transport http mnemodrome http://localhost:8765/mcp \
  --header "Authorization: Bearer $MNEMODROME_KEY"
```

**Claude Desktop** through the `mcp-remote` bridge:

```json
{
  "mcpServers": {
    "mnemodrome": {
      "command": "npx",
      "args": ["mcp-remote", "http://localhost:8765/mcp",
               "--header", "Authorization:${MNEMODROME_AUTH_HEADER}"],
      "env": { "MNEMODROME_AUTH_HEADER": "Bearer mnd_..." }
    }
  }
}
```

**Hooks and scripts** calling `/project-context`, `/validate-reasoning`, or `/validate-prompt` send the same header, for example `curl -H "Authorization: Bearer $MNEMODROME_KEY" ...`.

**Any other MCP client** that speaks streamable HTTP: configure a custom header `Authorization: Bearer <key>` on the server entry.

### 4. Roll out: off, warn, enforce

1. Create keys and configure every client while the mode is still `off`.
2. Set `MNEMODROME_AUTH_MODE=warn` and restart. Requests without a valid key are still served, but the server logs a warning (at most once a minute per client address and reason) and counts them. `GET /health` shows the counters under `auth` when the request carries a valid agent key (anonymous callers get liveness only):
   ```json
   "auth": {"mode": "warn", "authenticated": 120, "unauthenticated_missing": 3,
            "unauthenticated_invalid": 0, "rejected": 0}
   ```
3. When the unauthenticated counters stop growing, set `MNEMODROME_AUTH_MODE=enforce` and restart. Requests without a valid key now get `401 Unauthorized`.

`MNEMODROME_AUTH_EXEMPT_PATHS` (default `/health`) lists paths that never need a key, so Docker health checks keep working. Set it to an empty value to protect `/health` as well. In `enforce` mode with no keys configured, every other request is rejected and the server logs an error at startup.

### What is covered

- `/mcp` (streamable HTTP), the SSE endpoints when `MCP_TRANSPORT=sse`, and the hook routes (`/project-context`, `/validate-reasoning`, `/validate-prompt`).
- The stdio transport has no network listener and no keys.
- Keys travel in clear text over plain HTTP. Across an untrusted network, put a TLS reverse proxy in front of port 8765 (see the nginx example below; proxy `/mcp` with buffering off).

### Provenance

When a request is authenticated, the agent name is recorded:

- **Graph**: a `captured_by` property on facts (`RELATES_TO` relationships, including facts created by `update_fact` and `capture_context`), and on `Learning`, `Session`, and `Procedure` nodes. No new labels or relationship types. Captures made without a key (mode `off`, or unauthenticated requests in `warn`) have no `captured_by` property.
- **Event log**: the agent name is the `actor` of automatic tool-call events and of events that did not name an actor, and is added to every event's metadata as `agent`. Query it with `query_events(actor="claude-code")`.

### Legacy `MNEMODROME_HOOK_TOKEN` (deprecated)

The old single shared token still works the same way on the hook routes (`/project-context`, `/validate-reasoning`, `/validate-prompt`). In `warn` and `enforce` modes it is accepted on those routes only, as one key named `legacy-hook-token`; it does not authorize `/mcp`. The server logs a deprecation warning at startup. Move each hook client to its own key, then remove the token.

## Admin UI and Memory Browser

The Admin UI is a small FastAPI app that runs in the same process as the MCP server. It hosts the configuration editor, health dashboard, event viewer, service controls, and the **Memory Browser**, a read-only view of every memory layer (facts and entities from Neo4j, semantic search through Qdrant, events, and the Markdown files on disk).

It is **off by default**. To enable it, set in `.env`:

```bash
ADMIN_WEB_ENABLED=true
ADMIN_USERNAME=admin
ADMIN_PASSWORD=<long random password>   # e.g. openssl rand -base64 32
ADMIN_WEB_PORT=8083                      # optional
```

Restart the server and open `http://localhost:8083/admin/memory`. You get a normal sign-in page, so your password manager can save and fill it. The session lasts `ADMIN_SESSION_TTL_HOURS` (default 12) and ends with **Log out** or a password change. Set `ADMIN_SESSION_SECRET` to keep sessions across restarts.

Scripts and API clients can keep using HTTP Basic auth with the same credentials:

```bash
curl -u admin:<password> http://localhost:8083/admin/api/memory/overview
```

**Network exposure**

- Docker Compose publishes the port as `127.0.0.1:8083` on the host only.
- A bare-metal install binds to `127.0.0.1`. `ADMIN_WEB_HOST=0.0.0.0` opens it to the network; only do that behind a firewall or reverse proxy.
- The Admin UI speaks plain HTTP. Put a TLS reverse proxy in front of it to reach it from other machines. The proxy must send `X-Forwarded-Proto: https` so the session cookie is marked Secure.
- The Admin UI has one admin account and is meant for a single operator on a trusted network (LAN or VPN). Do not expose it to the public internet.

**nginx reverse proxy example**

```nginx
server {
    listen 443 ssl;
    server_name memory.example.internal;
    ssl_certificate     /etc/letsencrypt/live/memory.example.internal/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/memory.example.internal/privkey.pem;

    location / {
        proxy_pass         http://127.0.0.1:8083;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-Proto https;
        # Optional second factor: allow only your networks
        # allow 192.168.0.0/16; deny all;
    }
}
```

**Memory Browser notes**

- Lists are paginated on the server (max 200 rows per page) so 30,000+ items stay fast.
- File folders are scanned once into an in-memory index cached for `MEMORY_BROWSER_FILE_INDEX_TTL` seconds (default 300). The **Refresh index** button rescans on demand; use it after bulk imports.
- The Memory Browser never writes. Edits and deletes are not available from the UI.

## Running as Service

**Docker Compose is the recommended deployment method** for production. Use the systemd/launchd sections below only for manual (non-Docker) deployments.

### Systemd (Linux)

**File**: `/etc/systemd/system/mnemodrome.service`

```ini
[Unit]
Description=Mnemodrome MCP Server
After=network.target neo4j.service redis.service

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/mnemodrome
Environment="PATH=/path/to/mnemodrome/.venv/bin"
ExecStart=/path/to/mnemodrome/.venv/bin/python -m src.mcp_server
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable and start**:
```bash
sudo systemctl daemon-reload
sudo systemctl enable mnemodrome
sudo systemctl start mnemodrome
sudo systemctl status mnemodrome
```

### Launchd (macOS)

**File**: `~/Library/LaunchAgents/com.mnemodrome.server.plist`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.mnemodrome.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/mnemodrome/.venv/bin/python</string>
        <string>-m</string>
        <string>src.mcp_server</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/path/to/mnemodrome</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

**Load and start**:
```bash
launchctl load ~/Library/LaunchAgents/com.mnemodrome.server.plist
launchctl start com.mnemodrome.server
```

---

## Troubleshooting

### MCP Server Won't Start

**Check Python version**:
```bash
python --version  # Must be 3.11+
```

**Check dependencies**:
```bash
# Sync dependencies
uv sync

# Check for missing requirements
uv pip list | grep -E "pydantic|neo4j|fastapi"
```

**Check logs**:
```bash
# If running as service
sudo journalctl -u mnemodrome -f

# If running manually
python -m src.mcp_server  # Watch output
```

### Neo4j Connection Failed

**Verify Neo4j is running**:
```bash
sudo systemctl status neo4j  # Linux
brew services list | grep neo4j  # macOS
```

**Test connection**:
```bash
cypher-shell -u neo4j -p your-password
```

**Check credentials in .env**:
```bash
# Ensure NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD are correct
grep NEO4J .env
```

### Ollama Models Not Found

**Check Ollama is running**:
```bash
curl http://localhost:11434/api/tags
```

**List installed models**:
```bash
ollama list
```

**Pull missing models**:
```bash
ollama pull nomic-embed-text  # Required for embeddings
ollama pull gemma3:4b-it-qat  # Required for LLM (if using Ollama provider)
```

### Redis Connection Issues

**Verify Redis is running**:
```bash
redis-cli ping  # Should return "PONG"
```

**Start Redis**:
```bash
sudo systemctl start redis  # Linux
brew services start redis  # macOS
```

**Note**: Redis is optional. Mnemodrome will work without it, but performance will be slower.

### Storage Path Errors

**Create missing directories**:
```bash
mkdir -p ~/.mnemodrome/{facts,learnings,sessions,graphs,updates,procedures}
```

**"No storage path configured"**: set `MNEMODROME_STORAGE_PATH` (one root for all types) or the per-type `MNEMODROME_<TYPE>_PATH` variable named in the error.

**Check permissions**:
```bash
ls -la ~/.mnemodrome
# All directories should be writable by your user
```

**Verify paths in .env**:
```bash
# Check MNEMODROME_ environment variables
grep MNEMODROME_ .env
```

### MCP Client Can't Connect

**Check MCP server is running**:
```bash
curl http://localhost:8765/health
```

**Verify .mcp.json configuration**:
```bash
cat ~/.claude/.mcp.json
# Check URL is http://localhost:8765/mcp
```

**Restart client**:
- Claude Code: Quit and restart application
- Continue.dev: Reload VS Code window

**Check firewall**:
```bash
# Ensure port 8765 is not blocked
sudo ufw status  # Linux
```

### LLM Provider Errors

**Anthropic**:
- Verify API key is valid
- Check [console.anthropic.com](https://console.anthropic.com/) for usage limits

**OpenAI**:
- Verify API key is valid
- Check [platform.openai.com](https://platform.openai.com/) for usage limits

**Ollama**:
- Ensure Ollama is running: `ollama serve`
- Check model is pulled: `ollama list`
- Verify LLM_BASE_URL is correct (http://localhost:11434)

### Entity extraction produces no results

If `capture_fact`, `capture_context`, or similar tools appear to succeed but nothing is stored in the graph, and you are running a reasoning model (Qwen3, DeepSeek-R1, or any model with a "thinking" mode), the model likely exhausted its token budget on hidden reasoning before emitting JSON.

**Fix:**

1. Switch to a non-reasoning instruct model such as `gemma3:4b-it-qat` (recommended).
2. Or set `LLM_DISABLE_THINKING=true` in your `.env` and restart Mnemodrome. This flag suppresses thinking-token output for Ollama and OpenAI-compatible providers.
3. If you are on Ollama 0.20.3 or earlier, note that those versions ignore the no-think hint on the OpenAI-compatible endpoint. Upgrade Ollama, or raise `LLM_MAX_TOKENS` substantially so the model has enough budget to finish thinking and still emit JSON.

---

## Next Steps

- **[Configuration Reference](.env.example)** - All environment variables
- **MCP tool catalog** - Available at runtime: any MCP client can enumerate all tools, parameters, and descriptions through the standard MCP protocol

---

## Getting Help

- **GitHub Issues**: [Report problems](https://github.com/Ballast-Research/mnemodrome/issues)
- **Discussions**: [Ask questions](https://github.com/Ballast-Research/mnemodrome/discussions)

---

**Installation complete! Start using Mnemodrome through your MCP client.**
