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
5. [MCP Client Setup](#mcp-client-setup)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)

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

## Admin UI and Memory Browser

The Admin UI is a small FastAPI app that runs in the same process as the MCP server. It hosts the configuration editor, health dashboard, event viewer, service controls, and the **Memory Browser**, a read-only view of every memory layer (facts and entities from Neo4j, semantic search through Qdrant, events, and the Markdown files on disk).

It is **off by default**. To enable it, set in `.env`:

```bash
ADMIN_WEB_ENABLED=true
ADMIN_USERNAME=admin
ADMIN_PASSWORD=<long random password>   # e.g. openssl rand -base64 32
ADMIN_WEB_PORT=8083                      # optional
```

Restart the server and open `http://localhost:8083/admin/memory`. The browser asks for the username and password (HTTP Basic auth).

**Network exposure**

- Docker Compose publishes the port as `127.0.0.1:8083` on the host only.
- A bare-metal install binds to `127.0.0.1`. `ADMIN_WEB_HOST=0.0.0.0` opens it to the network; only do that behind a firewall or reverse proxy.
- The Admin UI speaks plain HTTP. Put a TLS reverse proxy in front of it to reach it from other machines.

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
