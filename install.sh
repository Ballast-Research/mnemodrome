#!/usr/bin/env bash
# Mnemodrome one-command installer
# Usage:
#   ./install.sh               — interactive (3 questions)
#   ./install.sh --defaults    — non-interactive, accept all defaults
#   ./install.sh --dry-run     — do everything except docker compose up
# Flags (can combine with --defaults or --dry-run):
#   --data-root PATH           — override Q2 bulk data location
#   --llm-endpoint URL         — override Ollama/LLM endpoint URL
#   --llm-provider X           — override LLM provider (ollama/openai/anthropic/openai-compatible)
#   --llm-api-key X            — API key for cloud providers (openai/anthropic/openai-compatible);
#                                also reads from the LLM_API_KEY env var; required when provider is
#                                a cloud provider in non-interactive/--defaults mode
#   --chat-model X             — override chat model
#   --embedding-model X        — override embedding model

set -euo pipefail

# ---------------------------------------------------------------------------
# Globals / colour helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()    { printf "${CYAN}==> ${NC}%s\n" "$*"; }
success() { printf "${GREEN}✓ ${NC}%s\n" "$*"; }
warn()    { printf "${YELLOW}⚠  ${NC}%s\n" "$*"; }
error()   { printf "${RED}✗  ${NC}%s\n" "$*" >&2; }
die()     { error "$*"; exit 1; }
bold()    { printf "${BOLD}%s${NC}" "$*"; }

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
DEFAULTS=false
DRY_RUN=false
ARG_DATA_ROOT=""
ARG_LLM_ENDPOINT=""
ARG_LLM_PROVIDER=""
ARG_LLM_API_KEY=""
ARG_CHAT_MODEL=""
ARG_EMBEDDING_MODEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --defaults)       DEFAULTS=true ;;
        --dry-run)        DRY_RUN=true ;;
        --data-root)      ARG_DATA_ROOT="${2:-}";      shift ;;
        --llm-endpoint)   ARG_LLM_ENDPOINT="${2:-}";   shift ;;
        --llm-provider)   ARG_LLM_PROVIDER="${2:-}";   shift ;;
        --llm-api-key)    ARG_LLM_API_KEY="${2:-}";    shift ;;
        --chat-model)     ARG_CHAT_MODEL="${2:-}";      shift ;;
        --embedding-model) ARG_EMBEDDING_MODEL="${2:-}"; shift ;;
        -h|--help)
            cat <<'HELP'
Usage: ./install.sh [OPTIONS]

  --defaults          Accept all defaults non-interactively (CI mode)
  --dry-run           Do everything except docker compose up
  --data-root PATH    Bulk data directory (Q2 default: <install-dir>/data)
  --llm-endpoint URL  LLM/Ollama API base URL
  --llm-provider X    Provider: ollama | openai | anthropic | openai-compatible
  --llm-api-key X     API key for cloud providers (openai/anthropic/openai-compatible).
                      Also reads from the LLM_API_KEY env var. Required in
                      non-interactive/--defaults mode when using a cloud provider.
  --chat-model X      Chat/LLM model name
  --embedding-model X Embedding model name
HELP
            exit 0
            ;;
        *) die "Unknown argument: $1" ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# Portable helpers (no GNU-specific flags)
# ---------------------------------------------------------------------------

# Portable absolute path without readlink -f (works on macOS + Linux)
portable_abspath() {
    local target="$1"
    # If it's a relative path, make it absolute from PWD
    case "$target" in
        /*) echo "$target" ;;
        *)  echo "$(pwd)/$target" ;;
    esac
}

# Portable disk-space check in GB (uses df -k, works on both Linux and macOS)
disk_free_gb() {
    local path="$1"
    # df -k outputs KB; col 4 is Available
    local kb
    kb=$(df -k "$path" 2>/dev/null | tail -1 | awk '{print $4}')
    if [[ -n "$kb" && "$kb" =~ ^[0-9]+$ ]]; then
        echo $(( kb / 1024 / 1024 ))
    else
        echo "?"
    fi
}

# Generate a secure random password (base64 URL-safe, 18 bytes → 24 chars)
gen_password() {
    if command -v openssl >/dev/null 2>&1; then
        openssl rand -base64 18 | tr '+/' '-_' | tr -d '='
    else
        # fallback: /dev/urandom + od
        dd if=/dev/urandom bs=18 count=1 2>/dev/null | od -A n -t x1 | tr -d ' \n' | head -c 24
    fi
}

# ---------------------------------------------------------------------------
# Step 0: Detect install directory
# ---------------------------------------------------------------------------
# 'cd ... && pwd' gives the canonical absolute path on both Linux and macOS.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

is_mnemodrome_dir() {
    local d="$1"
    # True if the directory contains a .git folder AND either:
    #   - pyproject.toml with name = "mnemodrome" (or "membrana")
    #   - docker-compose.yml
    [[ -d "$d/.git" ]] || return 1
    if [[ -f "$d/pyproject.toml" ]] && grep -q 'name.*=.*"mnemodrome"' "$d/pyproject.toml" 2>/dev/null; then
        return 0
    fi
    if [[ -f "$d/docker-compose.yml" ]]; then
        return 0
    fi
    return 1
}

INSTALL_DIR="$SCRIPT_DIR"
AUTO_INSTALL_DIR=false

if is_mnemodrome_dir "$SCRIPT_DIR"; then
    AUTO_INSTALL_DIR=true
fi

echo ""
BANNER_VER="$(grep -m1 '^version' "$SCRIPT_DIR/pyproject.toml" 2>/dev/null | cut -d'"' -f2)"
printf "${CYAN}"
cat <<'BANNER'
    __  ___                               __
   /  |/  /___  ___  ____ ___  ____  ____/ /________  ____ ___  ___
  / /|_/ / __ \/ _ \/ __ `__ \/ __ \/ __  / ___/ __ \/ __ `__ \/ _ \
 / /  / / / / /  __/ / / / / / /_/ / /_/ / /  / /_/ / / / / / /  __/
/_/  /_/_/ /_/\___/_/ /_/ /_/\____/\__,_/_/   \____/_/ /_/ /_/\___/
BANNER
printf "${NC}"
printf "   ${BOLD}multi-layer memory for AI agents${NC}  ·  v%s\n" "${BANNER_VER:-1.0.0}"
echo "────────────────────────────────────────────────────────"
echo ""

# ---------------------------------------------------------------------------
# Q1 — Install destination
# ---------------------------------------------------------------------------
if "$AUTO_INSTALL_DIR"; then
    if "$DEFAULTS"; then
        info "Q1: Install directory auto-detected: $INSTALL_DIR (--defaults accepted)"
        INSTALL_DIR="$INSTALL_DIR"   # already set
    else
        printf "$(bold 'Q1 — Install directory')  (detected git + docker-compose.yml)\n"
        printf "    Installing here: ${CYAN}%s${NC}\n" "$INSTALL_DIR"
        printf "    Press Enter to confirm, or type another path: "
        read -r Q1_INPUT
        if [[ -n "$Q1_INPUT" ]]; then
            INSTALL_DIR="$(portable_abspath "$Q1_INPUT")"
            info "Will copy source to: $INSTALL_DIR"
        else
            info "Using: $INSTALL_DIR"
        fi
    fi
else
    # Not in a mnemodrome directory; must ask or use fallback
    DEFAULT_INSTALL="$HOME/mnemodrome"
    if "$DEFAULTS"; then
        INSTALL_DIR="$DEFAULT_INSTALL"
        info "Q1: Install directory (--defaults): $INSTALL_DIR"
    else
        printf "$(bold 'Q1 — Install directory')\n"
        printf "    Default: ${CYAN}%s${NC}\n" "$DEFAULT_INSTALL"
        printf "    Enter path [%s]: " "$DEFAULT_INSTALL"
        read -r Q1_INPUT
        INSTALL_DIR="${Q1_INPUT:-$DEFAULT_INSTALL}"
        INSTALL_DIR="$(portable_abspath "$INSTALL_DIR")"
    fi
fi

# If INSTALL_DIR differs from SCRIPT_DIR, copy the tree there
if [[ "$INSTALL_DIR" != "$SCRIPT_DIR" ]]; then
    if [[ -d "$INSTALL_DIR" ]]; then
        warn "Directory $INSTALL_DIR already exists — installing into it (existing files preserved)"
    else
        info "Creating $INSTALL_DIR"
        mkdir -p "$INSTALL_DIR"
    fi
    # Rsync the source (exclude .git, .venv, __pycache__, data)
    if command -v rsync >/dev/null 2>&1; then
        rsync -a --exclude='.git' --exclude='.venv' --exclude='__pycache__' \
              --exclude='data' --exclude='*.pyc' \
              "$SCRIPT_DIR/" "$INSTALL_DIR/"
    else
        cp -a "$SCRIPT_DIR/." "$INSTALL_DIR/"
    fi
    cd "$INSTALL_DIR"
fi

# ---------------------------------------------------------------------------
# Q2 — Bulk data location
# ---------------------------------------------------------------------------
DEFAULT_DATA_ROOT="$INSTALL_DIR/data"

if [[ -n "$ARG_DATA_ROOT" ]]; then
    DATA_ROOT="$(portable_abspath "$ARG_DATA_ROOT")"
    info "Q2: Data root (--data-root): $DATA_ROOT"
elif "$DEFAULTS"; then
    DATA_ROOT="$DEFAULT_DATA_ROOT"
    info "Q2: Data root (--defaults): $DATA_ROOT"
else
    # Show available disk space for the default location
    # We check the parent since the directory may not exist yet
    DISK_DIR="$DEFAULT_DATA_ROOT"
    [[ -d "$DISK_DIR" ]] || DISK_DIR="$(dirname "$DISK_DIR")"
    [[ -d "$DISK_DIR" ]] || DISK_DIR="$HOME"
    FREE_GB=$(disk_free_gb "$DISK_DIR")
    echo ""
    printf "$(bold 'Q2 — Bulk data location') (Neo4j, Qdrant, Redis, Markdown files)\n"
    printf "    Default: ${CYAN}%s${NC}\n" "$DEFAULT_DATA_ROOT"
    printf "    Available disk space: ${FREE_GB}GB\n"
    printf "    Enter path [%s]: " "$DEFAULT_DATA_ROOT"
    read -r Q2_INPUT
    DATA_ROOT="${Q2_INPUT:-$DEFAULT_DATA_ROOT}"
    DATA_ROOT="$(portable_abspath "$DATA_ROOT")"
fi

# In dry-run mode we still validate the path but don't fail if directories
# cannot be created (e.g. root-owned ./data from a previous Docker run).
if "$DRY_RUN"; then
    success "Data root (dry-run, skipping mkdir): $DATA_ROOT"
else
    mkdir -p "$DATA_ROOT"
    success "Data root: $DATA_ROOT"

    # Create subdirectories that need to exist before containers start
    for subdir in neo4j "neo4j-logs" qdrant redis mnemodrome "mnemodrome-logs" "mnemodrome/facts" \
                  "mnemodrome/learnings" "mnemodrome/sessions" "mnemodrome/graphs" \
                  "mnemodrome/updates" "mnemodrome/procedures" "mnemodrome/events"; do
        mkdir -p "$DATA_ROOT/$subdir"
    done
fi

# ---------------------------------------------------------------------------
# Q3 — LLM endpoint
# ---------------------------------------------------------------------------
OLLAMA_DETECTED=false
OLLAMA_LOCAL_URL="http://localhost:11434"
OLLAMA_MODELS=""

# Auto-detect local Ollama
if curl -sf --connect-timeout 1 "${OLLAMA_LOCAL_URL}/api/tags" >/dev/null 2>&1; then
    OLLAMA_DETECTED=true
    # Get model list (best-effort)
    OLLAMA_MODELS=$(curl -sf --connect-timeout 2 "${OLLAMA_LOCAL_URL}/api/tags" 2>/dev/null \
        | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(' '.join(m['name'] for m in d.get('models', [])))
except Exception:
    pass
" 2>/dev/null || true)
fi

DEFAULT_EMBEDDING_MODEL="nomic-embed-text"
DEFAULT_CHAT_MODEL="gemma3:4b-it-qat"
LLM_PROVIDER=""
LLM_ENDPOINT=""
CHAT_MODEL=""
EMBEDDING_MODEL=""
# Capture any caller-exported LLM_API_KEY before we reset the variable below.
# Precedence (resolved after Q3): (a) --llm-api-key flag, (b) env var, (c) interactive read.
_ENV_LLM_API_KEY="${LLM_API_KEY:-}"
LLM_API_KEY=""
EMBEDDING_DIM=768

if [[ -n "$ARG_LLM_ENDPOINT" || -n "$ARG_LLM_PROVIDER" ]]; then
    # All LLM flags provided — skip interactive Q3
    LLM_PROVIDER="${ARG_LLM_PROVIDER:-ollama}"
    LLM_ENDPOINT="${ARG_LLM_ENDPOINT:-$OLLAMA_LOCAL_URL}"
    CHAT_MODEL="${ARG_CHAT_MODEL:-$DEFAULT_CHAT_MODEL}"
    EMBEDDING_MODEL="${ARG_EMBEDDING_MODEL:-$DEFAULT_EMBEDDING_MODEL}"
    info "Q3: LLM config from flags — provider=$LLM_PROVIDER endpoint=$LLM_ENDPOINT"
elif "$DEFAULTS"; then
    if "$OLLAMA_DETECTED"; then
        LLM_PROVIDER="ollama"
        LLM_ENDPOINT="$OLLAMA_LOCAL_URL"
        CHAT_MODEL="${ARG_CHAT_MODEL:-$DEFAULT_CHAT_MODEL}"
        EMBEDDING_MODEL="${ARG_EMBEDDING_MODEL:-$DEFAULT_EMBEDDING_MODEL}"
        info "Q3: Ollama detected at $LLM_ENDPOINT (--defaults)"
    else
        # Ollama not detected but --defaults given — warn and continue with
        # default values so the stack still starts.  Embedding/LLM features
        # will be unavailable until an Ollama (or compatible) endpoint is
        # reachable at OLLAMA_BASE_URL.
        LLM_PROVIDER="ollama"
        LLM_ENDPOINT="$OLLAMA_LOCAL_URL"
        CHAT_MODEL="${ARG_CHAT_MODEL:-$DEFAULT_CHAT_MODEL}"
        EMBEDDING_MODEL="${ARG_EMBEDDING_MODEL:-$DEFAULT_EMBEDDING_MODEL}"
        warn "Q3: Ollama not found at $OLLAMA_LOCAL_URL — using default endpoint anyway."
        warn "    The stack will start but embedding/LLM features won't work until"
        warn "    Ollama is running.  Override with --llm-endpoint / --llm-provider."
    fi
else
    echo ""
    printf "$(bold 'Q3 — LLM endpoint')\n"
    if "$OLLAMA_DETECTED"; then
        printf "    ${GREEN}Ollama detected at %s${NC}\n" "$OLLAMA_LOCAL_URL"
        if [[ -n "$OLLAMA_MODELS" ]]; then
            printf "    Available models: ${CYAN}%s${NC}\n" "$OLLAMA_MODELS"
        fi
        echo ""
        printf "    Embedding model [%s]: " "$DEFAULT_EMBEDDING_MODEL"
        read -r Q3_EMBED
        EMBEDDING_MODEL="${Q3_EMBED:-$DEFAULT_EMBEDDING_MODEL}"

        printf "    Chat model      [%s]: " "$DEFAULT_CHAT_MODEL"
        read -r Q3_CHAT
        CHAT_MODEL="${Q3_CHAT:-$DEFAULT_CHAT_MODEL}"

        LLM_PROVIDER="ollama"
        LLM_ENDPOINT="$OLLAMA_LOCAL_URL"

        # Pull hint for nomic-embed-text if missing
        if [[ -n "$OLLAMA_MODELS" ]] && ! echo "$OLLAMA_MODELS" | grep -q "nomic-embed-text"; then
            warn "nomic-embed-text not found in Ollama. Pull it with: ollama pull nomic-embed-text"
        fi
    else
        printf "    Ollama not found at $OLLAMA_LOCAL_URL.\n"
        printf "    Provider [ollama/openai/anthropic/openai-compatible]: "
        read -r Q3_PROVIDER
        LLM_PROVIDER="${Q3_PROVIDER:-ollama}"

        case "$LLM_PROVIDER" in
            ollama)
                printf "    Ollama endpoint URL [http://localhost:11434]: "
                read -r Q3_ENDPOINT
                LLM_ENDPOINT="${Q3_ENDPOINT:-http://localhost:11434}"
                ;;
            openai)
                LLM_ENDPOINT="https://api.openai.com/v1"
                ;;
            anthropic)
                LLM_ENDPOINT="https://api.anthropic.com"
                ;;
            openai-compatible)
                printf "    Endpoint URL (e.g. http://myserver:8000/v1): "
                read -r Q3_ENDPOINT
                LLM_ENDPOINT="${Q3_ENDPOINT:-http://localhost:8000/v1}"
                ;;
            *)
                die "Unknown provider: $LLM_PROVIDER"
                ;;
        esac

        printf "    Chat model [%s]: " "$DEFAULT_CHAT_MODEL"
        read -r Q3_CHAT
        CHAT_MODEL="${Q3_CHAT:-$DEFAULT_CHAT_MODEL}"

        printf "    Embedding model [%s]: " "$DEFAULT_EMBEDDING_MODEL"
        read -r Q3_EMBED
        EMBEDDING_MODEL="${Q3_EMBED:-$DEFAULT_EMBEDDING_MODEL}"

        # For cloud providers, prompt for API key with hidden input
        if [[ "$LLM_PROVIDER" == "openai" || "$LLM_PROVIDER" == "anthropic" || "$LLM_PROVIDER" == "openai-compatible" ]]; then
            printf "    API key (input hidden): "
            read -rs LLM_API_KEY
            echo ""
            if [[ -z "$LLM_API_KEY" ]]; then
                warn "No API key entered. LLM features requiring API calls will fail."
            fi
        fi
    fi
fi

# Apply flag overrides that weren't already applied
[[ -n "$ARG_CHAT_MODEL" ]]       && CHAT_MODEL="$ARG_CHAT_MODEL"
[[ -n "$ARG_EMBEDDING_MODEL" ]] && EMBEDDING_MODEL="$ARG_EMBEDDING_MODEL"

# ---------------------------------------------------------------------------
# API key resolution: flag > env var > interactive prompt (already captured)
# ---------------------------------------------------------------------------
# Cloud providers require an API key; ollama does not.
# After Q3, LLM_API_KEY holds the value from the interactive read (if any).
# Layer the --llm-api-key flag and the caller-exported LLM_API_KEY env var
# on top, then fail fast in non-interactive mode if still empty.
_is_cloud_provider() {
    case "$1" in
        openai|anthropic|openai-compatible) return 0 ;;
        *) return 1 ;;
    esac
}

if _is_cloud_provider "$LLM_PROVIDER"; then
    # (a) --llm-api-key flag wins.
    if [[ -n "$ARG_LLM_API_KEY" ]]; then
        LLM_API_KEY="$ARG_LLM_API_KEY"
    # (b) Caller-exported env var (captured before Q3 reset it to "").
    elif [[ -z "$LLM_API_KEY" && -n "$_ENV_LLM_API_KEY" ]]; then
        LLM_API_KEY="$_ENV_LLM_API_KEY"
    fi
    # (c) Interactive read already stored the value in LLM_API_KEY; nothing to do.

    # Fail fast in non-interactive / scripted mode with no key available.
    if [[ -z "$LLM_API_KEY" ]] && { "$DEFAULTS" || [[ -n "$ARG_LLM_PROVIDER" || -n "$ARG_LLM_ENDPOINT" ]]; }; then
        die "Provider '$LLM_PROVIDER' requires an API key in non-interactive mode. Pass --llm-api-key <key> or export LLM_API_KEY before running install.sh."
    fi
fi

success "LLM: provider=$LLM_PROVIDER  chat=$CHAT_MODEL  embed=$EMBEDDING_MODEL"

# ---------------------------------------------------------------------------
# Probe embedding dimension
# ---------------------------------------------------------------------------
# Translate endpoint URL for probing: if it's already host.docker.internal,
# convert back to localhost for the host-side curl.
PROBE_URL="$LLM_ENDPOINT"
PROBE_URL="${PROBE_URL//host.docker.internal/localhost}"

info "Probing embedding dimension (model: $EMBEDDING_MODEL)..."
EMBEDDING_DIM=768    # safe default
LLM_REACHABLE=false  # flipped true once a probe actually reaches the endpoint
                     # (the final layer report reflects reality, not the initial
                     #  localhost auto-detect)

if [[ "$LLM_PROVIDER" == "ollama" || "$LLM_PROVIDER" == "openai-compatible" ]]; then
    # Try Ollama /api/embeddings endpoint
    PROBE_RESP=$(curl -sf --connect-timeout 2 --max-time 10 \
        "${PROBE_URL}/api/embeddings" \
        -d "{\"model\":\"${EMBEDDING_MODEL}\",\"prompt\":\"ping\"}" 2>/dev/null || true)

    if [[ -n "$PROBE_RESP" ]]; then
        # Try python3 first, then pure shell
        DIM=$(python3 -c "
import json, sys
try:
    d = json.loads(sys.stdin.read())
    print(len(d.get('embedding', [])))
except Exception:
    pass
" <<< "$PROBE_RESP" 2>/dev/null || true)

        if [[ -z "$DIM" ]] || ! [[ "$DIM" =~ ^[0-9]+$ ]]; then
            # Shell fallback: count commas in the embedding array + 1
            DIM=$(echo "$PROBE_RESP" | tr -cd ',' | wc -c | tr -d ' ')
            DIM=$(( DIM + 1 ))
        fi

        if [[ "$DIM" =~ ^[0-9]+$ && "$DIM" -gt 0 ]]; then
            EMBEDDING_DIM="$DIM"
            LLM_REACHABLE=true
            success "Probed embedding dimension: $EMBEDDING_DIM"
        else
            warn "Could not parse embedding dimension from probe response; using default $EMBEDDING_DIM"
        fi
    else
        warn "Embedding probe failed (endpoint unreachable or model not loaded). Using default $EMBEDDING_DIM."
        warn "If your model produces a different dimension, update EMBEDDING_DIM in .env after install."
    fi
fi

# ---------------------------------------------------------------------------
# Extraction canary (early heads-up)
# ---------------------------------------------------------------------------
# Confirm the chosen chat model returns ENTITIES, not hidden reasoning. A
# reasoning model (Qwen3, DeepSeek-R1) emits <think> instead of JSON and every
# capture silently stores nothing. This is a friendly early warning only — the
# server runs the authoritative canary at startup and refuses to serve a model
# that can't extract. Uses Ollama's native /api/generate (which honours
# think:false); other providers are validated server-side.
if [[ "$LLM_PROVIDER" == "ollama" ]] && command -v python3 >/dev/null 2>&1; then
    info "Probing extraction (model: $CHAT_MODEL)..."
    CANARY_PAYLOAD=$(CHAT_MODEL="$CHAT_MODEL" python3 -c '
import json, os
print(json.dumps({
    "model": os.environ["CHAT_MODEL"],
    "prompt": "Extract entities as JSON. Context: Alice Johnson works at Acme Corporation in Paris. Return ONLY {\"entities\":[{\"name\":\"...\"}]}.",
    "stream": False,
    "think": False,
    "options": {"num_predict": 512},
}))
' 2>/dev/null || true)
    CANARY_RESP=$(curl -sf --connect-timeout 2 --max-time 30 \
        "${PROBE_URL}/api/generate" -d "$CANARY_PAYLOAD" 2>/dev/null || true)
    CANARY_OUT=$(printf '%s' "$CANARY_RESP" | python3 -c "
import json, sys
try:
    print(json.loads(sys.stdin.read()).get('response', ''))
except Exception:
    pass
" 2>/dev/null || true)
    if printf '%s' "$CANARY_OUT" | grep -qiE 'alice|acme|"name"'; then
        success "Extraction probe OK — '$CHAT_MODEL' returns entities."
        LLM_REACHABLE=true
    elif [[ -z "$CANARY_RESP" ]]; then
        warn "Extraction probe skipped — '$CHAT_MODEL' not reachable/loaded yet. The server re-checks at startup."
    else
        warn "Extraction probe: '$CHAT_MODEL' returned NO entities — it may be a REASONING model."
        warn "  Use a non-reasoning instruct model, e.g.: ollama pull gemma3:4b-it-qat"
        warn "  Mnemodrome refuses to start on a model that cannot extract (override: ALLOW_DEGRADED_EXTRACTION=true)."
    fi
fi

# ---------------------------------------------------------------------------
# Translate localhost → host.docker.internal for Docker-side vars
# ---------------------------------------------------------------------------
# The stack runs in Docker; URLs pointing at localhost/127.0.0.1 on the host
# must be host.docker.internal:<port> from inside the container.
translate_to_docker() {
    local url="$1"
    url="${url//localhost/host.docker.internal}"
    url="${url//127.0.0.1/host.docker.internal}"
    echo "$url"
}

OLLAMA_DOCKER_ENDPOINT=$(translate_to_docker "$LLM_ENDPOINT")

# ---------------------------------------------------------------------------
# Preflight: Docker
# ---------------------------------------------------------------------------
echo ""
info "Preflight checks..."

check_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        error "docker not found."
        echo "  → Linux: https://docs.docker.com/engine/install/"
        echo "  → macOS:  https://docs.docker.com/desktop/mac/install/"
        return 1
    fi

    # Check compose plugin
    if ! docker compose version >/dev/null 2>&1; then
        error "'docker compose' plugin not found (requires Docker Compose v2)."
        echo "  → Linux: sudo apt install docker-compose-plugin"
        echo "  → macOS:  included with Docker Desktop"
        return 1
    fi

    # Check daemon
    if ! docker info >/dev/null 2>&1; then
        error "Docker daemon is not running."
        echo "  → Linux:  sudo systemctl start docker"
        echo "  → macOS:  start Docker Desktop"
        return 1
    fi

    return 0
}

if ! check_docker; then
    exit 1
fi
success "Docker OK ($(docker --version | head -1))"
success "Docker Compose OK ($(docker compose version | head -1))"

# Warn (don't fail) if LLM endpoint is unreachable
if ! curl -sf --connect-timeout 2 "${PROBE_URL}/api/tags" >/dev/null 2>&1 \
   && ! curl -sf --connect-timeout 2 "${PROBE_URL}/health" >/dev/null 2>&1 \
   && [[ "$LLM_PROVIDER" == "ollama" ]]; then
    warn "LLM endpoint $LLM_ENDPOINT is not currently reachable."
    warn "The stack will start but embedding/LLM features won't work until Ollama is running."
fi

# ---------------------------------------------------------------------------
# Generate .env
# ---------------------------------------------------------------------------
echo ""
info "Generating .env..."

ENV_FILE="$INSTALL_DIR/.env"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [[ -f "$ENV_FILE" ]]; then
    cp "$ENV_FILE" "${ENV_FILE}.bak.${TIMESTAMP}"
    info "Backed up existing .env → .env.bak.${TIMESTAMP}"

    # Rotation: retain only the 3 newest timestamped backups, purge older ones.
    # The %Y%m%d_%H%M%S suffix sorts lexically, so glob expansion yields oldest
    # first; keep the last 3.  Without this, repeated runs (notably the
    # install-matrix test suite) accumulate .env.bak.* unbounded.
    shopt -s nullglob
    _env_baks=( "${ENV_FILE}.bak."* )
    shopt -u nullglob
    if (( ${#_env_baks[@]} > 3 )); then
        rm -f -- "${_env_baks[@]:0:${#_env_baks[@]}-3}"
        info "Pruned $(( ${#_env_baks[@]} - 3 )) old .env backup(s), kept 3 newest"
    fi
fi

# Generate secure random passwords — but REUSE existing values when .env
# already exists so that a re-run does not rotate credentials and break
# Neo4j / Redis data that was initialised with the original password.
_reuse_pw() {
    local key="$1" file="$2"
    # Extract key=value from an existing .env; handles optional surrounding whitespace.
    grep -E "^[[:space:]]*${key}[[:space:]]*=" "$file" 2>/dev/null \
        | head -1 | sed 's/^[^=]*=//; s/[[:space:]]*$//'
}
# Initialise to empty so they are always bound (required by set -u).
EXISTING_NEO4J_PASSWORD=""
EXISTING_ADMIN_PASSWORD=""
if [[ -f "$ENV_FILE" ]]; then
    EXISTING_NEO4J_PASSWORD="$(_reuse_pw NEO4J_PASSWORD "$ENV_FILE")"
    EXISTING_ADMIN_PASSWORD="$(_reuse_pw ADMIN_PASSWORD "$ENV_FILE")"
fi
NEO4J_PASSWORD="${EXISTING_NEO4J_PASSWORD:-$(gen_password)}"
ADMIN_PASSWORD="${EXISTING_ADMIN_PASSWORD:-$(gen_password)}"
[[ -n "$EXISTING_NEO4J_PASSWORD" ]] && info "Re-using existing NEO4J_PASSWORD from .env (data directory preserved)"

cat > "$ENV_FILE" <<ENVEOF
# Mnemodrome .env — generated by install.sh on $(date)
# Re-run install.sh to update (existing .env is backed up first).

# =============================================================================
# INSTALL — written by install.sh
# =============================================================================

# Where all persistent data lives on the host.
# docker-compose.yml uses bind mounts: \${DATA_ROOT:-./data}/<service>
DATA_ROOT=${DATA_ROOT}

# Embedding dimension — probed from live model during install.
# Change only if you switch embedding models; then delete Qdrant collections
# and re-embed your data. See src/services/vector_service.py for mismatch detection.
EMBEDDING_DIM=${EMBEDDING_DIM}

# LLM configuration
# NOTE: URLs below use host.docker.internal so they resolve from inside Docker.
# The host machine uses localhost instead (install.sh translates automatically).
LLM_PROVIDER=${LLM_PROVIDER}
CHAT_MODEL=${CHAT_MODEL}
LLM_API_KEY=${LLM_API_KEY}

# API base URL and Ollama endpoints — container-side URLs (host.docker.internal = Docker host)
LLM_BASE_URL=${OLLAMA_DOCKER_ENDPOINT}
OLLAMA_BASE_URL=${OLLAMA_DOCKER_ENDPOINT}
OLLAMA_HOST=${OLLAMA_DOCKER_ENDPOINT}
OLLAMA_EMBEDDING_HOST=${OLLAMA_DOCKER_ENDPOINT}
OLLAMA_EMBEDDING_MODEL=${EMBEDDING_MODEL}

# =============================================================================
# SECURITY — generated by install.sh (do not share these values)
# =============================================================================
NEO4J_PASSWORD=${NEO4J_PASSWORD}
ADMIN_PASSWORD=${ADMIN_PASSWORD}

# =============================================================================
# CORE DATABASE CONNECTIONS
# =============================================================================
# Service names resolve automatically within Docker Compose.
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_DATABASE=neo4j

REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0

QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334
QDRANT_ENABLED=true
QDRANT_SIMILARITY_THRESHOLD=0.45

# =============================================================================
# FILE STORAGE (Markdown Memory Files — Layer 5)
# =============================================================================
# Paths are inside the container (bound to \${DATA_ROOT}/mnemodrome on the host).
MNEMODROME_FACTS_PATH=/data/mnemodrome/facts
MNEMODROME_LEARNINGS_PATH=/data/mnemodrome/learnings
MNEMODROME_SESSIONS_PATH=/data/mnemodrome/sessions
MNEMODROME_GRAPHS_PATH=/data/mnemodrome/graphs
MNEMODROME_UPDATES_PATH=/data/mnemodrome/updates
MNEMODROME_PROCEDURES_PATH=/data/mnemodrome/procedures

# =============================================================================
# EVENT LOG (Layer 4)
# =============================================================================
# Default backend is SQLite (embedded, zero-config).
# For concurrent multi-agent or team deployments, set EVENT_LOG_BACKEND=postgres,
# add the PG_* vars (PG_HOST, PG_PORT, PG_DATABASE, PG_USER, PG_PASSWORD), and
# start the postgres service: docker compose --profile postgres up
EVENT_LOG_ENABLED=true
EVENT_LOG_AUTO_CAPTURE=true
EVENT_LOG_RETENTION_DAYS=90
EVENT_LOG_MAX_SIZE_MB=100
EVENT_LOG_SESSION_TIMEOUT_HOURS=4
EVENT_LOG_PERFORMANCE_SLA_MS=50
EVENT_LOG_QUERY_SLA_MS=100
EVENT_LOG_FAIL_LOUD=true

# =============================================================================
# ADMIN WEB UI
# =============================================================================
ADMIN_WEB_ENABLED=false
ADMIN_WEB_PORT=8083
ADMIN_USERNAME=admin

# =============================================================================
# LOGGING & TRANSPORT
# =============================================================================
LOG_LEVEL=INFO
LOG_FILE=/var/log/mnemodrome/mnemodrome.log
# MCP transport: streamable-http is the current standard (served at /mcp).
# Set to "sse" only for legacy clients on the deprecated /sse endpoint.
MCP_TRANSPORT=streamable-http
ENVEOF

success ".env written (passwords generated, not printed)"

# ---------------------------------------------------------------------------
# Docker compose up (skipped in dry-run)
# ---------------------------------------------------------------------------
if "$DRY_RUN"; then
    echo ""
    printf "${YELLOW}──── DRY RUN — skipping docker compose up ────${NC}\n"
    echo ""
    echo "  The following command would be executed:"
    echo "    cd $INSTALL_DIR && docker compose up -d --build"
    echo ""
    echo "  DATA_ROOT   : $DATA_ROOT"
    echo "  INSTALL_DIR : $INSTALL_DIR"
    echo "  LLM_PROVIDER: $LLM_PROVIDER"
    echo "  CHAT_MODEL  : $CHAT_MODEL"
    echo "  EMBEDDING   : $EMBEDDING_MODEL (dim=$EMBEDDING_DIM)"
    echo "  LLM_ENDPOINT: $LLM_ENDPOINT (docker: $OLLAMA_DOCKER_ENDPOINT)"
    echo ""
    success "Dry run complete."
    exit 0
fi

echo ""
info "Building Mnemodrome from source and starting the stack..."
printf "    ${CYAN}First run pulls the database images (Neo4j, Qdrant, Redis) and compiles\n"
printf "    the memory server, so it can take a few minutes. This is real\n"
printf "    infrastructure being assembled on your machine — not a mock.${NC}\n"
echo ""
cd "$INSTALL_DIR"
docker compose up -d --build

# ---------------------------------------------------------------------------
# Wait for health
# ---------------------------------------------------------------------------
HEALTH_URL="http://localhost:8765/health"
TIMEOUT=180
INTERVAL=5
ELAPSED=0

echo ""
info "Waiting for MCP server to become healthy (timeout: ${TIMEOUT}s)..."

while [[ $ELAPSED -lt $TIMEOUT ]]; do
    if curl -sf "$HEALTH_URL" 2>/dev/null | grep -q '"status"'; then
        echo ""
        success "Mnemodrome is healthy!"
        break
    fi
    printf "  [%3ds] waiting...\r" "$ELAPSED"
    sleep "$INTERVAL"
    ELAPSED=$(( ELAPSED + INTERVAL ))
done

if [[ $ELAPSED -ge $TIMEOUT ]]; then
    echo ""
    error "Health check timed out after ${TIMEOUT}s."
    echo ""
    echo "  Inspect server logs:"
    echo "    docker compose logs --tail 50 mcp-server"
    echo ""
    echo "  Check service status:"
    echo "    docker compose ps"
    exit 1
fi

# ---------------------------------------------------------------------------
# Warm up every memory layer + record the install as this instance's first memory
# ---------------------------------------------------------------------------
# /health is a plain HTTP route that never opens an MCP session, so the graph,
# vector, cache and event layers stay dormant until the first real MCP client
# connects. We open one session now to (a) initialize every layer and (b) write
# the install date as the instance's first memory. The mcp client library ships
# inside the server image, so no Python/uv is required on the host.
echo ""
info "Warming up memory layers and recording the install..."
INSTALL_STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date)"

STATUS_LINE="$( { docker compose exec -T mcp-server python - "$INSTALL_STAMP" <<'PYEOF'
import asyncio, sys, json, urllib.request
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession

stamp = sys.argv[1] if len(sys.argv) > 1 else "unknown"

async def main():
    name = ver = "?"
    recorded = "no"
    async with streamablehttp_client("http://localhost:8765/mcp") as (r, w, _):
        async with ClientSession(r, w) as s:
            init = await s.initialize()          # opens session -> inits all layers
            name = init.serverInfo.name
            ver = init.serverInfo.version
            await s.list_tools()
            try:
                # Explicit entity types skip the LLM classification path, so this
                # graph write succeeds even with no LLM endpoint configured.
                await s.call_tool("capture_fact", {
                    "entity1": "This Mnemodrome instance",
                    "relation": "WAS_INSTALLED_ON",
                    "entity2": stamp,
                    "context": "Recorded automatically by install.sh on first run.",
                    "entity1_type": "System",
                    "entity2_type": "Concept",
                })
                recorded = "yes"
            except Exception:
                recorded = "no"
    try:
        h = json.load(urllib.request.urlopen("http://localhost:8765/health", timeout=5))
        svc = h.get("services", {})
    except Exception:
        svc = {}
    print("STATUS "
          f"name={name} version={ver} recorded={recorded} "
          f"neo4j={svc.get('neo4j','unknown')} qdrant={svc.get('qdrant','unknown')} "
          f"redis={svc.get('redis','unknown')} event_log={svc.get('event_log','unknown')}")

asyncio.run(main())
PYEOF
} 2>/dev/null || true )"

# Always returns 0 (|| true) — a missing key must not trip `set -e` and fail an
# install whose server is already healthy.
get_kv() { printf '%s\n' "$STATUS_LINE" | tr ' ' '\n' | grep "^$1=" | cut -d= -f2- | head -1 || true; }
SRV_NAME="$(get_kv name)";   [[ -n "$SRV_NAME" ]] || SRV_NAME="Mnemodrome"
SRV_VER="$(get_kv version)"; [[ -n "$SRV_VER" ]] || SRV_VER="unknown"
REC="$(get_kv recorded)"
L_NEO="$(get_kv neo4j)"; L_QDR="$(get_kv qdrant)"; L_RED="$(get_kv redis)"; L_EVT="$(get_kv event_log)"

if [[ "$REC" == "yes" ]]; then
    success "Memory layers warmed; install recorded ($INSTALL_STAMP)"
elif [[ -n "$STATUS_LINE" ]]; then
    warn "Layers warmed, but the install-date record didn't complete (non-fatal; server is up)."
else
    warn "Warm-up didn't complete cleanly — server is up; layers initialize on your first client connection."
fi

# ---------------------------------------------------------------------------
# Per-layer status report — unambiguous up/down signal
# ---------------------------------------------------------------------------
report_layer() {  # $1 label  $2 state
    if [[ "$2" == "connected" ]]; then
        printf "    ${GREEN}\xe2\x9c\x93${NC} %-20s connected\n" "$1"
    else
        printf "    ${YELLOW}\xe2\x9a\xa0${NC} %-20s %s\n" "$1" "${2:-unknown}"
    fi
}
echo ""
info "Memory layer status:"
report_layer "Graph (Neo4j)"   "$L_NEO"
report_layer "Vector (Qdrant)" "$L_QDR"
report_layer "Cache (Redis)"   "$L_RED"
report_layer "Event Log"       "$L_EVT"
if [[ "$LLM_REACHABLE" == "true" ]]; then
    printf "    ${GREEN}\xe2\x9c\x93${NC} %-20s %s\n" "LLM ($LLM_PROVIDER)" "$LLM_ENDPOINT ($CHAT_MODEL / $EMBEDDING_MODEL)"
elif [[ ( "$LLM_PROVIDER" == "openai" || "$LLM_PROVIDER" == "anthropic" ) && -n "$LLM_API_KEY" ]]; then
    printf "    ${GREEN}\xe2\x9c\x93${NC} %-20s %s\n" "LLM ($LLM_PROVIDER)" "configured (cloud)"
elif [[ -n "$LLM_ENDPOINT" ]]; then
    printf "    ${YELLOW}\xe2\x9a\xa0${NC} %-20s %s\n" "LLM ($LLM_PROVIDER)" "$LLM_ENDPOINT — configured but unreachable at install; server retries at startup"
else
    printf "    ${YELLOW}\xe2\x9a\xa0${NC} %-20s %s\n" "LLM" "not configured — embeddings/extraction limited"
fi

echo ""
info "Container status:"
docker compose ps

# ---------------------------------------------------------------------------
# Success message
# ---------------------------------------------------------------------------
echo ""
echo "════════════════════════════════════════════════════════"
printf "${GREEN}${BOLD}  ✅  Mnemodrome v%s is LIVE and remembering.${NC}\n" "$SRV_VER"
echo "════════════════════════════════════════════════════════"
echo ""
printf "  MCP endpoint : ${CYAN}http://localhost:8765/mcp${NC}\n"
printf "  Health check : ${CYAN}http://localhost:8765/health${NC}\n"
printf "  Data root    : ${CYAN}%s${NC}\n" "$DATA_ROOT"
echo ""
echo "  Deep verification (recommended) — full MCP handshake + live tool round-trip:"
printf "    ${BOLD}uv run python scripts/verify_deployment.py${NC}\n"
echo "    uv is how we run Python — fast, reproducible, the modern standard."
echo "    Don't have it? One line:  curl -LsSf https://astral.sh/uv/install.sh | sh"
echo "    Docs: https://docs.astral.sh/uv/"
echo ""
echo "  To stop:    docker compose down"
echo "  To start:   docker compose up -d"
echo "  To rebuild: docker compose up -d --build"
echo ""
echo "  MCP client config snippets:"
echo ""
printf "  ${BOLD}Claude Code${NC} (~/.claude/.mcp.json or ~/.claude.json):\n"
cat <<'CLIENT1'
    {
      "mcpServers": {
        "mnemodrome": {
          "type": "http",
          "url": "http://localhost:8765/mcp"
        }
      }
    }
CLIENT1
echo ""
printf "  ${BOLD}Claude Desktop${NC} (claude_desktop_config.json):\n"
cat <<'CLIENT2'
    {
      "mcpServers": {
        "mnemodrome": {
          "url": "http://localhost:8765/mcp"
        }
      }
    }
CLIENT2
echo ""
printf "  ${BOLD}First steps${NC}\n"
echo "    1. Add the snippet above to your MCP client (Claude Code / Desktop), restart it."
echo "    2. Confirm the client lists Mnemodrome's tools (capture_fact, load_context, ...)."
echo "    3. Make your first memory — ask your agent to remember something, then recall it."
echo "    4. Check health anytime:  curl -s http://localhost:8765/health"
echo ""
