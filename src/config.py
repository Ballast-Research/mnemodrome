"""Configuration management for Mnemodrome using environment variables."""

import os
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file
load_dotenv()


class MemoryType(Enum):
    """Memory types for granular storage organization."""

    FACT = "facts"
    LEARNING = "learnings"
    SESSION = "sessions"
    GRAPH = "graphs"
    UPDATE = "updates"
    PROCEDURE = "procedures"


# Required environment variables (updated for granular paths)
REQUIRED_ENV_VARS = [
    "NEO4J_URI",
    "NEO4J_USER",
    "NEO4J_PASSWORD",
]


# Per-type storage path env vars (Feature 010). Feature 039 adds the
# MNEMODROME_STORAGE_PATH umbrella: each type defaults to <umbrella>/<type>.
MEMORY_PATH_ENV_VARS: dict[str, str] = {
    "facts": "MNEMODROME_FACTS_PATH",
    "learnings": "MNEMODROME_LEARNINGS_PATH",
    "sessions": "MNEMODROME_SESSIONS_PATH",
    "graphs": "MNEMODROME_GRAPHS_PATH",
    "updates": "MNEMODROME_UPDATES_PATH",
    "procedures": "MNEMODROME_PROCEDURES_PATH",
}


def get_memory_path(memory_type: MemoryType) -> Path:
    """
    Get the storage path for a specific memory type.

    Priority (Feature 039):
    1. Per-type env var (MNEMODROME_FACTS_PATH, MNEMODROME_LEARNINGS_PATH, ...)
       overrides that one folder.
    2. MNEMODROME_STORAGE_PATH umbrella: <umbrella>/<type> (facts, learnings, ...).
    3. ValueError if neither is set.

    Installs that set all six per-type vars keep working unchanged.

    Args:
        memory_type: The type of memory to get path for

    Returns:
        Path to the memory type directory

    Raises:
        ValueError: If no storage path configured
    """
    var = MEMORY_PATH_ENV_VARS[memory_type.value]
    path = os.getenv(var)
    if path:
        return Path(path)

    umbrella = os.getenv("MNEMODROME_STORAGE_PATH")
    if umbrella:
        return Path(umbrella) / memory_type.value

    # No configuration found
    raise ValueError(
        f"No storage path configured for {memory_type.value}. "
        f"Set MNEMODROME_STORAGE_PATH (umbrella, defaults to <path>/{memory_type.value}) "
        f"or {var} in your .env file."
    )


def validate_llm_config() -> None:
    """
    Validate LLM configuration is properly set for model-agnostic operation.

    Ensures users explicitly configure their LLM provider and model.
    """
    provider = os.getenv("LLM_PROVIDER", "")
    model = os.getenv("CHAT_MODEL", "")
    api_key = os.getenv("LLM_API_KEY", "")

    # Require explicit provider
    if not provider:
        raise ValueError(
            "LLM_PROVIDER is required. Set in .env file.\n"
            "Supported providers: anthropic, openai, ollama, openai-compatible\n\n"
            "Examples:\n"
            "  LLM_PROVIDER=anthropic\n"
            "  LLM_PROVIDER=openai\n"
            "  LLM_PROVIDER=ollama\n"
            "  LLM_PROVIDER=openai-compatible\n\n"
            "See .env.example for full configuration."
        )

    # Validate provider value
    if provider not in ["anthropic", "openai", "ollama", "openai-compatible"]:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: {provider}\n"
            "Supported providers: anthropic, openai, ollama, openai-compatible"
        )

    # Require explicit model
    if not model:
        raise ValueError(
            f"CHAT_MODEL is required for {provider} provider. Set in .env file.\n\n"
            "Examples:\n"
            "  Anthropic: CHAT_MODEL=claude-3-5-sonnet-20241022\n"
            "  OpenAI: CHAT_MODEL=gpt-4\n"
            "  Ollama: CHAT_MODEL=llama3.1\n"
            "  openai-compatible: CHAT_MODEL=mistral-7b-instruct\n\n"
            "See .env.example for full configuration."
        )

    # Require API key for cloud providers
    if provider in ["anthropic", "openai", "openai-compatible"] and not api_key:
        raise ValueError(
            f"LLM_API_KEY is required for {provider} provider.\n"
            f"Set your API key in .env file:\n"
            f"  LLM_API_KEY=your-api-key-here\n\n"
            f"Get your API key from:\n"
            f"  Anthropic: https://console.anthropic.com/\n"
            f"  OpenAI: https://platform.openai.com/api-keys\n"
            f"  openai-compatible: see your provider's documentation"
        )

    # Require base URL for Ollama (warn — defaults to localhost)
    if provider == "ollama" and not (
        os.getenv("LLM_BASE_URL") or os.getenv("OLLAMA_BASE_URL")
    ):
        logger.warning(
            "LLM_BASE_URL not set for Ollama. Defaulting to http://localhost:11434\n"
            "Set explicitly in .env if Ollama is running on a different host."
        )

    # Require base URL for openai-compatible (hard error — no sane default)
    if provider == "openai-compatible" and not (
        os.getenv("LLM_BASE_URL") or os.getenv("OLLAMA_BASE_URL")
    ):
        raise ValueError(
            "LLM_BASE_URL is required for openai-compatible provider.\n"
            "Set the API base URL in .env file:\n"
            "  LLM_BASE_URL=https://openrouter.ai/api/v1\n\n"
            "Examples:\n"
            "  OpenRouter:  LLM_BASE_URL=https://openrouter.ai/api/v1\n"
            "  Groq:        LLM_BASE_URL=https://api.groq.com/openai/v1\n"
            "  xAI/Grok:    LLM_BASE_URL=https://api.x.ai/v1\n"
            "  DeepSeek:    LLM_BASE_URL=https://api.deepseek.com/v1\n"
            "  vLLM/local:  LLM_BASE_URL=http://localhost:8000/v1"
        )


def validate_config() -> None:
    """
    Validate that all required environment variables are set.

    Updated for Feature 010: At least one granular storage path must be configured.
    """
    # Check required Neo4j variables
    missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Please check your .env file."
        )

    # Check that a storage path is configured: either the umbrella
    # MNEMODROME_STORAGE_PATH or at least one per-type path (Feature 039).
    has_storage = bool(os.getenv("MNEMODROME_STORAGE_PATH")) or any(
        os.getenv(var) for var in MEMORY_PATH_ENV_VARS.values()
    )

    if not has_storage:
        raise ValueError(
            "No storage paths configured. Please set:\n"
            "  MNEMODROME_STORAGE_PATH (umbrella; each type lives in <path>/<type>)\n"
            "or per-type paths:\n"
            "  MNEMODROME_FACTS_PATH, MNEMODROME_LEARNINGS_PATH, "
            "MNEMODROME_SESSIONS_PATH, MNEMODROME_GRAPHS_PATH, "
            "MNEMODROME_UPDATES_PATH, MNEMODROME_PROCEDURES_PATH\n"
            "See .env.example for configuration examples."
        )

    # Validate LLM configuration
    validate_llm_config()


# Validate on import
validate_config()

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Storage Configuration (Feature 039: umbrella path)
# MNEMODROME_STORAGE_PATH is the umbrella root; each memory type defaults to
# <umbrella>/<type>. Per-type MNEMODROME_*_PATH vars override one folder each.
# Always resolve paths through get_memory_path(MemoryType.X).
_umbrella_storage_path = os.getenv("MNEMODROME_STORAGE_PATH")
MNEMODROME_STORAGE_PATH = (
    Path(_umbrella_storage_path) if _umbrella_storage_path else None
)

# MCP Server Configuration
MCP_PORT = int(os.getenv("MCP_PORT", "8765"))
MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_TRANSPORT = os.getenv(
    "MCP_TRANSPORT", "streamable-http"
)  # streamable-http (current MCP standard) or sse (deprecated, legacy clients)

# GPU Cache Management (Feature 030)
GPU_CACHE_ENABLED = os.getenv("GPU_CACHE_ENABLED", "false").lower() == "true"

# Redis Configuration (Feature 005: Entity Cache)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# Vector Embedding Dimension
# Must match the model's actual output size. install.sh probes the endpoint and
# writes the correct value. If you change the embedding model after collections
# are created, update this value AND re-embed all data (collections cannot be
# resized in place — delete them and let Mnemodrome recreate them).
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

# Retry Queue Configuration
RETRY_QUEUE_PATH = Path(os.getenv("RETRY_QUEUE_PATH", ".mnemodrome_retry_queue.json"))

# LLM Configuration (T053: LLM Extraction Enhancement)
# REQUIRED: Configure in .env file (see .env.example for provider-specific examples)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "")  # Required: anthropic, openai, or ollama
CHAT_MODEL = os.getenv("CHAT_MODEL", "")  # Required: Model name for chosen provider
# Optional per-role model overrides — both fall back to CHAT_MODEL when unset,
# preserving identical single-model behaviour for users who only set CHAT_MODEL.
EXTRACTION_MODEL = (
    os.getenv("EXTRACTION_MODEL", "") or CHAT_MODEL
)  # entity / claim extraction
CLASSIFICATION_MODEL = (
    os.getenv("CLASSIFICATION_MODEL", "") or CHAT_MODEL
)  # entity classification
LLM_API_KEY = os.getenv(
    "LLM_API_KEY", ""
)  # Required for Anthropic/OpenAI, not needed for Ollama
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL", ""
)  # Back-compat: existing Ollama installs that set OLLAMA_BASE_URL
LLM_BASE_URL = (
    os.getenv("LLM_BASE_URL") or OLLAMA_BASE_URL
)  # Preferred: used for both ollama and openai-compatible; falls back to OLLAMA_BASE_URL
OLLAMA_FALLBACK_URL = os.getenv(
    "OLLAMA_FALLBACK_URL", ""
)  # Fallback Ollama (e.g., direct Ollama bypassing job manager)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))  # 0.0 for deterministic
LLM_MAX_TOKENS = int(
    os.getenv("LLM_MAX_TOKENS", "4096")
)  # Max response tokens. Entity extraction emits large structured JSON for dense inputs; 1000 truncates it (invalid JSON -> 0 entities).
LLM_TIMEOUT = int(
    float(os.getenv("LLM_TIMEOUT", "91"))
)  # Timeout in seconds for LLM calls
LLM_MAX_RETRIES = int(
    os.getenv("LLM_MAX_RETRIES", "1")
)  # Max retries for LLM calls (Feature 030)
# Suppress hidden <think> tokens on reasoning models (Qwen3, DeepSeek-R1, etc.).
# Passed as extra_body={"think": False} on ollama / openai-compatible calls only;
# NEVER sent to the real openai or anthropic providers (would error).
LLM_DISABLE_THINKING = os.getenv("LLM_DISABLE_THINKING", "true").lower() == "true"

# ── Extraction canary (startup self-test) ────────────────────────────────────
# On startup Mnemodrome runs ONE extraction against a known sentence and asserts
# entities come back. This catches the silent-failure trap where a reasoning
# model (Qwen3, DeepSeek-R1, etc.) emits hidden thinking instead of JSON and
# every capture stores nothing. If the probe returns zero entities, Mnemodrome
# auto-applies the no-think setting and raises the token budget, then retries —
# so a healthy model self-heals into a no-fail install. Only a model that STILL
# returns nothing trips the hard-fail below.
EXTRACTION_CANARY_ENABLED = (
    os.getenv("EXTRACTION_CANARY_ENABLED", "true").lower() == "true"
)
# When the canary cannot be satisfied even after self-heal, Mnemodrome refuses to
# start rather than silently lose memories. Advanced operators who knowingly
# accept degraded extraction can set this true to downgrade the hard-fail to a
# loud warning (NOT recommended).
ALLOW_DEGRADED_EXTRACTION = (
    os.getenv("ALLOW_DEGRADED_EXTRACTION", "false").lower() == "true"
)


# MCP Server Version — single source of truth is the package metadata, which is
# populated from pyproject.toml at build/install time (and, as a fast-follow,
# from the git tag via hatch-vcs). We deliberately do NOT hardcode a per-release
# literal here; a second copy of the version only drifts out of sync. Falls back
# to reading pyproject.toml directly for an uninstalled source checkout (dev).
def _resolve_mcp_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("mnemodrome")
        except PackageNotFoundError:
            pass
    except Exception:
        pass
    try:
        import pathlib
        import tomllib

        pyproject = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
        with pyproject.open("rb") as fh:
            return tomllib.load(fh)["project"]["version"]
    except Exception:
        return "0.0.0+unknown"


MCP_VERSION = _resolve_mcp_version()

# Admin Web UI Configuration (Feature 013)
# Off by default (Feature 039). The UI binds to loopback unless ADMIN_WEB_HOST
# is set; expose it through a reverse proxy with TLS, never directly.
ADMIN_WEB_ENABLED = os.getenv("ADMIN_WEB_ENABLED", "false").lower() == "true"
ADMIN_WEB_HOST = os.getenv("ADMIN_WEB_HOST", "127.0.0.1")
ADMIN_WEB_PORT = int(os.getenv("ADMIN_WEB_PORT", "8083"))
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")  # Required, no default for security

# Logging Configuration (Runtime-Editable via Admin UI)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Event Log Configuration (Feature 009: Phase 2)
EVENT_LOG_ENABLED = os.getenv("EVENT_LOG_ENABLED", "true").lower() == "true"
EVENT_LOG_AUTO_CAPTURE = os.getenv("EVENT_LOG_AUTO_CAPTURE", "true").lower() == "true"
EVENT_LOG_RETENTION_DAYS = int(os.getenv("EVENT_LOG_RETENTION_DAYS", "90"))
EVENT_LOG_MAX_SIZE_MB = int(os.getenv("EVENT_LOG_MAX_SIZE_MB", "100"))
EVENT_LOG_SESSION_TIMEOUT_HOURS = int(os.getenv("EVENT_LOG_SESSION_TIMEOUT_HOURS", "4"))
EVENT_LOG_PERFORMANCE_SLA_MS = int(os.getenv("EVENT_LOG_PERFORMANCE_SLA_MS", "50"))
EVENT_LOG_QUERY_SLA_MS = int(os.getenv("EVENT_LOG_QUERY_SLA_MS", "100"))
EVENT_LOG_FAIL_LOUD = os.getenv("EVENT_LOG_FAIL_LOUD", "true").lower() == "true"
# Event log database path (used by the default SQLite backend)
EVENT_LOG_DB_PATH = os.getenv("EVENT_LOG_DB_PATH", "/data/mnemodrome/events/events.db")

# Event log backend selector: "sqlite" (default, zero-config) or "postgres" (opt-in)
EVENT_LOG_BACKEND = os.getenv("EVENT_LOG_BACKEND", "sqlite").lower()

# PostgreSQL connection — used when EVENT_LOG_BACKEND=postgres
# Full DSN takes priority over individual PG_* vars
EVENT_LOG_POSTGRES_DSN = os.getenv("EVENT_LOG_POSTGRES_DSN", "")

# Individual PostgreSQL connection parameters (used when EVENT_LOG_POSTGRES_DSN is not set)
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DATABASE = os.getenv("PG_DATABASE", "event_logs")
PG_USER = os.getenv("PG_USER", "event_log_user")
PG_PASSWORD = os.getenv("PG_PASSWORD", "")

# PostgreSQL connection pool sizing
PG_POOL_MIN = int(os.getenv("PG_POOL_MIN", "5"))
PG_POOL_MAX = int(os.getenv("PG_POOL_MAX", "20"))

# Service Management Configuration (Feature 014: Service Management UI Controls)
DEPLOYMENT_TYPE = os.getenv(
    "DEPLOYMENT_TYPE", ""
)  # Optional: override with "systemd" or "docker"
SERVICE_NAME = os.getenv(
    "SERVICE_NAME", "mnemodrome-mcp.service"
)  # Systemd unit or docker container name
DOCKER_CONTAINER_NAME = os.getenv(
    "DOCKER_CONTAINER_NAME", "mnemodrome-mcp"
)  # Docker container name for control operations
SERVICE_CONTROL_TIMEOUT = int(
    os.getenv("SERVICE_CONTROL_TIMEOUT", "30")
)  # Timeout for service control commands (seconds)
LOG_RETRIEVAL_TIMEOUT = int(
    os.getenv("LOG_RETRIEVAL_TIMEOUT", "10")
)  # Timeout for log retrieval commands (seconds)

# Knowledge Deduplication Configuration (Feature 023)
DEDUP_AUTO_SKIP_THRESHOLD = float(os.getenv("DEDUP_AUTO_SKIP_THRESHOLD", "0.95"))
DEDUP_VALIDATION_THRESHOLD = float(os.getenv("DEDUP_VALIDATION_THRESHOLD", "0.90"))
DEDUP_MAX_VALIDATION_TIME_MS = float(os.getenv("DEDUP_MAX_VALIDATION_TIME_MS", "300.0"))
DEDUP_VECTOR_SEARCH_LIMIT = int(os.getenv("DEDUP_VECTOR_SEARCH_LIMIT", "5"))
DEDUP_FUZZY_MATCH_THRESHOLD = float(os.getenv("DEDUP_FUZZY_MATCH_THRESHOLD", "0.85"))
DEDUP_ENABLE_GRACEFUL_DEGRADATION = (
    os.getenv("DEDUP_ENABLE_GRACEFUL_DEGRADATION", "true").lower() == "true"
)
DEDUP_BOOTSTRAP_BATCH_SIZE = int(os.getenv("DEDUP_BOOTSTRAP_BATCH_SIZE", "1"))
DEDUP_BOOTSTRAP_MAX_FAILURES = int(os.getenv("DEDUP_BOOTSTRAP_MAX_FAILURES", "10"))

# Bidirectional Conflict Detection Configuration (Feature 025)
BIDIRECTIONAL_VALIDATION_ENABLED = (
    os.getenv("BIDIRECTIONAL_VALIDATION_ENABLED", "true").lower() == "true"
)
CONFLICT_SEVERITY_MINOR_MAX = int(os.getenv("CONFLICT_SEVERITY_MINOR_MAX", "40"))
CONFLICT_SEVERITY_MAJOR_MIN = int(os.getenv("CONFLICT_SEVERITY_MAJOR_MIN", "70"))
CONFLICT_AUTO_CORRECT_ENABLED = (
    os.getenv("CONFLICT_AUTO_CORRECT_ENABLED", "true").lower() == "true"
)
CONFLICT_AUTO_CORRECT_CONFIDENCE_MIN = float(
    os.getenv("CONFLICT_AUTO_CORRECT_CONFIDENCE_MIN", "0.9")
)
CONFLICT_INBOUND_TIMEOUT_MS = int(os.getenv("CONFLICT_INBOUND_TIMEOUT_MS", "1000"))
CONFLICT_OUTBOUND_TIMEOUT_MS = int(os.getenv("CONFLICT_OUTBOUND_TIMEOUT_MS", "1000"))

# Critical entities for conflict detection (infrastructure components)
# Override with CRITICAL_ENTITIES env var (comma-separated) to add your own host names
CRITICAL_ENTITIES = {
    "Mnemodrome",
    "Neo4j",
    "Qdrant",
    "Redis",
    "Ollama",
    "mnemodrome-mcp",
    "admin-web",
    "event-log",
}


def validate_conflict_detection_config() -> None:
    """Validate bidirectional conflict detection configuration values."""
    if CONFLICT_SEVERITY_MINOR_MAX < 0 or CONFLICT_SEVERITY_MINOR_MAX > 100:
        raise ValueError(
            f"CONFLICT_SEVERITY_MINOR_MAX must be between 0 and 100. "
            f"Current value: {CONFLICT_SEVERITY_MINOR_MAX}"
        )

    if CONFLICT_SEVERITY_MAJOR_MIN < 0 or CONFLICT_SEVERITY_MAJOR_MIN > 100:
        raise ValueError(
            f"CONFLICT_SEVERITY_MAJOR_MIN must be between 0 and 100. "
            f"Current value: {CONFLICT_SEVERITY_MAJOR_MIN}"
        )

    if CONFLICT_SEVERITY_MINOR_MAX >= CONFLICT_SEVERITY_MAJOR_MIN:
        raise ValueError(
            f"CONFLICT_SEVERITY_MINOR_MAX ({CONFLICT_SEVERITY_MINOR_MAX}) must be "
            f"< CONFLICT_SEVERITY_MAJOR_MIN ({CONFLICT_SEVERITY_MAJOR_MIN})"
        )

    if not 0.0 <= CONFLICT_AUTO_CORRECT_CONFIDENCE_MIN <= 1.0:
        raise ValueError(
            f"CONFLICT_AUTO_CORRECT_CONFIDENCE_MIN must be between 0.0 and 1.0. "
            f"Current value: {CONFLICT_AUTO_CORRECT_CONFIDENCE_MIN}"
        )

    if CONFLICT_INBOUND_TIMEOUT_MS < 100 or CONFLICT_INBOUND_TIMEOUT_MS > 5000:
        raise ValueError(
            f"CONFLICT_INBOUND_TIMEOUT_MS must be between 100 and 5000. "
            f"Current value: {CONFLICT_INBOUND_TIMEOUT_MS}"
        )

    if CONFLICT_OUTBOUND_TIMEOUT_MS < 100 or CONFLICT_OUTBOUND_TIMEOUT_MS > 5000:
        raise ValueError(
            f"CONFLICT_OUTBOUND_TIMEOUT_MS must be between 100 and 5000. "
            f"Current value: {CONFLICT_OUTBOUND_TIMEOUT_MS}"
        )


def validate_event_log_config() -> None:
    """Validate event log configuration values."""
    # Backend selector
    if EVENT_LOG_BACKEND not in {"sqlite", "postgres"}:
        raise ValueError(
            f"EVENT_LOG_BACKEND must be 'sqlite' or 'postgres'. "
            f"Current value: '{EVENT_LOG_BACKEND}'"
        )

    # PostgreSQL requires at minimum a DSN or a password
    if EVENT_LOG_BACKEND == "postgres":
        if not EVENT_LOG_POSTGRES_DSN and not PG_PASSWORD:
            raise ValueError(
                "EVENT_LOG_BACKEND=postgres requires either EVENT_LOG_POSTGRES_DSN "
                "or PG_PASSWORD to be set."
            )

    if EVENT_LOG_RETENTION_DAYS < 1:
        raise ValueError(
            "EVENT_LOG_RETENTION_DAYS must be >= 1. "
            f"Current value: {EVENT_LOG_RETENTION_DAYS}"
        )

    if EVENT_LOG_MAX_SIZE_MB < 10:
        raise ValueError(
            "EVENT_LOG_MAX_SIZE_MB must be >= 10 (minimum 10MB). "
            f"Current value: {EVENT_LOG_MAX_SIZE_MB}"
        )

    if EVENT_LOG_SESSION_TIMEOUT_HOURS < 0:
        raise ValueError(
            "EVENT_LOG_SESSION_TIMEOUT_HOURS must be >= 0. "
            f"Current value: {EVENT_LOG_SESSION_TIMEOUT_HOURS}"
        )

    if EVENT_LOG_PERFORMANCE_SLA_MS < 1:
        raise ValueError(
            "EVENT_LOG_PERFORMANCE_SLA_MS must be >= 1. "
            f"Current value: {EVENT_LOG_PERFORMANCE_SLA_MS}"
        )

    if EVENT_LOG_QUERY_SLA_MS < 1:
        raise ValueError(
            "EVENT_LOG_QUERY_SLA_MS must be >= 1. "
            f"Current value: {EVENT_LOG_QUERY_SLA_MS}"
        )


def validate_admin_config() -> None:
    """Validate admin web UI configuration."""
    if ADMIN_WEB_ENABLED:
        if not ADMIN_PASSWORD:
            raise ValueError(
                "ADMIN_PASSWORD is required when ADMIN_WEB_ENABLED=true. "
                "Please set a strong password in .env file for security."
            )

        if len(ADMIN_PASSWORD) < 12:
            logger.warning(
                f"ADMIN_PASSWORD should be at least 12 characters for security. "
                f"Current length: {len(ADMIN_PASSWORD)}"
            )

        if ADMIN_WEB_PORT < 1024 or ADMIN_WEB_PORT > 65535:
            raise ValueError(
                f"ADMIN_WEB_PORT must be between 1024 and 65535. "
                f"Current value: {ADMIN_WEB_PORT}"
            )

    # Validate LOG_LEVEL
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if LOG_LEVEL not in valid_log_levels:
        raise ValueError(
            f"LOG_LEVEL must be one of {valid_log_levels}. Current value: {LOG_LEVEL}"
        )


def validate_deduplication_config() -> None:
    """Validate knowledge deduplication configuration values."""
    if not 0.0 <= DEDUP_AUTO_SKIP_THRESHOLD <= 1.0:
        raise ValueError(
            f"DEDUP_AUTO_SKIP_THRESHOLD must be between 0.0 and 1.0. "
            f"Current value: {DEDUP_AUTO_SKIP_THRESHOLD}"
        )

    if not 0.0 <= DEDUP_VALIDATION_THRESHOLD <= 1.0:
        raise ValueError(
            f"DEDUP_VALIDATION_THRESHOLD must be between 0.0 and 1.0. "
            f"Current value: {DEDUP_VALIDATION_THRESHOLD}"
        )

    if DEDUP_VALIDATION_THRESHOLD > DEDUP_AUTO_SKIP_THRESHOLD:
        raise ValueError(
            f"DEDUP_VALIDATION_THRESHOLD ({DEDUP_VALIDATION_THRESHOLD}) must be "
            f"<= DEDUP_AUTO_SKIP_THRESHOLD ({DEDUP_AUTO_SKIP_THRESHOLD})"
        )

    if not 0.0 <= DEDUP_FUZZY_MATCH_THRESHOLD <= 1.0:
        raise ValueError(
            f"DEDUP_FUZZY_MATCH_THRESHOLD must be between 0.0 and 1.0. "
            f"Current value: {DEDUP_FUZZY_MATCH_THRESHOLD}"
        )

    if DEDUP_MAX_VALIDATION_TIME_MS < 1.0:
        raise ValueError(
            f"DEDUP_MAX_VALIDATION_TIME_MS must be >= 1.0. "
            f"Current value: {DEDUP_MAX_VALIDATION_TIME_MS}"
        )

    if DEDUP_VECTOR_SEARCH_LIMIT < 1 or DEDUP_VECTOR_SEARCH_LIMIT > 20:
        raise ValueError(
            f"DEDUP_VECTOR_SEARCH_LIMIT must be between 1 and 20. "
            f"Current value: {DEDUP_VECTOR_SEARCH_LIMIT}"
        )

    if DEDUP_BOOTSTRAP_BATCH_SIZE < 1:
        raise ValueError(
            f"DEDUP_BOOTSTRAP_BATCH_SIZE must be >= 1. "
            f"Current value: {DEDUP_BOOTSTRAP_BATCH_SIZE}"
        )

    if DEDUP_BOOTSTRAP_MAX_FAILURES < 0:
        raise ValueError(
            f"DEDUP_BOOTSTRAP_MAX_FAILURES must be >= 0. "
            f"Current value: {DEDUP_BOOTSTRAP_MAX_FAILURES}"
        )


# Run event log validation
validate_event_log_config()

# Run admin config validation
validate_admin_config()

# Run deduplication config validation
validate_deduplication_config()

# Run conflict detection config validation
validate_conflict_detection_config()

# Cross-Session Pattern Recognition Configuration (Feature 026)
PATTERN_RECOGNITION_ENABLED = (
    os.getenv("PATTERN_RECOGNITION_ENABLED", "true").lower() == "true"
)
PATTERN_EXTRACTION_MIN_SESSIONS = int(os.getenv("PATTERN_EXTRACTION_MIN_SESSIONS", "3"))
PATTERN_SIMILARITY_THRESHOLD = float(os.getenv("PATTERN_SIMILARITY_THRESHOLD", "0.90"))
PATTERN_LLM_MODEL = os.getenv("PATTERN_LLM_MODEL", CHAT_MODEL)
PATTERN_CACHE_TTL_SECONDS = int(os.getenv("PATTERN_CACHE_TTL_SECONDS", "3600"))


def validate_pattern_recognition_config() -> None:
    """Validate cross-session pattern recognition configuration values."""
    if not 0.0 <= PATTERN_SIMILARITY_THRESHOLD <= 1.0:
        raise ValueError(
            f"PATTERN_SIMILARITY_THRESHOLD must be between 0.0 and 1.0. "
            f"Current value: {PATTERN_SIMILARITY_THRESHOLD}"
        )

    if PATTERN_EXTRACTION_MIN_SESSIONS < 1:
        raise ValueError(
            f"PATTERN_EXTRACTION_MIN_SESSIONS must be >= 1. "
            f"Current value: {PATTERN_EXTRACTION_MIN_SESSIONS}"
        )

    if PATTERN_CACHE_TTL_SECONDS < 60:
        raise ValueError(
            f"PATTERN_CACHE_TTL_SECONDS must be >= 60 (1 minute). "
            f"Current value: {PATTERN_CACHE_TTL_SECONDS}"
        )


# Run pattern recognition config validation
validate_pattern_recognition_config()

# Procedural Memory Configuration (Feature 031)
PROCEDURE_DEDUP_THRESHOLD = float(os.getenv("PROCEDURE_DEDUP_THRESHOLD", "0.92"))
PROCEDURE_SEARCH_DEFAULT_LIMIT = int(os.getenv("PROCEDURE_SEARCH_DEFAULT_LIMIT", "5"))
PROCEDURE_ENRICHMENT_TIMEOUT_S = int(os.getenv("PROCEDURE_ENRICHMENT_TIMEOUT_S", "10"))

# Evolving Strategies Configuration (Feature 032)
EVOLUTION_ENABLED = os.getenv("EVOLUTION_ENABLED", "false").lower() == "true"
EVOLUTION_MIN_SUCCESS_RATE = float(os.getenv("EVOLUTION_MIN_SUCCESS_RATE", "0.8"))
EVOLUTION_MIN_OCCURRENCES = int(os.getenv("EVOLUTION_MIN_OCCURRENCES", "3"))
EVOLUTION_MAX_CANDIDATES = int(os.getenv("EVOLUTION_MAX_CANDIDATES", "10"))
EVOLUTION_BACKUP_RETENTION_DAYS = int(
    os.getenv("EVOLUTION_BACKUP_RETENTION_DAYS", "30")
)
EVOLUTION_SKILL_SCOPE = os.getenv(
    "EVOLUTION_SKILL_SCOPE", "~/.claude/skills/*/SKILL.md"
)

# ACT-R Cognitive Activation Configuration (Feature 034)
ACTR_ENABLED = os.getenv("ACTR_ENABLED", "true").lower() == "true"
ACTR_DECAY_EXPONENT = float(os.getenv("ACTR_DECAY_EXPONENT", "0.5"))
ACTR_CONTENT_WEIGHT = float(os.getenv("ACTR_CONTENT_WEIGHT", "0.7"))
ACTR_ACTIVATION_WEIGHT = float(os.getenv("ACTR_ACTIVATION_WEIGHT", "0.2"))
ACTR_HEBBIAN_WEIGHT = float(os.getenv("ACTR_HEBBIAN_WEIGHT", "0.1"))
ACTR_HEBBIAN_INCREMENT = float(os.getenv("ACTR_HEBBIAN_INCREMENT", "0.1"))
ACTR_CONFIDENCE_DEFAULT = float(os.getenv("ACTR_CONFIDENCE_DEFAULT", "0.8"))


# ---------------------------------------------------------------------------
# Predicate cardinality (supersession policy)
# ---------------------------------------------------------------------------
# A relationship auto-supersedes a prior same-(subject, predicate) edge ONLY
# when the predicate is FUNCTIONAL (single-valued): a subject can have at most
# one valid target at a time (e.g. you live in one place). CUMULATIVE / many-
# valued predicates (ORDERED, BOUGHT, OWNS, USES, ...) must NOT supersede —
# independent facts that happen to share subject+predicate must coexist.
#
# This is the safe default: any predicate NOT listed here is treated as
# cumulative, so an unknown/new predicate can never silently cause data loss
# (worst case: a harmless stale duplicate). See src/services/predicate_cardinality.py.
#
# MNEMODROME_FUNCTIONAL_PREDICATES (comma-separated) EXTENDS this built-in set;
# it never shrinks it, so the safe defaults can't be lost via misconfiguration.
FUNCTIONAL_PREDICATES_DEFAULT = frozenset(
    {
        "LIVES_IN",
        "LOCATED_IN",
        "RESIDES_IN",
        "BORN_IN",
        "BORN_ON",
        "HAS_STATUS",
        "CURRENT_STATUS",
    }
)
_extra_functional = {
    p.strip().upper()
    for p in os.getenv("MNEMODROME_FUNCTIONAL_PREDICATES", "").split(",")
    if p.strip()
}
FUNCTIONAL_PREDICATES = frozenset(FUNCTIONAL_PREDICATES_DEFAULT | _extra_functional)


def validate_actr_config() -> None:
    """Validate ACT-R cognitive activation configuration values."""
    if not 0.0 < ACTR_DECAY_EXPONENT <= 2.0:
        raise ValueError(
            f"ACTR_DECAY_EXPONENT must be between 0.0 (exclusive) and 2.0. "
            f"Current value: {ACTR_DECAY_EXPONENT}"
        )

    if ACTR_CONTENT_WEIGHT < 0.0:
        raise ValueError(
            f"ACTR_CONTENT_WEIGHT must be >= 0.0. Current value: {ACTR_CONTENT_WEIGHT}"
        )

    if ACTR_ACTIVATION_WEIGHT < 0.0:
        raise ValueError(
            f"ACTR_ACTIVATION_WEIGHT must be >= 0.0. "
            f"Current value: {ACTR_ACTIVATION_WEIGHT}"
        )

    if ACTR_HEBBIAN_WEIGHT < 0.0:
        raise ValueError(
            f"ACTR_HEBBIAN_WEIGHT must be >= 0.0. Current value: {ACTR_HEBBIAN_WEIGHT}"
        )

    if not 0.0 < ACTR_HEBBIAN_INCREMENT <= 1.0:
        raise ValueError(
            f"ACTR_HEBBIAN_INCREMENT must be between 0.0 (exclusive) and 1.0. "
            f"Current value: {ACTR_HEBBIAN_INCREMENT}"
        )

    if not 0.01 <= ACTR_CONFIDENCE_DEFAULT <= 0.99:
        raise ValueError(
            f"ACTR_CONFIDENCE_DEFAULT must be between 0.01 and 0.99. "
            f"Current value: {ACTR_CONFIDENCE_DEFAULT}"
        )


def validate_evolution_config() -> None:
    """Validate evolving strategies configuration values."""
    if not 0.0 <= EVOLUTION_MIN_SUCCESS_RATE <= 1.0:
        raise ValueError(
            f"EVOLUTION_MIN_SUCCESS_RATE must be between 0.0 and 1.0. "
            f"Current value: {EVOLUTION_MIN_SUCCESS_RATE}"
        )

    if EVOLUTION_MIN_OCCURRENCES < 1:
        raise ValueError(
            f"EVOLUTION_MIN_OCCURRENCES must be >= 1. "
            f"Current value: {EVOLUTION_MIN_OCCURRENCES}"
        )

    if EVOLUTION_MAX_CANDIDATES < 1:
        raise ValueError(
            f"EVOLUTION_MAX_CANDIDATES must be >= 1. "
            f"Current value: {EVOLUTION_MAX_CANDIDATES}"
        )

    if EVOLUTION_BACKUP_RETENTION_DAYS < 1:
        raise ValueError(
            f"EVOLUTION_BACKUP_RETENTION_DAYS must be >= 1. "
            f"Current value: {EVOLUTION_BACKUP_RETENTION_DAYS}"
        )


# Run evolution config validation
validate_evolution_config()

# Run ACT-R activation config validation
validate_actr_config()

# Feature 024: Project-Aware Session Init
MNEMODROME_HOOK_TOKEN = os.getenv("MNEMODROME_HOOK_TOKEN", "")
PROJECT_CONTEXT_ENDPOINT_ENABLED = (
    os.getenv("PROJECT_CONTEXT_ENDPOINT_ENABLED", "true").lower() == "true"
)
PROJECT_CONTEXT_DEFAULT_DAYS = int(os.getenv("PROJECT_CONTEXT_DEFAULT_DAYS", "7"))
PROJECT_CONTEXT_TEMPORAL_LIMIT = int(os.getenv("PROJECT_CONTEXT_TEMPORAL_LIMIT", "5"))
PROJECT_CONTEXT_SESSIONS_LIMIT = int(os.getenv("PROJECT_CONTEXT_SESSIONS_LIMIT", "3"))
PROJECT_CONTEXT_LEARNINGS_LIMIT = int(os.getenv("PROJECT_CONTEXT_LEARNINGS_LIMIT", "3"))

# Reasoning-Time Knowledge Validation (Feature 028)
REASONING_VALIDATION_ENABLED = (
    os.getenv("REASONING_VALIDATION_ENABLED", "true").lower() == "true"
)
REASONING_VALIDATION_SOFT_TIMEOUT_MS = int(
    os.getenv("REASONING_VALIDATION_SOFT_TIMEOUT_MS", "500")
)
REASONING_VALIDATION_HARD_TIMEOUT_MS = int(
    os.getenv("REASONING_VALIDATION_HARD_TIMEOUT_MS", "1000")
)
REASONING_VALIDATION_PER_QUERY_TIMEOUT_MS = int(
    os.getenv("REASONING_VALIDATION_PER_QUERY_TIMEOUT_MS", "300")
)
REASONING_VALIDATION_LEARNING_LIMIT = int(
    os.getenv("REASONING_VALIDATION_LEARNING_LIMIT", "5")
)
REASONING_VALIDATION_PATTERN_LIMIT = int(
    os.getenv("REASONING_VALIDATION_PATTERN_LIMIT", "5")
)
REASONING_VALIDATION_MIN_RELEVANCE_SCORE = float(
    os.getenv("REASONING_VALIDATION_MIN_RELEVANCE_SCORE", "0.75")
)


def validate_reasoning_validation_config() -> None:
    """Validate reasoning-time knowledge validation configuration."""
    if (
        REASONING_VALIDATION_SOFT_TIMEOUT_MS < 100
        or REASONING_VALIDATION_SOFT_TIMEOUT_MS > 3000
    ):
        raise ValueError(
            f"REASONING_VALIDATION_SOFT_TIMEOUT_MS must be 100-3000. "
            f"Current: {REASONING_VALIDATION_SOFT_TIMEOUT_MS}"
        )
    if REASONING_VALIDATION_HARD_TIMEOUT_MS < REASONING_VALIDATION_SOFT_TIMEOUT_MS:
        raise ValueError(
            "REASONING_VALIDATION_HARD_TIMEOUT_MS must be >= REASONING_VALIDATION_SOFT_TIMEOUT_MS"
        )
    if (
        REASONING_VALIDATION_PER_QUERY_TIMEOUT_MS
        >= REASONING_VALIDATION_HARD_TIMEOUT_MS
    ):
        raise ValueError(
            "REASONING_VALIDATION_PER_QUERY_TIMEOUT_MS must be < REASONING_VALIDATION_HARD_TIMEOUT_MS"
        )
    if not 0.0 <= REASONING_VALIDATION_MIN_RELEVANCE_SCORE <= 1.0:
        raise ValueError(
            f"REASONING_VALIDATION_MIN_RELEVANCE_SCORE must be 0.0-1.0. "
            f"Current: {REASONING_VALIDATION_MIN_RELEVANCE_SCORE}"
        )
    if (
        REASONING_VALIDATION_LEARNING_LIMIT < 1
        or REASONING_VALIDATION_LEARNING_LIMIT > 20
    ):
        raise ValueError(
            f"REASONING_VALIDATION_LEARNING_LIMIT must be 1-20. "
            f"Current: {REASONING_VALIDATION_LEARNING_LIMIT}"
        )


validate_reasoning_validation_config()
