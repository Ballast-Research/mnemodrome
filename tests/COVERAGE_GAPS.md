# Test Coverage Gaps & Remaining Work

**Created**: 2026-03-13 | **Branch**: 035-test-suite
**Current State**: 1382 passed, 4 skipped, 9 xfailed, 0 failures

## Completed

- [x] P1: MCP server handler smoke tests — 48 tests covering all 29 handlers (`tests/contract/test_mcp_tool_handlers.py`)
- [x] P2: Deduplication service unit tests — 57 tests, three-stage pipeline (`tests/unit/test_deduplication_service.py`)
- [x] P3: Validation history service unit tests — 59 tests (`tests/unit/test_validation_history_service.py`)
- [x] P4: Entity cache unit tests — 44 tests with mocked Redis (`tests/unit/test_entity_cache.py`)
- [x] P5: Claim extraction service unit tests — 33 tests with mocked LLM (`tests/unit/test_claim_extraction_service.py`)

## Remaining

- [ ] P6: GPU cache manager unit tests (LOW — disabled in Docker, optional infrastructure)

## Bugs Found During Testing

| Bug | File | Status |
|-----|------|--------|
| `_calculate_complexity_for_matching` dict path sets `created_at = None` after reading it | `validation_history_service.py:374-376` | Documented in tests |
| `importlib.reload(config)` creates duplicate MemoryType enum instances | `admin_service.py:642` | Fixed — `get_memory_path()` uses `.value` string keys |
| Admin UI variable shadowing (`status` overwriting FastAPI module) | `admin_ui.py` | Fixed — renamed to `service_status` |

## Infrastructure Fixes During Test Suite Work

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| MCP server crash on startup | `simcheck-website-staging` on port 8083 poisoned Python's ThreadPoolExecutor via Uvicorn bind failure | Changed Admin UI to port 8084, made bind failure non-fatal |
| Per-connection service teardown | MCP SDK runs lifespan per-SSE-connection, destroying Neo4j/Redis on disconnect | Added `_services_initialized` guard, no teardown on client disconnect |
| Ollama fallback bypassing job manager | `OLLAMA_FALLBACK_URL` pointed to direct Ollama (11434) instead of job manager (8086) | Cleared fallback URL |

## Docker Migration Lessons Learned

The systemd-to-Docker migration exposed three failure patterns that tests should guard against:

| Pattern | What Happened | Test That Would Catch It |
|---------|--------------|--------------------------|
| **Silent dependency gaps** | `qdrant-client` not in pyproject.toml, vector layer silently disabled | Handler smoke test: call `search_similar` handler, assert it doesn't return "disabled" |
| **Stale code paths** | `get_event_log_health` still used SQLite API after PostgreSQL migration | Handler test: call health endpoint, assert no `AttributeError` |
| **Network assumption changes** | `localhost` means container loopback, not host machine | Integration test with health checks against configured service endpoints |
