"""
Mnemodrome MCP Server - Process-Bound Lifecycle

Implements temporal-first context loading with optional keyword filtering.
Uses process-bound lifecycle with signal handlers for production deployment.
"""

import asyncio
import functools
import os
import secrets
import signal
import sys
import time
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from loguru import logger
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.config import (
    ACTR_CONFIDENCE_DEFAULT,
    ACTR_ENABLED,
    EVOLUTION_ENABLED,
    MCP_HOST,
    MCP_PORT,
    MCP_VERSION,
    MNEMODROME_HOOK_TOKEN,
    PATTERN_RECOGNITION_ENABLED,
    PROJECT_CONTEXT_DEFAULT_DAYS,
    PROJECT_CONTEXT_LEARNINGS_LIMIT,
    PROJECT_CONTEXT_SESSIONS_LIMIT,
    PROJECT_CONTEXT_TEMPORAL_LIMIT,
    REDIS_DB,
    REDIS_HOST,
    REDIS_PORT,
    MemoryType,
)
from src.models.deduplication_models import AppendResult, ValidationResult
from src.models.procedure_models import Procedure
from src.models.reasoning_validation_models import ReasoningValidationResult
from src.services.activation_service import (
    activation_service,
    calculate_activation,
    compute_final_score,
)
from src.services.auto_correction_service import AutoCorrectionService
from src.services.claim_extraction_service import ClaimExtractionService
from src.services.classification_service import classification_service
from src.services.conflict_detection_service import conflict_detection_service
from src.services.conflict_validation_service import ConflictValidationService
from src.services.context_formatter import format_context_markdown
from src.services.context_retrieval_service import get_context_retrieval_service
from src.services.entity_cache import entity_cache
from src.services.entity_factory import entity_factory
from src.services.event_log_service import event_log_service
from src.services.evolution_candidate_service import EvolutionCandidateService
from src.services.file_service import file_service
from src.services.graph_factory import graph_factory
from src.services.keyword_extractor import extract_keywords
from src.services.llm_service import llm_service as llm_service_singleton
from src.services.neo4j_fact_retrieval_service import Neo4jFactRetrievalService
from src.services.neo4j_service import neo4j_service
from src.services.pattern_extraction_service import PatternExtractionService
from src.services.pattern_query_service import PatternQueryService
from src.services.pattern_storage_service import PatternStorageService
from src.services.procedure_service import procedure_service
from src.services.reasoning_validation_service import ReasoningValidationService
from src.services.redis_service import redis_service
from src.services.relationship_factory import relationship_factory
from src.services.vector_service import get_vector_service

# Global initialization state
_services_initialized = False

# Background enrichment task tracking (fire-and-forget pattern)
# Captures return immediately; entity extraction + vector embedding run in background
_background_tasks: set[asyncio.Task] = set()


def _fire_and_forget(
    coro: Coroutine[Any, Any, Any], description: str = "background task"
):
    """Schedule a coroutine to run in the background without blocking.

    The task is tracked in _background_tasks to prevent garbage collection.
    On completion (success or failure), it auto-removes from the set.
    """
    task = asyncio.create_task(coro)
    _background_tasks.add(task)

    def _on_done(t):
        _background_tasks.discard(t)
        if t.cancelled():
            logger.debug(f"Background {description} was cancelled")
        elif t.exception():
            logger.warning(f"Background {description} failed: {t.exception()}")
        else:
            logger.debug(f"Background {description} completed successfully")

    task.add_done_callback(_on_done)
    return task


# ========================================
# Feature 034: ACT-R Activation Helpers
# ========================================


async def _apply_activation_scoring(
    results: list[dict], content_scores: list[float] | None = None
) -> list[dict]:
    """
    Apply ACT-R activation scoring to retrieval results and re-rank.

    Fetches activation metadata from Neo4j, calculates activation scores,
    blends with content match scores, and returns re-ranked results.
    Fires access tracking and Hebbian strengthening in background.

    Args:
        results: List of result dicts with source/target UUIDs.
        content_scores: Optional content match scores (0-1) per result.
                       If None, uses position-based normalization.

    Returns:
        Re-ranked results list (highest activation first).
    """
    if not results or not ACTR_ENABLED:
        return results

    try:
        from datetime import datetime as dt

        # Extract unique entity UUIDs from results
        entity_uuids = set()
        for r in results:
            if isinstance(r, dict):
                source = r.get("source", {})
                target = r.get("target", {})
                if isinstance(source, dict) and source.get("uuid"):
                    entity_uuids.add(source["uuid"])
                if isinstance(target, dict) and target.get("uuid"):
                    entity_uuids.add(target["uuid"])

        if not entity_uuids:
            return results

        uuid_list = list(entity_uuids)

        # Fetch activation metadata and Hebbian/confidence data concurrently
        activation_query = """
        UNWIND $uuids AS uuid
        MATCH (n {uuid: uuid})
        RETURN n.uuid AS uuid,
               coalesce(n.access_count, 0) AS access_count,
               n.last_accessed AS last_accessed,
               n.created_at AS created_at
        """

        # D7 (FR-012): scope all RELATES_TO reads to native edges (r.type IS NOT NULL)
        # Batch Hebbian weight lookup: avg hebbian_weight per entity (T019)
        hebbian_query = """
        UNWIND $uuids AS uuid
        MATCH (e {uuid: uuid})-[r:RELATES_TO]-(other)
        WHERE r.invalid_at IS NULL AND r.hebbian_weight IS NOT NULL
          AND r.type IS NOT NULL
        RETURN uuid, avg(r.hebbian_weight) AS avg_hebbian
        """

        # Batch confidence lookup per source-target pair (T016)
        # Build pairs from results for confidence fetch
        confidence_pairs = []
        for r in results:
            s_uuid = (
                r.get("source", {}).get("uuid")
                if isinstance(r.get("source"), dict)
                else None
            )
            t_uuid = (
                r.get("target", {}).get("uuid")
                if isinstance(r.get("target"), dict)
                else None
            )
            if s_uuid and t_uuid:
                confidence_pairs.append([s_uuid, t_uuid])

        # D7: scope to native edges
        confidence_query = """
        UNWIND $pairs AS pair
        MATCH (s {uuid: pair[0]})-[r:RELATES_TO]-(t {uuid: pair[1]})
        WHERE r.invalid_at IS NULL AND r.type IS NOT NULL
        RETURN pair[0] AS source_uuid, pair[1] AS target_uuid,
               coalesce(r.confidence, $default_confidence) AS confidence
        """

        # Run all three queries concurrently for performance (NFR-001: <50ms)
        queries = [
            neo4j_service.query(activation_query, {"uuids": uuid_list}),
            neo4j_service.query(hebbian_query, {"uuids": uuid_list}),
        ]
        if confidence_pairs:
            queries.append(
                neo4j_service.query(
                    confidence_query,
                    {
                        "pairs": confidence_pairs,
                        "default_confidence": ACTR_CONFIDENCE_DEFAULT,
                    },
                )
            )

        query_results = await asyncio.gather(*queries, return_exceptions=True)

        # Unpack results, treating exceptions as empty
        activation_data = (
            query_results[0] if not isinstance(query_results[0], Exception) else []
        )
        hebbian_data = (
            query_results[1] if not isinstance(query_results[1], Exception) else []
        )
        confidence_data = (
            query_results[2]
            if len(query_results) > 2 and not isinstance(query_results[2], Exception)
            else []
        )

        if isinstance(query_results[0], Exception):
            logger.warning(f"Activation metadata query failed: {query_results[0]}")
        if isinstance(query_results[1], Exception):
            logger.warning(f"Hebbian weight query failed: {query_results[1]}")
        if len(query_results) > 2 and isinstance(query_results[2], Exception):
            logger.warning(f"Confidence query failed: {query_results[2]}")

        # Build activation lookup map
        activation_map = {}
        for row in activation_data:
            last_accessed = row.get("last_accessed") or row.get("created_at")
            if last_accessed and not isinstance(last_accessed, dt):
                try:
                    last_accessed = dt.fromisoformat(
                        str(last_accessed).replace("Z", "+00:00")
                    )
                except (ValueError, TypeError):
                    last_accessed = dt.now(UTC)
            elif not last_accessed:
                last_accessed = dt.now(UTC)

            activation_map[row["uuid"]] = {
                "access_count": row.get("access_count", 0),
                "last_accessed": last_accessed,
            }

        # Build Hebbian weight lookup map (T019)
        hebbian_map: dict[str, float] = {}
        for row in hebbian_data:
            hebbian_map[row["uuid"]] = row.get("avg_hebbian", 0.0)

        # Build confidence lookup map keyed by (source, target) (T016)
        confidence_map: dict[tuple[str, str], float] = {}
        for row in confidence_data:
            key = (row["source_uuid"], row["target_uuid"])
            confidence_map[key] = row.get("confidence", ACTR_CONFIDENCE_DEFAULT)

        # Calculate activation scores and re-rank
        scored_results = []
        for i, r in enumerate(results):
            # Content score: from provided scores or position-based
            if content_scores and i < len(content_scores):
                content_match = content_scores[i]
            else:
                # Normalize position to 0-1 (first result = 1.0)
                content_match = 1.0 - (i / max(len(results), 1))

            # Average activation across source and target entities
            activations = []
            source_uuid = (
                r.get("source", {}).get("uuid")
                if isinstance(r.get("source"), dict)
                else None
            )
            target_uuid = (
                r.get("target", {}).get("uuid")
                if isinstance(r.get("target"), dict)
                else None
            )

            for uuid in [source_uuid, target_uuid]:
                if uuid and uuid in activation_map:
                    act = calculate_activation(
                        access_count=activation_map[uuid]["access_count"],
                        last_accessed=activation_map[uuid]["last_accessed"],
                    )
                    activations.append(act)

            avg_activation = sum(activations) / len(activations) if activations else 0.0

            # Hebbian boost: average of hebbian weights for entities in this result (T019)
            hebbian_values = []
            for uuid in [source_uuid, target_uuid]:
                if uuid and uuid in hebbian_map:
                    hebbian_values.append(hebbian_map[uuid])
            hebbian_boost = (
                sum(hebbian_values) / len(hebbian_values) if hebbian_values else 0.0
            )

            # Confidence modulation: demote low-confidence results (T016)
            confidence = ACTR_CONFIDENCE_DEFAULT
            if source_uuid and target_uuid:
                # Try both directions since RELATES_TO is matched bidirectionally
                confidence = confidence_map.get(
                    (source_uuid, target_uuid),
                    confidence_map.get(
                        (target_uuid, source_uuid), ACTR_CONFIDENCE_DEFAULT
                    ),
                )
            content_match *= confidence

            final_score = compute_final_score(
                content_match=content_match,
                activation=avg_activation,
                hebbian_boost=hebbian_boost,
            )

            scored_results.append((final_score, i, r))

        # Sort by final score descending
        scored_results.sort(key=lambda x: x[0], reverse=True)
        reranked = [r for _, _, r in scored_results]

        # Fire-and-forget: update access tracking and Hebbian links
        _fire_and_forget(
            activation_service.update_access_tracking(uuid_list),
            description="activation access tracking",
        )
        if len(uuid_list) >= 2:
            _fire_and_forget(
                activation_service.strengthen_hebbian_links(uuid_list),
                description="hebbian link strengthening",
            )

        return reranked

    except Exception as e:
        logger.warning(
            f"Activation scoring failed, returning original order (FR-008): {e}"
        )
        return results


# ========================================
# Feature 023: Deduplication Response Formatters
# ========================================


def _format_auto_skip_response(result: ValidationResult, total_time_ms: float) -> str:
    """Format response when learning is auto-skipped due to high similarity.

    Args:
        result: ValidationResult with status="auto_skipped"
        total_time_ms: Total operation time in milliseconds

    Returns:
        User-friendly response explaining the skip and offering alternatives
    """
    return f"""⏭️ Learning auto-skipped (duplicate detected)

**Similarity**: {result.similarity_score:.0%} match with existing learning
**Conflict Type**: {result.conflict_type}
**Existing UUID**: `{result.existing_learning_uuid}`
**Validation Stage**: {result.validation_stage}
**Performance**: {total_time_ms:.1f}ms

**Rationale**: {result.rationale}

**Options**:
- To append to existing: use `append_to_learning(learning_uuid="{result.existing_learning_uuid}", new_content="...")`
- To force capture anyway: use `capture_learning(..., auto_deduplicate=False)`
"""


def _format_validation_required_response(
    result: ValidationResult, total_time_ms: float
) -> str:
    """Format response when validation is required due to moderate similarity.

    Args:
        result: ValidationResult with status="validation_required"
        total_time_ms: Total operation time in milliseconds

    Returns:
        User-friendly response with options for user to choose
    """
    # Format existing items list
    items_list = ""
    if result.existing_items:
        for i, item in enumerate(result.existing_items[:3], 1):
            if isinstance(item, dict):
                uuid = item.get("uuid", "unknown")
                title = item.get("title", "Untitled")
                score = item.get("similarity_score", 0)
            else:
                uuid = item.uuid
                title = item.title
                score = item.similarity_score
            items_list += f"\n{i}. **{title}** ({score:.0%} similar)\n   UUID: `{uuid}`"

    options_list = ""
    if result.options:
        for action, description in result.options.items():
            options_list += f"\n- **{action}**: {description}"

    return f"""⚠️ Similar learning detected - validation required

**Similarity**: {result.similarity_score:.0%} match found
**Conflict Type**: {result.conflict_type}
**Recommendation**: {result.recommendation}
**Validation Stage**: {result.validation_stage}
**Performance**: {total_time_ms:.1f}ms

**Rationale**: {result.rationale}

**Similar Learnings Found**:{items_list}

**Available Actions**:{options_list}

**To proceed**:
- Append to existing: `append_to_learning(learning_uuid="<uuid>", new_content="...")`
- Force create: `capture_learning(..., auto_deduplicate=False)`
"""


def _format_append_success_response(result: AppendResult) -> str:
    """Format response for successful append operation.

    Args:
        result: AppendResult with status="success"

    Returns:
        User-friendly confirmation of append
    """
    embedding_status = "✓ Updated" if result.embedding_updated else "⚠ Pending"

    return f"""✅ Content appended successfully

**Learning UUID**: `{result.learning_uuid}`
**File Path**: `{result.file_path}`
**New Section**: {result.new_section_title}
**Append Count**: {result.append_count}
**Timestamp**: {result.timestamp}
**Embedding**: {embedding_status}

The learning has been updated in:
- Markdown file (new section added)
- Neo4j graph (metadata updated)
- Qdrant vectors ({"regenerated" if result.embedding_updated else "queued for update"})
"""


def _format_append_error_response(result: AppendResult) -> str:
    """Format response for failed append operation.

    Args:
        result: AppendResult with status="error"

    Returns:
        User-friendly error message with troubleshooting hints
    """
    return f"""❌ Append operation failed

**Learning UUID**: `{result.learning_uuid}`
**Error**: {result.error}
**Timestamp**: {result.timestamp}

**Troubleshooting**:
- Verify the UUID exists: use `load_context` to check
- Check file permissions on the markdown file
- Ensure Neo4j connection is healthy
"""


# ========================================
# Feature 026: Pattern Context Formatting
# ========================================


def _format_patterns_for_context(patterns: list) -> str:
    """Format stored patterns as markdown context block.

    Args:
        patterns: List of StoredPattern objects to format

    Returns:
        Markdown-formatted pattern context block
    """
    if not patterns:
        return ""

    markdown = "## Relevant Patterns from History\n\n"

    for pattern in patterns:
        # Format success indicator
        success_indicator = ""
        if pattern.success_rate >= 0.8:
            success_indicator = "✅"
        elif pattern.failure_count >= 2:
            success_indicator = "⚠️"

        # Format success rate
        success_pct = f"{pattern.success_rate:.0%}"

        # Format occurrence count
        occurrence_text = f"{pattern.occurrence_count} session{'s' if pattern.occurrence_count != 1 else ''}"

        # Format last occurred date
        last_occurred = (
            pattern.latest_occurrence_date.strftime("%Y-%m-%d")
            if pattern.latest_occurrence_date
            else "Unknown"
        )

        markdown += f"""### Pattern: {pattern.name}
- **Type**: {pattern.pattern_type}
- **Success Rate**: {success_pct} ({pattern.success_count}/{pattern.occurrence_count} sessions) {success_indicator}
- **Last Applied**: {last_occurred}
- **Confidence**: {pattern.confidence:.0%}

"""

        # Add failure warning if applicable
        if pattern.consecutive_failures >= 2:
            markdown += f"⚠️ **Warning**: This approach has failed {pattern.consecutive_failures} times in a row. Consider alternatives.\n\n"

        # Add problem and solution for context
        markdown += f"**Problem**: {pattern.problem[:100]}...\n"
        markdown += f"**Solution**: {pattern.solution[:100]}...\n\n"

    return markdown


def auto_capture_tool(func: Callable) -> Callable:
    """
    Decorator to automatically log MCP tool execution.

    Phase 2C: Auto-capture hook for event log layer.
    Wraps MCP tool execution and logs to event log service if auto-capture enabled.
    Uses fire-and-forget pattern to avoid blocking tool execution.
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        # Import config here to avoid circular import at module level
        from src import config

        if not config.EVENT_LOG_AUTO_CAPTURE:
            return await func(*args, **kwargs)

        tool_name = func.__name__
        start = time.perf_counter()
        success = False
        error_msg = None
        result = None

        try:
            result = await func(*args, **kwargs)
            success = True
            return result
        except Exception as e:
            error_msg = str(e)
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000

            # Fire-and-forget async logging (don't block tool execution)
            try:
                asyncio.create_task(
                    event_log_service.log_event(
                        event_type="tool_call",
                        action=tool_name,
                        actor="mcp_server",
                        context={"duration_ms": duration_ms},
                        result="success" if success else "error",
                        metadata={"error": error_msg} if error_msg else None,
                    )
                )
            except Exception as log_error:
                # Silently fail logging to prevent tool crashes
                logger.debug(
                    f"Auto-capture logging failed for {tool_name}: {log_error}"
                )

    return wrapper


@asynccontextmanager
async def server_lifespan(app) -> AsyncIterator[None]:
    """
    Lifespan context manager - initializes services in SAME event loop as server.

    Critical: The MCP SDK runs this lifespan PER SSE CONNECTION, not once at
    server startup. We guard with _services_initialized so services are only
    initialized on the first connection and NOT torn down when clients disconnect.
    Service cleanup only happens on actual server shutdown (process exit).
    """
    global _services_initialized

    if not _services_initialized:
        logger.info("MCP Server lifespan: Initializing services (first connection)...")

        # Initialize Neo4j (blocking but fast <2s)
        logger.info("Connecting to Neo4j...")
        await neo4j_service.connect()

        # Auto-initialize Neo4j schema (constraints + indexes)
        # All statements are idempotent (IF NOT EXISTS) — safe on every startup
        from src.database.neo4j_schema import init_neo4j_schema

        await init_neo4j_schema(neo4j_service)

        # Initialize Redis with graceful degradation
        logger.info(f"Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}...")
        try:
            await redis_service.connect(
                host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, max_connections=10
            )
            logger.info("✅ Redis connected")
        except Exception as e:
            logger.warning(f"⚠️  Redis unavailable (graceful degradation): {e}")

        # Initialize Event Log Service (episodic memory layer)
        logger.info("Connecting to Event Log Service...")
        try:
            await event_log_service.connect()
            logger.info("✅ Event Log Service connected")
        except Exception as e:
            logger.warning(
                f"⚠️  Event Log Service unavailable (graceful degradation): {e}"
            )

        # Initialize Vector Service (Feature 022: Semantic Vector Layer)
        logger.info("Initializing Vector Service...")
        try:
            vector_service = get_vector_service()
            await vector_service.initialize()
            logger.info("✅ Vector Service initialized")
        except Exception as e:
            logger.warning(f"⚠️  Vector Service unavailable (graceful degradation): {e}")

        # Initialize Context Retrieval Service (Feature 022: Retrieval Cascade)
        logger.info("Initializing Context Retrieval Service...")
        try:
            context_retrieval_service = get_context_retrieval_service()
            await context_retrieval_service.initialize()
            logger.info("✅ Context Retrieval Service initialized")
        except Exception as e:
            logger.warning(
                f"⚠️  Context Retrieval Service unavailable (graceful degradation): {e}"
            )

        await startup_health_check()
        logger.info("✅ All services initialized")

        _services_initialized = True

        # Background tasks start AFTER initialization (non-blocking)
        # Cache warming (runs in background after server is ready)
        if redis_service.is_connected:

            async def warm_cache():
                await asyncio.sleep(2)
                logger.info("Background: Checking cache status...")
                try:
                    stats = await entity_cache.warm_cache(neo4j_service)

                    if stats["skipped"] > 0 and stats["cached"] == 0:
                        logger.info(
                            f"✅ Cache already warm ({stats['skipped']} entities)"
                        )
                    else:
                        logger.info(
                            f"✅ Cache warmed: {stats['cached']} cached, "
                            f"{stats['skipped']} skipped ({stats['duration_ms']:.1f}ms)"
                        )
                except Exception as e:
                    logger.warning(f"Cache warming failed (non-critical): {e}")

            asyncio.create_task(warm_cache())

        # Phase 2D: Start retention scheduler if event log enabled
        from src import config

        if config.EVENT_LOG_ENABLED and event_log_service._initialized:
            logger.info("Starting Event Log retention scheduler...")
            retention_task = asyncio.create_task(
                event_log_service.retention_scheduler_loop()
            )
            event_log_service._retention_task = retention_task
            logger.info("✅ Retention scheduler started")
    else:
        logger.info("MCP Server lifespan: Services already initialized (reusing)")

    try:
        yield  # Server runs (accepting requests for this connection)
    finally:
        # Do NOT tear down services on client disconnect — the MCP SDK runs
        # this lifespan per-connection, so tearing down here would break
        # subsequent connections. Services are cleaned up on process exit.
        logger.info("SSE connection closed (services remain active)")


# Initialize FastMCP server with minimal lifespan
mcp = FastMCP(name="Mnemodrome", host=MCP_HOST, port=MCP_PORT, lifespan=server_lifespan)
# FastMCP exposes no `version=` constructor kwarg; left unset, the MCP
# `serverInfo.version` reported to every client defaults to the installed `mcp`
# framework version (e.g. 1.16.0) instead of our app version. Stamp the
# low-level server so the handshake reports MCP_VERSION.
mcp._mcp_server.version = MCP_VERSION


# ========================================
# Feature 024: Project-Aware Session Init HTTP Endpoint
# ========================================


async def _project_vector_search(project: str, search_type: str, limit: int) -> list:
    """Execute vector search for project context, returns empty list on failure."""
    try:
        vector_service = get_vector_service()
        if not vector_service or not vector_service.is_enabled():
            return []
        return await vector_service.search(
            query_text=project,
            search_type=search_type,
            limit=limit,
            min_score=0.4,  # Lower threshold for project context (broad recall)
        )
    except Exception as e:
        logger.warning(f"Vector search failed for project context ({search_type}): {e}")
        return []


def _format_project_context_response(
    project: str,
    temporal: list,
    sessions: list,
    learnings: list,
) -> str:
    """Format project context into injectable markdown block."""
    parts = [
        "<!-- PROJECT_AWARE_SESSION_INIT -->",
        f"# Session Context: {project}",
        "",
    ]

    # Temporal / graph relationships
    if temporal:
        temporal_md = format_context_markdown(temporal)
        if temporal_md.strip():
            parts.append("## Recent Activity")
            parts.append(temporal_md)

    # Recent sessions from vector search
    if sessions:
        parts.append("## Recent Sessions")
        for r in sessions:
            summary = r.get("summary", "")[:300]
            date = r.get("created_at", "")[:10] if r.get("created_at") else ""
            date_str = f" ({date})" if date else ""
            parts.append(f"- {summary}{date_str}")
        parts.append("")

    # Recent learnings from vector search
    if learnings:
        parts.append("## Recent Learnings")
        for r in learnings:
            title = r.get("title", "Untitled")
            date = r.get("created_at", "")[:10] if r.get("created_at") else ""
            date_str = f" ({date})" if date else ""
            parts.append(f"- **{title}**{date_str}")
        parts.append("")

    # Always close the block
    parts.append("<!-- END_PROJECT_AWARE_SESSION_INIT -->")

    return "\n".join(parts)


@mcp.custom_route("/health", methods=["GET"])
async def handle_health(request: Request) -> Response:
    """HTTP health endpoint for Docker healthchecks and monitoring.

    Returns lightweight JSON status without triggering event logging.
    Checks Neo4j, Redis, and Qdrant connectivity.

    Returns:
        200 application/json: {"status": "healthy", ...}
        503 application/json: {"status": "unhealthy", ...}
    """
    import time as _time

    checks = {}

    # Neo4j
    try:
        await neo4j_service.ping()
        checks["neo4j"] = "connected"
    except Exception:
        checks["neo4j"] = "disconnected"

    # Redis
    try:
        if redis_service.client:
            await redis_service.client.ping()
            checks["redis"] = "connected"
        else:
            checks["redis"] = "not_initialized"
    except Exception:
        checks["redis"] = "disconnected"

    # Event log (SQLite)
    try:
        if event_log_service._initialized:
            checks["event_log"] = "connected"
        else:
            checks["event_log"] = "not_initialized"
    except Exception:
        checks["event_log"] = "disconnected"

    # Qdrant
    try:
        vector_svc = get_vector_service()
        if vector_svc and vector_svc.client:
            import asyncio as _asyncio

            await _asyncio.to_thread(vector_svc.client.get_collections)
            checks["qdrant"] = "connected"
        else:
            checks["qdrant"] = "not_initialized"
    except Exception:
        checks["qdrant"] = "disconnected"

    connected_count = sum(1 for v in checks.values() if v == "connected")
    disconnected_count = sum(1 for v in checks.values() if v == "disconnected")

    if disconnected_count > 0 and connected_count == 0:
        overall = "unhealthy"
    elif disconnected_count > 0:
        overall = "degraded"
    else:
        overall = "healthy"

    status_code = 503 if overall == "unhealthy" else 200

    health_data = {
        "status": overall,
        "services": checks,
        "timestamp": _time.time(),
        "version": MCP_VERSION,
        "llm_extraction_degraded_count": llm_service_singleton.extraction_degraded_count,
        "llm_classification_degraded_count": llm_service_singleton.classification_degraded_count,
    }

    return JSONResponse(health_data, status_code=status_code)


@mcp.custom_route("/project-context", methods=["GET"])
async def handle_project_context(request: Request) -> Response:
    """Feature 024: Project-Aware Session Init context endpoint.

    Returns pre-formatted markdown context for the named project,
    combining temporal graph data and vector search results.
    Called by the load-dynamic-requirements.ts hook on first prompt.

    Query params:
        project (str, required): Project name to query context for
        days (int, optional): Days of history to include (default: PROJECT_CONTEXT_DEFAULT_DAYS)

    Headers:
        Authorization: Bearer <MNEMODROME_HOOK_TOKEN>  (only if token configured)

    Returns:
        200 text/markdown: Formatted project context
        400 application/json: {"error": "project param required"}
        401 application/json: {"error": "Unauthorized"}
        503 application/json: {"error": "Service unavailable"}
    """
    # Optional auth check
    if MNEMODROME_HOOK_TOKEN:
        auth = request.headers.get("Authorization", "")
        if not secrets.compare_digest(auth, f"Bearer {MNEMODROME_HOOK_TOKEN}"):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)

    project = request.query_params.get("project", "").strip()
    if not project:
        return JSONResponse({"error": "project param required"}, status_code=400)

    days = int(request.query_params.get("days", str(PROJECT_CONTEXT_DEFAULT_DAYS)))

    try:
        # Run all queries in parallel for minimum latency
        temporal_task = neo4j_service.temporal_query(
            keyword=project, days=days, limit=PROJECT_CONTEXT_TEMPORAL_LIMIT
        )
        sessions_task = _project_vector_search(
            project, search_type="sessions", limit=PROJECT_CONTEXT_SESSIONS_LIMIT
        )
        learnings_task = _project_vector_search(
            project, search_type="learnings", limit=PROJECT_CONTEXT_LEARNINGS_LIMIT
        )

        temporal_results, sessions_results, learnings_results = await asyncio.gather(
            temporal_task, sessions_task, learnings_task, return_exceptions=True
        )

        markdown = _format_project_context_response(
            project=project,
            temporal=temporal_results
            if not isinstance(temporal_results, Exception)
            else [],
            sessions=sessions_results
            if not isinstance(sessions_results, Exception)
            else [],
            learnings=learnings_results
            if not isinstance(learnings_results, Exception)
            else [],
        )

        return Response(content=markdown, media_type="text/markdown")

    except Exception as e:
        logger.error(f"Project context endpoint failed for '{project}': {e}")
        return JSONResponse({"error": str(e)}, status_code=503)


async def startup_health_check() -> bool:
    """
    Validate connectivity at server startup (NOT before every write).

    Note: Neo4j driver is already initialized by initialize_services(),
    so we just verify it works with a ping.

    Returns:
        True if all checks pass

    Raises:
        Exception if connectivity fails
    """
    logger.info("Running startup health check...")

    # Neo4j connectivity check (driver already connected by initialize_services)
    try:
        await neo4j_service.ping()
        logger.info("✅ Neo4j connectivity verified")
    except Exception as e:
        raise ConnectionError(f"Neo4j unavailable: {e}")

    # File storage accessibility check
    try:
        file_service.validate_path()
        logger.info("✅ File storage accessibility verified")
    except Exception as e:
        raise OSError(f"File storage not accessible: {e}")

    # Ollama connectivity check (non-fatal — embeddings degrade gracefully)
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        import httpx

        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{ollama_host}/api/tags")
            if resp.status_code == 200:
                logger.info(f"✅ Ollama reachable at {ollama_host}")
            else:
                logger.warning(
                    f"⚠️  Ollama returned status {resp.status_code} at {ollama_host}. "
                    "Vector embeddings may not work — semantic search will fall back to fuzzy matching."
                )
    except Exception as e:
        logger.warning(
            f"⚠️  Ollama not reachable at {ollama_host}: {e}\n"
            "   Vector embeddings will be unavailable. Semantic search falls back to fuzzy matching.\n"
            "   Install Ollama (https://ollama.com) or set OLLAMA_HOST in .env."
        )

    # Extraction canary — prove the configured model actually returns ENTITIES,
    # not merely that the LLM endpoint answers. Guards against the silent
    # reasoning-model trap where every capture stores nothing. Self-heals when it
    # can; otherwise hard-fails (refuses to serve) so memories are never lost
    # silently — unless ALLOW_DEGRADED_EXTRACTION=true.
    from src import config

    if config.EXTRACTION_CANARY_ENABLED:
        logger.info(
            "Running extraction canary (configured model must return entities)..."
        )
        canary = await llm_service_singleton.extraction_canary()
        if canary["ok"] and canary["healed"]:
            logger.warning(
                "✅ Extraction canary passed AFTER auto-healing. Mnemodrome adjusted: "
                f"{', '.join(canary['adjustments'])}. Make these permanent in your .env "
                "to silence this warning."
            )
        elif canary["ok"]:
            logger.info(
                f"✅ Extraction canary passed ({canary['entity_count']} entities from probe)"
            )
        elif canary["error"]:
            # Unreachable model = connectivity, not the trap. The Ollama check
            # above already warned; don't block startup on a transient hiccup.
            logger.warning(
                f"⚠️  Extraction canary could not reach the model ({canary['error']}). "
                "Skipping the extraction safety check — captures may fail until the "
                "model is reachable."
            )
        else:
            banner = (
                "\n" + "=" * 72 + "\n"
                "  EXTRACTION CANARY FAILED\n" + "=" * 72 + "\n"
                f"  The configured extraction model '{canary['model']}' returned ZERO\n"
                "  entities on a known-good sentence, even after auto-applying no-think\n"
                "  and raising the token budget. This is the reasoning-model trap: every\n"
                "  capture would silently store nothing.\n\n"
                "  FIX: set CHAT_MODEL (or EXTRACTION_MODEL) to a NON-reasoning instruct\n"
                "  model. Recommended: gemma3:4b-it-qat  (or gemma4:e4b-it-qat on newer Ollama)\n"
                "      ollama pull gemma3:4b-it-qat\n"
                "      # in .env:  CHAT_MODEL=gemma3:4b-it-qat\n\n"
                "  To start anyway with degraded extraction (NOT recommended), set\n"
                "  ALLOW_DEGRADED_EXTRACTION=true\n" + "=" * 72
            )
            if config.ALLOW_DEGRADED_EXTRACTION:
                logger.warning(
                    banner
                    + "\n  ALLOW_DEGRADED_EXTRACTION=true — starting anyway (DEGRADED)."
                )
            else:
                logger.error(banner)
                raise RuntimeError(
                    f"Extraction canary failed: model '{canary['model']}' returns no "
                    "entities even after self-heal. Set a non-reasoning model, or "
                    "ALLOW_DEGRADED_EXTRACTION=true to override."
                )

    logger.info("✅ Startup health check passed")
    return True


@mcp.tool()
@auto_capture_tool
async def echo(message: str) -> str:
    """Echo input message for transport verification."""
    return f"ECHO: {message}"


@mcp.tool()
@auto_capture_tool
async def verify_storage() -> str:
    """Test write to both storage layers using exception-based verification."""
    # Test Neo4j write - exception-based (industry standard)
    test_uuid = str(uuid4())
    try:
        await neo4j_service.create_entity(
            {
                "uuid": test_uuid,
                "name": "test_entity",
                "summary": "Storage verification test",
            }
        )
        logger.info(f"✅ Neo4j write succeeded: {test_uuid}")
    except Exception as e:
        raise Exception(f"Neo4j write failed: {e}")

    # Test markdown write - exception-based
    test_filename = f"test_{test_uuid}.md"
    try:
        test_path = file_service.write_file(
            test_filename, "# Test\nStorage verification"
        )
        logger.info(f"✅ Markdown write succeeded: {test_path}")
    except Exception as e:
        # Clean up Neo4j entity
        await neo4j_service.delete_entity(test_uuid)
        raise Exception(f"Markdown write failed: {e}")

    # Cleanup - also exception-based
    try:
        await neo4j_service.delete_entity(test_uuid)
        file_service.delete_file(file_service.storage_path / test_filename)
        logger.info("✅ Test cleanup completed")
    except Exception as e:
        logger.warning(f"Cleanup warning: {e}")

    return "✅ Storage layers verified (Neo4j + markdown) - no exceptions raised"


@mcp.tool()
@auto_capture_tool
async def health() -> str:
    """Health check reporting all 4 memory type storage paths."""
    import time

    # Check Neo4j connectivity
    neo4j_status = "disconnected"
    try:
        await neo4j_service.ping()
        neo4j_status = "connected"
    except Exception as e:
        logger.warning(f"Neo4j health check failed: {e}")

    # Validate all memory type storage paths (Feature 010)
    storage_paths = {}
    for memory_type in MemoryType:
        try:
            file_service.validate_path(memory_type)
            path = file_service._get_target_path(memory_type)
            storage_paths[memory_type.value] = str(path)
        except Exception as e:
            storage_paths[memory_type.value] = f"error: {e}"

    # Build health response
    health_data = {
        "status": "healthy" if neo4j_status == "connected" else "degraded",
        "neo4j": neo4j_status,
        "storage_paths": storage_paths,
        "timestamp": time.time(),
        "version": MCP_VERSION,
        "llm_extraction_degraded_count": llm_service_singleton.extraction_degraded_count,
        "llm_classification_degraded_count": llm_service_singleton.classification_degraded_count,
    }

    import json

    return json.dumps(health_data, indent=2)


@mcp.tool()
@auto_capture_tool
async def discover_schema() -> str:
    """Query production Neo4j to discover actual schema (source of truth)."""
    logger.info("Discovering production Neo4j schema...")

    # D7: scope to native edges only — never read foreign null-type edges.
    rel_query = """
    MATCH ()-[r:RELATES_TO]->()
    WHERE r.type IS NOT NULL
    RETURN r LIMIT 5
    """
    sample_rels = await neo4j_service.query(rel_query)

    # Check for entity properties
    entity_query = """
    MATCH (e)
    RETURN e LIMIT 5
    """
    sample_entities = await neo4j_service.query(entity_query)

    # Format schema report
    report = "# Production Neo4j Schema Discovery\n\n"

    report += "## Sample Relationships\n"
    if sample_rels:
        for i, record in enumerate(sample_rels[:3], 1):
            rel = record.get("r", {})
            report += f"\n### Relationship {i}\n"
            report += f"- Type: {type(rel).__name__}\n"
            if hasattr(rel, "type"):
                report += f"- Rel Type: {rel.type}\n"
            if isinstance(rel, dict):
                for key, value in rel.items():
                    report += f"- {key}: {type(value).__name__}\n"
    else:
        report += "No relationships found\n"

    report += "\n## Sample Entities\n"
    if sample_entities:
        for i, record in enumerate(sample_entities[:3], 1):
            entity = record.get("e", {})
            report += f"\n### Entity {i}\n"
            if isinstance(entity, dict):
                for key, value in entity.items():
                    report += f"- {key}: {type(value).__name__}\n"
    else:
        report += "No entities found\n"

    logger.info("Schema discovery complete")
    return report


@mcp.tool()
@auto_capture_tool
async def load_context(prompt: str = "", days: int = 7, limit: int = 10) -> str:
    """Load recent context using temporal-first strategy with entity-aware filtering. See /mnemodrome skill for guidance."""
    start_time = time.time()
    logger.info(
        f"Loading context (prompt={'provided' if prompt else 'none'}, days={days}, limit={limit})"
    )

    # Phase 4.5: Extract entities from prompt using GraphFactory (entity-aware)
    entity_uuids = []
    if prompt:
        try:
            # Use GraphFactory to extract entities from prompt
            extraction = await graph_factory._extract_with_llm(prompt)

            if extraction["entities"]:
                logger.debug(
                    f"Extracted {len(extraction['entities'])} entities from prompt"
                )

                # Classify entities
                classification_tasks = [
                    classification_service.classify_entity(
                        name=entity["name"],
                        context=prompt,
                        role=entity.get("role", "entity"),
                    )
                    for entity in extraction["entities"]
                ]
                entity_types = await asyncio.gather(*classification_tasks)

                # Resolve entities to UUIDs (deduplication-aware)
                resolution_tasks = [
                    entity_factory.create_or_resolve(
                        name=entity["name"],
                        entity_type=entity_type,
                        summary=prompt[:200],
                    )
                    for entity, entity_type in zip(
                        extraction["entities"], entity_types, strict=False
                    )
                ]
                entity_uuids = await asyncio.gather(*resolution_tasks)

                logger.debug(
                    f"Resolved entities to {len(entity_uuids)} UUIDs for context query"
                )

        except Exception as e:
            logger.warning(
                f"Entity extraction failed, falling back to keyword search: {e}"
            )

    # Extract keyword for fallback (always, not just when entity extraction fails)
    keyword = None
    if prompt:
        keyword = extract_keywords(prompt)
        if keyword:
            logger.debug(f"Extracted keyword for fallback: {keyword}")

    # Query temporal memory with entity UUIDs (preferred) or keyword filter (fallback)
    try:
        results = []

        if entity_uuids:
            # Entity-aware query (Phase 4.5)
            results = await neo4j_service.temporal_query_by_entities(
                entity_uuids=entity_uuids, days=days, limit=limit
            )

        # Fallback to keyword query if entity-aware returned empty
        # This handles cases where entity resolution creates new nodes with no relationships
        if not results:
            if entity_uuids:
                logger.info(
                    f"Entity-aware query returned 0 results for {len(entity_uuids)} UUIDs, "
                    f"falling back to keyword query"
                )
            results = await neo4j_service.temporal_query(
                keyword=keyword, days=days, limit=limit
            )

        # Feature 034: Apply activation scoring to re-rank results
        results = await _apply_activation_scoring(results)

        # Format results as markdown
        context_markdown = format_context_markdown(results)

        # Feature 026: Inject relevant patterns into context (fire-and-forget query)
        patterns_markdown = ""
        if PATTERN_RECOGNITION_ENABLED:
            try:
                query_service = PatternQueryService()

                # Query for high-success patterns relevant to the prompt
                # Using min_occurrences=2 as per spec (proactive surfacing threshold)
                patterns = await query_service.get_high_success_patterns(
                    min_success_rate=0.70,  # Include patterns with ≥70% success
                    min_occurrences=2,  # Only proven patterns
                    limit=5,  # Top 5 patterns max
                )

                if patterns:
                    # Format patterns as markdown
                    patterns_markdown = _format_patterns_for_context(patterns)
                    logger.debug(
                        f"Injected {len(patterns)} relevant patterns into context"
                    )

            except Exception as e:
                logger.warning(f"Pattern context injection failed (non-fatal): {e}")

        # Append pattern context if any patterns were found
        if patterns_markdown:
            context_markdown = (
                context_markdown + "\n\n" + patterns_markdown
                if context_markdown
                else patterns_markdown
            )

        query_time = (time.time() - start_time) * 1000  # Convert to ms
        logger.info(
            f"Context loaded in {query_time:.1f}ms ({len(results)} relationships)"
        )

        return context_markdown

    except Exception as e:
        logger.error(f"Failed to load context: {e}")
        raise Exception(f"Context loading failed: {e}")


@mcp.tool()
@auto_capture_tool
async def capture_learning(
    title: str, content: str, category: str = "general", auto_deduplicate: bool = True
) -> str:
    """Capture structured learning in dual memory layers with deduplication. See /mnemodrome skill for guidance."""
    start_time = time.time()
    logger.info(
        f"Capturing learning: {title} (category: {category}, dedup={auto_deduplicate})"
    )

    # Feature 023: Pre-capture validation
    if auto_deduplicate:
        try:
            from src.services.deduplication_service import get_deduplication_service

            dedup_service = get_deduplication_service(
                vector_service=get_vector_service(),
                neo4j_service=neo4j_service,
                file_service=file_service,
            )

            validation_result = await dedup_service.validate_learning(
                title=title, content=content
            )
            validation_time = (time.time() - start_time) * 1000

            # Handle auto_skipped - return early, do not capture
            if validation_result.status == "auto_skipped":
                return _format_auto_skip_response(validation_result, validation_time)

            # Handle validation_required - return options to user
            if validation_result.status == "validation_required":
                return _format_validation_required_response(
                    validation_result, validation_time
                )

            # Status is "proceed" - continue with capture
            logger.debug(
                f"Deduplication passed in {validation_result.validation_time_ms:.1f}ms"
            )

        except Exception as e:
            # Graceful degradation: log error and proceed with capture
            logger.warning(f"Deduplication check failed (proceeding): {e}")
    else:
        logger.info("Deduplication bypassed (auto_deduplicate=False)")

    # Generate UUID for learning node
    learning_uuid = str(uuid4())

    # Create file path first (needed for Neo4j)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    safe_title = title.lower().replace(" ", "-").replace("/", "-")[:50]
    filename = f"learning-{safe_title}-{timestamp}.md"
    file_path = file_service._get_target_path(MemoryType.LEARNING) / filename

    # Prepare markdown content (non-async)
    markdown_content = f"""# Learning: {title}

**Category**: {category}
**Captured**: {time.strftime("%Y-%m-%d %H:%M:%S")}
**UUID**: `{learning_uuid}`

---

{content}

---

*Captured via Mnemodrome MCP capture_learning tool*
"""

    # Optimization 1: Parallelize Neo4j and file storage writes (independent I/O operations)
    async def create_neo4j_learning():
        """Create learning node in Neo4j (episodic memory)"""
        cypher = """
        CREATE (l:Learning {
            uuid: $uuid,
            title: $title,
            content: $content,
            category: $category,
            file_path: $file_path,
            append_count: 0,
            created_at: datetime(),
            last_accessed: datetime()
        })
        RETURN l.uuid as uuid, l.title as title, l.category as category
        """

        result = await neo4j_service.query(
            cypher,
            {
                "uuid": learning_uuid,
                "title": title,
                "content": content,
                "category": category,
                "file_path": str(file_path),
            },
        )

        if not result:
            raise Exception("Failed to create learning node in Neo4j")

        logger.debug(f"Learning node created: {learning_uuid}")
        return result[0]

    async def write_markdown_file():
        """Write markdown file in markdown vault"""
        written_path = file_service.write_file(
            filename, markdown_content, MemoryType.LEARNING
        )
        logger.debug(f"Learning file written: {file_path}")
        return written_path

    # Execute both I/O operations in parallel
    try:
        neo4j_result, file_result = await asyncio.gather(
            create_neo4j_learning(), write_markdown_file(), return_exceptions=False
        )
    except Exception as e:
        logger.error(f"Learning creation failed: {e}")
        # Attempt cleanup (non-critical if it fails)
        try:
            await neo4j_service.query(
                "MATCH (l:Learning {uuid: $uuid}) DELETE l", {"uuid": learning_uuid}
            )
        except Exception as cleanup_error:
            logger.warning(f"Cleanup failed: {cleanup_error}")
        raise Exception(f"Failed to create learning: {e}")

    # Optimization 1B: Extract entities and store vector in parallel (independent enrichment operations)
    # Note: We can't parallelize entity extraction with graph linking since linking depends on extraction results
    entities_count = 0
    relationships_count = 0

    async def extract_and_link_entities():
        """Extract entities and link to learning node"""
        try:
            logger.debug("Extracting entities from learning content...")
            result = await graph_factory.build_from_extraction(content)
            extracted_entities = result.get("entities_created", 0)
            extracted_relationships = result.get("relationships_created", 0)

            # Link Learning node to extracted entities
            if extracted_entities > 0:
                # Query for entities created in the last few seconds (from our extraction)
                link_query = """
                MATCH (l:Learning {uuid: $learning_uuid})
                MATCH (e:Entity)
                WHERE e.created_at > datetime() - duration('PT5S')
                MERGE (l)-[:MENTIONS]->(e)
                RETURN count(e) as linked_count
                """
                link_result = await neo4j_service.query(
                    link_query, {"learning_uuid": learning_uuid}
                )
                linked_count = link_result[0]["linked_count"] if link_result else 0
                logger.debug(f"Linked learning to {linked_count} entities")

            logger.debug(
                f"Entity extraction: {extracted_entities} entities, {extracted_relationships} relationships"
            )
            return extracted_entities, extracted_relationships

        except Exception as e:
            logger.warning(f"Entity extraction failed (continuing): {e}")
            return 0, 0

    async def store_vector_embedding():
        """Store vector embedding for semantic search"""
        try:
            vector_service = get_vector_service()
            await vector_service.store_document(
                content=f"{title}\n\n{content}",
                metadata={
                    "type": "learning",
                    "uuid": learning_uuid,
                    "title": title,
                    "category": category,
                    "file_path": str(file_path),
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    # Note: Entity counts omitted here since this runs in parallel with extraction
                    # Metadata is for debugging only - semantic search uses content itself
                },
            )
            logger.debug(f"Vector embedding stored for learning {learning_uuid}")
        except Exception as e:
            logger.warning(f"Vector embedding failed (continuing): {e}")

    # Fire-and-forget: enrichment runs in background, capture returns immediately
    async def _background_enrichment():
        """Background task: entity extraction + vector embedding for learning."""
        enrichment_start = time.time()
        try:
            await asyncio.gather(
                extract_and_link_entities(),
                store_vector_embedding(),
                return_exceptions=False,
            )
            elapsed = (time.time() - enrichment_start) * 1000
            logger.info(
                f"Background enrichment complete for learning '{title}' in {elapsed:.0f}ms"
            )
        except Exception as e:
            logger.warning(
                f"Background enrichment failed for learning {learning_uuid}: {e}"
            )

    _fire_and_forget(
        _background_enrichment(), description=f"learning enrichment: {title[:50]}"
    )

    # Create response (returns immediately - enrichment continues in background)
    capture_time = (time.time() - start_time) * 1000
    logger.info(f"Learning captured in {capture_time:.1f}ms: {title}")

    response = f"""✅ Learning captured successfully

**Title**: {title}
**Category**: {category}
**Neo4j UUID**: `{learning_uuid}`
**File Path**: `{file_path}`
**Performance**: {capture_time:.1f}ms

The learning is now:
- Stored in Neo4j temporal graph for relationship discovery
- Saved as markdown in Obsidian vault for human review
- Available via load_context for future sessions
- Entity extraction and vector indexing queued (background)
"""

    return response


@mcp.tool()
@auto_capture_tool
async def append_to_learning(
    learning_uuid: str, new_content: str, new_section_title: str | None = None
) -> str:
    """Append content to an existing learning, updating Neo4j and Qdrant. See /mnemodrome skill for guidance."""
    logger.info(f"Appending to learning: {learning_uuid}")

    try:
        from src.services.deduplication_service import get_deduplication_service

        dedup_service = get_deduplication_service(
            vector_service=get_vector_service(),
            neo4j_service=neo4j_service,
            file_service=file_service,
        )

        result = await dedup_service.append_to_learning(
            learning_uuid=learning_uuid,
            new_content=new_content,
            new_section_title=new_section_title,
        )

        if result.status == "success":
            return _format_append_success_response(result)
        else:
            return _format_append_error_response(result)

    except Exception as e:
        logger.error(f"Append operation failed: {e}")
        raise Exception(f"Failed to append to learning: {e}")


@mcp.tool()
@auto_capture_tool
async def capture_session(
    conversation_summary: str, key_insights: list[str] | None = None
) -> str:
    """Capture end-of-session knowledge with graph-linked entity connections. See /mnemodrome skill for guidance."""
    start_time = time.time()
    logger.info("Synthesizing session knowledge...")

    # Generate UUID for session node
    session_uuid = str(uuid4())

    # Format insights as markdown list
    insights_text = ""
    if key_insights:
        insights_text = "\n".join(f"- {insight}" for insight in key_insights)

    # Create session node in Neo4j
    try:
        cypher_session = """
        CREATE (s:Session {
            uuid: $uuid,
            summary: $summary,
            insights: $insights,
            created_at: datetime(),
            session_date: date()
        })
        RETURN s.uuid as uuid, s.summary as summary
        """

        result = await neo4j_service.query(
            cypher_session,
            {
                "uuid": session_uuid,
                "summary": conversation_summary,
                "insights": insights_text,
            },
        )

        if not result:
            raise Exception("Failed to create session node in Neo4j")

        logger.debug(f"Session node created: {session_uuid}")

    except Exception as e:
        logger.error(f"Neo4j session creation failed: {e}")
        raise Exception(f"Failed to create session in graph: {e}")

    # Extract entities from summary using GraphFactory (entity-aware synthesis)
    entity_connections = []
    try:
        extraction = await graph_factory._extract_with_llm(conversation_summary)

        if extraction["entities"]:
            logger.debug(
                f"Extracted {len(extraction['entities'])} entities from session summary"
            )

            # Classify and resolve entities
            classification_tasks = [
                classification_service.classify_entity(
                    name=entity["name"],
                    context=conversation_summary,
                    role=entity.get("role", "entity"),
                )
                for entity in extraction["entities"]
            ]
            entity_types = await asyncio.gather(*classification_tasks)

            # Resolve entities to UUIDs
            resolution_tasks = [
                entity_factory.create_or_resolve(
                    name=entity["name"],
                    entity_type=entity_type,
                    summary=conversation_summary[:200],
                )
                for entity, entity_type in zip(
                    extraction["entities"], entity_types, strict=False
                )
            ]
            entity_uuids = await asyncio.gather(*resolution_tasks)

            # Link session to entities (DISCUSSED_IN relationships)
            for entity, entity_uuid in zip(
                extraction["entities"], entity_uuids, strict=False
            ):
                try:
                    cypher_link = """
                    MATCH (s:Session {uuid: $session_uuid})
                    MATCH (e {uuid: $entity_uuid})
                    CREATE (e)-[r:DISCUSSED_IN {
                        session_date: date(),
                        created_at: datetime()
                    }]->(s)
                    RETURN e.name as entity_name, e.entity_type as entity_type
                    """

                    link_result = await neo4j_service.query(
                        cypher_link,
                        {"session_uuid": session_uuid, "entity_uuid": entity_uuid},
                    )

                    if link_result:
                        entity_connections.append(
                            f"{link_result[0]['entity_name']} ({link_result[0]['entity_type']})"
                        )

                except Exception as e:
                    logger.warning(f"Failed to link entity {entity['name']}: {e}")

            logger.debug(f"Linked {len(entity_connections)} entities to session")

    except Exception as e:
        logger.warning(f"Entity extraction for session failed: {e}")

    # Create markdown file in markdown vault
    try:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"session-synthesis-{timestamp}.md"

        markdown_content = f"""# Session Synthesis

**Date**: {time.strftime("%Y-%m-%d %H:%M:%S")}
**UUID**: `{session_uuid}`

## Summary

{conversation_summary}
"""

        if key_insights:
            markdown_content += f"""
## Key Insights

{insights_text}
"""

        if entity_connections:
            markdown_content += f"""
## Entities Discussed ({len(entity_connections)})

{chr(10).join(f"- {entity}" for entity in entity_connections)}
"""

        markdown_content += """
---

*Captured via Mnemodrome MCP capture_session tool*
"""

        file_path = file_service.write_file(
            filename, markdown_content, MemoryType.SESSION
        )
        logger.debug(f"Session synthesis file written: {file_path}")

    except Exception as e:
        # Cleanup session node if file write fails
        await neo4j_service.query(
            "MATCH (s:Session {uuid: $uuid}) DETACH DELETE s", {"uuid": session_uuid}
        )
        logger.error(f"Session file write failed, cleaned up Neo4j node: {e}")
        raise Exception(f"Failed to write session file: {e}")

    # Feature 022b: Embed session in Qdrant for semantic search (fire-and-forget)
    async def _background_session_embedding():
        try:
            vs = get_vector_service()
            if vs and vs.is_enabled():
                await vs.store_session_embedding(
                    summary=conversation_summary,
                    session_uuid=session_uuid,
                    key_insights=key_insights,
                    metadata={"file_path": str(file_path)},
                )
                logger.debug(
                    f"Background: Session embedded in Qdrant: {session_uuid[:8]}..."
                )
        except Exception as e:
            logger.warning(
                f"Background session vector embedding failed (non-fatal): {e}"
            )

    _fire_and_forget(
        _background_session_embedding(),
        description=f"session embedding: {session_uuid[:8]}",
    )

    # Feature 026: Async pattern extraction on session capture (fire-and-forget)
    async def _background_pattern_extraction():
        """Extract patterns from session asynchronously with timeout."""
        if not PATTERN_RECOGNITION_ENABLED:
            logger.debug("Pattern recognition disabled, skipping extraction")
            return

        try:
            extraction_start = time.time()

            # Prepare session data for pattern extraction
            session_data = {
                "uuid": session_uuid,
                "conversation_summary": conversation_summary,
                "key_insights": key_insights or [],
            }

            # Extract patterns using PatternExtractionService
            extraction_service = PatternExtractionService()
            extracted_patterns = await asyncio.wait_for(
                extraction_service.extract_patterns_from_sessions([session_data]),
                timeout=30.0,  # 30s timeout as per spec
            )

            if not extracted_patterns:
                logger.debug(
                    f"No patterns extracted from session {session_uuid[:8]}..."
                )
                return

            # Store patterns using PatternStorageService
            storage_service = PatternStorageService()
            patterns_stored = []
            patterns_skipped = 0

            for pattern in extracted_patterns:
                try:
                    stored = await storage_service.store_pattern(
                        pattern=pattern, session_uuid=session_uuid
                    )
                    if stored:
                        patterns_stored.append(stored.pattern_uuid)
                    else:
                        patterns_skipped += 1
                except Exception as e:
                    logger.warning(f"Failed to store pattern: {e}")
                    patterns_skipped += 1

            extraction_duration_ms = (time.time() - extraction_start) * 1000

            # Log telemetry to event log
            try:
                await event_log_service.log_event(
                    event_type="pattern_extraction_complete",
                    action="extract_and_store",
                    actor="pattern_extraction_service",
                    target=f"session:{session_uuid[:8]}",
                    result="success",
                    metadata={
                        "patterns_extracted": len(extracted_patterns),
                        "patterns_stored": len(patterns_stored),
                        "patterns_skipped": patterns_skipped,
                        "duration_ms": round(extraction_duration_ms, 1),
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to log pattern extraction event: {e}")

            logger.info(
                f"Background pattern extraction complete: "
                f"{len(patterns_stored)} stored, {patterns_skipped} skipped in {extraction_duration_ms:.1f}ms"
            )

        except TimeoutError:
            logger.warning(
                f"Pattern extraction timed out after 30s for session {session_uuid[:8]}... "
                f"(non-fatal, session capture already completed)"
            )
        except Exception as e:
            logger.warning(f"Background pattern extraction failed (non-fatal): {e}")

    _fire_and_forget(
        _background_pattern_extraction(),
        description=f"pattern extraction: {session_uuid[:8]}",
    )

    # Create response
    synthesis_time = (time.time() - start_time) * 1000
    logger.info(f"Session synthesized in {synthesis_time:.1f}ms")

    response = f"""✅ Session knowledge synthesized

**Session UUID**: `{session_uuid}`
**File Path**: `{file_path}`
**Performance**: {synthesis_time:.1f}ms

## Session Summary
{conversation_summary}
"""

    if key_insights:
        response += f"""
## Key Insights ({len(key_insights)})
{insights_text}
"""

    if entity_connections:
        response += f"""
## Entity Connections ({len(entity_connections)})
{chr(10).join(f"- {entity}" for entity in entity_connections)}
"""

    response += """
The session synthesis is now:
- Stored in Neo4j with DISCUSSED_IN relationships to key entities
- Saved as markdown in Obsidian vault for human review
- Available via load_context for future sessions
- Linked to all entities mentioned for graph traversal
"""

    return response


@mcp.tool()
@auto_capture_tool
async def capture_fact(
    entity1: str,
    relation: str,
    entity2: str,
    context: str = "",
    entity1_type: str | None = None,
    entity2_type: str | None = None,
) -> str:
    """Capture entity relationship in dual-memory graph with Factory Pattern resolution. See /mnemodrome skill for guidance."""
    start_time = time.time()
    logger.info(f"Capturing fact: {entity1} -{relation}-> {entity2}")

    # Snapshot classification-degraded counter before any LLM classification
    # calls so we can detect if this specific capture_fact call triggered a
    # classification fallback.
    _pre_classify_degraded = llm_service_singleton.classification_degraded_count

    # Phase 4.5: Classify entities if types not provided
    # T057: Batch classify both entities in single LLM call for performance
    if not entity1_type or not entity2_type:
        entities_to_classify = []
        entity_indices = {}

        if not entity1_type:
            entity_indices["entity1"] = len(entities_to_classify)
            entities_to_classify.append(
                {"name": entity1, "role": "source", "context": context}
            )

        if not entity2_type:
            entity_indices["entity2"] = len(entities_to_classify)
            entities_to_classify.append(
                {"name": entity2, "role": "target", "context": context}
            )

        # Batch classify all missing types
        classified_types = await classification_service.classify_batch(
            entities_to_classify, context=context
        )

        # Assign types back
        if "entity1" in entity_indices:
            entity1_type = classified_types[entity_indices["entity1"]]
            logger.debug(f"Entity1 classified: {entity1} → {entity1_type}")

        if "entity2" in entity_indices:
            entity2_type = classified_types[entity_indices["entity2"]]
            logger.debug(f"Entity2 classified: {entity2} → {entity2_type}")

    # Phase 4.5: Resolve entities to UUIDs (creates or retrieves existing)
    entity1_uuid = await entity_factory.create_or_resolve(
        name=entity1,
        entity_type=entity1_type,
        summary=context or f"{entity1_type}: {entity1}",
    )
    logger.debug(f"Entity1 resolved: {entity1} → {entity1_uuid}")

    entity2_uuid = await entity_factory.create_or_resolve(
        name=entity2,
        entity_type=entity2_type,
        summary=context or f"{entity2_type}: {entity2}",
    )
    logger.debug(f"Entity2 resolved: {entity2} → {entity2_uuid}")

    # Phase 4.5: Create relationship using UUIDs (with dedup + auto-supersede)
    relationship_data = await relationship_factory.create(
        source_uuid=entity1_uuid,
        relation_type=relation,
        target_uuid=entity2_uuid,
        context=context,
    )

    dedup_status = relationship_data.pop("_dedup", None)
    superseded_targets = relationship_data.pop("_superseded_targets", [])

    # If exact duplicate, skip markdown/embedding and return early
    if dedup_status == "skipped":
        write_time = (time.time() - start_time) * 1000
        logger.info(f"Duplicate fact skipped in {write_time:.1f}ms")
        return f"""⏭️ Duplicate fact — already exists

**Relationship**:
{entity1} ({entity1_type}) -{relation}-> {entity2} ({entity2_type})

**Status**: Identical relationship already active in Neo4j. No new markdown or embedding created.

**Performance**: {write_time:.1f}ms
"""

    logger.debug(
        f"Relationship created: {relationship_data['source_name']} "
        f"-{relationship_data['relation_type']}-> {relationship_data['target_name']}"
    )

    # Prepare Neo4j data for sync service (relationship already created, just for logging)
    neo4j_data = {
        "operation": "relationship_created_via_factory",
        "data": {
            "entity1_name": entity1,
            "entity1_type": entity1_type,
            "entity1_uuid": entity1_uuid,
            "relation_type": relation,
            "entity2_name": entity2,
            "entity2_type": entity2_type,
            "entity2_uuid": entity2_uuid,
            "context": context,
        },
    }

    # Create markdown content (enhanced with entity types)
    markdown_content = f"""# Fact: {entity1} {relation} {entity2}

**Captured**: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Relationship (Phase 4.5 - Factory Pattern)
- **Source**: {entity1} ({entity1_type})
  - UUID: `{entity1_uuid}`
- **Relation**: {relation}
- **Target**: {entity2} ({entity2_type})
  - UUID: `{entity2_uuid}`

## Context
{context or "No additional context provided"}

## Technical Details
- Entity resolution via EntityFactory (prevents duplicates)
- Automatic classification via ClassificationService
- Temporal relationship via RelationshipFactory

---
*Captured via Mnemodrome MCP capture_fact tool (Amendment 7)*
"""

    # Generate filename with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"fact-{entity1.lower().replace(' ', '-')}-{timestamp}.md"

    # Write markdown file (Neo4j already written via factories)
    try:
        filepath = file_service.write_file(filename, markdown_content, MemoryType.FACT)
        logger.info(f"✅ Markdown documentation created: {filepath}")
        markdown_success = True
    except Exception as e:
        logger.warning(f"Markdown write failed (will queue): {e}")
        markdown_success = False
        # TODO: Queue for retry via sync_service

    # Feature 022b / D1 (FR-011): Embed fact in Qdrant using the graph uuid as point id.
    # The graph uuid was app-generated at capture boundary — no fallback uuid4() needed.
    async def _background_fact_embedding():
        try:
            vs = get_vector_service()
            if vs and vs.is_enabled():
                # D1: use the relationship_uuid returned from the graph (not a fallback)
                fact_uuid = relationship_data.get("uuid") or relationship_data.get(
                    "relationship_uuid"
                )
                if not fact_uuid:
                    logger.warning(
                        "D1: No uuid returned from RelationshipFactory.create(); skipping vector embed"
                    )
                    return
                await vs.store_fact_embedding(
                    entity1=entity1,
                    relation=relation,
                    entity2=entity2,
                    context=context,
                    fact_uuid=fact_uuid,
                    metadata={
                        "entity1_type": entity1_type,
                        "entity2_type": entity2_type,
                        "entity1_uuid": entity1_uuid,
                        "entity2_uuid": entity2_uuid,
                    },
                )
                logger.debug(
                    f"Background: Fact embedded in Qdrant: {entity1} -{relation}-> {entity2}"
                )
        except Exception as e:
            logger.warning(f"Background fact vector embedding failed (non-fatal): {e}")

    _fire_and_forget(
        _background_fact_embedding(),
        description=f"fact embedding: {entity1}-{relation}->{entity2}",
    )

    write_time = (time.time() - start_time) * 1000
    logger.info(f"Fact captured in {write_time:.1f}ms")

    # Format response message
    supersede_note = ""
    if superseded_targets:
        supersede_note = f"\n**Auto-superseded**: {', '.join(superseded_targets)} (old targets invalidated)\n"

    # Degraded-mode warning: emit in payload if any LLM call fell back.
    _degraded_note = ""
    if llm_service_singleton.classification_degraded_count > _pre_classify_degraded:
        _degraded_note = (
            "\n⚠️ **WARNING: Degraded Classification**\n"
            "Entity type classification ran in degraded mode — the LLM returned "
            "empty or unparseable output. Entity types above may be inaccurate. "
            "Search logs for `DEGRADED classification quality` to diagnose. "
            "Fix: use a non-reasoning model, set LLM_DISABLE_THINKING=true, "
            "or raise LLM_MAX_TOKENS.\n"
        )

    response = f"""✅ Fact captured successfully

**Entities** (Phase 4.5 - UUID Resolution):
- {entity1} ({entity1_type})
  → UUID: {entity1_uuid[:8]}...
- {entity2} ({entity2_type})
  → UUID: {entity2_uuid[:8]}...

**Relationship**:
{entity1} -{relation}-> {entity2}
{supersede_note}
**Storage**:
- Neo4j: ✓ (via RelationshipFactory)
- Markdown: {"✓" if markdown_success else "queued for retry"}

**Performance**: {write_time:.1f}ms
{_degraded_note}"""

    return response


@mcp.tool()
@auto_capture_tool
async def capture_context(context: str) -> str:
    """Extract and capture multi-entity graph from rich natural language context. See /mnemodrome skill for guidance."""
    start_time = time.time()
    logger.info(f"Capturing graph from context: {context[:100]}...")

    # Snapshot the degraded counter so we can tell the caller if extraction
    # silently failed (returned no entities) rather than report a hollow success.
    _pre_degraded = llm_service_singleton.extraction_degraded_count

    # Use GraphFactory to orchestrate extraction and creation
    try:
        result = await graph_factory.build_from_extraction(context)

        extraction_time = (time.time() - start_time) * 1000
        logger.info(
            f"Graph captured in {extraction_time:.1f}ms: "
            f"{result['entities_created']} entities, "
            f"{result['relationships_created']} relationships"
        )

        # Write markdown documentation
        markdown_content = f"""# Graph Capture: Multi-Entity Context

**Captured**: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Context
{context}

## Extracted Graph (Phase 4.5 - GraphFactory)

### Entities Created: {result["entities_created"]}
"""

        # List entities with types
        for entity_name, entity_uuid in result["entity_uuids"].items():
            # Find entity type from the list
            entity_idx = list(result["entity_uuids"].keys()).index(entity_name)
            entity_type = (
                result["entity_types"][entity_idx]
                if entity_idx < len(result["entity_types"])
                else "Entity"
            )
            markdown_content += (
                f"- **{entity_name}** ({entity_type})\n  - UUID: `{entity_uuid}`\n"
            )

        markdown_content += (
            f"\n### Relationships Created: {result['relationships_created']}\n"
        )

        # List relationships
        for rel in result["relationships"]:
            markdown_content += f"- {rel['source_name']} -{rel['relation_type']}-> {rel['target_name']}\n"

        markdown_content += """
## Technical Details
- Multi-entity extraction via GraphFactory (Phase 4.5)
- Parallel entity resolution (asyncio.gather)
- Entities-first transaction pattern
- Automatic entity typing via ClassificationService

---
*Captured via Mnemodrome MCP capture_context tool (Amendment 7B)*
"""

        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"graph-{timestamp}.md"

        # Write markdown file
        try:
            filepath = file_service.write_file(
                filename, markdown_content, MemoryType.GRAPH
            )
            logger.info(f"✅ Graph documentation created: {filepath}")
            markdown_success = True
        except Exception as e:
            logger.warning(f"Markdown write failed: {e}")
            markdown_success = False

        # Feature 022b: Embed each relationship/fact in Qdrant (fire-and-forget)
        relationships_to_embed = list(
            result["relationships"]
        )  # snapshot for background

        async def _background_context_embedding():
            try:
                vs = get_vector_service()
                if vs and vs.is_enabled():
                    embedded_count = 0
                    for rel in relationships_to_embed:
                        try:
                            # D1: use graph uuid, no fallback uuid4()
                            fact_uuid = rel.get("uuid") or rel.get("relationship_uuid")
                            if not fact_uuid:
                                logger.warning(
                                    f"D1: No uuid for relationship {rel}; skipping embed"
                                )
                                continue
                            await vs.store_fact_embedding(
                                entity1=rel["source_name"],
                                relation=rel["relation_type"],
                                entity2=rel["target_name"],
                                context=context,
                                fact_uuid=fact_uuid,
                                metadata={
                                    "source": "capture_context",
                                    "source_uuid": rel.get("source_uuid"),
                                    "target_uuid": rel.get("target_uuid"),
                                },
                            )
                            embedded_count += 1
                        except Exception as rel_e:
                            logger.warning(
                                f"Background: Failed to embed relationship: {rel_e}"
                            )
                    logger.info(
                        f"Background: Embedded {embedded_count} facts from context in Qdrant"
                    )
            except Exception as e:
                logger.warning(
                    f"Background context vector embedding failed (non-fatal): {e}"
                )

        _fire_and_forget(
            _background_context_embedding(),
            description=f"context embedding: {len(relationships_to_embed)} relationships",
        )

        # Detect silent extraction failure: zero entities, or the degraded
        # counter advanced during this call. Report it plainly instead of a
        # hollow success, so the caller knows nothing meaningful was stored.
        extraction_degraded = (
            result["entities_created"] == 0
            or llm_service_singleton.extraction_degraded_count > _pre_degraded
        )

        # Format response
        if extraction_degraded:
            response = f"""⚠️ Capture DEGRADED: no entities extracted

The model returned no usable entities for this context, so nothing was added to
the graph (the raw text was still saved to Markdown). This usually means the
configured extraction model produced empty or invalid output. Check the server
logs and confirm a non-reasoning instruct model with an adequate LLM_MAX_TOKENS.

**Entities Extracted**: {result["entities_created"]}
"""
        else:
            response = f"""✅ Graph captured successfully

**Entities Extracted**: {result["entities_created"]}
"""

        # Show first few entities as examples
        entity_examples = list(result["entity_uuids"].items())[:5]
        for entity_name, entity_uuid in entity_examples:
            entity_idx = list(result["entity_uuids"].keys()).index(entity_name)
            entity_type = (
                result["entity_types"][entity_idx]
                if entity_idx < len(result["entity_types"])
                else "Entity"
            )
            response += (
                f"  - {entity_name} ({entity_type}) → UUID: {entity_uuid[:8]}...\n"
            )

        if len(result["entity_uuids"]) > 5:
            response += f"  - ... and {len(result['entity_uuids']) - 5} more\n"

        response += f"\n**Relationships Created**: {result['relationships_created']}\n"

        # Show first few relationships
        rel_examples = result["relationships"][:5]
        for rel in rel_examples:
            response += f"  - {rel['source_name']} -{rel['relation_type']}-> {rel['target_name']}\n"

        if len(result["relationships"]) > 5:
            response += f"  - ... and {len(result['relationships']) - 5} more\n"

        response += f"""
**Storage**:
- Neo4j: ✓ (via GraphFactory)
- Markdown: {"✓" if markdown_success else "queued for retry"}

**Performance**: {extraction_time:.1f}ms
"""

        return response

    except Exception as e:
        logger.error(f"Graph capture failed: {e}")
        raise Exception(f"Failed to capture graph: {e}")


@mcp.tool()
@auto_capture_tool
async def invalidate_fact(
    entity1: str,
    relation: str,
    entity2: str,
    reason: str,
    invalidation_type: str = "both",
) -> dict:
    """Non-destructive fact invalidation with temporal validity and complexity scoring. See /mnemodrome skill for guidance."""
    start_time = time.time()
    logger.info(f"Invalidating fact: {entity1} -{relation}-> {entity2}")

    try:
        # Step 1: Find the relationship in Neo4j
        relationship = await neo4j_service.find_relationship(
            entity1=entity1, relation=relation, entity2=entity2
        )

        # Step 2: Check if relationship exists
        if not relationship:
            return {
                "status": "error",
                "error": "Fact not found",
                "details": f"No valid relationship found: {entity1} -{relation}-> {entity2}",
            }

        # Step 3: Check if already invalidated (shouldn't happen due to WHERE clause, but defensive)
        if relationship.get("invalid_at") or relationship.get("expired_at"):
            invalidated_at = relationship.get("invalid_at") or relationship.get(
                "expired_at"
            )
            return {
                "status": "error",
                "error": "Fact already invalidated",
                "details": f"Relationship was invalidated at {invalidated_at}",
            }

        # Step 4: Calculate complexity score to determine if validation needed
        old_fact_data = {
            "created_at": relationship["created_at"],
            "relation": relationship["relation_type"],
            "entity2": relationship["entity2_name"],
        }

        new_fact_data = {"entity1": entity1, "relation": relation, "entity2": entity2}

        # Get validation history for pattern matching (progressive automation)
        # Note: In Phase 1, we start with empty validation history
        # Phase 2 will integrate full pattern learning
        validation_history = []

        complexity_assessment = await conflict_detection_service.assess_complexity(
            old_fact=old_fact_data,
            new_fact=new_fact_data,
            validation_history=validation_history,
        )
        complexity_score = complexity_assessment["complexity_score"]

        logger.debug(f"Complexity score: {complexity_score} (threshold: 50)")

        # Step 5: If high complexity, require human validation
        COMPLEXITY_THRESHOLD = 50
        if complexity_score >= COMPLEXITY_THRESHOLD:
            # Generate recommendation rationale
            rationale_parts = []

            # Analyze complexity factors
            if hasattr(relationship["created_at"], "isoformat"):
                created_at = relationship["created_at"]
            else:
                from datetime import datetime as dt

                created_at = dt.fromisoformat(
                    str(relationship["created_at"]).replace("Z", "+00:00")
                )

            age_days = (datetime.now(UTC) - created_at.replace(tzinfo=UTC)).days

            if age_days > 30:
                rationale_parts.append(f"Fact is {age_days} days old (>30 days)")

            if len(validation_history) == 0:
                rationale_parts.append("No historical validation pattern found")

            if not rationale_parts:
                rationale_parts.append(
                    f"Complexity score {complexity_score} exceeds threshold"
                )

            return {
                "status": "validation_required",
                "conflict_detected": True,
                "complexity_score": complexity_score,
                "old_fact": {
                    "entity1": relationship["entity1_name"],
                    "relation": relationship["relation_type"],
                    "entity2": relationship["entity2_name"],
                    "created_at": relationship["created_at"].isoformat()
                    if hasattr(relationship["created_at"], "isoformat")
                    else str(relationship["created_at"]),
                },
                "recommendation": "supersede",
                "rationale": ". ".join(rationale_parts)
                + ". Human validation recommended.",
                "options": ["approve", "reject", "modify"],
            }

        # Step 6: Auto-resolve (low complexity) - invalidate the relationship
        try:
            invalidation_result = await neo4j_service.invalidate_relationship(
                relationship_uuid=relationship["uuid"],
                invalidation_type=invalidation_type,
                reason=reason,
                superseded_by=None,  # No replacement fact in simple invalidation
            )

            logger.info(
                f"✅ Neo4j invalidation successful: {relationship['uuid']} "
                f"(fields: {invalidation_result['fields_set']})"
            )

        except Exception as e:
            logger.error(f"Neo4j invalidation failed: {e}")
            return {
                "status": "error",
                "error": f"Neo4j operation failed: {str(e)}",
                "details": f"Failed to invalidate relationship {relationship['uuid']}",
            }

        # Step 7: Write to markdown (dual-memory pattern)
        dual_memory_status = "complete"
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = (
                f"invalidation-{entity1.lower().replace(' ', '-')}-{timestamp}.md"
            )

            markdown_content = f"""# Fact Invalidation

**Date**: {time.strftime("%Y-%m-%d %H:%M:%S")}
**Invalidation Type**: {invalidation_type}

## Invalidated Relationship
- **Source**: {relationship["entity1_name"]}
- **Relation**: {relationship["relation_type"]}
- **Target**: {relationship["entity2_name"]}
- **Original Creation**: {relationship["created_at"]}

## Invalidation Details
- **UUID**: `{relationship["uuid"]}`
- **Reason**: {reason}
- **Fields Set**: {", ".join(invalidation_result["fields_set"])}
- **Invalidated At**: {invalidation_result["invalidated_at"]}
- **Complexity Score**: {complexity_score} (auto-resolved, threshold: {COMPLEXITY_THRESHOLD})

## Temporal Validity
- **invalid_at**: {invalidation_result.get("invalid_at", "Not set")}
- **expired_at**: {invalidation_result.get("expired_at", "Not set")}

## Technical Details
- **Preservation**: Relationship preserved in graph (non-destructive operation)
- **Historical Query**: Accessible via temporal queries with scope "historical"
- **Progressive Automation**: Auto-resolved based on low complexity score

---
*Invalidated via Mnemodrome MCP invalidate_fact tool (Feature 006)*
"""

            filepath = file_service.write_file(
                filename=filename, content=markdown_content, memory_type=MemoryType.FACT
            )
            logger.info(f"✅ Markdown documentation created: {filepath}")

        except Exception as e:
            logger.warning(f"Markdown write failed (non-critical): {e}")
            dual_memory_status = "partial"

        # Step 8: Build success response
        invalidation_time = (time.time() - start_time) * 1000

        logger.info(
            f"Fact invalidated successfully in {invalidation_time:.1f}ms "
            f"(complexity: {complexity_score}, type: {invalidation_type})"
        )

        return {
            "status": "success",
            "fact_uuid": relationship["uuid"],
            "invalidated_at": invalidation_result["invalidated_at"],
            "fields_set": invalidation_result["fields_set"],
            "reason": reason,
            "preserved": True,
            "complexity_score": complexity_score,
            "dual_memory_status": dual_memory_status,
            "performance_ms": round(invalidation_time, 1),
        }

    except Exception as e:
        logger.error(f"Unexpected error in invalidate_fact: {e}")
        return {
            "status": "error",
            "error": str(e),
            "details": f"Unexpected error during invalidation: {type(e).__name__}",
        }


@mcp.tool()
@auto_capture_tool
async def update_fact(
    entity1: str,
    relation: str,
    entity2_old: str,
    entity2_new: str,
    context: str = "",
    conflict_resolution: str = "prompt",
) -> dict:
    """Atomically supersede a fact with a new one via supersession chain. See /mnemodrome skill for guidance."""
    start_time = time.time()
    logger.info(f"Updating fact: {entity1} -{relation}-> {entity2_old} → {entity2_new}")

    try:
        # Step 1: Find the old relationship
        old_relationship = await neo4j_service.find_relationship(
            entity1=entity1, relation=relation, entity2=entity2_old
        )

        if not old_relationship:
            return {
                "status": "error",
                "error": "Old fact not found",
                "details": f"No valid relationship found: {entity1} -{relation}-> {entity2_old}",
            }

        # Step 2: Calculate complexity score
        old_fact_data = {
            "created_at": old_relationship.get(
                "created_at",
                old_relationship.get("valid_at", datetime.now(UTC)),
            ),
            "relation": old_relationship.get(
                "relation_type", old_relationship.get("relation", relation)
            ),
            "entity2": old_relationship.get(
                "entity2_name", old_relationship.get("entity2", entity2_old)
            ),
        }

        new_fact_data = {
            "entity1": entity1,
            "relation": relation,
            "entity2": entity2_new,
        }

        validation_history = []  # Phase 1: empty validation history

        # Use assess_complexity wrapper (contract-compliant)
        complexity_assessment = await conflict_detection_service.assess_complexity(
            old_fact=old_fact_data,
            new_fact=new_fact_data,
            validation_history=validation_history,
        )

        complexity_score = complexity_assessment["complexity_score"]
        auto_resolvable = complexity_assessment["auto_resolvable"]

        logger.debug(
            f"Complexity score: {complexity_score} (mode: {conflict_resolution})"
        )

        if conflict_resolution == "error":
            # Error mode: fail on any conflict
            raise Exception(
                f"Conflict detected with conflict_resolution='error'. "
                f"Old fact exists: {entity1} -{relation}-> {entity2_old}"
            )

        elif conflict_resolution == "prompt":
            # Prompt mode: always require validation
            created_at_raw = old_relationship.get("created_at") or old_relationship.get(
                "valid_at"
            )
            if created_at_raw:
                if hasattr(created_at_raw, "isoformat"):
                    created_at = created_at_raw
                else:
                    try:
                        created_at = datetime.fromisoformat(
                            str(created_at_raw).replace("Z", "+00:00")
                        )
                    except (ValueError, AttributeError):
                        created_at = datetime.now(UTC)
            else:
                created_at = datetime.now(UTC)

            age_days = (datetime.now(UTC) - created_at.replace(tzinfo=UTC)).days

            rationale_parts = []
            if age_days > 30:
                rationale_parts.append(f"Fact is {age_days} days old (>30 days)")
            if len(validation_history) == 0:
                rationale_parts.append("No historical validation pattern found")
            if not rationale_parts:
                rationale_parts.append(f"Complexity score {complexity_score}")

            return {
                "status": "validation_required",
                "conflict_detected": True,
                "complexity_score": complexity_score,
                "old_fact": {
                    "entity1": old_relationship.get(
                        "entity1_name", old_relationship.get("entity1", entity1)
                    ),
                    "relation": old_relationship.get(
                        "relation_type", old_relationship.get("relation", relation)
                    ),
                    "entity2": old_relationship.get(
                        "entity2_name", old_relationship.get("entity2", entity2_old)
                    ),
                    "created_at": created_at.isoformat()
                    if hasattr(created_at, "isoformat")
                    else str(created_at),
                },
                "recommendation": "validate",
                "rationale": ". ".join(rationale_parts),
                "options": ["approve", "reject", "modify"],
            }

        elif conflict_resolution == "auto":
            # Auto mode: only auto-resolve if complexity < 50
            if not auto_resolvable:
                # Return validation_required for complex conflicts
                return {
                    "status": "validation_required",
                    "conflict_detected": True,
                    "complexity_score": complexity_score,
                    "old_fact": {
                        "entity1": old_relationship.get(
                            "entity1_name", old_relationship.get("entity1", entity1)
                        ),
                        "relation": old_relationship.get(
                            "relation_type", old_relationship.get("relation", relation)
                        ),
                        "entity2": old_relationship.get(
                            "entity2_name", old_relationship.get("entity2", entity2_old)
                        ),
                    },
                    "recommendation": "validate",
                    "rationale": f"Complexity score {complexity_score} exceeds threshold for auto-resolution",
                    "options": ["approve", "reject", "modify"],
                }

        # Step 4: Execute atomic update transaction (auto-resolved or auto mode with simple conflict)
        try:
            # Resolve entity UUIDs for new relationship.
            # entity1_uuid and entity2_uuid (old target) come from find_relationship so
            # we pass them distinctly (D2: old and new target UUIDs passed separately).
            source_uuid = old_relationship.get(
                "entity1_uuid"
            ) or await _resolve_entity_uuid(entity1)
            entity2_old_uuid = old_relationship.get("entity2_uuid")
            target_uuid = await _resolve_entity_uuid(entity2_new)
            old_uuid = old_relationship["uuid"]
            new_uuid = str(uuid4())

            # Use execute_update_transaction wrapper (contract-compliant)
            transaction_result = await neo4j_service.execute_update_transaction(
                old_fact_uuid=old_uuid,
                entity1_uuid=source_uuid,
                relation_type=relation,
                entity2_uuid=target_uuid,
                context=context,
                new_fact_uuid=new_uuid,
                entity2_old_uuid=entity2_old_uuid,
            )

            timestamp = datetime.fromisoformat(transaction_result["invalidated_at"])

            logger.info(f"✅ Atomic update successful: {old_uuid} → {new_uuid}")

        except Exception as e:
            logger.error(f"Atomic transaction failed: {e}")
            raise Exception(f"Transaction failed: {str(e)}")

        # D2b (FR-005): embed the NEW fact in Qdrant under new_uuid so graph
        # identity and vector identity remain aligned after supersession.
        async def _background_update_embedding():
            try:
                vs = get_vector_service()
                if vs and vs.is_enabled():
                    await vs.store_fact_embedding(
                        entity1=entity1,
                        relation=relation,
                        entity2=entity2_new,
                        context=context,
                        fact_uuid=new_uuid,
                        metadata={
                            "operation": "supersession",
                            "supersedes": old_uuid,
                            "entity1_uuid": source_uuid,
                            "entity2_uuid": target_uuid,
                        },
                    )
                    logger.debug(f"D2b: New fact embedded in Qdrant: {new_uuid}")
            except Exception as embed_err:
                logger.warning(
                    f"D2b: fact embedding after update failed (non-fatal): {embed_err}"
                )

        _fire_and_forget(
            _background_update_embedding(),
            description=f"D2b update embedding: {entity1}-{relation}->{entity2_new}",
        )

        # Step 5: Dual-memory sync (markdown documentation)
        dual_memory_status = {"neo4j": "success", "markdown": "success"}
        queued_retries = 0

        try:
            timestamp_str = time.strftime("%Y%m%d-%H%M%S")
            filename = f"update-{entity1.lower().replace(' ', '-')}-{timestamp_str}.md"

            markdown_content = f"""# Fact Update (Supersession)

**Date**: {time.strftime("%Y-%m-%d %H:%M:%S")}
**Operation**: Atomic fact supersession

## Old Fact (Invalidated)
- **Source**: {old_relationship.get("entity1_name", entity1)}
- **Relation**: {old_relationship.get("relation_type", relation)}
- **Target**: {old_relationship.get("entity2_name", entity2_old)}
- **UUID**: `{old_uuid}`
- **Original Creation**: {old_relationship.get("created_at", "Not recorded")}

## New Fact (Created)
- **Source**: {entity1}
- **Relation**: {relation}
- **Target**: {entity2_new}
- **UUID**: `{new_uuid}`
- **Context**: {context or "No additional context"}

## Supersession Chain
- Old UUID: `{old_uuid}`
- New UUID: `{new_uuid}`
- Superseded by field set on old fact

## Technical Details
- **Complexity Score**: {complexity_score} ({"auto-resolved" if conflict_resolution == "auto" and auto_resolvable else "validated"})
- **Conflict Resolution Mode**: {conflict_resolution}
- **Atomic Transaction**: Both operations completed successfully
- **Preservation**: Old fact preserved with superseded_by link

---
*Updated via Mnemodrome MCP update_fact tool (Feature 006)*
"""

            filepath = file_service.write_file(
                filename=filename,
                content=markdown_content,
                memory_type=MemoryType.UPDATE,
            )
            logger.info(f"✅ Markdown documentation created: {filepath}")

        except Exception as e:
            logger.warning(f"Markdown write failed (queued for retry): {e}")
            dual_memory_status["markdown"] = "queued"
            queued_retries = 1

        # Step 6: Build success response
        update_time = (time.time() - start_time) * 1000

        logger.info(
            f"Fact updated successfully in {update_time:.1f}ms "
            f"(complexity: {complexity_score}, mode: {conflict_resolution})"
        )

        return {
            "status": "success",
            "invalidated_fact_uuid": transaction_result["old_fact_uuid"],
            "new_fact_uuid": transaction_result["new_fact_uuid"],
            "invalidated_at": transaction_result["invalidated_at"],
            "supersession_chain": transaction_result["supersession_chain"],
            "complexity_score": complexity_score,
            "conflict_auto_resolved": conflict_resolution == "auto" and auto_resolvable,
            "dual_memory_status": dual_memory_status,
            "queued_retries": queued_retries,
            "performance_ms": round(update_time, 1),
        }

    except Exception as e:
        # Check if this is a deliberate error mode exception or transaction failure
        if "conflict_resolution='error'" in str(e) or "Transaction failed" in str(e):
            raise  # Propagate error mode exceptions and transaction failures

        logger.error(f"Unexpected error in update_fact: {e}")
        return {
            "status": "error",
            "error": str(e),
            "details": f"Unexpected error during update: {type(e).__name__}",
        }


async def _resolve_entity_uuid(entity_name: str) -> str:
    """Helper to resolve entity name to UUID."""
    # Try to find existing entity
    query = "MATCH (e {name: $name}) RETURN e.uuid as uuid LIMIT 1"
    results = await neo4j_service.query(query, {"name": entity_name})

    if results:
        return results[0]["uuid"]

    # Create new entity if not found
    new_uuid = str(uuid4())
    create_query = """
    CREATE (e:Entity {
        uuid: $uuid,
        name: $name,
        summary: $summary,
        created_at: datetime()
    })
    RETURN e.uuid as uuid
    """
    await neo4j_service.query(
        create_query,
        {"uuid": new_uuid, "name": entity_name, "summary": f"Entity: {entity_name}"},
    )

    return new_uuid


# ============================================================================
# EVENT LOG LAYER TOOLS (Episodic Memory)
# ============================================================================


@mcp.tool()
@auto_capture_tool
async def log_event(
    event_type: str,
    action: str,
    actor: str = "user",
    target: str = "",
    context: str = "",
    result: str = "",
    metadata: str = "",
) -> str:
    """Log an event to the episodic memory layer. See /mnemodrome skill for guidance."""
    import json

    # Parse JSON strings if provided
    context_dict = json.loads(context) if context else None
    metadata_dict = json.loads(metadata) if metadata else None

    # Log event
    event_id = await event_log_service.log_event(
        event_type=event_type,
        action=action,
        actor=actor or None,
        target=target or None,
        context=context_dict,
        result=result or None,
        metadata=metadata_dict,
    )

    return (
        f"Event logged successfully\n"
        f"Event ID: {event_id}\n"
        f"Session ID: {event_log_service.session_id}\n"
        f"Type: {event_type}/{action}"
    )


@mcp.tool()
@auto_capture_tool
async def query_events(
    session_id: str = "",
    event_type: str = "",
    action: str = "",
    actor: str = "",
    limit: int = 100,
    offset: int = 0,
) -> str:
    """Query events from the event log with flexible filtering. See /mnemodrome skill for guidance."""

    # Use current session if not specified
    query_session_id = session_id if session_id else event_log_service.session_id

    # Query events
    events = await event_log_service.query_events(
        session_id=query_session_id or None,
        event_type=event_type or None,
        action=action or None,
        actor=actor or None,
        limit=limit,
        offset=offset,
    )

    if not events:
        return f"No events found matching filters (session: {query_session_id[:8]}...)"

    # Format as markdown table
    lines = [
        "# Event Log Query Results\n",
        f"Session: {query_session_id}\n",
        f"Found: {len(events)} events\n",
        "\n",
        "| ID | Timestamp | Type | Action | Actor | Target | Result |\n",
        "|---|---|---|---|---|---|---|\n",
    ]

    for event in events:
        timestamp_short = event["timestamp"][:19] if event["timestamp"] else ""
        target_short = (
            (event["target"][:30] + "...")
            if event["target"] and len(event["target"]) > 30
            else (event["target"] or "")
        )

        lines.append(
            f"| {event['id']} | {timestamp_short} | {event['event_type']} | "
            f"{event['action']} | {event['actor'] or ''} | {target_short} | "
            f"{event['result'] or ''} |\n"
        )

    return "".join(lines)


@mcp.tool()
@auto_capture_tool
async def get_session_timeline(session_id: str = "", limit: int = 100) -> str:
    """Get complete chronological timeline for a session. See /mnemodrome skill for guidance."""
    # Use current session if not specified
    query_session_id = session_id if session_id else event_log_service.session_id

    timeline = await event_log_service.get_session_timeline(
        session_id=query_session_id or None, limit=limit
    )

    if not timeline:
        return f"No events found for session {query_session_id[:8]}..."

    # Format as markdown timeline
    lines = [
        "# Session Timeline\n",
        f"Session ID: {query_session_id}\n",
        f"Events: {len(timeline)}\n",
        "\n",
    ]

    for event in timeline:
        timestamp = event["timestamp"][:19] if event["timestamp"] else ""
        lines.append(f"## {timestamp} - {event['event_type']}/{event['action']}\n")

        if event["actor"]:
            lines.append(f"**Actor**: {event['actor']}\n")
        if event["target"]:
            lines.append(f"**Target**: {event['target']}\n")
        if event["result"]:
            lines.append(f"**Result**: {event['result']}\n")
        if event["context"]:
            lines.append(f"**Context**: {event['context']}\n")

        lines.append("\n")

    return "".join(lines)


@mcp.tool()
@auto_capture_tool
async def get_event_log_health() -> str:
    """Get Event Log health status and performance metrics."""
    # Import config here to avoid circular import
    from src import config

    metrics = event_log_service._metrics

    # Get database stats
    db_size = await event_log_service.get_database_size()
    event_count = await event_log_service.get_event_count()

    # Get session info
    session_status = await event_log_service.get_session_status()

    # Format health report
    report = f"""# Event Log Health Report

## Status
- **Enabled**: {config.EVENT_LOG_ENABLED}
- **Auto-Capture**: {config.EVENT_LOG_AUTO_CAPTURE}
- **Auto-Disabled**: {event_log_service._auto_disabled}

## Performance Metrics
- **Write Latency (p95)**: {metrics.get_p95_write():.2f}ms (SLA: {config.EVENT_LOG_PERFORMANCE_SLA_MS}ms)
- **Write Latency (p99)**: {metrics.get_p99_write():.2f}ms
- **Query Latency (p95)**: {metrics.get_p95_query():.2f}ms (SLA: {config.EVENT_LOG_QUERY_SLA_MS}ms)
- **Query Latency (p99)**: {metrics.get_p99_query():.2f}ms
- **SLA Violations**: {metrics.sla_violations}
- **Last Violation**: {metrics.last_violation_at or "Never"}

## Database
- **Size**: {db_size:.2f} MB (Max: {config.EVENT_LOG_MAX_SIZE_MB} MB)
- **Total Events**: {event_count}
- **Current Session**: {event_log_service.session_id[:8] if event_log_service.session_id else "None"}...
- **Session Status**: {session_status}

## Configuration
- **Retention Days**: {config.EVENT_LOG_RETENTION_DAYS}
- **Session Timeout**: {config.EVENT_LOG_SESSION_TIMEOUT_HOURS} hours
- **Fail Loud**: {config.EVENT_LOG_FAIL_LOUD}
- **Database**: {event_log_service.db.backend_description}

## Recommendations
"""

    # Add recommendations based on status
    recommendations = []

    if event_log_service._auto_disabled:
        recommendations.append(
            "- ⚠️ **Event log is auto-disabled due to SLA violations. Check database performance.**"
        )

    if db_size > config.EVENT_LOG_MAX_SIZE_MB * 0.9:
        recommendations.append(
            f"- ⚠️ **Database approaching size limit ({db_size:.1f}MB / {config.EVENT_LOG_MAX_SIZE_MB}MB). Retention will clean up soon.**"
        )

    if metrics.get_p95_write() > config.EVENT_LOG_PERFORMANCE_SLA_MS * 0.8:
        recommendations.append(
            "- ⚠️ **Write latency approaching SLA limit. Monitor for degradation.**"
        )

    if metrics.sla_violations > 0:
        recommendations.append(
            f"- **{metrics.sla_violations} recent SLA violations detected. {5 - metrics.sla_violations} more will trigger auto-disable.**"
        )

    if not recommendations:
        recommendations.append(
            "- ✅ **All systems nominal. Event log performing within SLA.**"
        )

    report += "\n".join(recommendations)

    return report


def _smart_truncate(text: str, max_length: int = 300) -> str:
    """Truncate text at sentence boundaries, never mid-word."""
    if not text or len(text) <= max_length:
        return text
    # Look for sentence boundary (". ") in the last 50 chars of the window
    truncated = text[:max_length]
    last_period = truncated.rfind(". ", max_length - 50)
    if last_period > 0:
        return truncated[: last_period + 1]
    # Fall back to last space (never mid-word)
    last_space = truncated.rfind(" ")
    if last_space > 0:
        return truncated[:last_space] + "..."
    return truncated + "..."


def _format_date_human(date_value) -> str:
    """Convert date/datetime/ISO string to human-readable format."""
    if not date_value:
        return ""
    try:
        if isinstance(date_value, datetime):
            dt = date_value
        elif isinstance(date_value, str):
            dt = datetime.fromisoformat(date_value.replace("Z", "+00:00"))
        else:
            return str(date_value)
        return dt.strftime("%B %-d, %Y")
    except (ValueError, TypeError):
        return str(date_value)


def _get_result_title(result: dict) -> str:
    """Extract a display title based on result type."""
    result_type = result.get("type", "")
    if result_type == "learning":
        return result.get("title", "Untitled Learning")
    elif result_type == "fact":
        e1 = result.get("entity1", "?")
        rel = result.get("relation", "?")
        e2 = result.get("entity2", "?")
        return f"{e1} → {rel} → {e2}"
    elif result_type == "session":
        summary = result.get("summary", "")
        if summary:
            return _smart_truncate(summary, 80)
        return "Session"
    elif result_type == "prompt":
        content = result.get("content", "")
        if content:
            return _smart_truncate(content, 80)
        return "Prompt"
    return result.get("title", "Untitled")


def _get_result_content(result: dict) -> str:
    """Extract the best content excerpt for a result."""
    result_type = result.get("type", "")
    if result_type == "learning":
        return result.get("content", "")
    elif result_type == "fact":
        return result.get("context", "")
    elif result_type == "session":
        return result.get("summary", "")
    elif result_type == "prompt":
        return result.get("content", "")
    return result.get("content", result.get("summary", ""))


def _get_type_badge(result_type: str) -> str:
    """Map result type to display badge."""
    badges = {
        "learning": "Learning",
        "fact": "Fact",
        "session": "Session",
        "prompt": "Prompt",
    }
    return badges.get(result_type, result_type.title() if result_type else "Unknown")


def _generate_synthesis(results: list, query: str) -> str:
    """Generate a template-based synthesis narrative from search results."""
    if not results:
        return "No results available for synthesis."

    # Count by type
    type_counts = {}
    for r in results:
        t = r.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    # Score spread
    scores = [r.get("score", 0) for r in results]
    high_score = max(scores)
    low_score = min(scores)

    # Date range
    dates = []
    for r in results:
        date_val = r.get("created_at")
        if date_val:
            try:
                if isinstance(date_val, datetime):
                    dates.append(date_val)
                elif isinstance(date_val, str):
                    dates.append(
                        datetime.fromisoformat(date_val.replace("Z", "+00:00"))
                    )
            except (ValueError, TypeError):
                pass

    # Dominant type
    dominant_type = max(type_counts, key=type_counts.get)
    dominant_badge = _get_type_badge(dominant_type)
    total = len(results)

    # Build synthesis sentences
    sentences = []

    # Sentence 1: Result summary
    type_parts = [
        f"{count} {_get_type_badge(t).lower()}{'s' if count > 1 else ''}"
        for t, count in sorted(type_counts.items(), key=lambda x: -x[1])
    ]
    sentences.append(
        f'Found {total} relevant result{"s" if total > 1 else ""} for "{query}": {", ".join(type_parts)}.'
    )

    # Sentence 2: Confidence
    if high_score >= 0.80:
        sentences.append(f"High confidence match (top score: {high_score:.3f}).")
    elif high_score >= 0.60:
        sentences.append(
            f"Moderate confidence (top score: {high_score:.3f}, spread: {high_score - low_score:.3f})."
        )
    else:
        sentences.append(
            f"Low confidence results (top score: {high_score:.3f}) — consider broadening search terms."
        )

    # Sentence 3: Temporal spread (if dates available)
    if len(dates) >= 2:
        earliest = min(dates)
        latest = max(dates)
        if earliest.date() != latest.date():
            sentences.append(
                f"Evidence spans from {_format_date_human(earliest)} to {_format_date_human(latest)}."
            )

    # Sentence 4: Dominant pattern
    if len(type_counts) > 1:
        sentences.append(
            f"Results are predominantly {dominant_badge.lower()}s, suggesting this topic is best documented in that layer."
        )

    return " ".join(sentences)


def _format_evidence_report(results: list, query: str, query_time: float) -> str:
    """Format search results as a Curated Evidence Report."""
    REPORT_THRESHOLD = 0.50
    FALLBACK_THRESHOLD = 0.45
    MAX_RESULTS = 7

    if not results:
        return f"# Curated Evidence Report\n\nNo matching content found for query: '{query}'\n\nTry with different keywords or a lower minimum similarity score."

    # Sort all results by score descending (cross-category)
    sorted_results = sorted(results, key=lambda r: r.get("score", 0), reverse=True)

    # Apply score threshold
    threshold = REPORT_THRESHOLD
    filtered = [r for r in sorted_results if r.get("score", 0) >= threshold]

    confidence_note = ""
    if len(filtered) < 3:
        threshold = FALLBACK_THRESHOLD
        filtered = [r for r in sorted_results if r.get("score", 0) >= threshold]
        if filtered and threshold < REPORT_THRESHOLD:
            confidence_note = f"\n> *Note: Threshold lowered to {threshold} (fewer than 3 results met {REPORT_THRESHOLD})*\n"

    # If still no results above fallback, include all with warning
    if not filtered:
        filtered = sorted_results
        confidence_note = "\n> *Note: All results below minimum confidence threshold — interpret with caution*\n"

    # Cap at max results
    filtered = filtered[:MAX_RESULTS]

    # Build report
    report = f"# Curated Evidence Report\n\n**Query**: {query}\n"
    if confidence_note:
        report += confidence_note
    report += "\n---\n\n"

    # Evidence blocks
    for r in filtered:
        title = _get_result_title(r)
        score = r.get("score", 0)
        content = _get_result_content(r)
        excerpt = _smart_truncate(content) if content else ""
        result_type = _get_type_badge(r.get("type", ""))
        date_str = _format_date_human(r.get("created_at"))

        report += f"> **{title}** (Score: {score:.3f})\n"
        if excerpt:
            report += f"> {excerpt}\n"
        attribution = f"*— {result_type}"
        if date_str:
            attribution += f" | {date_str}"
        attribution += "*"
        report += f"> {attribution}\n\n"

    # Synthesis section
    report += "---\n\n### Synthesis\n\n"
    report += _generate_synthesis(filtered, query)
    report += "\n\n"

    # Metadata footer
    report += f"---\n*{len(filtered)} result{'s' if len(filtered) != 1 else ''} "
    report += f"| threshold: {threshold} | {query_time:.1f}ms*\n"

    return report


@mcp.tool()
@auto_capture_tool
async def semantic_search(
    query: str,
    search_type: str = "all",
    limit: int = 5,
    min_score: float | None = None,
    format: str = "default",
) -> str:
    """Search for semantically similar content using vector embeddings. See /mnemodrome skill for guidance."""
    if format not in ("default", "report"):
        return f"# Semantic Search Error\n\nInvalid format '{format}'. Valid values: 'default', 'report'."

    start_time = time.time()
    logger.info(
        f"Executing semantic search: query='{query}', type={search_type}, limit={limit}, format={format}"
    )

    try:
        vector_service = get_vector_service()

        if not vector_service or not vector_service.is_enabled():
            return "# Semantic Search - Vector Layer Unavailable\n\nVector service is not enabled or initialized. Check vector layer configuration."

        # Execute semantic search
        results = await vector_service.search(
            query_text=query, search_type=search_type, limit=limit, min_score=min_score
        )

        # Feature 034: Apply activation scoring to semantic search results
        if ACTR_ENABLED and results:
            try:
                # Extract content scores from vector similarity
                content_scores = [r.get("score", 0.5) for r in results]

                # Extract entity UUIDs from results for access tracking
                result_uuids = [r.get("uuid") for r in results if r.get("uuid")]
                if result_uuids:
                    _fire_and_forget(
                        activation_service.update_access_tracking(result_uuids),
                        description="semantic search access tracking",
                    )
                    if len(result_uuids) >= 2:
                        _fire_and_forget(
                            activation_service.strengthen_hebbian_links(result_uuids),
                            description="semantic search hebbian strengthening",
                        )
            except Exception as e:
                logger.warning(
                    f"Semantic search activation scoring failed (FR-008): {e}"
                )

        # Format results as markdown
        query_time = (time.time() - start_time) * 1000  # Convert to ms
        logger.info(
            f"Semantic search completed in {query_time:.1f}ms ({len(results)} results)"
        )

        if format == "report":
            return _format_evidence_report(results, query, query_time)

        if not results:
            return f"# Semantic Search Results\n\nNo matching content found for query: '{query}'\n\nTry with different keywords or a lower minimum similarity score."

        markdown = f"# Semantic Search Results\n\nQuery: **{query}**\n\n"

        if search_type in ("prompts", "all"):
            prompt_results = [r for r in results if r.get("type") == "prompt"]
            if prompt_results:
                markdown += "## Similar Prompts\n\n"
                for result in prompt_results:
                    markdown += f"- **Score**: {result.get('score', 0):.3f}\n"
                    markdown += f"  **Content**: {result.get('content', '')[:200]}...\n"
                    if result.get("created_at"):
                        markdown += f"  **Date**: {result.get('created_at')}\n"
                    markdown += "\n"

        if search_type in ("learnings", "all"):
            learning_results = [r for r in results if r.get("type") == "learning"]
            if learning_results:
                markdown += "## Similar Learnings\n\n"
                for result in learning_results:
                    markdown += f"- **Score**: {result.get('score', 0):.3f}\n"
                    markdown += f"  **Title**: {result.get('title', 'Untitled')}\n"
                    markdown += f"  **Content**: {result.get('content', '')[:200]}...\n"
                    if result.get("created_at"):
                        markdown += f"  **Date**: {result.get('created_at')}\n"
                    markdown += "\n"

        if search_type in ("sessions", "all"):
            session_results = [r for r in results if r.get("type") == "session"]
            if session_results:
                markdown += "## Similar Sessions\n\n"
                for result in session_results:
                    markdown += f"- **Score**: {result.get('score', 0):.3f}\n"
                    summary = result.get("summary", "")
                    markdown += f"  **Summary**: {summary[:300]}{'...' if len(summary) > 300 else ''}\n"
                    if result.get("key_insights"):
                        insights = result.get("key_insights", [])[:3]
                        if insights:
                            markdown += f"  **Key Insights**: {', '.join(insights)}\n"
                    if result.get("created_at"):
                        markdown += f"  **Date**: {result.get('created_at')}\n"
                    markdown += "\n"

        if search_type in ("facts", "all"):
            fact_results = [r for r in results if r.get("type") == "fact"]
            if fact_results:
                markdown += "## Similar Facts\n\n"
                for result in fact_results:
                    markdown += f"- **Score**: {result.get('score', 0):.3f}\n"
                    entity1 = result.get("entity1", "?")
                    relation = result.get("relation", "?")
                    entity2 = result.get("entity2", "?")
                    markdown += f"  **Fact**: {entity1} → {relation} → {entity2}\n"
                    if result.get("context"):
                        context = result.get("context", "")
                        markdown += f"  **Context**: {context[:200]}{'...' if len(context) > 200 else ''}\n"
                    if result.get("created_at"):
                        markdown += f"  **Date**: {result.get('created_at')}\n"
                    markdown += "\n"

        markdown += f"\n**Search Time**: {query_time:.1f}ms"
        return markdown

    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        raise Exception(f"Semantic search failed: {e}")


def _format_context_cascade_result(
    cascade_result: dict[str, Any], query_time: float
) -> str:
    """
    Format a context cascade result dict as markdown.

    Every emitted item must carry readable text. Each layer's result shape
    differs (prompts use "matched_text", learnings use "title"/"content",
    facts use "entity1"/"relation"/"entity2"), so each section maps its own
    fields explicitly rather than assuming a shared key — the previous
    formatter read "text" for prompt hits, which vector_service never sets
    (it sets "matched_text"), silently producing an empty body for every
    prompt hit. Hits with no recoverable text (e.g. the underlying Neo4j
    node/markdown record no longer exists) are dropped rather than
    rendered empty.

    Args:
        cascade_result: Dict from ContextRetrievalService.retrieve_context_for_prompt
        query_time: Elapsed time in ms

    Returns:
        Markdown string for display ("" if no relevant context found)
    """
    # Check if greenfield (no relevant context found)
    is_greenfield = cascade_result.get("is_greenfield", True)

    if is_greenfield:
        return ""  # Return empty string - no relevant context found

    markdown = "# Relevant Context\n\n"

    # Similar prompts from vector search.
    # vector_service.search_similar_prompts(_by_embedding) returns the hit
    # text under "matched_text" (and raw payload under "payload"), never
    # under "text" — that mismatch was the root cause of the empty bodies.
    similar_prompts = cascade_result.get("similar_prompts", [])
    rendered_prompts = []
    for result in similar_prompts:
        text = (
            result.get("matched_text")
            or result.get("text")
            or (result.get("payload") or {}).get("text")
            or ""
        ).strip()
        if not text:
            # No recoverable text for this hit — drop it instead of
            # rendering an empty body with a similarity score attached.
            logger.warning(
                f"Dropping similar_prompt hit with no text (id={result.get('id')})"
            )
            continue
        rendered_prompts.append((result.get("score", 0), text[:200]))

    if rendered_prompts:
        markdown += "## Previous Similar Questions\n\n"
        for score, text in rendered_prompts:
            markdown += f"- **Similarity**: {score:.3f}\n"
            markdown += f"  {text}...\n\n"

    # Similar learnings (vector_service returns "title"/"content" keys).
    similar_learnings = cascade_result.get("similar_learnings", [])
    rendered_learnings = []
    for learning in similar_learnings:
        title = (learning.get("title") or "").strip()
        content = (learning.get("content") or "").strip()
        if not title and not content:
            # Underlying record has no retrievable title or content —
            # likely a stale Qdrant point whose source record is gone.
            logger.warning(
                f"Dropping similar_learning hit with no title/content "
                f"(id={learning.get('id')}, neo4j_uuid={learning.get('neo4j_uuid')})"
            )
            continue
        rendered_learnings.append(
            (
                title or "[Untitled]",
                content[:200] if content else "[content unavailable]",
            )
        )

    if rendered_learnings:
        markdown += "## Related Learnings\n\n"
        for title, content in rendered_learnings:
            markdown += f"### {title}\n{content}...\n\n"

    # Entity context from graph.
    entity_context = cascade_result.get("entity_context", [])
    rendered_entities = [
        e.get("name") for e in entity_context if (e.get("name") or "").strip()
    ]
    if rendered_entities:
        markdown += "## Related Entities\n\n"
        for name in rendered_entities:
            markdown += f"- {name}\n"

    # Related facts (Neo4j rows: entity1/relation/entity2).
    facts = cascade_result.get("facts", [])
    rendered_facts = [
        f
        for f in facts
        if (f.get("entity1") or "").strip()
        and (f.get("relation") or "").strip()
        and (f.get("entity2") or "").strip()
    ]
    if rendered_facts:
        markdown += "## Related Facts\n\n"
        for fact in rendered_facts:
            markdown += (
                f"- {fact['entity1']} **{fact['relation']}** {fact['entity2']}\n"
            )

    # Relevant procedures (Feature 031)
    procedures = cascade_result.get("procedures", [])
    if procedures:
        markdown += "\n## Relevant Procedures\n\n"
        for proc in procedures:
            steps_md = "\n".join(
                f"   {i}. {s}" for i, s in enumerate(proc.diagnostic_steps, 1)
            )
            techs = ", ".join(proc.technologies) if proc.technologies else "N/A"
            markdown += f"""### {proc.symptom[:80]} (similarity: {proc.similarity_score:.2f})
**Problem Type**: {proc.problem_type} | **Outcome**: {proc.outcome} | **Technologies**: {techs}

**Diagnostic Steps**:
{steps_md}

**Root Cause**: {proc.root_cause}
**Fix**: {proc.fix}
**Verification**: {proc.verification}
"""

    # Summary statistics — counts reflect what was actually rendered,
    # not raw hit counts (dropped empty-body hits are excluded).
    total_items = (
        len(rendered_prompts)
        + len(rendered_learnings)
        + len(rendered_entities)
        + len(rendered_facts)
        + len(procedures)
    )

    if total_items == 0:
        # Every hit was dropped for missing content — be explicit rather
        # than silently returning what looks like a normal greenfield "".
        return (
            "# Relevant Context\n\n"
            "No displayable context found (all candidate hits had "
            "missing or unavailable content).\n\n"
            f"---\n*Retrieved 0 items in {query_time:.1f}ms*"
        )

    markdown += f"\n---\n*Retrieved {total_items} items in {query_time:.1f}ms*"

    return markdown


@mcp.tool()
@auto_capture_tool
async def load_context_cascade(
    prompt: str,
    max_results: int = 5,
    include_learnings: bool = True,
    include_facts: bool = True,
) -> str:
    """Execute full context retrieval cascade (Vector -> Redis -> Neo4j -> Markdown). See /mnemodrome skill for guidance."""
    start_time = time.time()
    logger.info(f"Executing context cascade for prompt (max_results={max_results})")

    try:
        context_retrieval_service = get_context_retrieval_service()

        if not context_retrieval_service:
            return "# Context Cascade - Service Unavailable\n\nContext retrieval service is not initialized. Check service configuration."

        # Execute full cascade
        cascade_result = await context_retrieval_service.retrieve_context_for_prompt(
            prompt=prompt,
            max_results=max_results,
            include_learnings=include_learnings,
            include_facts=include_facts,
        )

        query_time = (time.time() - start_time) * 1000  # Convert to ms
        logger.info(f"Context cascade completed in {query_time:.1f}ms")

        # Feature 034: Fire-and-forget activation tracking for cascade results
        if ACTR_ENABLED:
            try:
                cascade_uuids = set()
                for entity in cascade_result.get("entity_context", []):
                    if entity.get("uuid"):
                        cascade_uuids.add(entity["uuid"])
                for fact in cascade_result.get("facts", []):
                    if fact.get("source_uuid"):
                        cascade_uuids.add(fact["source_uuid"])
                    if fact.get("target_uuid"):
                        cascade_uuids.add(fact["target_uuid"])

                if cascade_uuids:
                    uuid_list = list(cascade_uuids)
                    _fire_and_forget(
                        activation_service.update_access_tracking(uuid_list),
                        description="cascade access tracking",
                    )
                    if len(uuid_list) >= 2:
                        _fire_and_forget(
                            activation_service.strengthen_hebbian_links(uuid_list),
                            description="cascade hebbian strengthening",
                        )
            except Exception as e:
                logger.warning(f"Cascade activation tracking failed (FR-008): {e}")

        return _format_context_cascade_result(cascade_result, query_time)

    except Exception as e:
        logger.error(f"Context cascade failed: {e}")
        raise Exception(f"Context cascade failed: {e}")


@mcp.tool()
@auto_capture_tool
async def vector_health() -> str:
    """Get vector layer health status and statistics."""
    logger.info("Retrieving vector layer health status")

    try:
        vector_service = get_vector_service()
        context_retrieval_service = get_context_retrieval_service()

        # Check if services are available
        if not vector_service:
            return "# Vector Layer Health - Service Unavailable\n\nVector service is not initialized."

        # Get health status
        health_status = await vector_service.health_check()

        # Format health report
        report = "# Vector Layer Health Report\n\n"

        # Status section
        report += "## Status\n"
        report += f"- **Overall Status**: {health_status.get('status', 'unknown')}\n"
        report += (
            f"- **Vector Service Enabled**: {health_status.get('enabled', False)}\n"
        )
        report += f"- **Initialized**: {health_status.get('initialized', False)}\n"

        # Collection statistics
        if health_status.get("stats"):
            report += "\n## Collections\n"
            for collection_name, coll_stats in health_status["stats"].items():
                report += f"\n### {collection_name}\n"
                report += f"- **Points**: {coll_stats.get('points_count', 0)}\n"
                report += f"- **Indexed Vectors**: {coll_stats.get('indexed_vectors_count', 0)}\n"

        # Performance metrics
        if health_status.get("performance"):
            perf = health_status["performance"]
            report += "\n## Performance Metrics\n"
            report += f"- **Avg Search Time**: {perf.get('avg_search_ms', 0):.1f}ms\n"
            report += f"- **P95 Search Time**: {perf.get('p95_search_ms', 0):.1f}ms\n"
            report += f"- **P99 Search Time**: {perf.get('p99_search_ms', 0):.1f}ms\n"

        # Configuration
        report += "\n## Configuration\n"
        report += f"- **Qdrant**: {health_status.get('qdrant_host', 'Unknown')}\n"
        report += f"- **Ollama**: {health_status.get('ollama_host', 'Unknown')}\n"
        report += f"- **Embedding Model**: {health_status.get('embedding_model', 'Unknown')}\n"
        report += f"- **Embedding Dimension**: {health_status.get('embedding_dimension', 'Unknown')}\n"
        report += f"- **Similarity Threshold**: {health_status.get('similarity_threshold', 'Unknown')}\n"

        # Recommendations
        report += "\n## Recommendations\n"
        recommendations = []

        if health_status.get("status") != "healthy":
            recommendations.append(
                f"- ⚠️ **Service status is '{health_status.get('status')}'. Check logs for details.**"
            )

        if not health_status.get("initialized"):
            recommendations.append(
                "- ⚠️ **Vector service not initialized. Check startup logs.**"
            )

        if health_status.get("stats"):
            total_points = sum(
                c.get("points_count", 0) for c in health_status["stats"].values()
            )
            if total_points == 0:
                recommendations.append(
                    "- ℹ️ **No vectors indexed yet. Run initial indexing to populate collections.**"
                )

        if health_status.get("performance"):
            perf = health_status["performance"]
            if perf.get("p99_search_ms", 0) > 100:
                recommendations.append(
                    "- ⚠️ **Search latency is high. Consider optimizing collection or Qdrant configuration.**"
                )

        if not recommendations:
            recommendations.append(
                "- ✅ **Vector layer is healthy and performing within acceptable parameters.**"
            )

        report += "\n".join(recommendations)

        return report

    except Exception as e:
        logger.error(f"Vector health check failed: {e}")
        raise Exception(f"Vector health check failed: {e}")


@mcp.tool()
@auto_capture_tool
async def store_prompt_embedding(
    prompt_text: str,
    session_id: str = "unknown",
    metadata: dict[str, Any] | None = None,
) -> str:
    """Store prompt embedding in Qdrant for future similarity searches."""
    logger.info(
        f"Storing prompt embedding (session={session_id}, length={len(prompt_text)})"
    )

    try:
        vector_service = get_vector_service()

        if not vector_service:
            return "❌ Vector service unavailable - prompt not stored"

        # Prepare metadata
        prompt_metadata = {
            "session_id": session_id,
            "char_count": len(prompt_text),
            "timestamp": datetime.utcnow().isoformat(),
            **(metadata or {}),
        }

        # Store embedding
        result = await vector_service.store_prompt_embedding(
            text=prompt_text,
            prompt_id=None,  # Auto-generate UUID
            metadata=prompt_metadata,
        )

        if result:
            logger.info(f"Prompt embedding stored: {result}")
            return f"✅ Prompt stored successfully\n\n**UUID**: `{result}`\n**Length**: {len(prompt_text)} chars\n**Session**: {session_id}"
        else:
            return "❌ Failed to store prompt embedding"

    except Exception as e:
        logger.error(f"Store prompt embedding failed: {e}")
        return f"❌ Error storing prompt: {str(e)}"


@mcp.tool()
@auto_capture_tool
async def validate_prompt(prompt_text: str, timeout_ms: int = 1000) -> str:
    """Validate user prompt for conflicts with stored knowledge. See /mnemodrome skill for guidance."""
    start_time = time.time()
    logger.info(f"Validating prompt (timeout={timeout_ms}ms): {prompt_text[:100]}...")

    # Snapshot degraded counter before any LLM calls so we can detect if this
    # specific validate_prompt call triggered a degraded fallback.
    _pre_validate_degraded = (
        llm_service_singleton.extraction_degraded_count if llm_service_singleton else 0
    )

    try:
        # Initialize services
        if not llm_service_singleton:
            return "# Validation Result\n\n**Status**: Service Unavailable\n\nLLM service is not initialized. Cannot validate prompt."

        # Initialize claim extraction service
        claim_extraction_service = ClaimExtractionService(
            llm_service=llm_service_singleton,
            redis_service=redis_service if redis_service.is_connected else None,
        )

        # Step 1: Extract claims with timeout
        async def extract_claims_task():
            return await claim_extraction_service.extract_claims(prompt_text)

        try:
            claims = await asyncio.wait_for(
                extract_claims_task(), timeout=timeout_ms / 1000.0
            )
        except TimeoutError:
            logger.warning(f"Claim extraction timeout after {timeout_ms}ms")
            return f"# Validation Result\n\n**Status**: Timeout\n\nClaim extraction exceeded {timeout_ms}ms timeout. Validation aborted."

        if not claims:
            logger.info("No claims extracted from prompt")
            return "# Validation Result\n\n**Status**: No Conflicts Detected\n\nNo factual claims found in prompt to validate."

        # Step 2: Filter to assertions only
        assertions = [claim for claim in claims if claim.claim_type == "assertion"]
        if not assertions:
            logger.info(f"No assertions found among {len(claims)} claims")
            return f"# Validation Result\n\n**Status**: No Conflicts Detected\n\nNo assertions found ({len(claims)} non-assertion claims detected)."

        logger.debug(
            f"Filtered {len(assertions)} assertions from {len(claims)} total claims"
        )

        # Step 3: Extract entities from assertions
        try:
            entities = await claim_extraction_service.extract_entities_from_claims(
                assertions
            )
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            entities = []

        if not entities:
            logger.info("No entities extracted from claims")
            return "# Validation Result\n\n**Status**: No Conflicts Detected\n\nNo named entities found in assertions to validate against."

        # Step 4: Query Neo4j for stored facts
        try:
            fact_retrieval = Neo4jFactRetrievalService(neo4j_service)
            stored_facts = await fact_retrieval.get_facts_for_entities(entities)
        except Exception as e:
            logger.warning(f"Fact retrieval failed (graceful degradation): {e}")
            stored_facts = []

        if not stored_facts:
            logger.info(f"No stored facts found for {len(entities)} entities")
            return f"# Validation Result\n\n**Status**: No Conflicts Detected\n\nNo stored facts found for extracted entities ({len(entities)} entities)."

        # Step 5: Validate claims against stored facts
        conflict_validation = ConflictValidationService()
        conflicts = []

        for assertion in assertions:
            try:
                result = await conflict_validation.validate_claim(
                    assertion, stored_facts
                )
                if result and result.conflict_detected:
                    conflicts.append(result)
                    logger.debug(f"Conflict detected: severity={result.severity}")
            except Exception as e:
                logger.warning(f"Validation check failed: {e}")
                continue

        # Step 6: Format results as markdown
        elapsed_ms = (time.time() - start_time) * 1000

        # Degraded-mode warning: if any LLM call fell back during this
        # validation run, surface it in the returned payload.
        _vp_degraded_note = ""
        if (
            llm_service_singleton
            and llm_service_singleton.extraction_degraded_count > _pre_validate_degraded
        ):
            _vp_degraded_note = (
                "\n\n> ⚠️ **WARNING: Degraded Extraction** — The LLM returned "
                "empty or unparseable output during this validation run. Claim "
                "extraction or entity classification may be incomplete, so "
                "results above may miss conflicts. Search logs for "
                "`DEGRADED extraction quality` to diagnose. "
                "Fix: use a non-reasoning model, set LLM_DISABLE_THINKING=true, "
                "or raise LLM_MAX_TOKENS.\n"
            )

        if not conflicts:
            return f"""# Validation Result

**Status**: No conflicts detected

- **Claims analyzed**: {len(assertions)}
- **Entities checked**: {len(entities)}
- **Stored facts searched**: {len(stored_facts)}
- **Validation time**: {elapsed_ms:.1f}ms

All assertions in your prompt are consistent with stored knowledge.{_vp_degraded_note}"""

        # Format conflicts
        markdown = f"""# Validation Result

**Status**: {len(conflicts)} conflict(s) detected

**Summary**:
- **Total assertions**: {len(assertions)}
- **Entities checked**: {len(entities)}
- **Conflicts found**: {len(conflicts)}
- **Validation time**: {elapsed_ms:.1f}ms

## Conflicts

"""

        for i, conflict in enumerate(conflicts, 1):
            claim_desc = (
                f"{conflict.claim.entity1} → {conflict.claim.relation} → {conflict.claim.entity2}"
                if conflict.claim
                else "N/A"
            )
            fact_desc = (
                f"{conflict.stored_fact.entity1_name} → {conflict.stored_fact.relation} → {conflict.stored_fact.entity2_name}"
                if conflict.stored_fact
                else "N/A"
            )

            markdown += f"""### Conflict {i}

- **Type**: {conflict.conflict_type}
- **Severity**: {conflict.severity}/100
- **Claim**: {claim_desc}
- **Stored fact**: {fact_desc}
- **Recommendation**: {conflict.recommendation}

"""

        return markdown + _vp_degraded_note

    except Exception as e:
        logger.error(f"Prompt validation failed: {e}")
        raise Exception(f"Prompt validation failed: {e}")


# ========================================
# Feature 028: Reasoning-Time Knowledge Validation
# ========================================


def _format_reasoning_validation_result(result: ReasoningValidationResult) -> str:
    """
    Format reasoning validation result as markdown.

    Args:
        result: ReasoningValidationResult to format

    Returns:
        Markdown string for display
    """
    markdown = f"""# Reasoning Validation Result

**Status**: {result.validation_status}
**Elapsed**: {result.elapsed_ms:.0f}ms

## Summary
{result.summary}

## Conflicts Detected ({len(result.conflicts)})
"""

    for i, conflict in enumerate(result.conflicts, 1):
        markdown += f"""### Conflict {i}
- **Type**: {conflict.conflict_type}
- **Severity**: {conflict.severity}/100
- **Claim**: {conflict.claim}
- **Stored Fact**: {conflict.stored_fact}
- **Recommendation**: {conflict.recommendation}
"""
        if conflict.correction_suggestion:
            markdown += f"- **Correction**: {conflict.correction_suggestion}\n"
        markdown += "\n"

    markdown += f"""## Relevant Learnings ({len(result.relevant_learnings)})
"""
    for learning in result.relevant_learnings:
        created_str = learning.created_at.date() if learning.created_at else "unknown"
        markdown += f"- [{learning.uuid}] {learning.title} ({created_str}, similarity: {learning.similarity_score:.2f})\n"

    markdown += f"""
## Relevant Patterns ({len(result.relevant_patterns)})
"""
    for pattern in result.relevant_patterns:
        markdown += f"- [{pattern.pattern_uuid}] {pattern.name} (success_rate: {pattern.success_rate:.0%}, {pattern.occurrence_count} occurrences)\n"

    markdown += f"""
## Diagnostics
- Claims analyzed: {result.claims_analyzed}
- Entities checked: {result.entities_checked}
- Degraded queries: {result.degraded_queries if result.degraded_queries else "[]"}
"""

    return markdown


@mcp.tool()
@auto_capture_tool
async def validate_reasoning(
    proposed_response: str,
    timeout_ms: int = 1000,
) -> str:
    """Validate proposed LLM response against stored knowledge before delivery. See /mnemodrome skill for guidance."""
    logger.info(f"validate_reasoning called (timeout={timeout_ms}ms)")

    try:
        if not llm_service_singleton:
            return "# Reasoning Validation\n\n**Status**: Service Unavailable\n\nLLM service is not initialized. Cannot validate reasoning."

        vector_service = get_vector_service()
        service = ReasoningValidationService(
            llm_service=llm_service_singleton,
            neo4j_service=neo4j_service,
            vector_service=vector_service,
            redis_service=redis_service if redis_service.is_connected else None,
        )
        result = await service.validate_response(
            proposed_response, timeout_ms=timeout_ms
        )

        # Feature 034: Adjust confidence on relationships involved in conflicts
        if ACTR_ENABLED and result.conflicts:
            try:
                for conflict in result.conflicts:
                    rel_uuid = getattr(conflict, "relationship_uuid", None)
                    if rel_uuid:
                        is_supporting = conflict.conflict_type == "supporting"
                        evidence_strength = (
                            max(getattr(conflict, "severity", 50), 1) / 50.0
                        )
                        _fire_and_forget(
                            activation_service.update_confidence(
                                relationship_uuid=rel_uuid,
                                evidence_strength=evidence_strength,
                                is_supporting=is_supporting,
                            ),
                            description=f"confidence update for {rel_uuid[:8]}",
                        )
            except Exception as e:
                logger.warning(
                    f"Confidence adjustment in validate_reasoning failed (FR-008): {e}"
                )

        return _format_reasoning_validation_result(result)

    except Exception as e:
        logger.error(f"validate_reasoning failed: {e}")
        raise Exception(f"Reasoning validation failed: {e}")


@mcp.custom_route("/validate-reasoning", methods=["POST"])
async def handle_validate_reasoning_http(request: Request) -> Response:
    """Feature 028: Reasoning-time validation HTTP endpoint.

    POST endpoint bridging TypeScript hooks to ReasoningValidationService.
    Returns structured JSON (result.model_dump()) for hook consumption.

    Request body:
        proposed_response (str, required): Response text to validate
        timeout_ms (int, optional): Validation timeout in ms (default: 1000)

    Headers:
        Authorization: Bearer <MNEMODROME_HOOK_TOKEN>  (only if token configured)

    Returns:
        200 application/json: ReasoningValidationResult as JSON
        400 application/json: {"error": "proposed_response required"}
        401 application/json: {"error": "Unauthorized"}
        503 application/json: {"error": "Service unavailable"}
    """
    # Optional auth check
    if MNEMODROME_HOOK_TOKEN:
        auth = request.headers.get("Authorization", "")
        if not secrets.compare_digest(auth, f"Bearer {MNEMODROME_HOOK_TOKEN}"):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    proposed_response = body.get("proposed_response", "").strip()
    if not proposed_response:
        return JSONResponse({"error": "proposed_response required"}, status_code=400)

    timeout_ms = body.get("timeout_ms", 1000)

    try:
        if not llm_service_singleton:
            return JSONResponse({"error": "Service unavailable"}, status_code=503)

        vector_service = get_vector_service()
        service = ReasoningValidationService(
            llm_service=llm_service_singleton,
            neo4j_service=neo4j_service,
            vector_service=vector_service,
            redis_service=redis_service if redis_service.is_connected else None,
        )
        result = await service.validate_response(
            proposed_response, timeout_ms=timeout_ms
        )
        return JSONResponse(result.model_dump())

    except Exception as e:
        logger.error(f"validate_reasoning HTTP endpoint failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/validate-prompt", methods=["POST"])
async def handle_validate_prompt_http(request: Request) -> Response:
    """Feature 025: Prompt-time knowledge validation HTTP endpoint.

    POST endpoint bridging TypeScript UserPromptSubmit hooks to the same
    claim-extraction / conflict-validation pipeline used by the
    ``validate_prompt`` MCP tool.  Returns structured JSON so hook clients
    can filter conflicts by severity and inject a systemMessage.

    Request body:
        prompt_text (str, required): User prompt text to validate
        timeout_ms (int, optional): Claim-extraction timeout in ms (default: 1000)

    Headers:
        Authorization: Bearer <MNEMODROME_HOOK_TOKEN>  (only if token configured)

    Returns:
        200 application/json: PromptValidationResult JSON::

            {
                "validation_status": "clean" | "conflicts_detected" | "no_claims" | "timeout" | "error",
                "conflict_count": <int>,
                "conflicts": [
                    {
                        "conflict_type": <str>,
                        "severity": <int 0-100>,
                        "claim": <str>,
                        "stored_fact": <str>,
                        "recommendation": <str>
                    },
                    ...
                ],
                "summary": <str>,
                "elapsed_ms": <float>
            }

        400 application/json: {"error": "prompt_text required"}
        401 application/json: {"error": "Unauthorized"}
        503 application/json: {"error": "Service unavailable"}
    """
    # Optional auth check — identical to /validate-reasoning
    if MNEMODROME_HOOK_TOKEN:
        auth = request.headers.get("Authorization", "")
        if not secrets.compare_digest(auth, f"Bearer {MNEMODROME_HOOK_TOKEN}"):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    prompt_text = body.get("prompt_text", "").strip()
    if not prompt_text:
        return JSONResponse({"error": "prompt_text required"}, status_code=400)

    timeout_ms = body.get("timeout_ms", 1000)

    start_time = time.time()

    try:
        if not llm_service_singleton:
            return JSONResponse({"error": "Service unavailable"}, status_code=503)

        # Step 1: Extract claims with timeout (mirrors validate_prompt MCP tool)
        claim_extraction_service = ClaimExtractionService(
            llm_service=llm_service_singleton,
            redis_service=redis_service if redis_service.is_connected else None,
        )

        try:
            claims = await asyncio.wait_for(
                claim_extraction_service.extract_claims(prompt_text),
                timeout=timeout_ms / 1000.0,
            )
        except TimeoutError:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.warning(
                f"Claim extraction timeout after {timeout_ms}ms in /validate-prompt"
            )
            return JSONResponse(
                {
                    "validation_status": "timeout",
                    "conflict_count": 0,
                    "conflicts": [],
                    "summary": f"Claim extraction exceeded {timeout_ms}ms timeout.",
                    "elapsed_ms": elapsed_ms,
                }
            )

        # Step 2: Filter to assertions only
        assertions = [claim for claim in claims if claim.claim_type == "assertion"]
        if not assertions:
            elapsed_ms = (time.time() - start_time) * 1000
            return JSONResponse(
                {
                    "validation_status": "no_claims",
                    "conflict_count": 0,
                    "conflicts": [],
                    "summary": f"No assertions found ({len(claims)} non-assertion claims).",
                    "elapsed_ms": elapsed_ms,
                }
            )

        # Step 3: Extract entities from assertions
        try:
            entities = await claim_extraction_service.extract_entities_from_claims(
                assertions
            )
        except Exception as e:
            logger.warning(f"Entity extraction failed in /validate-prompt: {e}")
            entities = []

        if not entities:
            elapsed_ms = (time.time() - start_time) * 1000
            return JSONResponse(
                {
                    "validation_status": "no_claims",
                    "conflict_count": 0,
                    "conflicts": [],
                    "summary": "No named entities found in assertions to validate against.",
                    "elapsed_ms": elapsed_ms,
                }
            )

        # Step 4: Query Neo4j for stored facts
        try:
            fact_retrieval = Neo4jFactRetrievalService(neo4j_service)
            stored_facts = await fact_retrieval.get_facts_for_entities(entities)
        except Exception as e:
            logger.warning(
                f"Fact retrieval failed in /validate-prompt (graceful degradation): {e}"
            )
            stored_facts = []

        if not stored_facts:
            elapsed_ms = (time.time() - start_time) * 1000
            return JSONResponse(
                {
                    "validation_status": "no_claims",
                    "conflict_count": 0,
                    "conflicts": [],
                    "summary": f"No stored facts found for {len(entities)} entities.",
                    "elapsed_ms": elapsed_ms,
                }
            )

        # Step 5: Validate claims against stored facts
        conflict_validation = ConflictValidationService()
        conflicts_json = []

        for assertion in assertions:
            try:
                result = await conflict_validation.validate_claim(
                    assertion, stored_facts
                )
                if result and result.conflict_detected:
                    claim_desc = (
                        f"{result.claim.entity1} → {result.claim.relation} → {result.claim.entity2}"
                        if result.claim
                        else "N/A"
                    )
                    fact_desc = (
                        f"{result.stored_fact.entity1_name} → {result.stored_fact.relation} → {result.stored_fact.entity2_name}"
                        if result.stored_fact
                        else "N/A"
                    )
                    conflicts_json.append(
                        {
                            "conflict_type": result.conflict_type or "unknown",
                            "severity": result.severity,
                            "claim": claim_desc,
                            "stored_fact": fact_desc,
                            "recommendation": result.recommendation,
                        }
                    )
            except Exception as e:
                logger.warning(f"Conflict check failed in /validate-prompt: {e}")
                continue

        elapsed_ms = (time.time() - start_time) * 1000

        if conflicts_json:
            return JSONResponse(
                {
                    "validation_status": "conflicts_detected",
                    "conflict_count": len(conflicts_json),
                    "conflicts": conflicts_json,
                    "summary": (
                        f"{len(conflicts_json)} conflict(s) detected in {len(assertions)} assertion(s) "
                        f"across {len(entities)} entities in {elapsed_ms:.0f}ms."
                    ),
                    "elapsed_ms": elapsed_ms,
                }
            )

        return JSONResponse(
            {
                "validation_status": "clean",
                "conflict_count": 0,
                "conflicts": [],
                "summary": (
                    f"All {len(assertions)} assertion(s) consistent with stored knowledge "
                    f"({len(stored_facts)} facts checked in {elapsed_ms:.0f}ms)."
                ),
                "elapsed_ms": elapsed_ms,
            }
        )

    except Exception as e:
        logger.error(f"validate_prompt HTTP endpoint failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.tool()
@auto_capture_tool
async def auto_correct_response(response_text: str, timeout_ms: int = 1000) -> str:
    """Auto-correct LLM response based on stored knowledge. See /mnemodrome skill for guidance."""
    start_time = time.time()
    logger.info(
        f"Auto-correcting response (timeout={timeout_ms}ms): {response_text[:100]}..."
    )

    try:
        # Initialize services
        if not llm_service_singleton:
            return f"""# Auto-Correction Result

**Status**: Service Unavailable

LLM service is not initialized. Cannot correct response.

**Original Response**:
{response_text}"""

        # Initialize claim extraction service
        claim_extraction_service = ClaimExtractionService(
            llm_service=llm_service_singleton,
            redis_service=redis_service if redis_service.is_connected else None,
        )

        # Step 1: Extract claims with timeout
        async def extract_claims_task():
            return await claim_extraction_service.extract_claims(response_text)

        try:
            claims = await asyncio.wait_for(
                extract_claims_task(), timeout=timeout_ms / 1000.0
            )
        except TimeoutError:
            logger.warning(f"Claim extraction timeout after {timeout_ms}ms")
            return f"""# Auto-Correction Result

**Status**: Timeout

Claim extraction exceeded {timeout_ms}ms timeout. No corrections applied.

**Original Response**:
{response_text}"""

        if not claims:
            logger.info("No claims extracted from response")
            return f"""# Auto-Correction Result

**Status**: No Corrections Needed

No factual claims found in response to validate.

**Original Response**:
{response_text}"""

        # Step 2: Filter to assertions
        assertions = [claim for claim in claims if claim.claim_type == "assertion"]
        if not assertions:
            logger.info(f"No assertions found among {len(claims)} claims")
            return f"""# Auto-Correction Result

**Status**: No Corrections Needed

No assertions found ({len(claims)} non-assertion claims detected).

**Original Response**:
{response_text}"""

        logger.debug(
            f"Filtered {len(assertions)} assertions from {len(claims)} total claims"
        )

        # Step 3: Extract entities and query Neo4j
        try:
            entities = await claim_extraction_service.extract_entities_from_claims(
                assertions
            )
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            entities = []

        if not entities:
            logger.info("No entities extracted")
            return f"""# Auto-Correction Result

**Status**: No Corrections Needed

No named entities found in assertions.

**Original Response**:
{response_text}"""

        # Query Neo4j for facts
        try:
            fact_retrieval = Neo4jFactRetrievalService(neo4j_service)
            stored_facts = await fact_retrieval.get_facts_for_entities(entities)
        except Exception as e:
            logger.warning(f"Fact retrieval failed: {e}")
            stored_facts = []

        if not stored_facts:
            logger.info(f"No stored facts for {len(entities)} entities")
            return f"""# Auto-Correction Result

**Status**: No Corrections Needed

No stored facts found for validation.

**Original Response**:
{response_text}"""

        # Step 4: Validate claims
        conflict_validation = ConflictValidationService()
        conflicts = []

        for assertion in assertions:
            try:
                result = await conflict_validation.validate_claim(
                    assertion, stored_facts
                )
                if result and result.conflict_detected:
                    conflicts.append(result)
            except Exception as e:
                logger.warning(f"Validation failed: {e}")
                continue

        # Step 5: Apply corrections
        auto_correction = AutoCorrectionService(event_log_service=event_log_service)

        # Step 6: Return corrected text with audit log
        elapsed_ms = (time.time() - start_time) * 1000

        if not conflicts:
            return f"""# Auto-Correction Result

**Status**: No Corrections Needed

- **Claims analyzed**: {len(assertions)}
- **Entities checked**: {len(entities)}
- **Stored facts searched**: {len(stored_facts)}
- **Conflicts detected**: 0
- **Processing time**: {elapsed_ms:.1f}ms

**Original Response**:
{response_text}"""

        try:
            correction_result = await auto_correction.correct_text(
                original_text=response_text, conflicts=conflicts
            )
        except Exception as e:
            logger.warning(f"Correction application failed (graceful degradation): {e}")
            return f"""# Auto-Correction Result

**Status**: Correction Failed

Error during correction: {str(e)}

**Original Response**:
{response_text}"""

        # Format markdown result
        if not correction_result.corrections_applied:
            markdown = f"""# Auto-Correction Result

**Status**: {correction_result.action.capitalize()}

**Audit Log**:
- **Claims analyzed**: {len(assertions)}
- **Conflicts detected**: {len(conflicts)}
- **Corrections applied**: 0
- **Processing time**: {elapsed_ms:.1f}ms

**Original Response**:
{response_text}"""

            if correction_result.warning_message:
                markdown = f"""# Auto-Correction Result

**Status**: Warning Generated

{correction_result.warning_message}

**Audit Log**:
- **Claims analyzed**: {len(assertions)}
- **Conflicts detected**: {len(conflicts)}
- **Corrections applied**: 0
- **Processing time**: {elapsed_ms:.1f}ms

**Original Response**:
{response_text}"""

            return markdown

        markdown = f"""# Auto-Correction Result

**Status**: {len(correction_result.corrections_applied)} correction(s) applied

**Audit Log**:
- **Claims analyzed**: {len(assertions)}
- **Conflicts detected**: {len(conflicts)}
- **Corrections applied**: {len(correction_result.corrections_applied)}
- **Processing time**: {elapsed_ms:.1f}ms

## Corrections Applied

"""

        for i, correction in enumerate(correction_result.corrections_applied, 1):
            markdown += f"""### Correction {i}

- **Severity**: {correction.get("severity", "N/A")}/100
- **Conflict Type**: {correction.get("reason", "N/A")}
- **Original**: `{correction.get("original", "N/A")}`
- **Corrected**: `{correction.get("corrected", "N/A")}`
- **Confidence**: {correction.get("confidence", "N/A")}

"""

        if correction_result.warning_message:
            markdown += f"""## Warnings

{correction_result.warning_message}

"""

        markdown += f"""## Corrected Response

{correction_result.modified_text}

---

**Original Response** (for reference):
{response_text}"""

        return markdown

    except Exception as e:
        logger.error(f"Response auto-correction failed: {e}")
        raise Exception(f"Response auto-correction failed: {e}")


# ========================================
# Feature 026: Cross-Session Pattern Recognition Tools
# ========================================


@mcp.tool()
@auto_capture_tool
async def extract_patterns_from_recent_sessions(
    days_back: int = 30, min_sessions: int = 3
) -> str:
    """Extract recurring patterns from recent sessions. See /mnemodrome skill for guidance."""
    start_time = time.perf_counter()

    try:
        # Validate inputs
        if days_back < 1:
            return "**Error**: days_back must be >= 1"

        if min_sessions < 1:
            return "**Error**: min_sessions must be >= 1"

        logger.info(
            f"Extracting patterns from last {days_back} days "
            f"(min_sessions={min_sessions})"
        )

        # Step 1: Retrieve recent sessions from Neo4j
        from datetime import timedelta

        cutoff_date = (datetime.now(UTC) - timedelta(days=days_back)).isoformat()

        session_cypher = """
        MATCH (s:Session)
        WHERE s.created_at >= datetime($cutoff_date)
        RETURN {
            uuid: s.uuid,
            conversation_summary: s.conversation_summary,
            key_insights: s.key_insights
        } as session
        ORDER BY s.created_at DESC
        """

        session_results = await neo4j_service.query(
            session_cypher, {"cutoff_date": cutoff_date}
        )

        if not session_results:
            logger.info(f"No sessions found in last {days_back} days")
            return f"""# Pattern Extraction Results

**Status**: No sessions found

No sessions discovered in the last {days_back} days for pattern analysis.

**Configuration**:
- Days analyzed: {days_back}
- Minimum sessions required: {min_sessions}
"""

        # Extract session data
        sessions = [row.get("session") for row in session_results if row.get("session")]
        logger.debug(f"Retrieved {len(sessions)} sessions for analysis")

        if len(sessions) < min_sessions:
            logger.info(
                f"Found {len(sessions)} sessions, but minimum is {min_sessions}"
            )
            return f"""# Pattern Extraction Results

**Status**: Insufficient sessions

Found {len(sessions)} sessions, but minimum required is {min_sessions}.

**Recommendations**:
- Increase `days_back` parameter to analyze more history
- Lower `min_sessions` requirement
"""

        # Step 2: Extract patterns using PatternExtractionService
        extraction_service = PatternExtractionService()
        extracted_patterns = await extraction_service.extract_patterns_from_sessions(
            sessions
        )

        if not extracted_patterns:
            logger.info("No patterns extracted from sessions")
            return f"""# Pattern Extraction Results

**Status**: No patterns found

Analyzed {len(sessions)} sessions but no clear patterns emerged.

**Reasons**:
- Sessions may lack structured problem/solution statements
- Pattern confidence scores were too low (< 0.5 threshold)
- Insufficient detail in session summaries
"""

        # Step 3: Store patterns in Neo4j
        storage_service = PatternStorageService()
        stored_patterns = []
        store_errors = []

        for pattern in extracted_patterns:
            try:
                stored = await storage_service.store_pattern(
                    pattern=pattern, session_uuid=pattern.session_uuid or "unknown"
                )
                if stored:
                    stored_patterns.append(stored)
            except Exception as e:
                logger.warning(f"Failed to store pattern: {e}")
                store_errors.append(str(e))
                continue

        logger.info(
            f"Extracted {len(extracted_patterns)} patterns, "
            f"stored {len(stored_patterns)}"
        )

        # Step 4: Format results
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        markdown = f"""# Pattern Extraction Results

**Status**: Success

**Summary**:
- Sessions analyzed: {len(sessions)}
- Patterns extracted: {len(extracted_patterns)}
- Patterns stored: {len(stored_patterns)}
- Processing time: {elapsed_ms:.1f}ms

## Extracted Patterns

"""

        if stored_patterns:
            # Group by pattern type
            by_type = {}
            for pattern in stored_patterns:
                ptype = pattern.pattern_type
                if ptype not in by_type:
                    by_type[ptype] = []
                by_type[ptype].append(pattern)

            for ptype in sorted(by_type.keys()):
                patterns = by_type[ptype]
                markdown += f"""### {ptype.title()} ({len(patterns)} patterns)

"""
                for i, pattern in enumerate(patterns, 1):
                    markdown += f"""#### Pattern {i}: {pattern.name}

- **UUID**: `{pattern.pattern_uuid}`
- **Problem**: {pattern.problem[:200]}...
- **Solution**: {pattern.solution[:200]}...
- **Confidence**: {pattern.confidence:.0%}
- **Occurrence Count**: {pattern.occurrence_count}
- **Success Rate**: {pattern.success_rate:.0%}

"""
        else:
            markdown += "No patterns could be stored. Check logs for details.\n\n"

        if store_errors:
            markdown += f"""## Storage Errors

- {chr(10).join(f"- {err}" for err in store_errors[:5])}

"""

        return markdown

    except Exception as e:
        logger.error(f"Pattern extraction failed: {e}")
        raise Exception(f"Pattern extraction failed: {e}")


@mcp.tool()
@auto_capture_tool
async def query_patterns(
    problem_type: str = "", min_success_rate: float = 0.0, limit: int = 10
) -> str:
    """Query stored patterns with optional filters. See /mnemodrome skill for guidance."""
    start_time = time.perf_counter()

    try:
        # Validate inputs
        if not 0.0 <= min_success_rate <= 1.0:
            return "**Error**: min_success_rate must be between 0.0 and 1.0"

        if limit < 1 or limit > 100:
            return "**Error**: limit must be between 1 and 100"

        logger.info(
            f"Querying patterns: type='{problem_type}', "
            f"min_success_rate={min_success_rate:.0%}, limit={limit}"
        )

        # Query patterns
        query_service = PatternQueryService()
        patterns = await query_service.query_patterns(
            problem_type=problem_type,
            min_success_rate=min_success_rate,
            limit=limit,
            exclude_archived=True,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Format results
        markdown = f"""# Pattern Query Results

**Query Parameters**:
- Pattern type: {problem_type if problem_type else "all"}
- Minimum success rate: {min_success_rate:.0%}
- Limit: {limit}
- Processing time: {elapsed_ms:.1f}ms

**Results**: {len(patterns)} pattern(s) found

"""

        if not patterns:
            markdown += """No patterns match the specified filters.

**Suggestions**:
- Reduce `min_success_rate` threshold
- Remove or broaden `problem_type` filter
- Run pattern extraction first to build pattern library
"""
            return markdown

        # Format pattern results
        markdown += "## Patterns\n\n"

        for i, pattern in enumerate(patterns, 1):
            markdown += f"""### {i}. {pattern.name}

**Metadata**:
- **UUID**: `{pattern.pattern_uuid}`
- **Type**: {pattern.pattern_type}
- **Success Rate**: {pattern.success_rate:.0%}
- **Occurrences**: {pattern.occurrence_count}
- **Confidence**: {pattern.confidence:.0%}

**Problem**:
{pattern.problem}

**Solution**:
{pattern.solution}

**Stats**:
- Successful: {pattern.success_count}/{pattern.occurrence_count}
- Failed: {pattern.failure_count}
- Consecutive failures: {pattern.consecutive_failures}
- First seen: {pattern.first_occurrence_date.isoformat() if pattern.first_occurrence_date else "N/A"}
- Last seen: {pattern.latest_occurrence_date.isoformat() if pattern.latest_occurrence_date else "N/A"}

"""

        # Summary statistics
        avg_success = sum(p.success_rate for p in patterns) / len(patterns)
        total_occurrences = sum(p.occurrence_count for p in patterns)

        markdown += f"""## Summary Statistics

- **Average success rate**: {avg_success:.0%}
- **Total occurrences**: {total_occurrences}
- **Patterns with 100% success**: {sum(1 for p in patterns if p.success_rate == 1.0)}
- **Patterns with high confidence** (>= 0.9): {sum(1 for p in patterns if p.confidence >= 0.9)}

"""

        return markdown

    except Exception as e:
        logger.error(f"Pattern query failed: {e}")
        raise Exception(f"Pattern query failed: {e}")


# ========================================
# Feature 031: Procedural Memory Layer
# ========================================


@mcp.tool()
async def capture_procedure(
    symptom: str,
    diagnostic_steps: str,
    root_cause: str,
    fix: str,
    verification: str,
    problem_type: str = "general",
    technologies: str = "",
    outcome: str = "SUCCESS",
    pattern_uuid: str = "",
) -> str:
    """Capture structured diagnostic procedure in procedural memory. See /mnemodrome skill for guidance."""
    try:
        start_time = time.time()
        logger.info(f"Capturing procedure: {symptom[:60]}... (type={problem_type})")

        # Map "general" to "other" for model compatibility
        if problem_type == "general":
            problem_type = "other"

        # Parse diagnostic_steps from newline-separated string
        steps = [
            s.strip().lstrip("0123456789.-) ")
            for s in diagnostic_steps.strip().split("\n")
            if s.strip()
        ]
        if not steps:
            return "Error: At least one diagnostic step is required."

        # Parse technologies from comma-separated string
        tech_list = (
            [t.strip() for t in technologies.split(",") if t.strip()]
            if technologies
            else []
        )

        # Validate via Pydantic
        try:
            proc = Procedure(
                problem_type=problem_type,
                symptom=symptom,
                diagnostic_steps=steps,
                root_cause=root_cause,
                fix=fix,
                verification=verification,
                outcome=outcome,
                technologies=tech_list,
                pattern_uuid=pattern_uuid or None,
            )
        except Exception as e:
            logger.warning(f"Procedure validation failed: {e}")
            return f"Validation error: {e}"

        # Core write: dedup check + Neo4j node (synchronous)
        try:
            proc_uuid, is_new = await procedure_service.create_procedure(proc)
        except Exception as e:
            logger.error(f"Procedure creation failed: {e}")
            return f"Error creating procedure: {e}"

        if not is_new:
            capture_time = (time.time() - start_time) * 1000
            return f"""Duplicate procedure detected

**Existing UUID**: `{proc_uuid}`
**Similarity**: >= {0.92} (dedup threshold)
**Problem Type**: {problem_type}
**Time**: {capture_time:.0f}ms

The existing procedure matches your symptom and root cause. Use `find_procedure` to view it."""

        # Fire-and-forget: enrichment runs in background
        _fire_and_forget(
            procedure_service.enrich_procedure(proc, session_uuid=None),
            description=f"procedure enrichment: {symptom[:50]}",
        )

        capture_time = (time.time() - start_time) * 1000
        logger.info(f"Procedure captured in {capture_time:.1f}ms: {proc_uuid}")

        return f"""Procedure captured successfully

**UUID**: `{proc_uuid}`
**Problem Type**: {problem_type}
**Outcome**: {outcome}
**Technologies**: {", ".join(tech_list) if tech_list else "N/A"}
**Diagnostic Steps**: {len(steps)}
**Capture Time**: {capture_time:.0f}ms

Enrichment (embedding, markdown, entities) running in background."""

    except Exception as e:
        logger.error(f"Unexpected error in capture_procedure: {e}", exc_info=True)
        return f"Error capturing procedure: {e}"


@mcp.tool()
async def find_procedure(
    problem_description: str,
    problem_type: str = "",
    min_success_rate: float = 0.0,
    limit: int = 5,
) -> str:
    """Search procedural memory for matching diagnostic procedures. See /mnemodrome skill for guidance."""
    try:
        start_time = time.time()
        logger.info(f"Searching procedures: {problem_description[:60]}...")

        try:
            matches = await procedure_service.find_procedures(
                query_text=problem_description,
                problem_type=problem_type or None,
                min_success_rate=min_success_rate,
                limit=limit,
            )
        except Exception as e:
            logger.error(f"Procedure search failed: {e}")
            return f"Error searching procedures: {e}"

        search_time = (time.time() - start_time) * 1000

        if not matches:
            return f"""No matching procedures found

**Query**: {problem_description[:100]}
**Problem Type Filter**: {problem_type or "any"}
**Search Time**: {search_time:.0f}ms

No stored procedures match this problem description. Use `capture_procedure` to record solutions."""

        # Format results as markdown
        markdown = f"## Matching Procedures ({len(matches)} found)\n\n"
        markdown += f"**Query**: {problem_description[:100]}\n"
        markdown += f"**Search Time**: {search_time:.0f}ms\n\n---\n\n"

        for i, m in enumerate(matches, 1):
            steps_md = "\n".join(
                f"   {j}. {step}" for j, step in enumerate(m.diagnostic_steps, 1)
            )
            techs_md = ", ".join(m.technologies) if m.technologies else "N/A"

            markdown += f"""### {i}. {m.symptom[:80]} (score: {m.similarity_score:.2f})

**UUID**: `{m.procedure_uuid}`
**Problem Type**: {m.problem_type}
**Outcome**: {m.outcome}
**Technologies**: {techs_md}

**Symptom**: {m.symptom}

**Diagnostic Steps**:
{steps_md}

**Root Cause**: {m.root_cause}

**Fix**: {m.fix}

**Verification**: {m.verification}

"""
            if m.linked_pattern_name:
                markdown += f"**Linked Pattern**: {m.linked_pattern_name} (`{m.linked_pattern_uuid}`)\n\n"

            markdown += "---\n\n"

        return markdown

    except Exception as e:
        logger.error(f"Unexpected error in find_procedure: {e}", exc_info=True)
        return f"Error searching procedures: {e}"


@mcp.tool()
@auto_capture_tool
async def suggest_instruction_updates(
    skill_path: str | None = None,
    min_success_rate: float = 0.8,
    min_occurrences: int = 3,
    max_candidates: int = 10,
) -> str:
    """Query high-confidence patterns and suggest instruction updates. See /mnemodrome skill for guidance."""
    if not EVOLUTION_ENABLED:
        return """Evolution disabled (EVOLUTION_ENABLED=false)

Evolution feature is not enabled. Set EVOLUTION_ENABLED=true in .env to enable instruction update suggestions."""

    try:
        start_time = time.time()
        logger.info(
            f"Generating evolution candidates: "
            f"min_success_rate={min_success_rate}, min_occurrences={min_occurrences}"
        )

        # Initialize service
        service = EvolutionCandidateService()

        # Generate candidates
        result = await service.generate_candidates(
            skill_path=skill_path,
            min_success_rate=min_success_rate,
            min_occurrences=min_occurrences,
            max_candidates=max_candidates,
        )

        elapsed_ms = (time.time() - start_time) * 1000

        # Format response
        candidates = result.get("candidates", [])
        skills_analyzed = result.get("skills_analyzed", 0)
        patterns_evaluated = result.get("patterns_evaluated", 0)
        error = result.get("error")

        if error and not candidates:
            return f"""Error generating candidates

**Error**: {error}

Evolution candidates could not be generated. Check service availability."""

        if not candidates:
            return f"""No evolution candidates found

**Skills Analyzed**: {skills_analyzed}
**Patterns Evaluated**: {patterns_evaluated}
**Time**: {elapsed_ms:.0f}ms

All skills are up to date with current patterns. No instruction changes suggested."""

        # Format candidates for display
        markdown = f"## Evolution Candidates ({len(candidates)} found)\n\n"
        markdown += f"**Skills Analyzed**: {skills_analyzed}\n"
        markdown += f"**Patterns Evaluated**: {patterns_evaluated}\n"
        markdown += f"**Time**: {elapsed_ms:.0f}ms\n\n---\n\n"

        for i, c in enumerate(candidates, 1):
            markdown += f"""### {i}. {c["skill_name"]} (confidence: {c["confidence_score"]:.0%})

**UUID**: `{c["uuid"]}`
**Skill Path**: `{c["skill_path"]}`
**Status**: {c["status"]}

**Proposed Change**:
```diff
{c["proposed_diff"][:500]}
```

**Rationale**: {c["rationale"][:200]}...

**Evidence**:
- Patterns: {", ".join(c["pattern_uuids"])}
- Pattern Summary: {c["pattern_summary"]}
- Success Rate: {c["success_rate"]:.0%}
- Occurrences: {c["occurrence_count"]}

---

"""

        return markdown

    except Exception as e:
        logger.error(f"Candidate generation failed: {e}", exc_info=True)
        return f"Error generating candidates: {e}"


# ========================================
# Feature 034: ACT-R Activation Diagnostics
# ========================================


@mcp.tool()
@auto_capture_tool
async def get_activation_diagnostics(
    entity_name: str = "",
    entity_uuid: str = "",
) -> str:
    """Returns activation metrics for specified entities (FR-009). Accepts entity name or UUID."""
    if not entity_name and not entity_uuid:
        return "# Activation Diagnostics\n\nPlease provide either `entity_name` or `entity_uuid`."

    try:
        # Find entity
        if entity_uuid:
            entity_query = """
            MATCH (e {uuid: $uuid})
            RETURN e.uuid AS uuid, e.name AS name, e.entity_type AS entity_type,
                   coalesce(e.access_count, 0) AS access_count,
                   e.last_accessed AS last_accessed, e.created_at AS created_at
            """
            entities = await neo4j_service.query(entity_query, {"uuid": entity_uuid})
        else:
            entity_query = """
            MATCH (e {name: $name})
            RETURN e.uuid AS uuid, e.name AS name, e.entity_type AS entity_type,
                   coalesce(e.access_count, 0) AS access_count,
                   e.last_accessed AS last_accessed, e.created_at AS created_at
            """
            entities = await neo4j_service.query(entity_query, {"name": entity_name})

        if not entities:
            return f"# Activation Diagnostics\n\nEntity not found: {entity_name or entity_uuid}"

        markdown = "# Activation Diagnostics\n\n"

        for entity in entities:
            from datetime import datetime as dt

            last_accessed = entity.get("last_accessed") or entity.get("created_at")
            if last_accessed and not isinstance(last_accessed, dt):
                try:
                    last_accessed = dt.fromisoformat(
                        str(last_accessed).replace("Z", "+00:00")
                    )
                except (ValueError, TypeError):
                    last_accessed = dt.now(UTC)
            elif not last_accessed:
                last_accessed = dt.now(UTC)

            activation = calculate_activation(
                access_count=entity.get("access_count", 0),
                last_accessed=last_accessed,
            )

            markdown += f"""## {entity.get("name", "Unknown")} ({entity.get("entity_type", "Unknown")})

| Property | Value |
|----------|-------|
| UUID | `{entity.get("uuid", "N/A")}` |
| Access Count | {entity.get("access_count", 0)} |
| Last Accessed | {entity.get("last_accessed", "Never")} |
| Created At | {entity.get("created_at", "Unknown")} |
| **Base Activation** | **{activation:.3f}** |

"""
            # D7 (FR-012): scope to native edges only
            rel_query = """
            MATCH (e {uuid: $uuid})-[r:RELATES_TO]-(other)
            WHERE r.invalid_at IS NULL AND r.type IS NOT NULL
            RETURN r.type AS relation_type,
                   other.name AS other_name,
                   coalesce(r.confidence, 'N/A') AS confidence,
                   coalesce(r.hebbian_weight, 0.0) AS hebbian_weight,
                   r.last_coactivation AS last_coactivation
            LIMIT 20
            """
            rels = await neo4j_service.query(rel_query, {"uuid": entity["uuid"]})

            if rels:
                markdown += "### Relationships\n\n"
                markdown += "| Relation | Other Entity | Confidence | Hebbian Weight | Last Co-activation |\n"
                markdown += "|----------|-------------|------------|----------------|--------------------|\n"
                for rel in rels:
                    markdown += (
                        f"| {rel.get('relation_type', '?')} "
                        f"| {rel.get('other_name', '?')} "
                        f"| {rel.get('confidence', 'N/A')} "
                        f"| {rel.get('hebbian_weight', 0.0):.2f} "
                        f"| {rel.get('last_coactivation', 'Never')} |\n"
                    )
                markdown += "\n"

        markdown += f"\n*ACTR_ENABLED: {ACTR_ENABLED}*"
        return markdown

    except Exception as e:
        logger.error(f"Activation diagnostics failed: {e}")
        raise Exception(f"Activation diagnostics failed: {e}")


def shutdown_handler(sig, frame):
    """
    Graceful shutdown on SIGTERM/SIGINT.

    Ensures Neo4j driver is properly closed before process exit.
    """
    signal_name = "SIGTERM" if sig == signal.SIGTERM else "SIGINT"
    logger.info(f"Received {signal_name}, initiating graceful shutdown...")

    try:
        # Close Redis connection
        if redis_service.is_connected:
            logger.info("Closing Redis connection...")
            asyncio.run(redis_service.disconnect())
            logger.info("✅ Redis connection closed")

        # Close Neo4j driver
        logger.info("Closing Neo4j driver...")
        asyncio.run(neo4j_service.close())
        logger.info("✅ Neo4j driver closed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    finally:
        logger.info("Mnemodrome MCP Server - Shutdown complete")
        sys.exit(0)


async def run_admin_server():
    """
    Run Admin Web UI server (FastAPI).

    Feature 013: Dual-server architecture - runs Admin UI alongside MCP server.
    Imported and started by main() when ADMIN_WEB_ENABLED=true.
    """
    import uvicorn

    from src import config
    from src.admin_ui import app

    logger.info(
        f"Starting Admin Web UI on {config.ADMIN_WEB_HOST}:{config.ADMIN_WEB_PORT}..."
    )

    # Run uvicorn server (loopback by default; see ADMIN_WEB_HOST)
    uvicorn_config = uvicorn.Config(
        app, host=config.ADMIN_WEB_HOST, port=config.ADMIN_WEB_PORT, log_level="info"
    )
    server = uvicorn.Server(uvicorn_config)
    await server.serve()


async def main():
    """
    Main entry point - runs both MCP server and Admin UI concurrently.

    Feature 013: Dual-Server Architecture
    - MCP Server: SSE transport on port 8765 (existing functionality)
    - Admin UI: FastAPI HTTP server on port 8083 (new feature)

    Both servers run in same process using asyncio.TaskGroup for:
    - Shared service lifecycle (Neo4j, Redis, Event Log)
    - Graceful concurrent shutdown
    - Resource efficiency (single process)
    """
    from src import config

    logger.info("Initializing Mnemodrome services...")

    # Note: Services are initialized in server_lifespan context manager
    # (not here) to ensure they run in same event loop as FastMCP server

    # Start both servers concurrently
    logger.info("Starting dual-server architecture (MCP + Admin UI)...")

    try:
        async with asyncio.TaskGroup() as tg:
            # MCP Server (configurable transport on port 8765)
            logger.info(
                f"Starting MCP Server on {config.MCP_HOST}:{config.MCP_PORT} (transport={config.MCP_TRANSPORT})..."
            )
            mcp_task = tg.create_task(
                asyncio.to_thread(mcp.run, transport=config.MCP_TRANSPORT)
            )

            # Admin HTTP Server - conditional, non-fatal if port unavailable
            if config.ADMIN_WEB_ENABLED:
                logger.info(f"Admin Web UI enabled on port {config.ADMIN_WEB_PORT}")

                async def safe_admin_server():
                    try:
                        await run_admin_server()
                    except OSError as e:
                        logger.warning(f"⚠️  Admin UI failed to start (non-fatal): {e}")
                        logger.warning("MCP server continues without Admin UI")

                admin_task = tg.create_task(safe_admin_server())
            else:
                logger.info("Admin Web UI disabled (ADMIN_WEB_ENABLED=false)")

    except* Exception as eg:
        # TaskGroup collects exceptions from all tasks
        for exc in eg.exceptions:
            logger.error(f"Server error: {exc}")
        raise


if __name__ == "__main__":
    # Register signal handlers for emergency shutdown
    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)
    logger.info("Signal handlers registered (SIGTERM, SIGINT)")

    # Start dual-server architecture (MCP + Admin UI)
    # Services initialized in lifespan context (same event loop as servers)
    logger.info("Starting Mnemodrome MCP Server with Admin UI...")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
