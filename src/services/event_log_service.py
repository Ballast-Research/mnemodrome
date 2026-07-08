"""Event Log Service for Mnemodrome episodic memory layer.

Supports a selectable backend (SQLite by default, PostgreSQL opt-in):
- Zero-configuration SQLite setup (auto-creates database and schema)
- Lightweight single-file storage (no external service required for SQLite)
- WAL mode for concurrent read performance (SQLite)
- Built-in retention with age-based and size-based policies
- PostgreSQL backend enabled by EVENT_LOG_BACKEND=postgres
"""

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from loguru import logger

# Import config for Phase 2 settings
from src import config


def _create_db_connection():
    """Factory: return the appropriate database connection for the configured backend.

    SQLite path preserves the existing module-level singleton so that all code
    that imported sqlite_connection directly still references the same object.
    PostgreSQL path lazily imports asyncpg (optional dependency) so that the
    default SQLite path never touches it.
    """
    if config.EVENT_LOG_BACKEND == "postgres":
        from src.database.postgresql import PostgreSQLConnection  # noqa: PLC0415

        return PostgreSQLConnection()
    # Default: SQLite singleton (preserves exact existing behaviour)
    from src.database.sqlite import sqlite_connection  # noqa: PLC0415

    return sqlite_connection


# Module-level connection — instantiated once at import time
_db_connection = _create_db_connection()

# Re-export sqlite_connection at module level so that existing patches of
# `src.services.event_log_service.sqlite_connection` continue to work.
# When the postgres backend is active this resolves to the PostgreSQL instance,
# but patches in the SQLite unit tests always run under the default backend.
sqlite_connection = _db_connection


@dataclass
class PerformanceMetrics:
    """Performance metrics for event log operations (Phase 2B)."""

    write_latencies: deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    query_latencies: deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    sla_violations: int = 0
    last_violation_at: datetime | None = None

    def add_write(self, latency_ms: float) -> None:
        """Record write latency."""
        self.write_latencies.append(latency_ms)

    def add_query(self, latency_ms: float) -> None:
        """Record query latency."""
        self.query_latencies.append(latency_ms)

    def get_percentile(self, values: deque[float], percentile: int) -> float:
        """Calculate percentile from deque of values."""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * (percentile / 100.0))
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    def get_p95_write(self) -> float:
        """Get p95 write latency."""
        return self.get_percentile(self.write_latencies, 95)

    def get_p95_query(self) -> float:
        """Get p95 query latency."""
        return self.get_percentile(self.query_latencies, 95)

    def get_p99_write(self) -> float:
        """Get p99 write latency."""
        return self.get_percentile(self.write_latencies, 99)

    def get_p99_query(self) -> float:
        """Get p99 query latency."""
        return self.get_percentile(self.query_latencies, 99)


class EventLogService:
    """Service for SQLite-based event logging (episodic memory layer).

    Uses aiosqlite for async event logging with:
    - Auto-bootstrapping schema (CREATE TABLE IF NOT EXISTS)
    - WAL mode for concurrent reads during writes
    - Zero external dependencies (no database service required)
    """

    def __init__(self):
        """Initialize Event Log Service with the configured backend."""
        self.db = _db_connection
        self.session_id: str | None = None
        self._initialized = False

        # Phase 2B: Performance monitoring
        self._metrics = PerformanceMetrics()
        self._auto_disabled = False

        # Phase 2A: Session heartbeat debouncing
        self._last_heartbeat: datetime | None = None

        # Phase 2D: Retention scheduler task reference
        self._retention_task: asyncio.Task | None = None

        # Phase 2B: Database size cache
        self._db_size_cache: tuple[float, datetime] | None = (
            None  # (size_mb, cached_at)
        )

    async def connect(self) -> None:
        """Establish connection to SQLite and initialize session.

        Schema is auto-created on first connect via CREATE TABLE IF NOT EXISTS.
        Phase 2A: Uses session resumption instead of always creating new session.
        """
        if self._initialized:
            return

        try:
            # Connect to SQLite (auto-creates file and schema)
            await self.db.connect()

            logger.info("✅ SQLite event log schema initialized")

            # Phase 2A: Resume or create session
            self.session_id = await self.resume_or_create_session()

            self._initialized = True
            logger.info(
                f"✅ Event Log Service initialized with session {self.session_id[:8]}..."
            )

        except Exception as e:
            logger.error(f"Failed to initialize Event Log Service: {e}")
            raise

    async def resume_or_create_session(self) -> str:
        """Resume previous session if within timeout window, else create new.

        Phase 2A: Hybrid session management.

        Returns:
            Session ID (either resumed or newly created)
        """
        # Query last active session
        row = await self.db.fetchrow(
            """
            SELECT session_id, last_active_at, status
            FROM sessions
            WHERE status IN ('active', 'resumed')
            ORDER BY last_active_at DESC
            LIMIT 1
            """
        )

        if row:
            session_id = row["session_id"]
            last_active_str = row["last_active_at"]

            last_active = last_active_str
            if isinstance(last_active, str):
                last_active = datetime.fromisoformat(last_active.replace("Z", "+00:00"))

            now = datetime.now(UTC)
            time_diff = (now - last_active).total_seconds() / 3600  # hours

            # Resume if within timeout window
            if time_diff < config.EVENT_LOG_SESSION_TIMEOUT_HOURS:
                # Update session status
                await self.db.execute(
                    """
                    UPDATE sessions
                    SET status = 'resumed',
                        last_active_at = ?
                    WHERE session_id = ?
                    """,
                    now.isoformat(),
                    session_id,
                )

                logger.info(
                    f"📋 Resumed session {session_id[:8]}... "
                    f"(inactive for {time_diff:.1f}h)"
                )
                return session_id

        # Create new session
        new_session_id = str(uuid4())
        now = datetime.now(UTC)
        await self.db.execute(
            """
            INSERT INTO sessions (session_id, created_at, last_active_at, status, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            new_session_id,
            now.isoformat(),
            now.isoformat(),
            "active",
            json.dumps({"server_version": config.MCP_VERSION}),
        )

        logger.info(f"📋 Created new session {new_session_id[:8]}...")
        return new_session_id

    async def update_session_heartbeat(self) -> None:
        """Update last_active_at for current session.

        Phase 2A: Session heartbeat with debouncing.
        Updates at most once per minute to reduce writes.
        """
        now = datetime.now(UTC)

        # Debounce: only update if >1 minute since last heartbeat
        if self._last_heartbeat:
            time_since_last = (now - self._last_heartbeat).total_seconds()
            if time_since_last < 60:  # Less than 1 minute
                return

        await self.db.execute(
            """
            UPDATE sessions
            SET last_active_at = ?
            WHERE session_id = ?
            """,
            now.isoformat(),
            self.session_id,
        )

        self._last_heartbeat = now

    async def get_database_size(self) -> float:
        """Get database size in megabytes.

        Phase 2B: Database size monitoring with caching.
        Delegates to the backend's get_size_mb() method so the logic stays
        backend-specific (file size for SQLite, pg_total_relation_size for PG).

        Returns:
            Database size in MB
        """
        now = datetime.now(UTC)

        # Check cache (TTL: 5 minutes)
        if self._db_size_cache:
            cached_size, cached_at = self._db_size_cache
            age = (now - cached_at).total_seconds()
            if age < 300:  # 5 minutes
                return cached_size

        try:
            size_mb = await self.db.get_size_mb()

            # Update cache
            self._db_size_cache = (size_mb, now)
            return size_mb

        except Exception as e:
            logger.warning(f"Failed to get database size: {e}")
            return 0.0

    async def prune_old_events(self, retention_days: int) -> int:
        """Delete events older than retention period.

        Phase 2D: Age-based retention policy.

        Args:
            retention_days: Age threshold in days

        Returns:
            Number of events deleted
        """
        await self.connect()

        try:
            cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()

            # Get count before deleting
            count = await self.db.fetchval(
                "SELECT COUNT(*) FROM events WHERE timestamp < ?",
                cutoff,
            )

            if count and count > 0:
                await self.db.execute(
                    "DELETE FROM events WHERE timestamp < ?",
                    cutoff,
                )
                logger.info(
                    f"Retention: Deleted {count} events older than {retention_days} days"
                )
                # Invalidate size cache after deletion
                self._db_size_cache = None

            return count or 0

        except Exception as e:
            logger.error(f"Retention pruning error: {e}")
            return 0

    async def prune_by_size(self, max_size_mb: int) -> int:
        """Delete oldest events until database under size limit.

        Phase 2D: Size-based retention policy.

        Args:
            max_size_mb: Maximum database size in megabytes

        Returns:
            Number of events deleted
        """
        current_size_mb = await self.get_database_size()
        if current_size_mb <= max_size_mb:
            return 0

        try:
            total_deleted = 0
            batch_size = 1000
            max_iterations = 100  # Safety limit

            for _ in range(max_iterations):
                if current_size_mb <= max_size_mb:
                    break

                # Check remaining events to avoid infinite loop
                remaining = await self.db.fetchval("SELECT COUNT(*) FROM events")
                if remaining == 0:
                    break

                to_delete = min(batch_size, remaining)
                await self.db.execute(
                    """
                    DELETE FROM events WHERE id IN (
                        SELECT id FROM events ORDER BY timestamp ASC LIMIT ?
                    )
                    """,
                    to_delete,
                )
                total_deleted += to_delete

                # Reclaim disk space; no-op for PostgreSQL (autovacuum handles it)
                await self.db.vacuum()

                # Invalidate and recheck size
                self._db_size_cache = None
                current_size_mb = await self.get_database_size()

            if total_deleted > 0:
                logger.info(
                    f"Retention: Deleted {total_deleted} events to reduce size "
                    f"to {current_size_mb:.2f}MB"
                )

            return total_deleted

        except Exception as e:
            logger.error(f"Size-based retention error: {e}")
            return 0

    async def attempt_recovery(self) -> bool:
        """Attempt to recover from auto-disabled state.

        Tests a simple SELECT query to verify SQLite is responsive,
        then resets the SLA violation counter and re-enables event logging.

        Returns:
            True if recovery succeeded, False otherwise.
        """
        if not self._auto_disabled:
            return True

        try:
            start = time.perf_counter()
            await self.db.fetchval("SELECT 1")
            latency_ms = (time.perf_counter() - start) * 1000

            if latency_ms < config.EVENT_LOG_PERFORMANCE_SLA_MS:
                self._auto_disabled = False
                self._metrics.sla_violations = 0
                self._metrics.write_latencies.clear()
                logger.info(
                    f"✅ Event log recovered from auto-disabled state "
                    f"(test latency: {latency_ms:.1f}ms)"
                )
                return True
            else:
                logger.warning(
                    f"Event log recovery check still slow: {latency_ms:.1f}ms "
                    f"(SLA: {config.EVENT_LOG_PERFORMANCE_SLA_MS}ms)"
                )
                return False
        except Exception as e:
            logger.warning(f"Event log recovery check failed: {e}")
            return False

    async def retention_scheduler_loop(self) -> None:
        """Background task for event log retention management.

        Phase 2D: Runs cleanup based on age and size policies every 24 hours.
        """
        while True:
            try:
                # Attempt recovery if auto-disabled
                if self._auto_disabled:
                    await self.attempt_recovery()

                # Age-based pruning
                deleted_age = await self.prune_old_events(
                    config.EVENT_LOG_RETENTION_DAYS
                )

                # Size-based pruning (if needed)
                db_size_mb = await self.get_database_size()
                deleted_size = 0
                if db_size_mb > config.EVENT_LOG_MAX_SIZE_MB:
                    deleted_size = await self.prune_by_size(
                        config.EVENT_LOG_MAX_SIZE_MB
                    )
                    db_size_mb = await self.get_database_size()  # Refresh after cleanup

                # Log cleanup summary (only if not auto-disabled)
                if not self._auto_disabled:
                    await self.log_event(
                        event_type="system",
                        action="retention_cleanup",
                        actor="retention_scheduler",
                        result="success",
                        metadata={
                            "deleted_age_based": deleted_age,
                            "deleted_size_based": deleted_size,
                            "database_size_mb": db_size_mb,
                        },
                    )

                # Sleep for 24 hours
                await asyncio.sleep(86400)

            except Exception as e:
                logger.error(f"Retention scheduler error: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour

    async def close(self) -> None:
        """Close database connection and stop background tasks."""
        # Cancel retention scheduler if running
        if self._retention_task and not self._retention_task.done():
            self._retention_task.cancel()
            try:
                await self._retention_task
            except asyncio.CancelledError:
                pass

        await self.db.disconnect()
        self._initialized = False
        logger.info("Closed Event Log Service connection")

    async def log_event(
        self,
        event_type: str,
        action: str,
        actor: str | None = None,
        target: str | None = None,
        context: dict[str, Any] | None = None,
        result: str | None = None,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> int:
        """Log an event to the SQLite event log.

        Phase 2B: Includes performance tracking and auto-disable logic.

        Args:
            event_type: Event category (e.g., "tool_call", "memory_operation", "error")
            action: Specific action taken (e.g., "capture_fact", "query_context")
            actor: Who/what triggered the event (e.g., "user", "system", "claude_code")
            target: What was acted upon (e.g., entity UUID, file path)
            context: Additional context as JSON-serializable dict
            result: Outcome of the action (e.g., "success", "failure", error message)
            metadata: Additional metadata as JSON-serializable dict
            session_id: Override session ID (defaults to service session_id)

        Returns:
            Event ID from database

        Raises:
            Exception if database write fails or event log auto-disabled
        """
        # Phase 2B: Check if auto-disabled
        if self._auto_disabled:
            if config.EVENT_LOG_FAIL_LOUD:
                raise Exception(
                    "Event log auto-disabled due to SLA violations. "
                    f"p95 write latency: {self._metrics.get_p95_write():.1f}ms "
                    f"(SLA: {config.EVENT_LOG_PERFORMANCE_SLA_MS}ms)"
                )
            return -1  # Silent failure if not configured to fail loud

        await self.connect()

        # Phase 2B: Performance tracking
        start = time.perf_counter()

        try:
            # Use service session ID if not provided
            if session_id is None:
                session_id = self.session_id

            now = datetime.now(UTC).isoformat()

            # events.session_id has a FK to sessions: a caller-supplied session
            # id that was never registered needs a sessions row first.
            # Portable check-then-insert (INSERT OR IGNORE is SQLite-only).
            if session_id != self.session_id:
                known = await self.db.fetchval(
                    "SELECT 1 FROM sessions WHERE session_id = ?", session_id
                )
                if not known:
                    await self.db.execute(
                        """
                        INSERT INTO sessions
                            (session_id, created_at, last_active_at, status, metadata)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        session_id,
                        now,
                        now,
                        "active",
                        json.dumps({"auto_created": "log_event"}),
                    )

            # Insert event
            event_id = await self.db.fetchval(
                """
                INSERT INTO events (
                    timestamp, event_type, session_id, actor, action,
                    target, context, result, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
                """,
                now,
                event_type,
                session_id,
                actor,
                action,
                target,
                json.dumps(context) if context else None,
                result,
                json.dumps(metadata) if metadata else None,
            )

            # Phase 2B: Track performance
            duration_ms = (time.perf_counter() - start) * 1000
            self._metrics.add_write(duration_ms)

            # Phase 2B: Check SLA violation
            if duration_ms > config.EVENT_LOG_PERFORMANCE_SLA_MS:
                self._metrics.sla_violations += 1
                self._metrics.last_violation_at = datetime.now(UTC)

                # Auto-disable if too many consecutive violations
                if self._metrics.sla_violations >= 5:
                    self._auto_disabled = True
                    logger.error(
                        f"⚠️ EVENT LOG AUTO-DISABLED: "
                        f"SLA violated {self._metrics.sla_violations} times. "
                        f"p95 write latency: {self._metrics.get_p95_write():.1f}ms "
                        f"(SLA: {config.EVENT_LOG_PERFORMANCE_SLA_MS}ms)"
                    )
            else:
                # Reset violation counter on success
                if self._metrics.sla_violations > 0:
                    self._metrics.sla_violations = max(
                        0, self._metrics.sla_violations - 1
                    )

            # Phase 2A: Update session heartbeat
            await self.update_session_heartbeat()

            logger.debug(
                f"Event logged: {event_type}/{action} "
                f"(id={event_id}, session={session_id[:8]}..., {duration_ms:.1f}ms)"
            )

            return event_id

        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            self._metrics.add_write(duration_ms)
            logger.error(f"Failed to log event: {e}")
            raise

    async def query_events(
        self,
        session_id: str | None = None,
        event_type: str | None = None,
        action: str | None = None,
        actor: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query events with flexible filtering.

        Phase 2B: Includes performance tracking.

        Args:
            session_id: Filter by session ID
            event_type: Filter by event type
            action: Filter by action
            actor: Filter by actor
            start_time: Filter events after this timestamp
            end_time: Filter events before this timestamp
            limit: Maximum number of events to return
            offset: Number of events to skip (for pagination)

        Returns:
            List of event dictionaries with deserialized JSON fields
        """
        await self.connect()

        # Phase 2B: Performance tracking
        start = time.perf_counter()

        # Build dynamic query with ? placeholders
        where_clauses = []
        params = []

        if session_id:
            where_clauses.append("session_id = ?")
            params.append(session_id)

        if event_type:
            where_clauses.append("event_type = ?")
            params.append(event_type)

        if action:
            where_clauses.append("action = ?")
            params.append(action)

        if actor:
            where_clauses.append("actor = ?")
            params.append(actor)

        if start_time:
            where_clauses.append("timestamp >= ?")
            params.append(
                start_time.isoformat()
                if isinstance(start_time, datetime)
                else start_time
            )

        if end_time:
            where_clauses.append("timestamp <= ?")
            params.append(
                end_time.isoformat() if isinstance(end_time, datetime) else end_time
            )

        # Construct WHERE clause
        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        # Add limit and offset
        params.extend([limit, offset])

        query = f"""
            SELECT
                id, timestamp, event_type, session_id, actor, action,
                target, context, result, metadata
            FROM events
            WHERE {where_sql}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """

        rows = await self.db.fetch(query, *params)

        # Convert to list of dicts with deserialized JSON
        events = []
        for row in rows:
            event = {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "event_type": row["event_type"],
                "session_id": row["session_id"],
                "actor": row["actor"],
                "action": row["action"],
                "target": row["target"],
                "context": json.loads(row["context"]) if row["context"] else None,
                "result": row["result"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
            }
            events.append(event)

        # Phase 2B: Track query performance
        duration_ms = (time.perf_counter() - start) * 1000
        self._metrics.add_query(duration_ms)

        logger.debug(f"Query returned {len(events)} events ({duration_ms:.1f}ms)")
        return events

    async def get_session_timeline(
        self, session_id: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get complete chronological timeline for a session.

        Args:
            session_id: Session to retrieve (defaults to current session)
            limit: Maximum events to return

        Returns:
            List of events in chronological order (oldest first)
        """
        if session_id is None:
            session_id = self.session_id

        events = await self.query_events(session_id=session_id, limit=limit)

        # Reverse to get chronological order (oldest first)
        return list(reversed(events))

    async def get_event_count(
        self,
        session_id: str | None = None,
        event_type: str | None = None,
    ) -> int:
        """Get count of events matching filters.

        Args:
            session_id: Filter by session ID
            event_type: Filter by event type

        Returns:
            Count of matching events
        """
        await self.connect()

        where_clauses = []
        params = []

        if session_id:
            where_clauses.append("session_id = ?")
            params.append(session_id)

        if event_type:
            where_clauses.append("event_type = ?")
            params.append(event_type)

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        query = f"SELECT COUNT(*) FROM events WHERE {where_sql}"

        count = await self.db.fetchval(query, *params)
        return count or 0

    async def get_session_status(self) -> str:
        """Get the status of the current session.

        Returns:
            Session status string (e.g., 'active', 'resumed') or 'Unknown'
        """
        if not self.session_id:
            return "Unknown"

        try:
            row = await self.db.fetchrow(
                "SELECT status FROM sessions WHERE session_id = ?",
                self.session_id,
            )
            return row["status"] if row else "Unknown"
        except Exception:
            return "Unknown"

    async def get_event_log_health(self) -> dict[str, Any]:
        """Get comprehensive health metrics for event log service.

        Returns health data structure expected by admin UI health dashboard.

        Returns:
            Dictionary with health metrics:
                - status: "healthy" or "error"
                - total_events: Total number of events in database
                - oldest_event: Timestamp of oldest event (ISO format)
                - newest_event: Timestamp of newest event (ISO format)
                - storage_size: Database size in bytes
                - retention_days: Configured retention period
                - max_size_mb: Configured maximum size
        """
        try:
            await self.connect()

            # Get total event count
            total_events = await self.get_event_count()

            # Get oldest and newest event timestamps
            oldest_row = await self.db.fetchrow(
                "SELECT timestamp FROM events ORDER BY timestamp ASC LIMIT 1"
            )
            oldest_event = oldest_row["timestamp"] if oldest_row else None

            newest_row = await self.db.fetchrow(
                "SELECT timestamp FROM events ORDER BY timestamp DESC LIMIT 1"
            )
            newest_event = newest_row["timestamp"] if newest_row else None

            # Get database size (in bytes for formatBytes() function in UI)
            size_mb = await self.get_database_size()
            storage_size = int(size_mb * 1024 * 1024)  # Convert MB to bytes

            return {
                "status": "healthy",
                "total_events": total_events,
                "oldest_event": oldest_event,
                "newest_event": newest_event,
                "storage_size": storage_size,
                "retention_days": config.EVENT_LOG_RETENTION_DAYS,
                "max_size_mb": config.EVENT_LOG_MAX_SIZE_MB,
            }

        except Exception as e:
            logger.error(f"Failed to get event log health: {e}")
            return {
                "status": "error",
                "error": str(e),
                "total_events": 0,
                "storage_size": 0,
                "retention_days": config.EVENT_LOG_RETENTION_DAYS,
                "max_size_mb": config.EVENT_LOG_MAX_SIZE_MB,
            }


# Singleton instance
event_log_service = EventLogService()
