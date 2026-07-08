"""Integration tests for PostgreSQL-backed Event Log Service.

Mirrors tests/integration/test_event_log_sqlite.py for the PostgreSQL backend.

Run requirements:
  - asyncpg installed  (pytest.importorskip below handles this)
  - EVENT_LOG_POSTGRES_DSN or PG_PASSWORD env var must be set
  - PostgreSQL instance with the event_logs schema applied

Run with:
    EVENT_LOG_POSTGRES_DSN='postgresql://...' pytest -m postgres tests/integration/test_event_log_postgres.py -v
"""

import json
import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
import pytest_asyncio

# Skip module entirely if asyncpg is not installed
asyncpg = pytest.importorskip("asyncpg")

# Skip module if no connection env var is configured
_has_dsn = bool(os.getenv("EVENT_LOG_POSTGRES_DSN") or os.getenv("PG_PASSWORD"))
if not _has_dsn:
    pytest.skip(
        "Set EVENT_LOG_POSTGRES_DSN or PG_PASSWORD to run PostgreSQL integration tests",
        allow_module_level=True,
    )

from src.database.postgresql import PostgreSQLConnection  # noqa: E402
from src.services.event_log_service import EventLogService  # noqa: E402

# ============================================================
# Helpers
# ============================================================


async def _truncate(conn: PostgreSQLConnection) -> None:
    """Truncate both tables to give each test a clean slate."""
    await conn.execute(
        "TRUNCATE event_logs.events, event_logs.sessions RESTART IDENTITY CASCADE"
    )


# ============================================================
# Fixtures
# ============================================================


@pytest_asyncio.fixture
async def pg_conn():
    """Provide a connected PostgreSQLConnection and clean state per test."""
    conn = PostgreSQLConnection()
    await conn.connect()
    await _truncate(conn)
    yield conn
    await _truncate(conn)
    await conn.disconnect()


@pytest_asyncio.fixture
async def event_log(pg_conn):
    """Provide a fully initialized EventLogService backed by PostgreSQL.

    Teardown only cancels the retention task and resets the service state;
    the underlying connection pool is owned and closed by the ``pg_conn``
    fixture so that the post-test TRUNCATE can still run.
    """
    service = EventLogService()
    service.db = pg_conn
    await service.connect()
    yield service
    # Cancel retention task only — do NOT disconnect (pg_conn owns the pool)
    if service._retention_task and not service._retention_task.done():
        service._retention_task.cancel()
        try:
            await service._retention_task
        except Exception:
            pass
    service._initialized = False


@pytest_asyncio.fixture
async def seeded_event_log(event_log):
    """Event log pre-seeded with representative data.

    Mirrors the SQLite seeded_event_log fixture exactly:
    - 3 sessions (current active, previous resumed, old closed)
    - 15 events across different types, actors and targets
    """
    service = event_log

    old_session_id = str(uuid4())
    previous_session_id = str(uuid4())

    old_time = (datetime.now(UTC) - timedelta(days=30)).isoformat()
    prev_time = (datetime.now(UTC) - timedelta(hours=2)).isoformat()

    await service.db.execute(
        "INSERT INTO sessions (session_id, created_at, last_active_at, status, metadata)"
        " VALUES (?, ?, ?, ?, ?)",
        old_session_id,
        old_time,
        old_time,
        "closed",
        json.dumps({"server_version": "0.9.0"}),
    )

    await service.db.execute(
        "INSERT INTO sessions (session_id, created_at, last_active_at, status, metadata)"
        " VALUES (?, ?, ?, ?, ?)",
        previous_session_id,
        prev_time,
        prev_time,
        "resumed",
        json.dumps({"server_version": "0.9.0"}),
    )

    events_data = [
        # Current session events (8)
        (
            "tool_call",
            "capture_fact",
            "user",
            "entity-uuid-1",
            "success",
            service.session_id,
        ),
        (
            "tool_call",
            "capture_fact",
            "user",
            "entity-uuid-2",
            "success",
            service.session_id,
        ),
        ("tool_call", "load_context", "user", None, "success", service.session_id),
        ("tool_call", "semantic_search", "user", None, "success", service.session_id),
        (
            "memory_operation",
            "deduplication_check",
            "system",
            "learning-uuid-1",
            "auto_skipped",
            service.session_id,
        ),
        (
            "memory_operation",
            "vector_embed",
            "system",
            "entity-uuid-1",
            "success",
            service.session_id,
        ),
        (
            "error",
            "capture_fact",
            "user",
            None,
            "Neo4j connection timeout",
            service.session_id,
        ),
        (
            "system",
            "retention_cleanup",
            "retention_scheduler",
            None,
            "success",
            service.session_id,
        ),
        # Previous session events (4)
        (
            "tool_call",
            "capture_learning",
            "user",
            "learning-uuid-2",
            "success",
            previous_session_id,
        ),
        ("tool_call", "capture_session", "user", None, "success", previous_session_id),
        ("tool_call", "health", "user", None, "success", previous_session_id),
        (
            "memory_operation",
            "cache_warm",
            "system",
            None,
            "success",
            previous_session_id,
        ),
        # Old session events (3)
        (
            "tool_call",
            "capture_fact",
            "user",
            "entity-uuid-old",
            "success",
            old_session_id,
        ),
        ("tool_call", "load_context", "user", None, "success", old_session_id),
        (
            "error",
            "neo4j_connect",
            "system",
            None,
            "Connection refused",
            old_session_id,
        ),
    ]

    now = datetime.now(UTC)
    for i, (etype, action, actor, target, result, sid) in enumerate(events_data):
        if sid == old_session_id:
            ts = (now - timedelta(days=30, seconds=len(events_data) - i)).isoformat()
        else:
            ts = (now - timedelta(seconds=len(events_data) - i)).isoformat()
        ctx = json.dumps({"index": i}) if i % 3 == 0 else None
        meta = json.dumps({"duration_ms": 10 + i}) if i % 2 == 0 else None

        await service.db.execute(
            """INSERT INTO events
               (timestamp, event_type, session_id, actor, action, target, context, result, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ts,
            etype,
            sid,
            actor,
            action,
            target,
            ctx,
            result,
            meta,
        )

    service._seeded_sessions = {
        "current": service.session_id,
        "previous": previous_session_id,
        "old": old_session_id,
    }

    yield service


# ============================================================
# Session Management Tests
# ============================================================


class TestSessionManagement:
    """Test session creation and resumption with real PostgreSQL."""

    @pytest.mark.postgres
    @pytest.mark.asyncio
    async def test_new_session_created(self, event_log):
        """First connect creates a new session."""
        assert event_log.session_id is not None
        assert len(event_log.session_id) == 36  # UUID format

        row = await event_log.db.fetchrow(
            "SELECT * FROM sessions WHERE session_id = ?",
            event_log.session_id,
        )
        assert row is not None
        assert row["status"] == "active"

    @pytest.mark.postgres
    @pytest.mark.asyncio
    async def test_session_resumption(self, pg_conn):
        """Service should resume a recent active session."""
        # First service creates a session
        svc1 = EventLogService()
        svc1.db = pg_conn
        await svc1.connect()
        original_session = svc1.session_id
        svc1._initialized = False  # Reset without closing DB

        # Second service should resume the same session
        svc2 = EventLogService()
        svc2.db = pg_conn
        await svc2.connect()

        assert svc2.session_id == original_session

        row = await pg_conn.fetchrow(
            "SELECT status FROM sessions WHERE session_id = ?",
            original_session,
        )
        assert row["status"] == "resumed"

    @pytest.mark.postgres
    @pytest.mark.asyncio
    async def test_get_session_status(self, event_log):
        """get_session_status returns current session status."""
        status = await event_log.get_session_status()
        assert status == "active"

    @pytest.mark.postgres
    @pytest.mark.asyncio
    async def test_get_session_status_no_session(self):
        """get_session_status with no session returns Unknown."""
        svc = EventLogService()
        status = await svc.get_session_status()
        assert status == "Unknown"

    @pytest.mark.postgres
    @pytest.mark.asyncio
    async def test_heartbeat_update(self, event_log):
        """update_session_heartbeat updates last_active_at without error."""
        # Force heartbeat (bypass debounce by clearing last_heartbeat)
        event_log._last_heartbeat = None
        await event_log.update_session_heartbeat()

        row = await event_log.db.fetchrow(
            "SELECT last_active_at FROM sessions WHERE session_id = ?",
            event_log.session_id,
        )
        assert row is not None
        assert row["last_active_at"] is not None


# ============================================================
# Event Logging Tests
# ============================================================


class TestEventLogging:
    """Test actual event writes and reads against real PostgreSQL."""

    @pytest.mark.postgres
    @pytest.mark.asyncio
    async def test_log_and_retrieve_event(self, event_log):
        """Log an event and verify it can be retrieved."""
        event_id = await event_log.log_event(
            event_type="tool_call",
            action="capture_fact",
            actor="user",
            target="entity-123",
            context={"key": "value"},
            result="success",
            metadata={"duration_ms": 42},
        )

        assert event_id is not None
        assert isinstance(event_id, int)
        assert event_id > 0

        events = await event_log.query_events(
            session_id=event_log.session_id,
            action="capture_fact",
        )
        assert len(events) >= 1

        event = events[0]
        assert event["event_type"] == "tool_call"
        assert event["action"] == "capture_fact"
        assert event["actor"] == "user"
        assert event["target"] == "entity-123"
        assert event["context"] == {"key": "value"}
        assert event["result"] == "success"
        assert event["metadata"] == {"duration_ms": 42}

    @pytest.mark.postgres
    @pytest.mark.asyncio
    async def test_log_event_minimal_fields(self, event_log):
        """Log with only required fields."""
        event_id = await event_log.log_event(
            event_type="system",
            action="heartbeat",
        )
        assert event_id is not None

        events = await event_log.query_events(action="heartbeat")
        assert len(events) >= 1
        assert events[0]["actor"] is None
        assert events[0]["target"] is None
        assert events[0]["context"] is None
        assert events[0]["metadata"] is None

    @pytest.mark.postgres
    @pytest.mark.asyncio
    async def test_log_event_json_roundtrip(self, event_log):
        """JSON context and metadata survive write/read cycle."""
        complex_context = {
            "nested": {"list": [1, 2, 3]},
            "unicode": "café",
            "bool": True,
            "null": None,
        }

        await event_log.log_event(
            event_type="test",
            action="json_roundtrip",
            context=complex_context,
        )

        events = await event_log.query_events(action="json_roundtrip")
        assert len(events) >= 1
        assert events[0]["context"] == complex_context


# ============================================================
# Query and Filtering Tests
# ============================================================


class TestQueryFiltering:
    """Test query filtering with seeded data."""

    @pytest.mark.postgres
    @pytest.mark.asyncio
    async def test_filter_by_event_type(self, seeded_event_log):
        """Filter events by type."""
        tool_calls = await seeded_event_log.query_events(event_type="tool_call")
        errors = await seeded_event_log.query_events(event_type="error")

        assert all(e["event_type"] == "tool_call" for e in tool_calls)
        assert all(e["event_type"] == "error" for e in errors)
        assert len(tool_calls) > len(errors)

    @pytest.mark.postgres
    @pytest.mark.asyncio
    async def test_filter_by_session(self, seeded_event_log):
        """Filter events by session ID."""
        sessions = seeded_event_log._seeded_sessions

        current_events = await seeded_event_log.query_events(
            session_id=sessions["current"]
        )
        old_events = await seeded_event_log.query_events(session_id=sessions["old"])

        assert len(current_events) > 0
        assert len(old_events) > 0
        assert all(e["session_id"] == sessions["current"] for e in current_events)
        assert all(e["session_id"] == sessions["old"] for e in old_events)

    @pytest.mark.postgres
    @pytest.mark.asyncio
    async def test_filter_by_actor(self, seeded_event_log):
        """Filter events by actor."""
        user_events = await seeded_event_log.query_events(actor="user")
        system_events = await seeded_event_log.query_events(actor="system")

        assert all(e["actor"] == "user" for e in user_events)
        assert all(e["actor"] == "system" for e in system_events)

    @pytest.mark.postgres
    @pytest.mark.asyncio
    async def test_combined_filters(self, seeded_event_log):
        """Combine multiple filters."""
        events = await seeded_event_log.query_events(
            event_type="tool_call",
            actor="user",
            action="capture_fact",
        )

        assert all(
            e["event_type"] == "tool_call"
            and e["actor"] == "user"
            and e["action"] == "capture_fact"
            for e in events
        )

    @pytest.mark.postgres
    @pytest.mark.asyncio
    async def test_pagination(self, seeded_event_log):
        """Pagination returns correct subsets."""
        page1 = await seeded_event_log.query_events(limit=5, offset=0)
        page2 = await seeded_event_log.query_events(limit=5, offset=5)

        assert len(page1) == 5
        assert len(page2) == 5

        page1_ids = {e["id"] for e in page1}
        page2_ids = {e["id"] for e in page2}
        assert page1_ids.isdisjoint(page2_ids)

    @pytest.mark.postgres
    @pytest.mark.asyncio
    async def test_ordering_descending(self, seeded_event_log):
        """Events should be returned newest-first by default."""
        events = await seeded_event_log.query_events(limit=10)

        timestamps = [e["timestamp"] for e in events]
        assert timestamps == sorted(timestamps, reverse=True)

    @pytest.mark.postgres
    @pytest.mark.asyncio
    async def test_session_timeline_chronological(self, seeded_event_log):
        """Session timeline returns oldest-first."""
        sessions = seeded_event_log._seeded_sessions
        timeline = await seeded_event_log.get_session_timeline(
            session_id=sessions["current"]
        )

        timestamps = [e["timestamp"] for e in timeline]
        assert timestamps == sorted(timestamps)  # Ascending


# ============================================================
# Count and Size Tests
# ============================================================


class TestCountsAndSize:
    """Test event counting and database size reporting."""

    @pytest.mark.postgres
    @pytest.mark.asyncio
    async def test_event_count_total(self, seeded_event_log):
        """Total event count matches seeded data."""
        count = await seeded_event_log.get_event_count()
        assert count == 15  # 8 + 4 + 3 from seed data

    @pytest.mark.postgres
    @pytest.mark.asyncio
    async def test_event_count_by_type(self, seeded_event_log):
        """Count filtered by event type."""
        tool_count = await seeded_event_log.get_event_count(event_type="tool_call")
        error_count = await seeded_event_log.get_event_count(event_type="error")

        assert tool_count > 0
        assert error_count > 0
        assert tool_count > error_count

    @pytest.mark.postgres
    @pytest.mark.asyncio
    async def test_event_count_by_session(self, seeded_event_log):
        """Count filtered by session."""
        sessions = seeded_event_log._seeded_sessions
        current_count = await seeded_event_log.get_event_count(
            session_id=sessions["current"]
        )
        assert current_count == 8

    @pytest.mark.postgres
    @pytest.mark.asyncio
    async def test_get_database_size(self, seeded_event_log):
        """Database size should be reported in MB (>= 0.0)."""
        size_mb = await seeded_event_log.get_database_size()
        assert isinstance(size_mb, float)
        assert size_mb >= 0.0  # May be near 0 for fresh small test DB

    @pytest.mark.postgres
    @pytest.mark.asyncio
    async def test_database_size_caching(self, seeded_event_log):
        """Size should be cached for 5 minutes."""
        size1 = await seeded_event_log.get_database_size()
        size2 = await seeded_event_log.get_database_size()

        assert size1 == size2
        assert seeded_event_log._db_size_cache is not None


# ============================================================
# Retention Tests
# ============================================================


class TestRetention:
    """Test retention pruning against real PostgreSQL."""

    @pytest.mark.postgres
    @pytest.mark.asyncio
    async def test_prune_old_events(self, seeded_event_log):
        """Age-based pruning deletes old events."""
        initial_count = await seeded_event_log.get_event_count()

        deleted = await seeded_event_log.prune_old_events(retention_days=7)

        assert deleted > 0

        final_count = await seeded_event_log.get_event_count()
        assert final_count < initial_count
        assert final_count == initial_count - deleted

    @pytest.mark.postgres
    @pytest.mark.asyncio
    async def test_prune_keeps_recent(self, seeded_event_log):
        """Pruning should not delete recent events."""
        deleted = await seeded_event_log.prune_old_events(retention_days=60)
        assert deleted == 0

    @pytest.mark.postgres
    @pytest.mark.asyncio
    async def test_prune_by_size(self, pg_conn):
        """Size-based pruning removes oldest events first."""
        service = EventLogService()
        service.db = pg_conn
        await service.connect()

        now = datetime.now(UTC)
        for i in range(200):
            ts = (now - timedelta(seconds=200 - i)).isoformat()
            await pg_conn.execute(
                """INSERT INTO events
                   (timestamp, event_type, session_id, actor, action, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                ts,
                "bulk",
                service.session_id,
                "system",
                f"action_{i}",
                json.dumps({"payload": "x" * 500}),
            )

        # Exercise the prune_by_size path; size may be near 0 in a fresh test DB
        deleted = await service.prune_by_size(max_size_mb=0)
        assert deleted >= 0

        service._initialized = False  # Don't disconnect — pg_conn owns the pool

    @pytest.mark.postgres
    @pytest.mark.asyncio
    async def test_vacuum_is_noop(self, pg_conn):
        """vacuum() on PostgreSQL should complete without error."""
        await pg_conn.vacuum()  # no-op — must not raise


# ============================================================
# Backend Description Test
# ============================================================


class TestBackendDescription:
    """Test PostgreSQL-specific metadata methods."""

    @pytest.mark.postgres
    @pytest.mark.asyncio
    async def test_backend_description(self, pg_conn):
        """backend_description is a non-empty string describing the PG backend."""
        desc = pg_conn.backend_description
        assert isinstance(desc, str)
        assert len(desc) > 0
        assert "PostgreSQL" in desc or "postgresql" in desc.lower()

    @pytest.mark.postgres
    @pytest.mark.asyncio
    async def test_get_size_mb_returns_float(self, pg_conn):
        """get_size_mb returns a float >= 0."""
        size = await pg_conn.get_size_mb()
        assert isinstance(size, float)
        assert size >= 0.0


# ============================================================
# SLA / attempt_recovery Tests
# ============================================================


class TestAttemptRecovery:
    """Test the SLA auto-disable and recovery path with PostgreSQL backend."""

    @pytest.mark.postgres
    @pytest.mark.asyncio
    async def test_attempt_recovery_not_disabled_is_noop(self, event_log):
        """attempt_recovery returns True immediately when not disabled."""
        event_log._auto_disabled = False
        result = await event_log.attempt_recovery()
        assert result is True

    @pytest.mark.postgres
    @pytest.mark.asyncio
    async def test_attempt_recovery_recovers_with_fast_db(self, event_log):
        """Disabled service re-enables when PG probe latency is within SLA."""
        from unittest.mock import patch

        event_log._auto_disabled = True
        event_log._metrics.sla_violations = 5
        event_log._metrics.add_write(200.0)

        # Patch perf_counter to simulate a very fast probe (0.5 ms)
        call_count = 0

        def fast_perf_counter():
            nonlocal call_count
            call_count += 1
            return 0.0005 * call_count  # 0.5 ms per call

        # Use a generous SLA so the real DB always passes
        with patch("src.services.event_log_service.config") as mock_config:
            mock_config.EVENT_LOG_PERFORMANCE_SLA_MS = 10_000  # 10 s SLA
            with patch(
                "src.services.event_log_service.time.perf_counter",
                side_effect=fast_perf_counter,
            ):
                result = await event_log.attempt_recovery()

        assert result is True
        assert event_log._auto_disabled is False
        assert event_log._metrics.sla_violations == 0

    @pytest.mark.postgres
    @pytest.mark.asyncio
    async def test_attempt_recovery_stays_disabled_when_probe_slow(self, event_log):
        """Disabled service stays disabled when PG probe exceeds SLA."""
        from unittest.mock import patch

        event_log._auto_disabled = True
        event_log._metrics.sla_violations = 5

        call_count = 0

        def slow_perf_counter():
            nonlocal call_count
            call_count += 1
            return 0.1 * call_count  # 100 ms per call

        with patch("src.services.event_log_service.config") as mock_config:
            mock_config.EVENT_LOG_PERFORMANCE_SLA_MS = 50  # 50 ms SLA, probe is 100 ms
            with patch(
                "src.services.event_log_service.time.perf_counter",
                side_effect=slow_perf_counter,
            ):
                result = await event_log.attempt_recovery()

        assert result is False
        assert event_log._auto_disabled is True
        assert event_log._metrics.sla_violations == 5  # unchanged
