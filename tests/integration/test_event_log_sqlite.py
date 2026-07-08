"""Integration tests for SQLite-backed Event Log Service.

Unlike unit tests that mock the database, these tests use a real SQLite
database (temp file) to verify:
- Schema auto-creation
- Actual SQL query correctness
- Seed data ingestion and retrieval
- Pagination and filtering
- Retention pruning (age-based and size-based)
- Session management (creation and resumption)
- Performance metrics tracking
- Database size reporting
"""

import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
import pytest_asyncio

from src.database.sqlite import SQLiteConnection
from src.services.event_log_service import EventLogService

# ============================================================
# Fixtures
# ============================================================


@pytest_asyncio.fixture
async def tmp_db(tmp_path):
    """Provide a real SQLite database in a temp directory."""
    db_path = str(tmp_path / "test_events.db")
    conn = SQLiteConnection(db_path=db_path)
    await conn.connect()
    yield conn
    await conn.disconnect()


@pytest_asyncio.fixture
async def event_log(tmp_path):
    """Provide a fully initialized EventLogService with real SQLite."""
    db_path = str(tmp_path / "test_events.db")
    conn = SQLiteConnection(db_path=db_path)

    service = EventLogService()
    service.db = conn
    await service.connect()

    yield service

    await service.close()


@pytest_asyncio.fixture
async def seeded_event_log(event_log):
    """Event log pre-seeded with representative data.

    Creates:
    - 3 sessions (current active, previous resumed, old closed)
    - 20 events across different types, actors, and targets
    """
    service = event_log

    # Seed additional sessions
    old_session_id = str(uuid4())
    previous_session_id = str(uuid4())

    old_time = (datetime.now(UTC) - timedelta(days=30)).isoformat()
    prev_time = (datetime.now(UTC) - timedelta(hours=2)).isoformat()

    await service.db.execute(
        "INSERT INTO sessions (session_id, created_at, last_active_at, status, metadata) VALUES (?, ?, ?, ?, ?)",
        old_session_id,
        old_time,
        old_time,
        "closed",
        json.dumps({"server_version": "0.9.0"}),
    )

    await service.db.execute(
        "INSERT INTO sessions (session_id, created_at, last_active_at, status, metadata) VALUES (?, ?, ?, ?, ?)",
        previous_session_id,
        prev_time,
        prev_time,
        "resumed",
        json.dumps({"server_version": "0.9.0"}),
    )

    # Seed events across sessions and types
    events_data = [
        # Current session events
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
        # Previous session events
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
        # Old session events
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
        # Old session events get 30-day-old timestamps; others are recent
        if sid == old_session_id:
            ts = (now - timedelta(days=30, seconds=len(events_data) - i)).isoformat()
        else:
            ts = (now - timedelta(seconds=len(events_data) - i)).isoformat()
        ctx = json.dumps({"index": i}) if i % 3 == 0 else None
        meta = json.dumps({"duration_ms": 10 + i}) if i % 2 == 0 else None

        await service.db.execute(
            """INSERT INTO events (timestamp, event_type, session_id, actor, action, target, context, result, metadata)
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
# Schema Tests
# ============================================================


class TestSchemaCreation:
    """Verify SQLite schema auto-creates correctly."""

    @pytest.mark.asyncio
    async def test_tables_created(self, tmp_db):
        """Both sessions and events tables should exist."""
        tables = await tmp_db.fetch(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        names = [t["name"] for t in tables]
        assert "events" in names
        assert "sessions" in names

    @pytest.mark.asyncio
    async def test_indexes_created(self, tmp_db):
        """All four indexes should exist."""
        indexes = await tmp_db.fetch(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%' ORDER BY name"
        )
        names = [i["name"] for i in indexes]
        assert "idx_events_session_id" in names
        assert "idx_events_timestamp" in names
        assert "idx_events_event_type" in names
        assert "idx_events_action" in names

    @pytest.mark.asyncio
    async def test_wal_mode_enabled(self, tmp_db):
        """WAL journal mode should be active."""
        mode = await tmp_db.fetchval("PRAGMA journal_mode")
        assert mode == "wal"

    @pytest.mark.asyncio
    async def test_idempotent_schema_init(self, tmp_db):
        """Calling _init_schema again should not fail or duplicate."""
        await tmp_db._init_schema()  # Second call
        tables = await tmp_db.fetch(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='events'"
        )
        assert len(tables) == 1


# ============================================================
# Session Management Tests
# ============================================================


class TestSessionManagement:
    """Test session creation and resumption with real database."""

    @pytest.mark.asyncio
    async def test_new_session_created(self, event_log):
        """First connect creates a new session."""
        assert event_log.session_id is not None
        assert len(event_log.session_id) == 36  # UUID format

        # Session should exist in database
        row = await event_log.db.fetchrow(
            "SELECT * FROM sessions WHERE session_id = ?",
            event_log.session_id,
        )
        assert row is not None
        assert row["status"] == "active"

    @pytest.mark.asyncio
    async def test_session_resumption(self, tmp_path):
        """Service should resume a recent active session."""
        db_path = str(tmp_path / "resume_test.db")
        conn = SQLiteConnection(db_path=db_path)

        # First service creates a session
        svc1 = EventLogService()
        svc1.db = conn
        await svc1.connect()
        original_session = svc1.session_id
        svc1._initialized = False  # Reset without closing DB

        # Second service should resume the same session
        svc2 = EventLogService()
        svc2.db = conn
        await svc2.connect()

        assert svc2.session_id == original_session

        # Verify status updated to 'resumed'
        row = await conn.fetchrow(
            "SELECT status FROM sessions WHERE session_id = ?",
            original_session,
        )
        assert row["status"] == "resumed"

        await conn.disconnect()

    @pytest.mark.asyncio
    async def test_get_session_status(self, event_log):
        """get_session_status should return current session status."""
        status = await event_log.get_session_status()
        assert status == "active"

    @pytest.mark.asyncio
    async def test_get_session_status_no_session(self):
        """get_session_status with no session returns Unknown."""
        svc = EventLogService()
        status = await svc.get_session_status()
        assert status == "Unknown"


# ============================================================
# Event Logging Tests
# ============================================================


class TestEventLogging:
    """Test actual event writes and reads against real SQLite."""

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

        # Retrieve and verify
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

    @pytest.mark.asyncio
    async def test_log_event_minimal_fields(self, event_log):
        """Log with only required fields."""
        event_id = await event_log.log_event(
            event_type="system",
            action="heartbeat",
        )
        assert event_id is not None

        events = await event_log.query_events(action="heartbeat")
        assert len(events) == 1
        assert events[0]["actor"] is None
        assert events[0]["target"] is None
        assert events[0]["context"] is None
        assert events[0]["metadata"] is None

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
        assert events[0]["context"] == complex_context


# ============================================================
# Query and Filtering Tests
# ============================================================


class TestQueryFiltering:
    """Test query filtering with seeded data."""

    @pytest.mark.asyncio
    async def test_filter_by_event_type(self, seeded_event_log):
        """Filter events by type."""
        tool_calls = await seeded_event_log.query_events(event_type="tool_call")
        errors = await seeded_event_log.query_events(event_type="error")

        assert all(e["event_type"] == "tool_call" for e in tool_calls)
        assert all(e["event_type"] == "error" for e in errors)
        assert len(tool_calls) > len(errors)

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

    @pytest.mark.asyncio
    async def test_filter_by_actor(self, seeded_event_log):
        """Filter events by actor."""
        user_events = await seeded_event_log.query_events(actor="user")
        system_events = await seeded_event_log.query_events(actor="system")

        assert all(e["actor"] == "user" for e in user_events)
        assert all(e["actor"] == "system" for e in system_events)

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

    @pytest.mark.asyncio
    async def test_pagination(self, seeded_event_log):
        """Pagination returns correct subsets."""
        all_events = await seeded_event_log.query_events(limit=100)
        page1 = await seeded_event_log.query_events(limit=5, offset=0)
        page2 = await seeded_event_log.query_events(limit=5, offset=5)

        assert len(page1) == 5
        assert len(page2) == 5

        # Pages should not overlap
        page1_ids = {e["id"] for e in page1}
        page2_ids = {e["id"] for e in page2}
        assert page1_ids.isdisjoint(page2_ids)

    @pytest.mark.asyncio
    async def test_ordering_descending(self, seeded_event_log):
        """Events should be returned newest-first by default."""
        events = await seeded_event_log.query_events(limit=10)

        timestamps = [e["timestamp"] for e in events]
        assert timestamps == sorted(timestamps, reverse=True)

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

    @pytest.mark.asyncio
    async def test_event_count_total(self, seeded_event_log):
        """Total event count matches seeded data."""
        count = await seeded_event_log.get_event_count()
        assert count == 15  # 8 + 4 + 3 from seed data

    @pytest.mark.asyncio
    async def test_event_count_by_type(self, seeded_event_log):
        """Count filtered by event type."""
        tool_count = await seeded_event_log.get_event_count(event_type="tool_call")
        error_count = await seeded_event_log.get_event_count(event_type="error")

        assert tool_count > 0
        assert error_count > 0
        assert tool_count > error_count

    @pytest.mark.asyncio
    async def test_event_count_by_session(self, seeded_event_log):
        """Count filtered by session."""
        sessions = seeded_event_log._seeded_sessions
        current_count = await seeded_event_log.get_event_count(
            session_id=sessions["current"]
        )
        assert current_count == 8

    @pytest.mark.asyncio
    async def test_database_size(self, seeded_event_log):
        """Database size should be reported in MB."""
        size_mb = await seeded_event_log.get_database_size()
        assert isinstance(size_mb, float)
        assert size_mb > 0  # Should have some data

    @pytest.mark.asyncio
    async def test_database_size_caching(self, seeded_event_log):
        """Size should be cached for 5 minutes."""
        size1 = await seeded_event_log.get_database_size()
        size2 = await seeded_event_log.get_database_size()

        # Both should return same cached value
        assert size1 == size2
        assert seeded_event_log._db_size_cache is not None


# ============================================================
# Retention Tests
# ============================================================


class TestRetention:
    """Test retention pruning against real data."""

    @pytest.mark.asyncio
    async def test_prune_old_events(self, seeded_event_log):
        """Age-based pruning deletes old events."""
        # Old session events are 30 days old
        initial_count = await seeded_event_log.get_event_count()

        # Prune events older than 7 days
        deleted = await seeded_event_log.prune_old_events(retention_days=7)

        assert deleted > 0

        final_count = await seeded_event_log.get_event_count()
        assert final_count < initial_count
        assert final_count == initial_count - deleted

    @pytest.mark.asyncio
    async def test_prune_keeps_recent(self, seeded_event_log):
        """Pruning should not delete recent events."""
        sessions = seeded_event_log._seeded_sessions

        # Prune with generous retention (keep everything from last 60 days)
        deleted = await seeded_event_log.prune_old_events(retention_days=60)

        assert deleted == 0

    @pytest.mark.asyncio
    async def test_prune_by_size(self, tmp_path):
        """Size-based pruning removes oldest events first."""
        db_path = str(tmp_path / "size_test.db")
        conn = SQLiteConnection(db_path=db_path)
        service = EventLogService()
        service.db = conn
        await service.connect()

        # Bulk-insert directly via SQL (bypassing log_event overhead)
        now = datetime.now(UTC)
        for i in range(200):
            ts = (now - timedelta(seconds=200 - i)).isoformat()
            await conn.execute(
                """INSERT INTO events (timestamp, event_type, session_id, actor, action, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                ts,
                "bulk",
                service.session_id,
                "system",
                f"action_{i}",
                json.dumps({"payload": "x" * 500}),
            )

        size_before = await service.get_database_size()

        # Set a very small size limit to force pruning
        if size_before > 0.01:  # Only test if DB has measurable size
            deleted = await service.prune_by_size(max_size_mb=0)
            assert deleted > 0

        await service.close()


# ============================================================
# Health Check Tests
# ============================================================


class TestHealthCheck:
    """Test the health report endpoint."""

    @pytest.mark.asyncio
    async def test_health_report_structure(self, seeded_event_log):
        """Health report should have all expected fields."""
        health = await seeded_event_log.get_event_log_health()

        assert health["status"] == "healthy"
        assert health["total_events"] == 15
        assert health["oldest_event"] is not None
        assert health["newest_event"] is not None
        assert isinstance(health["storage_size"], int)
        assert health["storage_size"] > 0
        assert "retention_days" in health
        assert "max_size_mb" in health

    @pytest.mark.asyncio
    async def test_health_timestamps_ordered(self, seeded_event_log):
        """Oldest event should be before newest event."""
        health = await seeded_event_log.get_event_log_health()
        assert health["oldest_event"] < health["newest_event"]


# ============================================================
# Performance Metrics Tests
# ============================================================


class TestPerformanceMetrics:
    """Test that performance metrics track real operations."""

    @pytest.mark.asyncio
    async def test_write_latency_tracked(self, event_log):
        """Writing events should record latency."""
        await event_log.log_event(event_type="test", action="perf_test")

        assert len(event_log._metrics.write_latencies) >= 1
        # Real SQLite should be fast (< 100ms typically)
        assert event_log._metrics.get_p95_write() < 500

    @pytest.mark.asyncio
    async def test_query_latency_tracked(self, event_log):
        """Querying events should record latency."""
        await event_log.log_event(event_type="test", action="perf_query")
        await event_log.query_events(event_type="test")

        assert len(event_log._metrics.query_latencies) >= 1
        assert event_log._metrics.get_p95_query() < 500
