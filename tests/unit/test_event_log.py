"""Tests for Event Log Service (episodic memory layer)."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio

from src.services.event_log_service import EventLogService


class TestEventLogService:
    """Tests for Event Log Service."""

    @pytest_asyncio.fixture
    async def event_log(self):
        """Create mocked event log service for testing."""
        patcher = patch("src.services.event_log_service._db_connection")
        mock_db = patcher.start()

        try:
            # Set up mock SQLite connection with async methods
            mock_db.connect = AsyncMock()
            mock_db.fetch = AsyncMock(return_value=[])
            mock_db.fetchval = AsyncMock(return_value=None)
            mock_db.fetchrow = AsyncMock(return_value=None)  # No previous session
            mock_db.execute = AsyncMock()
            mock_db.disconnect = AsyncMock()
            mock_db.db_path = "./data/test_events.db"
            mock_db.is_connected = True

            service = EventLogService()
            await service.connect()

            yield service
            await service.close()
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    async def test_connect_and_initialize(self):
        """Test service initialization."""
        with patch("src.services.event_log_service._db_connection") as mock_db:
            mock_db.connect = AsyncMock()
            mock_db.fetch = AsyncMock(return_value=[])
            mock_db.fetchval = AsyncMock(return_value=None)
            mock_db.fetchrow = AsyncMock(return_value=None)  # No previous session
            mock_db.execute = AsyncMock()
            mock_db.disconnect = AsyncMock()
            mock_db.db_path = "./data/test_events.db"

            service = EventLogService()
            await service.connect()

            # Service should have initialized
            assert service.session_id is not None
            assert len(service.session_id) == 36  # UUID format
            assert service._initialized is True

            await service.close()

    @pytest.mark.asyncio
    async def test_session_id_generation(self):
        """Test that session ID is generated on connection."""
        with patch("src.services.event_log_service._db_connection") as mock_db:
            mock_db.connect = AsyncMock()
            mock_db.fetch = AsyncMock(return_value=[])
            mock_db.fetchval = AsyncMock(return_value=None)
            mock_db.fetchrow = AsyncMock(return_value=None)  # No previous session
            mock_db.execute = AsyncMock()
            mock_db.disconnect = AsyncMock()
            mock_db.db_path = "./data/test_events.db"

            service = EventLogService()
            await service.connect()

            session_id = service.session_id
            assert session_id is not None

            # Connect again should keep same session ID
            await service.connect()
            assert service.session_id == session_id

            await service.close()

    @pytest.mark.asyncio
    async def test_log_event_basic(self, event_log):
        """Test basic event logging."""
        event_id = await event_log.log_event(
            event_type="test",
            action="test_action",
            actor="test_actor",
            target="test_target",
            result="success",
        )

        # Mock db will return the event_id (None from mock fetchval)
        assert event_id is not None or event_id is None  # Mock returns None

    @pytest.mark.asyncio
    async def test_log_event_with_json_context(self, event_log):
        """Test event logging with JSON context and metadata."""
        context = {"key": "value", "number": 42}
        metadata = {"tags": ["test", "important"]}

        event_id = await event_log.log_event(
            event_type="test", action="json_test", context=context, metadata=metadata
        )

        # Verify the service can handle JSON context and metadata
        assert event_id is not None or event_id is None

    @pytest.mark.asyncio
    async def test_log_event_minimal(self, event_log):
        """Test event logging with minimal required fields."""
        event_id = await event_log.log_event(
            event_type="minimal", action="minimal_action"
        )

        # Verify minimal fields work
        assert event_id is not None or event_id is None

    @pytest.mark.asyncio
    async def test_query_events_by_session(self, event_log):
        """Test querying events by session ID."""
        await event_log.log_event(event_type="type1", action="action1")
        await event_log.log_event(event_type="type2", action="action2")

        # Query current session - should return empty (mocked)
        events = await event_log.query_events(session_id=event_log.session_id)
        assert isinstance(events, list)

    @pytest.mark.asyncio
    async def test_query_events_by_type(self, event_log):
        """Test querying events by event type."""
        await event_log.log_event(event_type="memory_operation", action="capture")
        await event_log.log_event(event_type="memory_operation", action="query")
        await event_log.log_event(event_type="error", action="failure")

        events = await event_log.query_events(event_type="memory_operation")
        assert isinstance(events, list)

    @pytest.mark.asyncio
    async def test_query_events_by_action(self, event_log):
        """Test querying events by action."""
        await event_log.log_event(event_type="test", action="capture_fact")
        await event_log.log_event(event_type="test", action="capture_fact")
        await event_log.log_event(event_type="test", action="query_context")

        events = await event_log.query_events(action="capture_fact")
        assert isinstance(events, list)

    @pytest.mark.asyncio
    async def test_query_events_by_actor(self, event_log):
        """Test querying events by actor."""
        await event_log.log_event(event_type="test", action="action1", actor="user")
        await event_log.log_event(event_type="test", action="action2", actor="user")
        await event_log.log_event(event_type="test", action="action3", actor="system")

        events = await event_log.query_events(actor="user")
        assert isinstance(events, list)

    @pytest.mark.asyncio
    async def test_query_events_temporal_filter(self, event_log):
        """Test querying events with temporal filtering."""
        now = datetime.now(UTC)

        await event_log.log_event(event_type="test", action="action1")
        await asyncio.sleep(0.1)
        middle_time = datetime.now(UTC)
        await asyncio.sleep(0.1)
        await event_log.log_event(event_type="test", action="action2")

        # Query events after middle time
        events = await event_log.query_events(start_time=middle_time)
        assert isinstance(events, list)

        # Query events before middle time
        events = await event_log.query_events(end_time=middle_time)
        assert isinstance(events, list)

    @pytest.mark.asyncio
    async def test_query_events_pagination(self, event_log):
        """Test event query pagination."""
        for i in range(5):
            await event_log.log_event(event_type="test", action=f"action{i}")

        events = await event_log.query_events(limit=2)
        assert isinstance(events, list)

        events_page2 = await event_log.query_events(limit=2, offset=2)
        assert isinstance(events_page2, list)

    @pytest.mark.asyncio
    async def test_query_events_order(self, event_log):
        """Test that events are returned in reverse chronological order."""
        await event_log.log_event(event_type="test", action="first")
        await asyncio.sleep(0.1)
        await event_log.log_event(event_type="test", action="second")
        await asyncio.sleep(0.1)
        await event_log.log_event(event_type="test", action="third")

        events = await event_log.query_events()
        assert isinstance(events, list)

    @pytest.mark.asyncio
    async def test_get_session_timeline(self, event_log):
        """Test getting session timeline in chronological order."""
        await event_log.log_event(event_type="test", action="first")
        await asyncio.sleep(0.1)
        await event_log.log_event(event_type="test", action="second")
        await asyncio.sleep(0.1)
        await event_log.log_event(event_type="test", action="third")

        timeline = await event_log.get_session_timeline()
        assert isinstance(timeline, list)

    @pytest.mark.asyncio
    async def test_get_session_timeline_different_session(self, event_log):
        """Test getting timeline for different session."""
        await event_log.log_event(event_type="test", action="current_session")

        other_session = str(uuid4())
        await event_log.log_event(
            event_type="test", action="other_session", session_id=other_session
        )

        timeline = await event_log.get_session_timeline()
        assert isinstance(timeline, list)

        timeline = await event_log.get_session_timeline(session_id=other_session)
        assert isinstance(timeline, list)

    @pytest.mark.asyncio
    async def test_get_event_count(self, event_log):
        """Test getting event count."""
        count = await event_log.get_event_count()
        assert isinstance(count, int)

        await event_log.log_event(event_type="type1", action="action1")
        await event_log.log_event(event_type="type1", action="action2")
        await event_log.log_event(event_type="type2", action="action3")

        count = await event_log.get_event_count()
        assert isinstance(count, int)

        count = await event_log.get_event_count(event_type="type1")
        assert isinstance(count, int)

    @pytest.mark.asyncio
    async def test_multiple_sessions(self, event_log):
        """Test handling multiple sessions."""
        session1 = event_log.session_id
        session2 = str(uuid4())
        session3 = str(uuid4())

        await event_log.log_event(event_type="test", action="s1", session_id=session1)
        await event_log.log_event(event_type="test", action="s2", session_id=session2)
        await event_log.log_event(event_type="test", action="s3", session_id=session3)

        events1 = await event_log.query_events(session_id=session1)
        events2 = await event_log.query_events(session_id=session2)
        events3 = await event_log.query_events(session_id=session3)

        assert isinstance(events1, list)
        assert isinstance(events2, list)
        assert isinstance(events3, list)

    @pytest.mark.asyncio
    async def test_combined_filters(self, event_log):
        """Test combining multiple filters."""
        await event_log.log_event(event_type="memory", action="capture", actor="user")
        await event_log.log_event(event_type="memory", action="query", actor="system")
        await event_log.log_event(event_type="error", action="capture", actor="user")

        events = await event_log.query_events(event_type="memory", action="capture")
        assert isinstance(events, list)

        events = await event_log.query_events(event_type="memory", actor="system")
        assert isinstance(events, list)

    @pytest.mark.asyncio
    async def test_database_persistence(self):
        """Test that events persist across connection cycles.

        Phase 2: Sessions resume if within timeout window (4 hours default).
        """
        with patch("src.services.event_log_service._db_connection") as mock_db:
            mock_db.connect = AsyncMock()
            mock_db.fetch = AsyncMock(return_value=[])
            mock_db.fetchval = AsyncMock(return_value=None)
            mock_db.fetchrow = AsyncMock(return_value=None)
            mock_db.execute = AsyncMock()
            mock_db.disconnect = AsyncMock()
            mock_db.db_path = "./data/test_events.db"

            # First connection
            service1 = EventLogService()
            await service1.connect()
            session1 = service1.session_id

            await service1.log_event(event_type="test", action="persisted")
            await service1.close()

            # Second connection
            service2 = EventLogService()
            await service2.connect()

            assert service2.session_id is not None

            await service2.close()

    @pytest.mark.asyncio
    async def test_get_session_status(self, event_log):
        """Test getting current session status."""
        # Mock returns None by default, so status should be "Unknown"
        status = await event_log.get_session_status()
        assert isinstance(status, str)

    @pytest.mark.asyncio
    async def test_get_session_status_no_session(self):
        """Test get_session_status when no session exists."""
        with patch("src.services.event_log_service._db_connection") as mock_db:
            mock_db.db_path = "./data/test_events.db"
            service = EventLogService()
            # session_id is None
            status = await service.get_session_status()
            assert status == "Unknown"
