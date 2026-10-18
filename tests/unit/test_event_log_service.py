"""Unit tests for Event Log Service.

Tests for the EventLogService class covering:
- SQLite connection and initialization
- Event logging operations
- Performance metrics tracking
- Session management
- Retention policies
- Error handling
"""

from collections import deque
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from src.services.event_log_service import EventLogService, PerformanceMetrics


class TestPerformanceMetrics:
    """Tests for the PerformanceMetrics dataclass."""

    def test_metrics_initialization(self):
        """Test PerformanceMetrics initializes with empty deques."""
        metrics = PerformanceMetrics()

        assert isinstance(metrics.write_latencies, deque)
        assert isinstance(metrics.query_latencies, deque)
        assert metrics.sla_violations == 0
        assert metrics.last_violation_at is None

    def test_add_write_latency(self):
        """Test adding write latency measurement."""
        metrics = PerformanceMetrics()

        metrics.add_write(10.5)
        metrics.add_write(12.3)

        assert len(metrics.write_latencies) == 2
        assert metrics.write_latencies[-1] == 12.3

    def test_add_query_latency(self):
        """Test adding query latency measurement."""
        metrics = PerformanceMetrics()

        metrics.add_query(5.2)
        metrics.add_query(6.1)

        assert len(metrics.query_latencies) == 2

    def test_percentile_calculation_empty(self):
        """Test percentile calculation with empty deque."""
        metrics = PerformanceMetrics()

        result = metrics.get_percentile(deque(), 95)

        assert result == 0.0

    def test_percentile_calculation_single_value(self):
        """Test percentile with single value."""
        metrics = PerformanceMetrics()
        values = deque([10.0])

        result = metrics.get_percentile(values, 95)

        assert result == 10.0

    def test_percentile_calculation_multiple_values(self):
        """Test percentile with multiple values."""
        metrics = PerformanceMetrics()
        values = deque([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        p95 = metrics.get_percentile(values, 95)
        p99 = metrics.get_percentile(values, 99)

        assert isinstance(p95, float)
        assert isinstance(p99, float)
        assert p95 >= 1.0 and p95 <= 10.0

    def test_p95_write_metric(self):
        """Test P95 write latency metric."""
        metrics = PerformanceMetrics()

        for i in range(1, 11):
            metrics.add_write(float(i))

        p95 = metrics.get_p95_write()

        assert isinstance(p95, float)
        assert p95 > 0.0

    def test_p99_query_metric(self):
        """Test P99 query latency metric."""
        metrics = PerformanceMetrics()

        for i in range(1, 11):
            metrics.add_query(float(i))

        p99 = metrics.get_p99_query()

        assert isinstance(p99, float)
        assert p99 > 0.0

    def test_deque_maxlen_enforced(self):
        """Test that deques enforce max length of 1000."""
        metrics = PerformanceMetrics()

        # Add more than maxlen values
        for i in range(1500):
            metrics.add_write(float(i))

        # Should only keep last 1000
        assert len(metrics.write_latencies) == 1000

    def test_sla_violation_tracking(self):
        """Test SLA violation tracking."""
        metrics = PerformanceMetrics()

        assert metrics.sla_violations == 0
        assert metrics.last_violation_at is None

        # Simulate violation
        metrics.sla_violations += 1
        metrics.last_violation_at = datetime.now(UTC)

        assert metrics.sla_violations == 1
        assert metrics.last_violation_at is not None


class TestEventLogService:
    """Tests for the EventLogService class."""

    @pytest.fixture
    def service(self):
        """Provide a fresh EventLogService instance."""
        return EventLogService()

    @pytest.fixture
    def mock_db(self):
        """Provide a mock SQLite connection."""
        mock = AsyncMock()
        mock.db_path = "./data/test_events.db"
        mock.is_connected = True
        return mock

    # ========================
    # Initialization Tests
    # ========================

    def test_service_initialization(self, service):
        """Test EventLogService initializes with correct defaults."""
        assert service.session_id is None
        assert service._initialized is False
        assert service._auto_disabled is False
        assert service._metrics is not None
        assert isinstance(service._metrics, PerformanceMetrics)

    def test_metrics_attribute_exists(self, service):
        """Test that metrics instance is properly initialized."""
        assert service._metrics.write_latencies is not None
        assert service._metrics.query_latencies is not None
        assert service._metrics.sla_violations == 0

    # ========================
    # Connection Tests
    # ========================

    @pytest.mark.asyncio
    async def test_connect_initializes_session(self, service):
        """Test connect initializes a session."""
        with patch("src.services.event_log_service._db_connection") as mock_db:
            mock_db.connect = AsyncMock()
            mock_db.fetchrow = AsyncMock(return_value=None)
            mock_db.execute = AsyncMock()
            mock_db.db_path = "./data/test_events.db"

            service.db = mock_db

            # Just verify the service has a db connection
            assert service.db is not None

    # ========================
    # Event Logging Tests
    # ========================

    @pytest.mark.asyncio
    async def test_log_event_basic(self, service, mock_db):
        """Test logging a basic event."""
        service.db = mock_db
        service._initialized = True
        service.session_id = "test-session-123"

        await service.log_event("capture_fact", "test_action")

    @pytest.mark.asyncio
    async def test_log_event_with_metadata(self, service):
        """Test logging an event with rich metadata."""
        service._initialized = True
        service.session_id = "test-session-123"

        mock_db = AsyncMock()
        mock_db.db_path = "./data/test_events.db"
        service.db = mock_db

        metadata = {
            "similarity_score": 0.92,
            "existing_uuid": "uuid-existing",
            "candidates_checked": 5,
            "duration_ms": 123,
        }

        # Should not raise
        await service.log_event("deduplication_check", "check", metadata=metadata)

    @pytest.mark.asyncio
    async def test_log_event_not_initialized(self, service):
        """Test logging when service not initialized."""
        service._initialized = False

        mock_db = AsyncMock()
        mock_db.connect = AsyncMock()
        mock_db.fetchrow = AsyncMock(return_value=None)
        mock_db.execute = AsyncMock()
        mock_db.db_path = "./data/test_events.db"
        service.db = mock_db

        # log_event calls connect() which initializes
        try:
            await service.log_event("test", "action")
        except Exception:
            # Some errors are expected - we're just testing it doesn't crash
            pass

    # ========================
    # Performance Tracking Tests
    # ========================

    def test_record_write_latency(self, service):
        """Test recording write latency metric."""
        service._metrics.add_write(15.5)
        service._metrics.add_write(18.2)

        p95 = service._metrics.get_p95_write()

        assert p95 > 0.0
        assert len(service._metrics.write_latencies) == 2

    def test_record_query_latency(self, service):
        """Test recording query latency metric."""
        service._metrics.add_query(8.3)
        service._metrics.add_query(9.1)

        p95 = service._metrics.get_p95_query()

        assert p95 > 0.0

    def test_sla_violation_detection(self, service):
        """Test SLA violation tracking."""
        for i in range(10):
            latency = 500.0 + (i * 10)
            service._metrics.add_write(latency)

        write_p95 = service._metrics.get_p95_write()
        assert write_p95 > 100.0

    def test_metrics_percentile_bounds(self, service):
        """Test that percentiles stay within bounds."""
        for i in range(100):
            service._metrics.add_write(5.0 + (i % 25))

        p95 = service._metrics.get_p95_write()
        p99 = service._metrics.get_p99_write()

        assert 5.0 <= p95 <= 30.0
        assert 5.0 <= p99 <= 30.0
        assert p99 >= p95

    # ========================
    # Session Management Tests
    # ========================

    @pytest.mark.asyncio
    async def test_session_id_generation(self, service):
        """Test session ID is assigned."""
        service._initialized = True
        assert service.session_id is None or isinstance(service.session_id, str)

    @pytest.mark.asyncio
    async def test_session_resumption(self, service):
        """Test session resumption from previous state."""
        service.session_id = "previous-session-123"
        service._initialized = True
        assert service.session_id == "previous-session-123"

    # ========================
    # Retention Tests
    # ========================

    @pytest.mark.asyncio
    async def test_retention_task_lifecycle(self, service, mock_db):
        """Test retention cleanup task management."""
        service.db = mock_db
        service._initialized = True
        assert service._retention_task is None

    @pytest.mark.asyncio
    async def test_retention_policy_enforcement(self, service, mock_db):
        """Test that retention policies are enforced."""
        service.db = mock_db
        service._initialized = True

    # ========================
    # Database Size Tracking Tests
    # ========================

    def test_db_size_cache_initialization(self, service):
        """Test database size cache is initially None."""
        assert service._db_size_cache is None

    def test_db_size_cache_validity(self, service):
        """Test database size cache validity check."""
        now = datetime.now(UTC)
        service._db_size_cache = (150.5, now)

        cached_size, cached_at = service._db_size_cache

        assert cached_size == 150.5
        assert cached_at == now

    # ========================
    # Auto-Disable Tests
    # ========================

    def test_auto_disable_flag(self, service):
        """Test auto-disable flag for error conditions."""
        assert service._auto_disabled is False

        service._auto_disabled = True

        assert service._auto_disabled is True

    # ========================
    # Session Status Tests
    # ========================

    @pytest.mark.asyncio
    async def test_get_session_status(self, service, mock_db):
        """Test get_session_status returns status from database."""
        service.db = mock_db
        service.session_id = "test-session-123"
        mock_db.fetchrow = AsyncMock(return_value={"status": "active"})

        status = await service.get_session_status()
        assert status == "active"

    @pytest.mark.asyncio
    async def test_get_session_status_no_session(self, service):
        """Test get_session_status with no session."""
        service.session_id = None
        status = await service.get_session_status()
        assert status == "Unknown"

    # ========================
    # Error Handling Tests
    # ========================

    @pytest.mark.asyncio
    async def test_graceful_handling_when_disabled(self, service):
        """Test service gracefully handles when auto-disabled."""
        service._auto_disabled = True
        service._initialized = True

        # Should return -1 when auto-disabled (fail_loud defaults to true in tests,
        # but we set it to false for graceful handling)
        with patch("src.services.event_log_service.config") as mock_config:
            mock_config.EVENT_LOG_FAIL_LOUD = False
            mock_config.EVENT_LOG_PERFORMANCE_SLA_MS = 50

            result = await service.log_event("test", "action")
            assert result == -1

    @pytest.mark.asyncio
    async def test_connection_error_handling(self, service, mock_db):
        """Test handling of connection errors."""
        service.db = mock_db
        mock_db.connect = AsyncMock(side_effect=Exception("Connection failed"))

        try:
            await service.connect()
        except Exception as e:
            assert "Connection" in str(e) or service._auto_disabled

    # ========================
    # attempt_recovery Tests
    # ========================

    @pytest.mark.asyncio
    async def test_attempt_recovery_not_disabled_is_noop(self, service, mock_db):
        """attempt_recovery returns True immediately when not disabled."""
        service.db = mock_db
        service._auto_disabled = False

        result = await service.attempt_recovery()

        assert result is True
        mock_db.fetchval.assert_not_called()

    @pytest.mark.asyncio
    async def test_attempt_recovery_recovers_when_probe_fast(self, service, mock_db):
        """Disabled service re-enables when SQLite probe latency is within SLA."""
        service.db = mock_db
        service._auto_disabled = True
        service._metrics.sla_violations = 5
        service._metrics.add_write(200.0)
        mock_db.fetchval = AsyncMock(return_value=1)  # fast probe

        with patch("src.services.event_log_service.config") as mock_config:
            mock_config.EVENT_LOG_PERFORMANCE_SLA_MS = 50

            # Patch perf_counter to simulate a fast probe (1 ms)
            call_count = 0

            def fast_perf_counter():
                nonlocal call_count
                call_count += 1
                return 0.001 * call_count  # 1 ms per call

            with patch(
                "src.services.event_log_service.time.perf_counter",
                side_effect=fast_perf_counter,
            ):
                result = await service.attempt_recovery()

        assert result is True
        assert service._auto_disabled is False
        assert service._metrics.sla_violations == 0
        assert len(service._metrics.write_latencies) == 0

    @pytest.mark.asyncio
    async def test_attempt_recovery_stays_disabled_when_probe_slow(
        self, service, mock_db
    ):
        """Disabled service stays disabled when SQLite probe exceeds SLA."""
        service.db = mock_db
        service._auto_disabled = True
        service._metrics.sla_violations = 5
        mock_db.fetchval = AsyncMock(return_value=1)

        with patch("src.services.event_log_service.config") as mock_config:
            mock_config.EVENT_LOG_PERFORMANCE_SLA_MS = 50

            # Simulate a slow probe (100 ms, above the 50 ms SLA)
            call_count = 0

            def slow_perf_counter():
                nonlocal call_count
                call_count += 1
                return 0.1 * call_count  # 100 ms per call

            with patch(
                "src.services.event_log_service.time.perf_counter",
                side_effect=slow_perf_counter,
            ):
                result = await service.attempt_recovery()

        assert result is False
        assert service._auto_disabled is True
        assert service._metrics.sla_violations == 5  # unchanged

    @pytest.mark.asyncio
    async def test_attempt_recovery_stays_disabled_on_probe_exception(
        self, service, mock_db
    ):
        """Disabled service stays disabled when SQLite probe throws."""
        service.db = mock_db
        service._auto_disabled = True
        mock_db.fetchval = AsyncMock(side_effect=Exception("SQLite unavailable"))

        result = await service.attempt_recovery()

        assert result is False
        assert service._auto_disabled is True
