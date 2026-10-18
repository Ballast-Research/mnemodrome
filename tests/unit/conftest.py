"""Unit test fixtures — pure mocks, no I/O, no network."""

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(autouse=True)
def _reset_shared_sqlite_connection():
    """Forget the shared SQLite event-log connection after each test.

    The connection is a module singleton. Tests that go through a TestClient
    (Admin UI, Memory Browser) open it inside that client's event loop, which
    closes when the client exits; the connection still says "connected" but
    its worker can no longer answer, so the next test that uses it waits
    forever. Production runs one event loop and is not affected.
    """
    yield
    from src.database.sqlite import sqlite_connection

    sqlite_connection._db = None
    sqlite_connection._connected = False


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j async driver returning empty results."""
    driver = AsyncMock()
    session = AsyncMock()
    tx = AsyncMock()

    # session context manager
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    # transaction context manager
    tx.__aenter__ = AsyncMock(return_value=tx)
    tx.__aexit__ = AsyncMock(return_value=False)

    # Default: queries return empty results
    result = AsyncMock()
    result.single = AsyncMock(return_value=None)
    result.data = AsyncMock(return_value=[])
    result.values = AsyncMock(return_value=[])
    tx.run = AsyncMock(return_value=result)
    session.run = AsyncMock(return_value=result)
    session.begin_transaction = AsyncMock(return_value=tx)

    driver.session = MagicMock(return_value=session)
    driver.close = AsyncMock()

    return driver


@pytest.fixture
def mock_neo4j_service(mock_neo4j_driver):
    """Mock Neo4jService with driver pre-configured."""
    service = AsyncMock()
    service.driver = mock_neo4j_driver
    service.verify_connectivity = AsyncMock(return_value=True)
    service.execute_query = AsyncMock(return_value=[])
    service.execute_write = AsyncMock(return_value=None)
    return service


@pytest.fixture
def mock_redis_client():
    """Mock Redis async client with in-memory behavior."""
    client = AsyncMock()
    _store = {}

    async def mock_get(key):
        return _store.get(key)

    async def mock_set(key, value, *args, **kwargs):
        _store[key] = value
        return True

    async def mock_delete(*keys):
        for k in keys:
            _store.pop(k, None)
        return len(keys)

    async def mock_exists(*keys):
        return sum(1 for k in keys if k in _store)

    client.get = AsyncMock(side_effect=mock_get)
    client.set = AsyncMock(side_effect=mock_set)
    client.delete = AsyncMock(side_effect=mock_delete)
    client.exists = AsyncMock(side_effect=mock_exists)
    client.ping = AsyncMock(return_value=True)
    client.close = AsyncMock()
    client._store = _store  # Expose for test assertions

    return client


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client returning empty search results."""
    client = AsyncMock()
    client.search = AsyncMock(return_value=[])
    client.upsert = AsyncMock(return_value=None)
    client.delete = AsyncMock(return_value=None)
    client.get_collection = AsyncMock(return_value=MagicMock(points_count=0))
    client.collection_exists = AsyncMock(return_value=True)
    return client


@pytest.fixture
def mock_event_log():
    """Mock EventLogService that no-ops all operations."""
    service = AsyncMock()
    service.log_event = AsyncMock(return_value=None)
    service.connect = AsyncMock(return_value=None)
    service.close = AsyncMock(return_value=None)
    service.session_id = "test-session-00000000"
    service._auto_disabled = False
    service._metrics = MagicMock()
    service._metrics.get_p95_write = MagicMock(return_value=1.0)
    service._metrics.get_p99_write = MagicMock(return_value=2.0)
    service._metrics.get_p95_query = MagicMock(return_value=1.0)
    service._metrics.get_p99_query = MagicMock(return_value=2.0)
    service._metrics.sla_violations = 0
    service._metrics.last_violation_at = None
    return service


@pytest.fixture
def mock_llm_service():
    """Mock LLM service returning canned classification responses."""
    service = AsyncMock()
    service.classify_entity = AsyncMock(return_value="Concept")
    service.extract_entities = AsyncMock(return_value=[])
    service.generate = AsyncMock(return_value="Test response")
    service.chat = AsyncMock(return_value="Test response")
    return service


@pytest.fixture
def mock_file_service(tmp_path):
    """Mock file service using tmp_path for file operations."""
    service = AsyncMock()
    service.write_file = AsyncMock(return_value=str(tmp_path / "test.md"))
    service.read_file = AsyncMock(return_value="# Test Content")
    service.list_files = AsyncMock(return_value=[])
    service.facts_path = tmp_path / "facts"
    service.learnings_path = tmp_path / "learnings"
    service.sessions_path = tmp_path / "sessions"
    service.graphs_path = tmp_path / "graphs"

    for p in [
        service.facts_path,
        service.learnings_path,
        service.sessions_path,
        service.graphs_path,
    ]:
        p.mkdir(parents=True, exist_ok=True)

    return service


@pytest.fixture
def mock_vector_service():
    """Mock VectorService with no-op operations."""
    service = AsyncMock()
    service.enabled = False
    service.initialized = False
    service.search = AsyncMock(return_value=[])
    service.upsert = AsyncMock(return_value=None)
    service.search_similar_learnings = AsyncMock(return_value=[])
    service.generate_embedding = AsyncMock(return_value=[0.0] * 768)
    return service
