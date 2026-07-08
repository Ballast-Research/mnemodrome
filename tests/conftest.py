"""pytest configuration and fixtures.

IMPORTANT: Environment defaults MUST be set before any src imports.
The config module reads env vars at import time and will fail without these.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

# ========================================
# Environment defaults for test isolation
# MUST be set before ANY src imports
# ========================================
_TEST_ENV_DEFAULTS = {
    "LLM_PROVIDER": "ollama",
    "LLM_BASE_URL": "http://localhost:11434",
    "CHAT_MODEL": "test-model",
    "NEO4J_URI": "bolt://localhost:7687",
    "NEO4J_HOST": "localhost",
    "NEO4J_USER": "neo4j",
    "NEO4J_PASSWORD": "test-password",
    "NEO4J_DATABASE": "neo4j",
    "REDIS_HOST": "localhost",
    "REDIS_PORT": "6379",
    "QDRANT_HOST": "localhost",
    "QDRANT_PORT": "6333",
    "QDRANT_ENABLED": "false",
    "GPU_CACHE_ENABLED": "false",
    "ADMIN_PASSWORD": "test-password",
    "ADMIN_WEB_ENABLED": "false",
    "EVENT_LOG_ENABLED": "false",
    "EVENT_LOG_FAIL_LOUD": "false",
    "ACTR_ENABLED": "true",
    "MCP_TRANSPORT": "streamable-http",
    "OLLAMA_HOST": "http://localhost:11434",
    "OLLAMA_EMBEDDING_HOST": "http://localhost:11434",
    "OLLAMA_EMBEDDING_MODEL": "nomic-embed-text",
    # Granular storage paths — required by config.validate_config() at import time.
    # Set here so the suite is self-contained and never depends on a developer's .env.
    "MNEMODROME_FACTS_PATH": "/tmp/mnemodrome-test/facts",
    "MNEMODROME_LEARNINGS_PATH": "/tmp/mnemodrome-test/learnings",
    "MNEMODROME_SESSIONS_PATH": "/tmp/mnemodrome-test/sessions",
    "MNEMODROME_GRAPHS_PATH": "/tmp/mnemodrome-test/graphs",
    "MNEMODROME_UPDATES_PATH": "/tmp/mnemodrome-test/updates",
    "MNEMODROME_PROCEDURES_PATH": "/tmp/mnemodrome-test/procedures",
}

for key, value in _TEST_ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)


import pytest
import pytest_asyncio


def pytest_configure(config):
    """Register custom markers so pytest --strict-markers doesn't reject them."""
    config.addinivalue_line(
        "markers",
        "postgres: tests requiring a live PostgreSQL instance",
    )


# Add the project root to sys.path so tests can import src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import config

# ========================================
# Feature 023: Deduplication Fixtures
# ========================================


@pytest.fixture
def dedup_config():
    """Provide DeduplicationConfig with test values."""
    # Return a dict of config values that tests can use
    return {
        "auto_skip_threshold": config.DEDUP_AUTO_SKIP_THRESHOLD,
        "validation_threshold": config.DEDUP_VALIDATION_THRESHOLD,
        "max_validation_time_ms": config.DEDUP_MAX_VALIDATION_TIME_MS,
        "vector_search_limit": config.DEDUP_VECTOR_SEARCH_LIMIT,
        "fuzzy_match_threshold": config.DEDUP_FUZZY_MATCH_THRESHOLD,
        "enable_graceful_degradation": config.DEDUP_ENABLE_GRACEFUL_DEGRADATION,
        "bootstrap_batch_size": config.DEDUP_BOOTSTRAP_BATCH_SIZE,
        "bootstrap_max_failures": config.DEDUP_BOOTSTRAP_MAX_FAILURES,
    }


@pytest_asyncio.fixture
async def dedup_service():
    """Provide initialized DeduplicationService instance.

    Note: This fixture will fail until DeduplicationService is implemented (T011-T018).
    """
    # Will be implemented after DeduplicationService exists
    pytest.skip("DeduplicationService not yet implemented (pending T011-T018)")


@pytest.fixture
def existing_learning() -> dict:
    """Provide test learning data for contract tests (mocked scenarios).

    Returns:
        dict: Learning metadata with uuid, title, content, file_path

    Note: This is a simple fixture for contract tests using mocks.
    For integration tests with real services, use existing_learning_real fixture.
    """
    learning_uuid = "test-learning-existing-12345"
    learning_title = "Test Learning: Factory Pattern Implementation"
    learning_content = """
    The Factory Pattern is a creational design pattern that provides an interface
    for creating objects without specifying their exact classes. This pattern is
    particularly useful when dealing with entity creation in graph databases like Neo4j.
    """
    learning_file_path = f"/tmp/test-learnings/learning-{learning_uuid}.md"

    return {
        "uuid": learning_uuid,
        "title": learning_title,
        "content": learning_content,
        "file_path": learning_file_path,
    }


@pytest.fixture
def similar_learning() -> dict:
    """Provide test learning data with ~92% similarity for contract tests.

    Returns:
        dict: Learning metadata with uuid, title, content, file_path

    Note: This is a simple fixture for contract tests using mocks.
    For integration tests with real services, use similar_learning_real fixture.
    """
    learning_uuid = "test-learning-similar-67890"
    learning_title = "Test Learning: Factory Pattern Usage"
    learning_content = """
    The Factory Pattern provides a creational design interface for object creation
    without specifying exact classes. This is especially valuable for entity creation
    in graph databases such as Neo4j.
    """
    learning_file_path = f"/tmp/test-learnings/learning-{learning_uuid}.md"

    return {
        "uuid": learning_uuid,
        "title": learning_title,
        "content": learning_content,
        "file_path": learning_file_path,
    }


@pytest.fixture
def mock_qdrant_unavailable():
    """Mock VectorService to raise connection error (Qdrant unavailable scenario).

    Usage:
        def test_graceful_degradation(mock_qdrant_unavailable, dedup_service):
            with mock_qdrant_unavailable:
                result = await dedup_service.validate_learning(title, content)
                assert result.validation_stage == "fuzzy_fallback"
    """
    return patch(
        "src.services.vector_service.VectorService.search_similar_learnings",
        side_effect=ConnectionError("Qdrant unavailable: connection refused"),
    )


@pytest.fixture
def test_learnings_dir(tmp_path) -> Path:
    """Provide temporary directory for test markdown files.

    Returns:
        Path: Temporary directory path for test learning files
    """
    learnings_dir = tmp_path / "test_learnings"
    learnings_dir.mkdir(parents=True, exist_ok=True)
    return learnings_dir
