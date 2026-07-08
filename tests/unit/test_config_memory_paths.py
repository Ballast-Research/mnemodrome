"""Unit tests for memory path configuration (Feature 010)."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import MemoryType, get_memory_path, validate_config

# All storage-related env vars that must be controlled in tests
ALL_STORAGE_VARS = [
    "MNEMODROME_FACTS_PATH",
    "MNEMODROME_LEARNINGS_PATH",
    "MNEMODROME_SESSIONS_PATH",
    "MNEMODROME_GRAPHS_PATH",
    "MNEMODROME_UPDATES_PATH",
    "MNEMODROME_PROCEDURES_PATH",
]


def _clear_storage_vars():
    """Remove all storage env vars to ensure test isolation."""
    for var in ALL_STORAGE_VARS:
        os.environ.pop(var, None)


class TestMemoryTypeEnum:
    """Test MemoryType enum definition."""

    def test_memory_type_has_six_types(self):
        """Verify MemoryType enum has exactly 6 types."""
        assert len(MemoryType) == 6

    def test_memory_type_values(self):
        """Verify MemoryType enum values match directory names."""
        assert MemoryType.FACT.value == "facts"
        assert MemoryType.LEARNING.value == "learnings"
        assert MemoryType.SESSION.value == "sessions"
        assert MemoryType.GRAPH.value == "graphs"
        assert MemoryType.UPDATE.value == "updates"
        assert MemoryType.PROCEDURE.value == "procedures"

    def test_memory_type_unique_values(self):
        """Verify all MemoryType values are unique."""
        values = [mt.value for mt in MemoryType]
        assert len(values) == len(set(values))


class TestGetMemoryPathWithMnemodromePaths:
    """Test get_memory_path() with MNEMODROME_ prefix."""

    @patch.dict(
        os.environ,
        {
            "MNEMODROME_FACTS_PATH": "/data/facts",
        },
        clear=False,
    )
    def test_mnemodrome_prefix_facts(self):
        """Test MNEMODROME_ prefix returns correct path."""
        path = get_memory_path(MemoryType.FACT)
        assert path == Path("/data/facts")

    @patch.dict(
        os.environ,
        {
            "MNEMODROME_LEARNINGS_PATH": "/data/learnings",
        },
        clear=False,
    )
    def test_mnemodrome_prefix_learnings(self):
        path = get_memory_path(MemoryType.LEARNING)
        assert path == Path("/data/learnings")

    @patch.dict(
        os.environ,
        {
            "MNEMODROME_SESSIONS_PATH": "/data/sessions",
        },
        clear=False,
    )
    def test_mnemodrome_prefix_sessions(self):
        path = get_memory_path(MemoryType.SESSION)
        assert path == Path("/data/sessions")

    @patch.dict(
        os.environ,
        {
            "MNEMODROME_GRAPHS_PATH": "/data/graphs",
        },
        clear=False,
    )
    def test_mnemodrome_prefix_graphs(self):
        path = get_memory_path(MemoryType.GRAPH)
        assert path == Path("/data/graphs")


class TestGetMemoryPathErrorHandling:
    """Test get_memory_path() error handling when no paths configured."""

    @patch.dict(os.environ, {}, clear=False)
    def test_raises_value_error_when_no_config_facts(self):
        """Test ValueError raised when no path configured for FACT."""
        _clear_storage_vars()
        with pytest.raises(ValueError) as exc_info:
            get_memory_path(MemoryType.FACT)
        assert "No storage path configured" in str(exc_info.value)
        assert "MNEMODROME_FACTS_PATH" in str(exc_info.value)

    @patch.dict(os.environ, {}, clear=False)
    def test_raises_value_error_when_no_config_learnings(self):
        """Test ValueError raised when no path configured for LEARNING."""
        _clear_storage_vars()
        with pytest.raises(ValueError) as exc_info:
            get_memory_path(MemoryType.LEARNING)
        assert "No storage path configured" in str(exc_info.value)
        assert "MNEMODROME_LEARNINGS_PATH" in str(exc_info.value)


class TestValidateConfig:
    """Test validate_config() function."""

    @patch.dict(
        os.environ,
        {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
            "MNEMODROME_FACTS_PATH": "/data/facts",
            "LLM_PROVIDER": "ollama",
            "CHAT_MODEL": "test-model",
        },
        clear=False,
    )
    def test_validate_config_passes_with_mnemodrome_paths(self):
        """Test validate_config passes with MNEMODROME_ prefix."""
        validate_config()

    @patch.dict(
        os.environ,
        {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
            "LLM_PROVIDER": "ollama",
            "CHAT_MODEL": "test-model",
        },
        clear=False,
    )
    def test_validate_config_fails_without_storage_paths(self):
        """Test validate_config fails when no storage paths configured."""
        _clear_storage_vars()
        with pytest.raises(ValueError) as exc_info:
            validate_config()
        assert "No storage paths configured" in str(exc_info.value)

    @patch.dict(
        os.environ,
        {
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
            "MNEMODROME_FACTS_PATH": "/data/facts",
            "LLM_PROVIDER": "ollama",
            "CHAT_MODEL": "test-model",
        },
        clear=False,
    )
    def test_validate_config_fails_without_neo4j_uri(self):
        """Test validate_config fails when NEO4J_URI missing."""
        os.environ.pop("NEO4J_URI", None)
        with pytest.raises(ValueError) as exc_info:
            validate_config()
        assert "Missing required environment variables" in str(exc_info.value)
        assert "NEO4J_URI" in str(exc_info.value)
