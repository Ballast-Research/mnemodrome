"""Unit tests for FileService memory type routing (Feature 010)."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import MemoryType
from src.services.file_service import FileService

# MNEMODROME_ env vars drive storage paths in config.py.
# Tests must clear them to avoid interference from a real .env.
MNEMODROME_VARS = [
    "MNEMODROME_FACTS_PATH",
    "MNEMODROME_LEARNINGS_PATH",
    "MNEMODROME_SESSIONS_PATH",
    "MNEMODROME_GRAPHS_PATH",
    "MNEMODROME_UPDATES_PATH",
    "MNEMODROME_PROCEDURES_PATH",
]


@pytest.fixture(autouse=True)
def _clear_mnemodrome_env():
    """Clear MNEMODROME_ env vars so each test controls its own paths."""
    saved = {k: os.environ.pop(k, None) for k in MNEMODROME_VARS}
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v


class TestFileServiceInitialization:
    """Test FileService initialization with memory types."""

    @patch.dict(
        os.environ,
        {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
            "MNEMODROME_FACTS_PATH": "/tmp/test_facts",
            "MNEMODROME_LEARNINGS_PATH": "/tmp/test_learnings",
            "MNEMODROME_SESSIONS_PATH": "/tmp/test_sessions",
            "MNEMODROME_GRAPHS_PATH": "/tmp/test_graphs",
            "MNEMODROME_PROCEDURES_PATH": "/tmp/test_procedures",
            "MNEMODROME_UPDATES_PATH": "/tmp/test_updates",
        },
        clear=True,
    )
    def test_initialization_caches_all_memory_paths(self):
        """Test that FileService caches all 4 memory type paths on init."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Update env vars to use temp directory
            os.environ["MNEMODROME_FACTS_PATH"] = f"{tmpdir}/facts"
            os.environ["MNEMODROME_LEARNINGS_PATH"] = f"{tmpdir}/learnings"
            os.environ["MNEMODROME_SESSIONS_PATH"] = f"{tmpdir}/sessions"
            os.environ["MNEMODROME_GRAPHS_PATH"] = f"{tmpdir}/graphs"
            os.environ["MNEMODROME_PROCEDURES_PATH"] = f"{tmpdir}/procedures"
            os.environ["MNEMODROME_UPDATES_PATH"] = f"{tmpdir}/updates"

            service = FileService()

            # Verify all paths are cached
            assert MemoryType.FACT in service.memory_paths
            assert MemoryType.LEARNING in service.memory_paths
            assert MemoryType.SESSION in service.memory_paths
            assert MemoryType.GRAPH in service.memory_paths

            # Verify paths are correct
            assert service.memory_paths[MemoryType.FACT] == Path(f"{tmpdir}/facts")
            assert service.memory_paths[MemoryType.LEARNING] == Path(
                f"{tmpdir}/learnings"
            )
            assert service.memory_paths[MemoryType.SESSION] == Path(
                f"{tmpdir}/sessions"
            )
            assert service.memory_paths[MemoryType.GRAPH] == Path(f"{tmpdir}/graphs")

    @patch.dict(
        os.environ,
        {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
            "MNEMODROME_FACTS_PATH": "/tmp/test_facts",
            "MNEMODROME_LEARNINGS_PATH": "/tmp/test_learnings",
            "MNEMODROME_SESSIONS_PATH": "/tmp/test_sessions",
            "MNEMODROME_GRAPHS_PATH": "/tmp/test_graphs",
            "MNEMODROME_PROCEDURES_PATH": "/tmp/test_procedures",
            "MNEMODROME_UPDATES_PATH": "/tmp/test_updates",
        },
        clear=True,
    )
    def test_initialization_creates_all_directories(self):
        """Test that FileService creates directories for all memory types on init."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Update env vars to use temp directory
            os.environ["MNEMODROME_FACTS_PATH"] = f"{tmpdir}/facts"
            os.environ["MNEMODROME_LEARNINGS_PATH"] = f"{tmpdir}/learnings"
            os.environ["MNEMODROME_SESSIONS_PATH"] = f"{tmpdir}/sessions"
            os.environ["MNEMODROME_GRAPHS_PATH"] = f"{tmpdir}/graphs"
            os.environ["MNEMODROME_PROCEDURES_PATH"] = f"{tmpdir}/procedures"
            os.environ["MNEMODROME_UPDATES_PATH"] = f"{tmpdir}/updates"

            service = FileService()

            # Verify all directories were created
            assert Path(f"{tmpdir}/facts").exists()
            assert Path(f"{tmpdir}/facts").is_dir()
            assert Path(f"{tmpdir}/learnings").exists()
            assert Path(f"{tmpdir}/learnings").is_dir()
            assert Path(f"{tmpdir}/sessions").exists()
            assert Path(f"{tmpdir}/sessions").is_dir()
            assert Path(f"{tmpdir}/graphs").exists()
            assert Path(f"{tmpdir}/graphs").is_dir()


class TestFileServiceWriteFile:
    """Test write_file() routes to correct directories."""

    @patch.dict(
        os.environ,
        {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
        },
        clear=True,
    )
    def test_write_file_routes_to_facts_directory(self):
        """Test that write_file routes to facts directory for FACT memory type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["MNEMODROME_FACTS_PATH"] = f"{tmpdir}/facts"
            os.environ["MNEMODROME_LEARNINGS_PATH"] = f"{tmpdir}/learnings"
            os.environ["MNEMODROME_SESSIONS_PATH"] = f"{tmpdir}/sessions"
            os.environ["MNEMODROME_GRAPHS_PATH"] = f"{tmpdir}/graphs"
            os.environ["MNEMODROME_PROCEDURES_PATH"] = f"{tmpdir}/procedures"
            os.environ["MNEMODROME_UPDATES_PATH"] = f"{tmpdir}/updates"

            service = FileService()
            filepath = service.write_file(
                "test_fact.md", "fact content", MemoryType.FACT
            )

            assert filepath == Path(f"{tmpdir}/facts/test_fact.md")
            assert filepath.exists()
            assert filepath.read_text() == "fact content"

    @patch.dict(
        os.environ,
        {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
        },
        clear=True,
    )
    def test_write_file_routes_to_learnings_directory(self):
        """Test that write_file routes to learnings directory for LEARNING memory type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["MNEMODROME_FACTS_PATH"] = f"{tmpdir}/facts"
            os.environ["MNEMODROME_LEARNINGS_PATH"] = f"{tmpdir}/learnings"
            os.environ["MNEMODROME_SESSIONS_PATH"] = f"{tmpdir}/sessions"
            os.environ["MNEMODROME_GRAPHS_PATH"] = f"{tmpdir}/graphs"
            os.environ["MNEMODROME_PROCEDURES_PATH"] = f"{tmpdir}/procedures"
            os.environ["MNEMODROME_UPDATES_PATH"] = f"{tmpdir}/updates"

            service = FileService()
            filepath = service.write_file(
                "test_learning.md", "learning content", MemoryType.LEARNING
            )

            assert filepath == Path(f"{tmpdir}/learnings/test_learning.md")
            assert filepath.exists()
            assert filepath.read_text() == "learning content"

    @patch.dict(
        os.environ,
        {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
        },
        clear=True,
    )
    def test_write_file_routes_to_sessions_directory(self):
        """Test that write_file routes to sessions directory for SESSION memory type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["MNEMODROME_FACTS_PATH"] = f"{tmpdir}/facts"
            os.environ["MNEMODROME_LEARNINGS_PATH"] = f"{tmpdir}/learnings"
            os.environ["MNEMODROME_SESSIONS_PATH"] = f"{tmpdir}/sessions"
            os.environ["MNEMODROME_GRAPHS_PATH"] = f"{tmpdir}/graphs"
            os.environ["MNEMODROME_PROCEDURES_PATH"] = f"{tmpdir}/procedures"
            os.environ["MNEMODROME_UPDATES_PATH"] = f"{tmpdir}/updates"

            service = FileService()
            filepath = service.write_file(
                "test_session.md", "session content", MemoryType.SESSION
            )

            assert filepath == Path(f"{tmpdir}/sessions/test_session.md")
            assert filepath.exists()
            assert filepath.read_text() == "session content"

    @patch.dict(
        os.environ,
        {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
        },
        clear=True,
    )
    def test_write_file_routes_to_graphs_directory(self):
        """Test that write_file routes to graphs directory for GRAPH memory type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["MNEMODROME_FACTS_PATH"] = f"{tmpdir}/facts"
            os.environ["MNEMODROME_LEARNINGS_PATH"] = f"{tmpdir}/learnings"
            os.environ["MNEMODROME_SESSIONS_PATH"] = f"{tmpdir}/sessions"
            os.environ["MNEMODROME_GRAPHS_PATH"] = f"{tmpdir}/graphs"
            os.environ["MNEMODROME_PROCEDURES_PATH"] = f"{tmpdir}/procedures"
            os.environ["MNEMODROME_UPDATES_PATH"] = f"{tmpdir}/updates"

            service = FileService()
            filepath = service.write_file(
                "test_graph.md", "graph content", MemoryType.GRAPH
            )

            assert filepath == Path(f"{tmpdir}/graphs/test_graph.md")
            assert filepath.exists()
            assert filepath.read_text() == "graph content"

    @patch.dict(
        os.environ,
        {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
        },
        clear=True,
    )
    def test_write_file_defaults_to_facts(self):
        """Test that write_file defaults to FACT memory type when not specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["MNEMODROME_FACTS_PATH"] = f"{tmpdir}/facts"
            os.environ["MNEMODROME_LEARNINGS_PATH"] = f"{tmpdir}/learnings"
            os.environ["MNEMODROME_SESSIONS_PATH"] = f"{tmpdir}/sessions"
            os.environ["MNEMODROME_GRAPHS_PATH"] = f"{tmpdir}/graphs"
            os.environ["MNEMODROME_PROCEDURES_PATH"] = f"{tmpdir}/procedures"
            os.environ["MNEMODROME_UPDATES_PATH"] = f"{tmpdir}/updates"

            service = FileService()
            # Call without memory_type parameter
            filepath = service.write_file("default_test.md", "default content")

            assert filepath == Path(f"{tmpdir}/facts/default_test.md")
            assert filepath.exists()


class TestFileServiceValidatePath:
    """Test validate_path() for different memory types."""

    @patch.dict(
        os.environ,
        {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
        },
        clear=True,
    )
    def test_validate_path_succeeds_for_all_memory_types(self):
        """Test that validate_path succeeds for all memory types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["MNEMODROME_FACTS_PATH"] = f"{tmpdir}/facts"
            os.environ["MNEMODROME_LEARNINGS_PATH"] = f"{tmpdir}/learnings"
            os.environ["MNEMODROME_SESSIONS_PATH"] = f"{tmpdir}/sessions"
            os.environ["MNEMODROME_GRAPHS_PATH"] = f"{tmpdir}/graphs"
            os.environ["MNEMODROME_PROCEDURES_PATH"] = f"{tmpdir}/procedures"
            os.environ["MNEMODROME_UPDATES_PATH"] = f"{tmpdir}/updates"

            service = FileService()

            # All should pass since __init__ creates directories
            assert service.validate_path(MemoryType.FACT) is True
            assert service.validate_path(MemoryType.LEARNING) is True
            assert service.validate_path(MemoryType.SESSION) is True
            assert service.validate_path(MemoryType.GRAPH) is True

    @patch.dict(
        os.environ,
        {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
        },
        clear=True,
    )
    def test_validate_path_raises_ioerror_when_path_missing(self):
        """Test that validate_path raises IOError when directory doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["MNEMODROME_FACTS_PATH"] = f"{tmpdir}/facts"
            os.environ["MNEMODROME_LEARNINGS_PATH"] = f"{tmpdir}/learnings"
            os.environ["MNEMODROME_SESSIONS_PATH"] = f"{tmpdir}/sessions"
            os.environ["MNEMODROME_GRAPHS_PATH"] = f"{tmpdir}/graphs"
            os.environ["MNEMODROME_PROCEDURES_PATH"] = f"{tmpdir}/procedures"
            os.environ["MNEMODROME_UPDATES_PATH"] = f"{tmpdir}/updates"

            service = FileService()

            # Remove a directory
            Path(f"{tmpdir}/facts").rmdir()

            with pytest.raises(IOError) as exc_info:
                service.validate_path(MemoryType.FACT)

            assert "does not exist" in str(exc_info.value)
