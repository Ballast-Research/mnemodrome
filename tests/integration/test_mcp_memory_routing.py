"""Integration tests for MCP memory type routing (Feature 010)."""

import os
import tempfile
from unittest.mock import patch

import pytest

# These tests verify that the MemoryType parameter is correctly passed to file service
# by each MCP tool. They use temp directories and environment variable patching.


class TestMCPMemoryRouting:
    """Test that memory type routing works correctly in file service integration."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directories for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set up environment variables for granular paths
            env_vars = {
                "NEO4J_URI": "bolt://localhost:7687",
                "NEO4J_USER": "neo4j",
                "NEO4J_PASSWORD": "password",
                "MNEMODROME_FACTS_PATH": f"{tmpdir}/facts",
                "MNEMODROME_LEARNINGS_PATH": f"{tmpdir}/learnings",
                "MNEMODROME_SESSIONS_PATH": f"{tmpdir}/sessions",
                "MNEMODROME_GRAPHS_PATH": f"{tmpdir}/graphs",
                "MNEMODROME_PROCEDURES_PATH": f"{tmpdir}/procedures",
            }
            with patch.dict(os.environ, env_vars, clear=True):
                yield tmpdir

    def test_health_check_reports_all_paths(self, temp_storage):
        """Test that health check reports all 4 storage paths."""
        # Import after environment is set

        from src.config import MemoryType
        from src.services.file_service import FileService

        # Create FileService instance
        service = FileService()

        # Simulate health check logic
        storage_paths = {}
        for memory_type in MemoryType:
            try:
                service.validate_path(memory_type)
                path = service._get_target_path(memory_type)
                storage_paths[memory_type.value] = str(path)
            except Exception as e:
                storage_paths[memory_type.value] = f"error: {e}"

        # Verify all 4 memory type paths are reported
        assert "facts" in storage_paths
        assert "learnings" in storage_paths
        assert "sessions" in storage_paths
        assert "graphs" in storage_paths

        # Verify paths are correct
        assert f"{temp_storage}/facts" in storage_paths["facts"]
        assert f"{temp_storage}/learnings" in storage_paths["learnings"]
        assert f"{temp_storage}/sessions" in storage_paths["sessions"]
        assert f"{temp_storage}/graphs" in storage_paths["graphs"]

    def test_write_file_routes_to_learnings(self, temp_storage):
        """Test that write_file with MemoryType.LEARNING routes to learnings/ directory."""
        from src.config import MemoryType
        from src.services.file_service import FileService

        service = FileService()
        filepath = service.write_file(
            filename="test_learning.md",
            content="# Test Learning\n\nContent here",
            memory_type=MemoryType.LEARNING,
        )

        # Verify file was written to learnings directory
        assert f"{temp_storage}/learnings" in str(filepath)
        assert filepath.exists()
        assert "Test Learning" in filepath.read_text()

    def test_write_file_routes_to_sessions(self, temp_storage):
        """Test that write_file with MemoryType.SESSION routes to sessions/ directory."""
        from src.config import MemoryType
        from src.services.file_service import FileService

        service = FileService()
        filepath = service.write_file(
            filename="test_session.md",
            content="# Test Session\n\nSession summary",
            memory_type=MemoryType.SESSION,
        )

        # Verify file was written to sessions directory
        assert f"{temp_storage}/sessions" in str(filepath)
        assert filepath.exists()
        assert "Test Session" in filepath.read_text()

    def test_write_file_routes_to_graphs(self, temp_storage):
        """Test that write_file with MemoryType.GRAPH routes to graphs/ directory."""
        from src.config import MemoryType
        from src.services.file_service import FileService

        service = FileService()
        filepath = service.write_file(
            filename="test_graph.md",
            content="# Test Graph\n\nGraph context",
            memory_type=MemoryType.GRAPH,
        )

        # Verify file was written to graphs directory
        assert f"{temp_storage}/graphs" in str(filepath)
        assert filepath.exists()
        assert "Test Graph" in filepath.read_text()

    def test_write_file_routes_to_facts(self, temp_storage):
        """Test that write_file with MemoryType.FACT routes to facts/ directory."""
        from src.config import MemoryType
        from src.services.file_service import FileService

        service = FileService()
        filepath = service.write_file(
            filename="test_fact.md",
            content="# Test Fact\n\nFactual relationship",
            memory_type=MemoryType.FACT,
        )

        # Verify file was written to facts directory
        assert f"{temp_storage}/facts" in str(filepath)
        assert filepath.exists()
        assert "Test Fact" in filepath.read_text()

    def test_write_file_defaults_to_facts(self, temp_storage):
        """Test that write_file without memory_type defaults to FACT."""
        from src.services.file_service import FileService

        service = FileService()
        filepath = service.write_file(
            filename="test_default.md",
            content="# Default Test\n\nNo memory type specified",
        )

        # Verify file was written to facts directory (default)
        assert f"{temp_storage}/facts" in str(filepath)
        assert filepath.exists()


# Note: Full MCP tool integration tests with Neo4j would require a running
# Neo4j instance. These tests verify the file service routing layer which is
# the core of Feature 010's memory type organization.
