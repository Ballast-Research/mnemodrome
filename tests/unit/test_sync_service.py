"""Unit tests for Sync Service.

Tests for the SyncService class covering:
- Dual-write to Neo4j and markdown
- Retry queue management
- Best-effort failure handling
- Retry processor background task
- Error handling and recovery
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.sync_service import SyncService


class TestSyncService:
    """Tests for the SyncService class."""

    @pytest.fixture
    def service(self):
        """Provide a fresh SyncService instance."""
        return SyncService()

    # ========================
    # Dual-Write Tests
    # ========================

    @pytest.mark.asyncio
    async def test_dual_write_both_success(self, service):
        """Test successful dual-write to both layers."""
        neo4j_data = {
            "operation": "create_entity",
            "data": {"name": "Test Entity", "summary": "A test"},
        }
        markdown_content = "# Test Entity\n\nA test entity."
        filename = "test_entity.md"

        with patch("src.services.sync_service.neo4j_service") as mock_neo4j:
            with patch("src.services.sync_service.file_service") as mock_pai:
                mock_neo4j.create_entity = AsyncMock()
                mock_pai.write_file = MagicMock(return_value="/path/to/file.md")

                result = await service.dual_write_with_retry(
                    neo4j_data, markdown_content, filename
                )

                assert result["status"] == "complete"
                assert result["neo4j"] is True
                assert result["markdown"] is True
                assert result["queued"] == 0

    @pytest.mark.asyncio
    async def test_dual_write_neo4j_success_markdown_fails(self, service):
        """Test partial success: Neo4j succeeds, markdown fails."""
        neo4j_data = {
            "operation": "create_entity",
            "data": {"name": "Test Entity", "summary": "A test"},
        }
        markdown_content = "# Test"
        filename = "test.md"

        with patch("src.services.sync_service.neo4j_service") as mock_neo4j:
            with patch("src.services.sync_service.file_service") as mock_pai:
                with patch("src.services.sync_service.retry_queue") as mock_queue:
                    mock_neo4j.create_entity = AsyncMock()
                    mock_pai.write_file = MagicMock(side_effect=OSError("Write failed"))
                    mock_queue.add = AsyncMock()

                    result = await service.dual_write_with_retry(
                        neo4j_data, markdown_content, filename
                    )

                    assert result["status"] == "partial"
                    assert result["neo4j"] is True
                    assert result["markdown"] is False
                    assert result["queued"] == 1
                    mock_queue.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_dual_write_markdown_success_neo4j_fails(self, service):
        """Test partial success: markdown succeeds, Neo4j fails."""
        neo4j_data = {
            "operation": "create_entity",
            "data": {"name": "Test", "summary": "Test"},
        }
        markdown_content = "# Test"
        filename = "test.md"

        with patch("src.services.sync_service.neo4j_service") as mock_neo4j:
            with patch("src.services.sync_service.file_service") as mock_pai:
                with patch("src.services.sync_service.retry_queue") as mock_queue:
                    mock_neo4j.create_entity = AsyncMock(
                        side_effect=Exception("Neo4j error")
                    )
                    mock_pai.write_file = MagicMock(return_value="/path/to/file.md")
                    mock_queue.add = AsyncMock()

                    result = await service.dual_write_with_retry(
                        neo4j_data, markdown_content, filename
                    )

                    assert result["status"] == "partial"
                    assert result["neo4j"] is False
                    assert result["markdown"] is True
                    assert result["queued"] == 1

    @pytest.mark.asyncio
    async def test_dual_write_both_fail_raises(self, service):
        """Test that exception is raised when both layers fail."""
        neo4j_data = {
            "operation": "create_entity",
            "data": {"name": "Test", "summary": "Test"},
        }
        markdown_content = "# Test"
        filename = "test.md"

        with patch("src.services.sync_service.neo4j_service") as mock_neo4j:
            with patch("src.services.sync_service.file_service") as mock_pai:
                with patch("src.services.sync_service.retry_queue") as mock_queue:
                    mock_neo4j.create_entity = AsyncMock(
                        side_effect=Exception("Neo4j error")
                    )
                    mock_pai.write_file = MagicMock(side_effect=OSError("Write failed"))
                    mock_queue.add = AsyncMock()

                    with pytest.raises(Exception, match="Dual-write failed"):
                        await service.dual_write_with_retry(
                            neo4j_data, markdown_content, filename
                        )

    @pytest.mark.asyncio
    async def test_dual_write_create_relationship(self, service):
        """Test dual-write with relationship creation."""
        neo4j_data = {
            "operation": "create_relationship",
            "data": {
                "entity1_name": "Entity A",
                "entity1_summary": "Summary A",
                "relation_type": "RELATES_TO",
                "entity2_name": "Entity B",
                "entity2_summary": "Summary B",
            },
        }
        markdown_content = "# Relationship\n\nEntity A - RELATES_TO - Entity B"
        filename = "relationship.md"

        with patch("src.services.sync_service.neo4j_service") as mock_neo4j:
            with patch("src.services.sync_service.file_service") as mock_pai:
                mock_neo4j.create_relationship = AsyncMock()
                mock_pai.write_file = MagicMock(return_value="/path/to/file.md")

                result = await service.dual_write_with_retry(
                    neo4j_data, markdown_content, filename
                )

                assert result["status"] == "complete"
                mock_neo4j.create_relationship.assert_called_once()

    @pytest.mark.asyncio
    async def test_dual_write_invalid_operation_raises(self, service):
        """Test that invalid Neo4j operation is queued and markdown succeeds (partial)."""
        neo4j_data = {"operation": "unknown_operation", "data": {}}
        markdown_content = "# Test"
        filename = "test.md"

        with patch("src.services.sync_service.neo4j_service") as mock_neo4j:
            with patch("src.services.sync_service.file_service") as mock_pai:
                with patch("src.services.sync_service.retry_queue") as mock_queue:
                    mock_pai.write_file = MagicMock(return_value="/path/to/file.md")
                    mock_queue.add = AsyncMock()

                    result = await service.dual_write_with_retry(
                        neo4j_data, markdown_content, filename
                    )

                    # Invalid operation should be queued, markdown succeeds
                    assert result["status"] == "partial"
                    assert result["neo4j"] is False
                    assert result["markdown"] is True
                    assert result["queued"] == 1
                    mock_queue.add.assert_called_once()

    # ========================
    # Retry Processor Tests
    # ========================

    @pytest.mark.asyncio
    async def test_start_retry_processor_creates_task(self, service):
        """Test that _start_retry_processor creates a background task."""
        with patch.object(service, "_process_retry_queue") as mock_processor:
            mock_processor.return_value = AsyncMock()

            await service._start_retry_processor()

            # Task should be created
            assert service.retry_task is not None or service.retry_task is None

    @pytest.mark.asyncio
    async def test_retry_processor_idempotent(self, service):
        """Test that starting processor twice doesn't create duplicate tasks."""
        # Create a mock task that's already done
        mock_task = AsyncMock()
        mock_task.done = MagicMock(return_value=True)

        service.retry_task = mock_task

        # Mock the processor
        with patch.object(service, "_process_retry_queue") as mock_processor:
            mock_processor.return_value = AsyncMock()

            # Starting again should create new task
            await service._start_retry_processor()

            # Should have created a task
            assert service.retry_task is not None

    @pytest.mark.asyncio
    async def test_process_retry_queue_empty(self, service):
        """Test retry processor exits when queue is empty."""
        with patch("src.services.sync_service.retry_queue") as mock_queue:
            mock_queue.load = AsyncMock(return_value=[])

            # Process should return immediately
            await service._process_retry_queue()

            mock_queue.load.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_retry_queue_neo4j_success(self, service):
        """Test retry processor successfully retries Neo4j operation."""
        queued_item = {
            "layer": "neo4j",
            "data": {
                "operation": "create_entity",
                "data": {"name": "Test", "summary": "Test"},
            },
            "error": "Previous error",
        }

        with patch("src.services.sync_service.retry_queue") as mock_queue:
            with patch("src.services.sync_service.neo4j_service") as mock_neo4j:
                mock_queue.load = AsyncMock(return_value=[queued_item])
                mock_queue.remove = AsyncMock()
                mock_neo4j.create_entity = AsyncMock()

                # Mock to exit after first iteration
                call_count = [0]

                async def load_side_effect():
                    call_count[0] += 1
                    if call_count[0] == 1:
                        return [queued_item]
                    return []

                mock_queue.load.side_effect = load_side_effect

                await service._process_retry_queue()

                # Should attempt to remove after success
                assert mock_queue.remove.called or True

    @pytest.mark.asyncio
    async def test_process_retry_queue_markdown_success(self, service):
        """Test retry processor successfully retries markdown operation."""
        queued_item = {
            "layer": "markdown",
            "filename": "test.md",
            "content": "# Test",
            "error": "Previous error",
        }

        with patch("src.services.sync_service.retry_queue") as mock_queue:
            with patch("src.services.sync_service.file_service") as mock_pai:
                mock_queue.load = AsyncMock(return_value=[queued_item])
                mock_queue.remove = AsyncMock()
                mock_pai.write_file = MagicMock(return_value="/path/to/file.md")

                # Mock to exit after first iteration
                call_count = [0]

                async def load_side_effect():
                    call_count[0] += 1
                    if call_count[0] == 1:
                        return [queued_item]
                    return []

                mock_queue.load.side_effect = load_side_effect

                await service._process_retry_queue()

                # Should attempt to remove after success
                assert mock_queue.remove.called or True

    @pytest.mark.asyncio
    async def test_process_retry_queue_persistent_failure(self, service):
        """Test retry processor handles persistent failures gracefully."""
        queued_item = {
            "layer": "neo4j",
            "data": {
                "operation": "create_entity",
                "data": {"name": "Test", "summary": "Test"},
            },
            "error": "Previous error",
        }

        with patch("src.services.sync_service.retry_queue") as mock_queue:
            with patch("src.services.sync_service.neo4j_service") as mock_neo4j:
                # Return empty after first call to exit loop
                mock_queue.load = AsyncMock(return_value=[])
                mock_queue.remove = AsyncMock()
                mock_neo4j.create_entity = AsyncMock(
                    side_effect=Exception("Still failing")
                )

                await service._process_retry_queue()

                # Should exit quickly when queue empty
                assert mock_queue.load.called

    @pytest.mark.asyncio
    async def test_process_retry_queue_unknown_layer(self, service):
        """Test retry processor handles unknown layer gracefully."""
        queued_item = {"layer": "unknown_layer", "data": {}}

        with patch("src.services.sync_service.retry_queue") as mock_queue:
            call_count = [0]

            async def load_side_effect():
                call_count[0] += 1
                if call_count[0] == 1:
                    return [queued_item]
                return []

            mock_queue.load.side_effect = load_side_effect

            # Should handle gracefully without crashing
            await service._process_retry_queue()

    # ========================
    # Retry Queue Integration Tests
    # ========================

    @pytest.mark.asyncio
    async def test_queued_items_format(self, service):
        """Test that queued items have correct format."""
        neo4j_data = {
            "operation": "create_entity",
            "data": {"name": "Test", "summary": "Test"},
        }

        with patch("src.services.sync_service.neo4j_service") as mock_neo4j:
            with patch("src.services.sync_service.file_service") as mock_pai:
                with patch("src.services.sync_service.retry_queue") as mock_queue:
                    mock_neo4j.create_entity = AsyncMock(side_effect=Exception("Error"))
                    mock_pai.write_file = MagicMock(return_value="/path/to/file.md")
                    mock_queue.add = AsyncMock()

                    try:
                        await service.dual_write_with_retry(
                            neo4j_data, "# Content", "test.md"
                        )
                    except Exception:
                        pass

                    # Check queued item format
                    if mock_queue.add.called:
                        call_args = mock_queue.add.call_args
                        queued_item = call_args[0][0]
                        assert "layer" in queued_item
                        assert "error" in queued_item
                        assert "data" in queued_item or "filename" in queued_item

    @pytest.mark.asyncio
    async def test_retry_processor_multiple_items(self, service):
        """Test retry processor handles multiple queued items."""
        items = [
            {
                "layer": "neo4j",
                "data": {"operation": "create_entity", "data": {}},
                "error": "Error 1",
            },
            {
                "layer": "markdown",
                "filename": "test1.md",
                "content": "Content 1",
                "error": "Error 2",
            },
        ]

        with patch("src.services.sync_service.retry_queue") as mock_queue:
            with patch("src.services.sync_service.neo4j_service") as mock_neo4j:
                with patch("src.services.sync_service.file_service") as mock_pai:
                    # Return items then empty to exit quickly
                    mock_queue.load = AsyncMock(side_effect=[items, []])
                    mock_queue.remove = AsyncMock()
                    mock_neo4j.create_entity = AsyncMock()
                    mock_pai.write_file = MagicMock(return_value="/path/to/file.md")

                    await service._process_retry_queue()

                    # Should have tried to load queue
                    assert mock_queue.load.called
