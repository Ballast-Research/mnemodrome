"""Integration tests for dual-write sync service with retry queue.

Tests the best-effort dual-write pattern:
- Both layers succeed
- One layer fails (partial success)
- Both layers fail (exception raised)
- Retry queue processing
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.sync_service import sync_service


@pytest.mark.asyncio
class TestDualWriteIntegration:
    """Test dual-write with retry queue integration."""

    async def test_dual_write_both_succeed(self):
        """Test successful write to both layers."""
        with (
            patch("src.services.sync_service.neo4j_service") as mock_neo4j,
            patch("src.services.sync_service.file_service") as mock_pai,
            patch("src.services.sync_service.retry_queue") as mock_queue,
        ):
            # Mock successful writes
            mock_neo4j.create_entity = AsyncMock()
            mock_pai.write_file = MagicMock(return_value=Path("/test/path.md"))
            mock_queue.add = AsyncMock()
            mock_queue.load = AsyncMock(return_value=[])

            neo4j_data = {
                "operation": "create_entity",
                "data": {"name": "Test", "summary": "Test entity"},
            }

            result = await sync_service.dual_write_with_retry(
                neo4j_data=neo4j_data, markdown_content="# Test", filename="test.md"
            )

            # Verify both succeeded
            assert result["status"] == "complete"
            assert result["neo4j"] is True
            assert result["markdown"] is True
            assert result["queued"] == 0

            # Verify no items queued
            mock_queue.add.assert_not_called()

    async def test_dual_write_relationship_succeeds(self):
        """Test successful relationship creation."""
        with (
            patch("src.services.sync_service.neo4j_service") as mock_neo4j,
            patch("src.services.sync_service.file_service") as mock_pai,
            patch("src.services.sync_service.retry_queue") as mock_queue,
        ):
            # Mock successful writes
            mock_neo4j.create_relationship = AsyncMock(
                return_value={
                    "entity1_uuid": "uuid1",
                    "entity2_uuid": "uuid2",
                    "relation_type": "IMPLEMENTS",
                }
            )
            mock_pai.write_file = MagicMock(return_value=Path("/test/path.md"))
            mock_queue.add = AsyncMock()
            mock_queue.load = AsyncMock(return_value=[])

            neo4j_data = {
                "operation": "create_relationship",
                "data": {
                    "entity1_name": "Test1",
                    "entity1_summary": "Summary1",
                    "relation_type": "IMPLEMENTS",
                    "entity2_name": "Test2",
                    "entity2_summary": "Summary2",
                    "context": "Test context",
                },
            }

            result = await sync_service.dual_write_with_retry(
                neo4j_data=neo4j_data, markdown_content="# Test", filename="test.md"
            )

            # Verify both succeeded
            assert result["status"] == "complete"
            assert result["neo4j"] is True
            assert result["markdown"] is True
            assert result["queued"] == 0

            # Verify relationship was created
            mock_neo4j.create_relationship.assert_called_once()

    async def test_dual_write_neo4j_fails(self):
        """Test partial success when Neo4j fails."""
        with (
            patch("src.services.sync_service.neo4j_service") as mock_neo4j,
            patch("src.services.sync_service.file_service") as mock_pai,
            patch("src.services.sync_service.retry_queue") as mock_queue,
        ):
            # Mock Neo4j failure, markdown success
            mock_neo4j.create_entity = AsyncMock(side_effect=Exception("Neo4j down"))
            mock_pai.write_file = MagicMock(return_value=Path("/test/path.md"))
            mock_queue.add = AsyncMock()
            mock_queue.load = AsyncMock(return_value=[{"layer": "neo4j"}])

            neo4j_data = {
                "operation": "create_entity",
                "data": {"name": "Test", "summary": "Test entity"},
            }

            result = await sync_service.dual_write_with_retry(
                neo4j_data=neo4j_data, markdown_content="# Test", filename="test.md"
            )

            # Verify partial success
            assert result["status"] == "partial"
            assert result["neo4j"] is False
            assert result["markdown"] is True
            assert result["queued"] == 1

            # Verify Neo4j write was queued
            mock_queue.add.assert_called_once()
            queued_item = mock_queue.add.call_args[0][0]
            assert queued_item["layer"] == "neo4j"
            assert "Neo4j down" in queued_item["error"]

    async def test_dual_write_markdown_fails(self):
        """Test partial success when markdown fails."""
        with (
            patch("src.services.sync_service.neo4j_service") as mock_neo4j,
            patch("src.services.sync_service.file_service") as mock_pai,
            patch("src.services.sync_service.retry_queue") as mock_queue,
        ):
            # Mock Neo4j success, markdown failure
            mock_neo4j.create_entity = AsyncMock()
            mock_pai.write_file = MagicMock(side_effect=OSError("Disk full"))
            mock_queue.add = AsyncMock()
            mock_queue.load = AsyncMock(return_value=[{"layer": "markdown"}])

            neo4j_data = {
                "operation": "create_entity",
                "data": {"name": "Test", "summary": "Test entity"},
            }

            result = await sync_service.dual_write_with_retry(
                neo4j_data=neo4j_data, markdown_content="# Test", filename="test.md"
            )

            # Verify partial success
            assert result["status"] == "partial"
            assert result["neo4j"] is True
            assert result["markdown"] is False
            assert result["queued"] == 1

            # Verify markdown write was queued
            mock_queue.add.assert_called_once()
            queued_item = mock_queue.add.call_args[0][0]
            assert queued_item["layer"] == "markdown"
            assert "Disk full" in queued_item["error"]

    async def test_dual_write_both_fail(self):
        """Test exception raised when both layers fail."""
        with (
            patch("src.services.sync_service.neo4j_service") as mock_neo4j,
            patch("src.services.sync_service.file_service") as mock_pai,
            patch("src.services.sync_service.retry_queue") as mock_queue,
        ):
            # Mock both failures
            mock_neo4j.create_entity = AsyncMock(side_effect=Exception("Neo4j down"))
            mock_pai.write_file = MagicMock(side_effect=OSError("Disk full"))
            mock_queue.add = AsyncMock()

            neo4j_data = {
                "operation": "create_entity",
                "data": {"name": "Test", "summary": "Test entity"},
            }

            # Should raise exception
            with pytest.raises(Exception) as exc_info:
                await sync_service.dual_write_with_retry(
                    neo4j_data=neo4j_data, markdown_content="# Test", filename="test.md"
                )

            assert "BOTH layers failed" in str(exc_info.value)

            # Verify both writes were queued
            assert mock_queue.add.call_count == 2

    async def test_retry_queue_processing(self):
        """Test retry processor reconciles queued writes."""
        with (
            patch("src.services.sync_service.neo4j_service") as mock_neo4j,
            patch("src.services.sync_service.retry_queue") as mock_queue,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):  # Skip sleep delays
            # Mock queue with one pending item
            queued_item = {
                "layer": "neo4j",
                "data": {
                    "operation": "create_entity",
                    "data": {"name": "Test", "summary": "Queued entity"},
                },
            }

            # First load returns item, second load returns empty (queue cleared)
            mock_queue.load = AsyncMock(
                side_effect=[
                    [queued_item],  # First cycle: item present
                    [],  # Second cycle: queue empty, processor exits
                ]
            )
            mock_queue.remove = AsyncMock()
            mock_neo4j.create_entity = AsyncMock()

            # Run retry processor
            await sync_service._process_retry_queue()

            # Verify item was retried and removed
            mock_neo4j.create_entity.assert_called_once()
            mock_queue.remove.assert_called_once_with(queued_item)

    async def test_unknown_operation_handling(self):
        """Test handling of unknown Neo4j operations."""
        with (
            patch("src.services.sync_service.neo4j_service") as mock_neo4j,
            patch("src.services.sync_service.file_service") as mock_pai,
            patch("src.services.sync_service.retry_queue") as mock_queue,
        ):
            mock_pai.write_file = MagicMock(return_value=Path("/test/path.md"))
            mock_queue.add = AsyncMock()
            mock_queue.load = AsyncMock(return_value=[])

            neo4j_data = {
                "operation": "unknown_operation",  # Invalid
                "data": {"name": "Test"},
            }

            result = await sync_service.dual_write_with_retry(
                neo4j_data=neo4j_data, markdown_content="# Test", filename="test.md"
            )

            # Should fail Neo4j but succeed markdown
            assert result["status"] == "partial"
            assert result["neo4j"] is False
            assert result["markdown"] is True

            # Verify error was queued
            queued_item = mock_queue.add.call_args[0][0]
            assert "Unknown Neo4j operation" in queued_item["error"]
