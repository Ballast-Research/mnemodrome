"""Integration tests for dual-memory synchronization of fact invalidation operations.

Tests the best-effort dual-write pattern for Feature 006 (Fact Invalidation):
- Both layers succeed (Neo4j + Markdown)
- Neo4j succeeds, markdown fails (partial success)
- Neo4j fails, markdown succeeds (partial success)
- Both layers fail (exception raised)
- Retry queue integration
- Invalidation event log format validation

Dual-Memory Architecture:
- Neo4j: Set temporal validity fields (invalid_at, expired_at) on relationships
- Markdown: Write invalidation event logs to file service vault with YAML frontmatter
- Best-effort: If one fails, queue for retry; if both fail, raise exception
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.sync_service import sync_service


@pytest.mark.asyncio
class TestInvalidationDualMemorySync:
    """Test dual-memory synchronization for fact invalidation operations."""

    async def test_invalidation_both_layers_succeed(self):
        """Test successful invalidation write to both Neo4j and markdown layers."""
        with (
            patch("src.services.sync_service.neo4j_service") as mock_neo4j,
            patch("src.services.sync_service.file_service") as mock_file_service,
            patch("src.services.sync_service.retry_queue") as mock_queue,
        ):
            # Mock successful Neo4j invalidation
            mock_neo4j.invalidate_relationship = AsyncMock(
                return_value={
                    "fact_uuid": "abc123",
                    "invalid_at": "2025-10-08T14:23:00Z",
                    "expired_at": "2025-10-08T14:23:00Z",
                    "reason": "Infrastructure migrated to server-01",
                }
            )

            # Mock successful markdown write
            mock_file_service.write_file = MagicMock(
                return_value=Path("/test/invalidation_event.md")
            )
            mock_queue.add = AsyncMock()
            mock_queue.load = AsyncMock(return_value=[])

            # Neo4j operation data
            neo4j_data = {
                "operation": "invalidate_relationship",
                "data": {
                    "fact_uuid": "abc123",
                    "invalid_at": "2025-10-08T14:23:00Z",
                    "expired_at": "2025-10-08T14:23:00Z",
                    "invalidation_reason": "Infrastructure migrated to server-01",
                },
            }

            # Markdown event log content
            markdown_content = """---
event_type: fact_invalidation
fact_uuid: abc123
entity1: Mnemodrome
relation: RUNS_ON
entity2: server-02
invalidated_at: 2025-10-08T14:23:00Z
reason: Infrastructure migrated to server-01
---

# Fact Invalidation Event

**Fact**: Mnemodrome RUNS_ON server-02
**Reason**: Infrastructure migrated to server-01
**Timestamp**: 2025-10-08T14:23:00Z
"""

            result = await sync_service.dual_write_with_retry(
                neo4j_data=neo4j_data,
                markdown_content=markdown_content,
                filename="invalidation_event_abc123.md",
            )

            # Verify both layers succeeded
            assert result["status"] == "complete"
            assert result["neo4j"] is True
            assert result["markdown"] is True
            assert result["queued"] == 0

            # Verify no retry queue items
            mock_queue.add.assert_not_called()

            # Verify Neo4j invalidation was called
            mock_neo4j.invalidate_relationship.assert_called_once()

    async def test_invalidation_neo4j_succeeds_markdown_fails(self):
        """Test partial success when Neo4j invalidation succeeds but markdown write fails."""
        with (
            patch("src.services.sync_service.neo4j_service") as mock_neo4j,
            patch("src.services.sync_service.file_service") as mock_file_service,
            patch("src.services.sync_service.retry_queue") as mock_queue,
        ):
            # Mock Neo4j success
            mock_neo4j.invalidate_relationship = AsyncMock(
                return_value={
                    "fact_uuid": "def456",
                    "invalid_at": "2025-10-08T15:00:00Z",
                    "expired_at": "2025-10-08T15:00:00Z",
                }
            )

            # Mock markdown failure (disk full scenario)
            mock_file_service.write_file = MagicMock(
                side_effect=OSError("Disk full - cannot write to file service vault")
            )
            mock_queue.add = AsyncMock()
            mock_queue.load = AsyncMock(return_value=[{"layer": "markdown"}])

            neo4j_data = {
                "operation": "invalidate_relationship",
                "data": {
                    "fact_uuid": "def456",
                    "invalid_at": "2025-10-08T15:00:00Z",
                    "expired_at": "2025-10-08T15:00:00Z",
                    "invalidation_reason": "Corrected erroneous fact",
                },
            }

            markdown_content = """---
event_type: fact_invalidation
fact_uuid: def456
invalidated_at: 2025-10-08T15:00:00Z
reason: Corrected erroneous fact
---

# Fact Invalidation Event

**Reason**: Corrected erroneous fact
**Timestamp**: 2025-10-08T15:00:00Z
"""

            result = await sync_service.dual_write_with_retry(
                neo4j_data=neo4j_data,
                markdown_content=markdown_content,
                filename="invalidation_event_def456.md",
            )

            # Verify partial success (Neo4j succeeded, markdown failed)
            assert result["status"] == "partial"
            assert result["neo4j"] is True
            assert result["markdown"] is False
            assert result["queued"] == 1

            # Verify markdown write was queued for retry
            mock_queue.add.assert_called_once()
            queued_item = mock_queue.add.call_args[0][0]
            assert queued_item["layer"] == "markdown"
            assert "Disk full" in queued_item["error"]
            assert queued_item["filename"] == "invalidation_event_def456.md"

    async def test_invalidation_neo4j_fails_markdown_succeeds(self):
        """Test partial success when Neo4j invalidation fails but markdown write succeeds."""
        with (
            patch("src.services.sync_service.neo4j_service") as mock_neo4j,
            patch("src.services.sync_service.file_service") as mock_file_service,
            patch("src.services.sync_service.retry_queue") as mock_queue,
        ):
            # Mock Neo4j failure (connection error)
            mock_neo4j.invalidate_relationship = AsyncMock(
                side_effect=Exception("Neo4j connection timeout - database unreachable")
            )

            # Mock markdown success
            mock_file_service.write_file = MagicMock(
                return_value=Path("/test/invalidation_event_ghi789.md")
            )
            mock_queue.add = AsyncMock()
            mock_queue.load = AsyncMock(return_value=[{"layer": "neo4j"}])

            neo4j_data = {
                "operation": "invalidate_relationship",
                "data": {
                    "fact_uuid": "ghi789",
                    "invalid_at": "2025-10-08T16:30:00Z",
                    "expired_at": "2025-10-08T16:30:00Z",
                    "invalidation_reason": "System detected contradiction",
                },
            }

            markdown_content = """---
event_type: fact_invalidation
fact_uuid: ghi789
invalidated_at: 2025-10-08T16:30:00Z
reason: System detected contradiction
---

# Fact Invalidation Event

**Reason**: System detected contradiction
**Timestamp**: 2025-10-08T16:30:00Z
"""

            result = await sync_service.dual_write_with_retry(
                neo4j_data=neo4j_data,
                markdown_content=markdown_content,
                filename="invalidation_event_ghi789.md",
            )

            # Verify partial success (markdown succeeded, Neo4j failed)
            assert result["status"] == "partial"
            assert result["neo4j"] is False
            assert result["markdown"] is True
            assert result["queued"] == 1

            # Verify Neo4j write was queued for retry
            mock_queue.add.assert_called_once()
            queued_item = mock_queue.add.call_args[0][0]
            assert queued_item["layer"] == "neo4j"
            assert "Neo4j connection timeout" in queued_item["error"]
            assert queued_item["data"]["operation"] == "invalidate_relationship"

    async def test_invalidation_both_layers_fail(self):
        """Test exception raised when both Neo4j and markdown invalidation writes fail."""
        with (
            patch("src.services.sync_service.neo4j_service") as mock_neo4j,
            patch("src.services.sync_service.file_service") as mock_file_service,
            patch("src.services.sync_service.retry_queue") as mock_queue,
        ):
            # Mock both failures
            mock_neo4j.invalidate_relationship = AsyncMock(
                side_effect=Exception("Neo4j server down")
            )
            mock_file_service.write_file = MagicMock(
                side_effect=OSError("Filesystem readonly")
            )
            mock_queue.add = AsyncMock()

            neo4j_data = {
                "operation": "invalidate_relationship",
                "data": {
                    "fact_uuid": "jkl012",
                    "invalid_at": "2025-10-08T17:00:00Z",
                    "expired_at": "2025-10-08T17:00:00Z",
                    "invalidation_reason": "Critical error",
                },
            }

            markdown_content = """---
event_type: fact_invalidation
fact_uuid: jkl012
invalidated_at: 2025-10-08T17:00:00Z
reason: Critical error
---

# Fact Invalidation Event
"""

            # Should raise exception when both layers fail
            with pytest.raises(Exception) as exc_info:
                await sync_service.dual_write_with_retry(
                    neo4j_data=neo4j_data,
                    markdown_content=markdown_content,
                    filename="invalidation_event_jkl012.md",
                )

            # Verify exception message
            assert "BOTH layers failed" in str(exc_info.value)

            # Verify both writes were queued for retry (2 items)
            assert mock_queue.add.call_count == 2

            # Verify queue contains both failed writes
            queued_layers = [
                call[0][0]["layer"] for call in mock_queue.add.call_args_list
            ]
            assert "neo4j" in queued_layers
            assert "markdown" in queued_layers

    async def test_invalidation_event_log_format_validation(self):
        """Test that invalidation events are logged with correct YAML frontmatter format."""
        with (
            patch("src.services.sync_service.neo4j_service") as mock_neo4j,
            patch("src.services.sync_service.file_service") as mock_file_service,
            patch("src.services.sync_service.retry_queue") as mock_queue,
        ):
            # Mock successful writes
            mock_neo4j.invalidate_relationship = AsyncMock()
            mock_queue.add = AsyncMock()
            mock_queue.load = AsyncMock(return_value=[])

            # Capture markdown content written
            written_content = None
            written_filename = None

            def capture_write(filename: str, content: str) -> Path:
                nonlocal written_content, written_filename
                written_content = content
                written_filename = filename
                return Path(f"/test/{filename}")

            mock_file_service.write_file = MagicMock(side_effect=capture_write)

            # Test invalidation event with full context
            neo4j_data = {
                "operation": "invalidate_relationship",
                "data": {
                    "fact_uuid": "mno345",
                    "invalid_at": "2025-10-08T18:00:00Z",
                    "expired_at": "2025-10-08T18:00:00Z",
                    "invalidation_reason": "Server migration completed",
                },
            }

            markdown_content = """---
event_type: fact_invalidation
fact_uuid: mno345
entity1: Mnemodrome
relation: RUNS_ON
entity2: old-server
invalidated_at: 2025-10-08T18:00:00Z
reason: Server migration completed
superseded_by: pqr678
---

# Fact Invalidation Event

**Fact**: Mnemodrome RUNS_ON old-server
**Reason**: Server migration completed
**Superseded By**: New fact with UUID pqr678
**Timestamp**: 2025-10-08T18:00:00Z

## Context

This fact was invalidated as part of infrastructure migration.
The relationship has been superseded by a new fact pointing to the updated server.
"""

            result = await sync_service.dual_write_with_retry(
                neo4j_data=neo4j_data,
                markdown_content=markdown_content,
                filename="invalidation_event_mno345.md",
            )

            # Verify write succeeded
            assert result["status"] == "complete"

            # Verify markdown content format
            assert written_content is not None
            assert written_filename == "invalidation_event_mno345.md"

            # Validate YAML frontmatter structure
            assert "---" in written_content
            assert "event_type: fact_invalidation" in written_content
            assert "fact_uuid: mno345" in written_content
            assert "entity1: Mnemodrome" in written_content
            assert "relation: RUNS_ON" in written_content
            assert "entity2: old-server" in written_content
            assert "invalidated_at: 2025-10-08T18:00:00Z" in written_content
            assert "reason: Server migration completed" in written_content
            assert "superseded_by: pqr678" in written_content

            # Validate markdown body structure
            assert "# Fact Invalidation Event" in written_content
            assert "**Fact**: Mnemodrome RUNS_ON old-server" in written_content
            assert "**Reason**: Server migration completed" in written_content
            assert "**Superseded By**: New fact with UUID pqr678" in written_content

    async def test_invalidation_retry_queue_integration(self):
        """Test retry queue processes invalidation writes correctly."""
        with (
            patch("src.services.sync_service.neo4j_service") as mock_neo4j,
            patch("src.services.sync_service.retry_queue") as mock_queue,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):  # Skip sleep delays
            # Mock queue with pending invalidation operation
            queued_invalidation = {
                "layer": "neo4j",
                "data": {
                    "operation": "invalidate_relationship",
                    "data": {
                        "fact_uuid": "stu901",
                        "invalid_at": "2025-10-08T19:00:00Z",
                        "expired_at": "2025-10-08T19:00:00Z",
                        "invalidation_reason": "Queued invalidation from retry",
                    },
                },
            }

            # First load returns queued item, second load returns empty (queue cleared)
            mock_queue.load = AsyncMock(
                side_effect=[
                    [queued_invalidation],  # First cycle: item present
                    [],  # Second cycle: queue empty, processor exits
                ]
            )
            mock_queue.remove = AsyncMock()
            mock_neo4j.invalidate_relationship = AsyncMock()

            # Run retry processor
            await sync_service._process_retry_queue()

            # Verify invalidation was retried
            mock_neo4j.invalidate_relationship.assert_called_once()

            # Verify item was removed from queue after success
            mock_queue.remove.assert_called_once_with(queued_invalidation)

    async def test_invalidation_update_fact_supersession_chain(self):
        """Test dual-memory sync for update_fact operation (invalidate + create new)."""
        with (
            patch("src.services.sync_service.neo4j_service") as mock_neo4j,
            patch("src.services.sync_service.file_service") as mock_file_service,
            patch("src.services.sync_service.retry_queue") as mock_queue,
        ):
            # Mock successful update (invalidate old + create new)
            mock_neo4j.update_relationship = AsyncMock(
                return_value={
                    "invalidated_fact_uuid": "old123",
                    "new_fact_uuid": "new456",
                    "supersession_chain": ["old123", "new456"],
                }
            )

            mock_file_service.write_file = MagicMock(
                return_value=Path("/test/update_event.md")
            )
            mock_queue.add = AsyncMock()
            mock_queue.load = AsyncMock(return_value=[])

            # Update operation (invalidate + create)
            neo4j_data = {
                "operation": "update_relationship",
                "data": {
                    "old_fact_uuid": "old123",
                    "new_fact_uuid": "new456",
                    "invalid_at": "2025-10-08T20:00:00Z",
                    "expired_at": "2025-10-08T20:00:00Z",
                    "superseded_by": "new456",
                    "reason": "Updated infrastructure location",
                },
            }

            markdown_content = """---
event_type: fact_update
old_fact_uuid: old123
new_fact_uuid: new456
invalidated_at: 2025-10-08T20:00:00Z
superseded_by: new456
reason: Updated infrastructure location
---

# Fact Update Event

**Old Fact UUID**: old123 (invalidated)
**New Fact UUID**: new456 (created)
**Supersession Chain**: old123 → new456
**Reason**: Updated infrastructure location
**Timestamp**: 2025-10-08T20:00:00Z
"""

            result = await sync_service.dual_write_with_retry(
                neo4j_data=neo4j_data,
                markdown_content=markdown_content,
                filename="update_event_old123_new456.md",
            )

            # Verify both layers succeeded
            assert result["status"] == "complete"
            assert result["neo4j"] is True
            assert result["markdown"] is True
            assert result["queued"] == 0

            # Verify update operation was executed
            mock_neo4j.update_relationship.assert_called_once()

    async def test_invalidation_unknown_operation_handling(self):
        """Test handling of unknown invalidation-related operations."""
        with (
            patch("src.services.sync_service.neo4j_service") as mock_neo4j,
            patch("src.services.sync_service.file_service") as mock_file_service,
            patch("src.services.sync_service.retry_queue") as mock_queue,
        ):
            mock_file_service.write_file = MagicMock(
                return_value=Path("/test/unknown_op.md")
            )
            mock_queue.add = AsyncMock()
            mock_queue.load = AsyncMock(return_value=[])

            # Unknown operation type
            neo4j_data = {
                "operation": "unknown_invalidation_operation",
                "data": {"fact_uuid": "xyz999"},
            }

            markdown_content = "# Unknown operation test"

            result = await sync_service.dual_write_with_retry(
                neo4j_data=neo4j_data,
                markdown_content=markdown_content,
                filename="unknown_op.md",
            )

            # Should fail Neo4j but succeed markdown
            assert result["status"] == "partial"
            assert result["neo4j"] is False
            assert result["markdown"] is True
            assert result["queued"] == 1

            # Verify unknown operation error was queued
            queued_item = mock_queue.add.call_args[0][0]
            assert queued_item["layer"] == "neo4j"
            assert (
                "Unknown Neo4j operation" in queued_item["error"]
                or "unknown_invalidation_operation" in queued_item["error"]
            )


@pytest.fixture(autouse=True)
async def reset_sync_service():
    """Reset sync service state before each test."""
    # Cancel any running retry tasks
    if sync_service.retry_task and not sync_service.retry_task.done():
        sync_service.retry_task.cancel()
        try:
            await sync_service.retry_task
        except Exception:
            pass

    sync_service.retry_task = None
    yield
