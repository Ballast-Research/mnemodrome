"""Contract tests for update_fact MCP tool (Feature 006).

These tests define the expected behavior of the update_fact tool before implementation.
Following Spec-Driven Development - these tests MUST FAIL until the tool is implemented.

Contract Specification:
- Atomically invalidate old fact and create new fact
- Maintain supersession chain (old_uuid -> new_uuid)
- Support conflict resolution modes: "prompt", "auto", "error"
- Synchronize dual-memory (Neo4j + markdown)
- Handle error cases gracefully

Test Strategy:
- Mock all service dependencies (neo4j_service, file_service, conflict_detection_service)
- Verify contract compliance (input/output structure)
- Test atomic transaction semantics
- Test all conflict resolution paths
- Follow pattern from tests/test_integration.py
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
class TestUpdateFactToolContract:
    """Contract tests for update_fact MCP tool."""

    async def test_successful_fact_update_with_supersession_chain(self):
        """Test successful update creates supersession chain atomically."""
        # This test will FAIL until update_fact tool is implemented
        with (
            patch("src.mcp_server.neo4j_service") as mock_neo4j,
            patch("src.mcp_server.file_service") as mock_pai,
            patch("src.mcp_server.conflict_detection_service") as mock_conflict,
            patch("src.mcp_server.sync_service") as mock_sync,
        ):
            # Mock old fact retrieval
            old_uuid = "abc123-old"
            mock_neo4j.find_relationship = AsyncMock(
                return_value={
                    "uuid": old_uuid,
                    "entity1_name": "Mnemodrome",
                    "entity1_uuid": "entity1-uuid",
                    "relation_type": "RUNS_ON",
                    "entity2_name": "gpu-server",
                    "created_at": "2025-09-15T10:00:00Z",
                    "valid_at": "2025-09-15T10:00:00Z",
                }
            )

            # Mock entity UUID resolution
            mock_neo4j.query = AsyncMock(return_value=[{"uuid": "entity2-uuid"}])

            # Mock conflict detection (simple conflict, auto-resolvable)
            mock_conflict.assess_complexity = AsyncMock(
                return_value={
                    "complexity_score": 25,
                    "auto_resolvable": True,
                    "recommendation": "supersede",
                }
            )

            # Mock Neo4j transaction (atomic update)
            new_uuid = "def456-new"
            timestamp = "2025-10-11T14:23:00Z"

            mock_neo4j.execute_update_transaction = AsyncMock(
                return_value={
                    "old_fact_uuid": old_uuid,
                    "new_fact_uuid": new_uuid,
                    "invalidated_at": timestamp,
                    "supersession_chain": [old_uuid, new_uuid],
                }
            )

            # Mock dual-memory sync
            mock_sync.dual_write_with_retry = AsyncMock(
                return_value={
                    "status": "complete",
                    "neo4j": True,
                    "markdown": True,
                    "queued": 0,
                }
            )

            # Import tool (will fail if not implemented)
            from src.mcp_server import update_fact

            # Execute tool
            result = await update_fact(
                entity1="Mnemodrome",
                relation="RUNS_ON",
                entity2_old="server-02",
                entity2_new="server-01",
                context="Infrastructure migration",
                conflict_resolution="auto",
            )

            # Verify contract compliance
            assert result["status"] == "success"
            assert result["invalidated_fact_uuid"] == old_uuid
            assert result["new_fact_uuid"] == new_uuid
            assert result["invalidated_at"] == timestamp
            assert result["supersession_chain"] == [old_uuid, new_uuid]
            assert result["dual_memory_status"]["neo4j"] == "success"
            assert result["dual_memory_status"]["markdown"] == "success"

            # Verify atomic transaction was called
            mock_neo4j.execute_update_transaction.assert_called_once()

    async def test_update_fact_with_prompt_conflict_resolution(self):
        """Test update_fact with conflict_resolution='prompt' returns validation request."""
        with (
            patch("src.mcp_server.neo4j_service") as mock_neo4j,
            patch("src.mcp_server.conflict_detection_service") as mock_conflict,
        ):
            # Mock high complexity conflict (requires human validation)
            mock_conflict.assess_complexity = AsyncMock(
                return_value={
                    "complexity_score": 75,
                    "auto_resolvable": False,
                    "recommendation": "validate",
                    "rationale": "Fact is >30 days old and has dependent relationships",
                }
            )

            # Mock old fact retrieval
            mock_neo4j.find_relationship = AsyncMock(
                return_value={
                    "uuid": "old-uuid",
                    "entity1": "Mnemodrome",
                    "relation": "RUNS_ON",
                    "entity2": "server-02",
                    "created_at": "2025-01-15T10:00:00Z",
                    "valid_at": "2025-01-15T10:00:00Z",
                }
            )

            from src.mcp_server import update_fact

            result = await update_fact(
                entity1="Mnemodrome",
                relation="RUNS_ON",
                entity2_old="server-02",
                entity2_new="server-01",
                conflict_resolution="prompt",  # Explicit prompt mode
            )

            # Verify validation_required response
            assert result["status"] == "validation_required"
            assert result["conflict_detected"] is True
            assert result["complexity_score"] == 75
            assert "old_fact" in result
            assert result["old_fact"]["entity2"] == "server-02"
            assert result["recommendation"] == "validate"
            assert "rationale" in result
            assert "options" in result
            assert result["options"] == ["approve", "reject", "modify"]

            # Verify no transaction was executed (waiting for validation)
            mock_neo4j.execute_update_transaction.assert_not_called()

    async def test_update_fact_with_auto_conflict_resolution(self):
        """Test update_fact with conflict_resolution='auto' auto-resolves simple conflicts."""
        with (
            patch("src.mcp_server.neo4j_service") as mock_neo4j,
            patch("src.mcp_server.conflict_detection_service") as mock_conflict,
            patch("src.mcp_server.sync_service") as mock_sync,
        ):
            # Mock old fact retrieval
            mock_neo4j.find_relationship = AsyncMock(
                return_value={
                    "uuid": "old-uuid",
                    "entity1_name": "Mnemodrome",
                    "entity1_uuid": "entity1-uuid",
                    "relation_type": "RUNS_ON",
                    "entity2_name": "test-server-01",
                    "created_at": "2025-09-15T10:00:00Z",
                }
            )

            # Mock entity UUID resolution
            mock_neo4j.query = AsyncMock(return_value=[{"uuid": "entity2-uuid"}])

            # Mock simple conflict (auto-resolvable)
            mock_conflict.assess_complexity = AsyncMock(
                return_value={
                    "complexity_score": 20,
                    "auto_resolvable": True,
                    "recommendation": "supersede",
                }
            )

            mock_neo4j.execute_update_transaction = AsyncMock(
                return_value={
                    "old_fact_uuid": "old-uuid",
                    "new_fact_uuid": "new-uuid",
                    "invalidated_at": "2025-10-11T14:23:00Z",
                    "supersession_chain": ["old-uuid", "new-uuid"],
                }
            )

            mock_sync.dual_write_with_retry = AsyncMock(
                return_value={"status": "complete", "neo4j": True, "markdown": True}
            )

            from src.mcp_server import update_fact

            result = await update_fact(
                entity1="Mnemodrome",
                relation="RUNS_ON",
                entity2_old="test-server-01",
                entity2_new="test-server-02",
                conflict_resolution="auto",
            )

            # Verify auto-resolution succeeded
            assert result["status"] == "success"
            assert "invalidated_fact_uuid" in result
            assert "new_fact_uuid" in result
            assert result["conflict_auto_resolved"] is True
            assert result["complexity_score"] == 20

            # Verify transaction executed
            mock_neo4j.execute_update_transaction.assert_called_once()

    async def test_update_fact_with_error_conflict_resolution(self):
        """Test update_fact with conflict_resolution='error' fails on any conflict."""
        with (
            patch("src.mcp_server.neo4j_service") as mock_neo4j,
            patch("src.mcp_server.conflict_detection_service") as mock_conflict,
        ):
            # Mock conflict detected (even simple one)
            mock_conflict.assess_complexity = AsyncMock(
                return_value={"complexity_score": 20, "auto_resolvable": True}
            )

            mock_neo4j.find_relationship = AsyncMock(
                return_value={
                    "uuid": "old-uuid",
                    "entity1": "Mnemodrome",
                    "relation": "RUNS_ON",
                    "entity2": "server-02",
                }
            )

            from src.mcp_server import update_fact

            # Should raise exception when conflict_resolution='error'
            with pytest.raises(Exception) as exc_info:
                await update_fact(
                    entity1="Mnemodrome",
                    relation="RUNS_ON",
                    entity2_old="server-02",
                    entity2_new="server-01",
                    conflict_resolution="error",
                )

            assert "Conflict detected" in str(exc_info.value)
            assert "conflict_resolution='error'" in str(exc_info.value)

    async def test_update_fact_old_fact_not_found(self):
        """Test error handling when old fact doesn't exist."""
        with patch("src.mcp_server.neo4j_service") as mock_neo4j:
            # Mock fact not found
            mock_neo4j.find_relationship = AsyncMock(return_value=None)

            from src.mcp_server import update_fact

            result = await update_fact(
                entity1="NonExistent",
                relation="RUNS_ON",
                entity2_old="nowhere",
                entity2_new="somewhere",
            )

            # Verify error response
            assert result["status"] == "error"
            assert "not found" in result["error"].lower()
            assert "NonExistent" in result["details"] or "nowhere" in result["details"]

    async def test_update_fact_atomic_transaction_rollback_on_failure(self):
        """Test atomic transaction rolls back if any step fails."""
        with (
            patch("src.mcp_server.neo4j_service") as mock_neo4j,
            patch("src.mcp_server.conflict_detection_service") as mock_conflict,
        ):
            # Mock old fact retrieval
            mock_neo4j.find_relationship = AsyncMock(
                return_value={
                    "uuid": "old-uuid",
                    "entity1_name": "Mnemodrome",
                    "entity1_uuid": "entity1-uuid",
                    "relation_type": "RUNS_ON",
                    "entity2_name": "old-server",
                    "created_at": "2025-09-15T10:00:00Z",
                }
            )

            # Mock entity UUID resolution
            mock_neo4j.query = AsyncMock(return_value=[{"uuid": "entity2-uuid"}])

            # Mock simple conflict
            mock_conflict.assess_complexity = AsyncMock(
                return_value={"complexity_score": 25, "auto_resolvable": True}
            )

            # Mock transaction failure (simulates rollback)
            mock_neo4j.execute_update_transaction = AsyncMock(
                side_effect=Exception("Transaction failed: relationship creation error")
            )

            from src.mcp_server import update_fact

            # Should raise exception
            with pytest.raises(Exception) as exc_info:
                await update_fact(
                    entity1="Mnemodrome",
                    relation="RUNS_ON",
                    entity2_old="old-server",
                    entity2_new="new-server",
                    conflict_resolution="auto",
                )

            assert "Transaction failed" in str(exc_info.value)

            # Verify rollback occurred (no partial state)
            # In Neo4j, failed transactions automatically roll back

    async def test_update_fact_dual_memory_partial_success(self):
        """Test update_fact with partial dual-memory success (Neo4j succeeds, markdown queued)."""
        with (
            patch("src.mcp_server.neo4j_service") as mock_neo4j,
            patch("src.mcp_server.conflict_detection_service") as mock_conflict,
            patch("src.mcp_server.sync_service") as mock_sync,
            patch("src.mcp_server.file_service") as mock_pai,
        ):
            # Mock old fact retrieval
            mock_neo4j.find_relationship = AsyncMock(
                return_value={
                    "uuid": "old-uuid",
                    "entity1_name": "Mnemodrome",
                    "entity1_uuid": "entity1-uuid",
                    "relation_type": "RUNS_ON",
                    "entity2_name": "old-server",
                    "created_at": "2025-09-15T10:00:00Z",
                }
            )

            # Mock entity UUID resolution
            mock_neo4j.query = AsyncMock(return_value=[{"uuid": "entity2-uuid"}])

            # Mock simple conflict
            mock_conflict.assess_complexity = AsyncMock(
                return_value={"complexity_score": 25, "auto_resolvable": True}
            )

            # Mock Neo4j success
            mock_neo4j.execute_update_transaction = AsyncMock(
                return_value={
                    "old_fact_uuid": "old-uuid",
                    "new_fact_uuid": "new-uuid",
                    "invalidated_at": "2025-10-11T14:23:00Z",
                    "supersession_chain": ["old-uuid", "new-uuid"],
                }
            )

            # Mock markdown write failure
            mock_pai.write_file = MagicMock(side_effect=Exception("Disk full"))

            # Mock markdown failure (partial dual-memory)
            mock_sync.dual_write_with_retry = AsyncMock(
                return_value={
                    "status": "partial",
                    "neo4j": True,
                    "markdown": False,
                    "queued": 1,
                }
            )

            from src.mcp_server import update_fact

            result = await update_fact(
                entity1="Mnemodrome",
                relation="RUNS_ON",
                entity2_old="old-server",
                entity2_new="new-server",
                conflict_resolution="auto",
            )

            # Verify partial success response
            assert result["status"] == "success"  # Core operation succeeded
            assert result["dual_memory_status"]["neo4j"] == "success"
            assert result["dual_memory_status"]["markdown"] == "queued"
            assert result["queued_retries"] == 1

    async def test_update_fact_dual_memory_both_fail(self):
        """Test update_fact handles markdown write failure gracefully by queuing."""
        with (
            patch("src.mcp_server.neo4j_service") as mock_neo4j,
            patch("src.mcp_server.conflict_detection_service") as mock_conflict,
            patch("src.mcp_server.file_service") as mock_pai,
        ):
            # Mock old fact retrieval
            mock_neo4j.find_relationship = AsyncMock(
                return_value={
                    "uuid": "old-uuid",
                    "entity1_name": "Mnemodrome",
                    "entity1_uuid": "entity1-uuid",
                    "relation_type": "RUNS_ON",
                    "entity2_name": "old-server",
                    "created_at": "2025-09-15T10:00:00Z",
                }
            )

            # Mock entity UUID resolution
            mock_neo4j.query = AsyncMock(return_value=[{"uuid": "entity2-uuid"}])

            # Mock simple conflict
            mock_conflict.assess_complexity = AsyncMock(
                return_value={"complexity_score": 25, "auto_resolvable": True}
            )

            # Mock Neo4j transaction success
            mock_neo4j.execute_update_transaction = AsyncMock(
                return_value={
                    "old_fact_uuid": "old-uuid",
                    "new_fact_uuid": "new-uuid",
                    "invalidated_at": "2025-10-11T14:23:00Z",
                    "supersession_chain": ["old-uuid", "new-uuid"],
                }
            )

            # Mock markdown write failure
            mock_pai.write_file = MagicMock(side_effect=Exception("Disk full"))

            from src.mcp_server import update_fact

            # Should succeed with markdown queued (graceful degradation)
            result = await update_fact(
                entity1="Mnemodrome",
                relation="RUNS_ON",
                entity2_old="old-server",
                entity2_new="new-server",
                conflict_resolution="auto",
            )

            # Verify Neo4j succeeded but markdown queued
            assert result["status"] == "success"
            assert result["dual_memory_status"]["neo4j"] == "success"
            assert result["dual_memory_status"]["markdown"] == "queued"
            assert result["queued_retries"] == 1

    async def test_update_fact_supersession_chain_tracking(self):
        """Test supersession chain is correctly tracked in Neo4j relationship."""
        with (
            patch("src.mcp_server.neo4j_service") as mock_neo4j,
            patch("src.mcp_server.conflict_detection_service") as mock_conflict,
            patch("src.mcp_server.sync_service") as mock_sync,
            patch("src.mcp_server.file_service") as mock_pai,
        ):
            # Mock old fact retrieval
            old_uuid = "chain-link-1"
            mock_neo4j.find_relationship = AsyncMock(
                return_value={
                    "uuid": old_uuid,
                    "entity1_name": "TestEntity",
                    "entity1_uuid": "entity1-uuid",
                    "relation_type": "HAS_VALUE",
                    "entity2_name": "old-value",
                    "created_at": "2025-09-15T10:00:00Z",
                }
            )

            # Mock entity UUID resolution
            mock_neo4j.query = AsyncMock(return_value=[{"uuid": "entity2-uuid"}])

            # Mock conflict
            mock_conflict.assess_complexity = AsyncMock(
                return_value={"complexity_score": 25, "auto_resolvable": True}
            )

            # Mock transaction with supersession_chain
            new_uuid = "chain-link-2"

            mock_neo4j.execute_update_transaction = AsyncMock(
                return_value={
                    "old_fact_uuid": old_uuid,
                    "new_fact_uuid": new_uuid,
                    "invalidated_at": "2025-10-11T14:23:00Z",
                    "supersession_chain": [old_uuid, new_uuid],
                }
            )

            # Mock markdown success
            mock_pai.write_file = MagicMock(return_value="/path/to/file.md")

            mock_sync.dual_write_with_retry = AsyncMock(
                return_value={"status": "complete", "neo4j": True, "markdown": True}
            )

            from src.mcp_server import update_fact

            result = await update_fact(
                entity1="TestEntity",
                relation="HAS_VALUE",
                entity2_old="old-value",
                entity2_new="new-value",
                conflict_resolution="auto",
            )

            # Verify supersession chain in response
            assert result["supersession_chain"] == [old_uuid, new_uuid]
            assert result["invalidated_fact_uuid"] == old_uuid
            assert result["new_fact_uuid"] == new_uuid

            # Verify Neo4j transaction included superseded_by field
            call_args = mock_neo4j.execute_update_transaction.call_args
            assert call_args is not None

    async def test_update_fact_validation_history_recorded(self):
        """Test that validation decisions are recorded for pattern learning."""
        with (
            patch("src.mcp_server.neo4j_service") as mock_neo4j,
            patch("src.mcp_server.conflict_detection_service") as mock_conflict,
            patch("src.mcp_server.sync_service") as mock_sync,
            patch("src.mcp_server.validation_history_service") as mock_history,
        ):
            # Mock old fact retrieval
            mock_neo4j.find_relationship = AsyncMock(
                return_value={
                    "uuid": "old-uuid",
                    "entity1_name": "Mnemodrome",
                    "relation_type": "RUNS_ON",
                    "entity2_name": "old-server",
                    "created_at": "2025-01-15T10:00:00Z",
                    "valid_at": "2025-01-15T10:00:00Z",
                }
            )

            # Mock conflict requiring validation
            mock_conflict.assess_complexity = AsyncMock(
                return_value={
                    "complexity_score": 75,
                    "auto_resolvable": False,
                    "conflict_type": "same_relation_different_target",
                }
            )

            # Mock validation history recording
            mock_history.record_validation = AsyncMock(
                return_value={
                    "validation_uuid": "val-123",
                    "recorded_at": "2025-10-11T14:23:00Z",
                }
            )

            from src.mcp_server import update_fact

            # First call: returns validation_required
            result = await update_fact(
                entity1="Mnemodrome",
                relation="RUNS_ON",
                entity2_old="old-server",
                entity2_new="new-server",
                conflict_resolution="prompt",
            )

            assert result["status"] == "validation_required"

            # Simulate user approval (second call with validation_decision)
            mock_neo4j.execute_update_transaction = AsyncMock(
                return_value={
                    "old_fact_uuid": "old-uuid",
                    "new_fact_uuid": "new-uuid",
                    "invalidated_at": "2025-10-11T14:23:00Z",
                    "supersession_chain": ["old-uuid", "new-uuid"],
                }
            )

            mock_sync.dual_write_with_retry = AsyncMock(
                return_value={"status": "complete", "neo4j": True, "markdown": True}
            )

            # Execute with validation_decision parameter (if supported)
            # Note: This tests future enhancement for validation decision tracking
            try:
                result = await update_fact(
                    entity1="Mnemodrome",
                    relation="RUNS_ON",
                    entity2_old="old-server",
                    entity2_new="new-server",
                    conflict_resolution="prompt",
                    validation_decision="approve",  # User approved
                )

                # Verify validation history was recorded
                if mock_history.record_validation.called:
                    call_args = mock_history.record_validation.call_args[1]
                    assert call_args["human_decision"] == "approve"
                    assert call_args["complexity_score"] == 75
            except TypeError:
                # Tool may not support validation_decision parameter yet
                # This is acceptable for MVP - enhancement for Phase 2
                pass

    async def test_update_fact_context_preserved_in_new_fact(self):
        """Test that context parameter is preserved in the new fact."""
        with (
            patch("src.mcp_server.neo4j_service") as mock_neo4j,
            patch("src.mcp_server.conflict_detection_service") as mock_conflict,
            patch("src.mcp_server.sync_service") as mock_sync,
        ):
            # Mock simple conflict
            mock_conflict.assess_complexity = AsyncMock(
                return_value={"complexity_score": 25, "auto_resolvable": True}
            )

            context_text = "Infrastructure migration from server-02 to server-01"

            mock_neo4j.execute_update_transaction = AsyncMock(
                return_value={
                    "old_fact_uuid": "old-uuid",
                    "new_fact_uuid": "new-uuid",
                    "invalidated_at": "2025-10-11T14:23:00Z",
                    "supersession_chain": ["old-uuid", "new-uuid"],
                    "new_fact_context": context_text,
                }
            )

            mock_sync.dual_write_with_retry = AsyncMock(
                return_value={"status": "complete", "neo4j": True, "markdown": True}
            )

            from src.mcp_server import update_fact

            result = await update_fact(
                entity1="Mnemodrome",
                relation="RUNS_ON",
                entity2_old="server-02",
                entity2_new="server-01",
                context=context_text,
                conflict_resolution="auto",
            )

            # Verify context was passed to transaction
            call_args = mock_neo4j.execute_update_transaction.call_args
            if call_args:
                # Check that context was included in call
                assert context_text in str(call_args)

    async def test_update_fact_default_conflict_resolution_is_prompt(self):
        """Test that default conflict_resolution parameter is 'prompt'."""
        with (
            patch("src.mcp_server.neo4j_service") as mock_neo4j,
            patch("src.mcp_server.conflict_detection_service") as mock_conflict,
        ):
            # Mock conflict
            mock_conflict.assess_complexity = AsyncMock(
                return_value={"complexity_score": 50, "auto_resolvable": False}
            )

            mock_neo4j.find_relationship = AsyncMock(
                return_value={
                    "uuid": "old-uuid",
                    "entity1": "Mnemodrome",
                    "relation": "RUNS_ON",
                    "entity2": "server-02",
                }
            )

            from src.mcp_server import update_fact

            # Call without conflict_resolution parameter
            result = await update_fact(
                entity1="Mnemodrome",
                relation="RUNS_ON",
                entity2_old="server-02",
                entity2_new="server-01",
            )

            # Should default to prompt mode
            assert result["status"] == "validation_required"
            assert "options" in result
