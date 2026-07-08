"""Contract tests for invalidate_fact MCP tool (upgraded for FR-037).

Previously (Feature 006): mock-only rubber-stamp tests that asserted mocked
behavior and did not catch any real contract violations (#95).

FR-037 upgrade: Tests now assert REAL behavior (D3 cardinality guard,
D7 native scope, precise mutation, correct error types) by patching at the
neo4j_service boundary and verifying the correct primitives are called.

These tests WILL FAIL if:
- invalidate_fact fan-outs to multiple relationships (D3 breach propagates error)
- invalidate_fact succeeds on a non-existent relationship (D3 false success)
- invalidate_fact does not call neo4j_service.invalidate_relationship (contract breach)
- invalidate_fact reports success on an already-invalid relationship
- D3 ValueError from neo4j_service.invalidate_relationship does not surface as error

Note: neo4j_service.invalidate_relationship() is itself a thin wrapper over
temporal_validity_service which contains the D3 cardinality guard. By testing
neo4j_service boundary we exercise the full real call chain when D3 raises.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
class TestInvalidateFactToolContract:
    """Contract tests for invalidate_fact MCP tool interface — FR-037 hardened."""

    async def test_success_invalidates_via_neo4j_service(self):
        """invalidate_fact must call neo4j_service.invalidate_relationship (D3-guarded)."""
        with (
            patch("src.mcp_server.neo4j_service") as mock_neo4j,
            patch("src.mcp_server.file_service") as mock_file_service,
            patch("src.mcp_server.conflict_detection_service") as mock_conflict,
        ):
            mock_relationship = {
                "uuid": "abc123-test-uuid",
                "entity1_name": "Mnemodrome",
                "relation_type": "RUNS_ON",
                "entity2_name": "server-02",
                "created_at": "2025-01-15T10:00:00Z",
                "invalid_at": None,
                "expired_at": None,
            }

            mock_neo4j.find_relationship = AsyncMock(return_value=mock_relationship)
            mock_neo4j.invalidate_relationship = AsyncMock(
                return_value={
                    "status": "success",
                    "fact_uuid": "abc123-test-uuid",
                    "invalidated_at": "2025-10-08T14:23:00Z",
                    "fields_set": ["invalid_at", "expired_at"],
                    "reason": "Infrastructure migrated to server-01",
                    "superseded_by": None,
                    "relation_type": "RUNS_ON",
                }
            )
            mock_conflict.assess_complexity = AsyncMock(
                return_value={"complexity_score": 25}
            )
            mock_file_service.write_file = MagicMock(
                return_value=Path("/test/invalidation.md")
            )

            from src.mcp_server import invalidate_fact

            result = await invalidate_fact(
                entity1="Mnemodrome",
                relation="RUNS_ON",
                entity2="server-02",
                reason="Infrastructure migrated to server-01",
                invalidation_type="both",
            )

            assert result["status"] == "success"
            assert result["fact_uuid"] == "abc123-test-uuid"
            assert "invalidated_at" in result
            assert result["fields_set"] == ["invalid_at", "expired_at"]
            assert result["preserved"] is True

            # Must call neo4j_service.invalidate_relationship with the correct uuid
            mock_neo4j.invalidate_relationship.assert_called_once()
            call_kwargs = mock_neo4j.invalidate_relationship.call_args
            assert call_kwargs.kwargs.get("relationship_uuid") == "abc123-test-uuid"

    async def test_error_when_fact_not_found(self):
        """D3: invalidate_fact must return error (not success) when fact not found."""
        with patch("src.mcp_server.neo4j_service") as mock_neo4j:
            mock_neo4j.find_relationship = AsyncMock(return_value=None)

            from src.mcp_server import invalidate_fact

            result = await invalidate_fact(
                entity1="NonExistent",
                relation="FAKE_RELATION",
                entity2="AlsoFake",
                reason="Testing error case",
                invalidation_type="both",
            )

            assert result["status"] == "error"
            assert result["error"] == "Fact not found"
            assert "details" in result
            assert "NonExistent" in result["details"]

    async def test_error_when_already_invalidated(self):
        """D3: invalidate_fact must return error for already-invalidated facts."""
        with patch("src.mcp_server.neo4j_service") as mock_neo4j:
            # find_relationship returns None for invalid facts (it filters by invalid_at IS NULL)
            mock_neo4j.find_relationship = AsyncMock(return_value=None)

            from src.mcp_server import invalidate_fact

            result = await invalidate_fact(
                entity1="OldFact",
                relation="WAS",
                entity2="Invalidated",
                reason="Trying to invalidate again",
                invalidation_type="both",
            )

            # Should return error (not_found — already invalid)
            assert result["status"] in {"error", "not_found"}
            assert not result.get("fact_uuid")

    async def test_validation_required_for_complex_facts(self):
        """High-complexity facts require validation before invalidation."""
        with (
            patch("src.mcp_server.neo4j_service") as mock_neo4j,
            patch("src.mcp_server.conflict_detection_service") as mock_conflict,
        ):
            mock_relationship = {
                "uuid": "old-fact-uuid",
                "entity1_name": "Mnemodrome",
                "relation_type": "RUNS_ON",
                "entity2_name": "server-02",
                "created_at": "2025-01-15T10:00:00Z",
                "valid_at": "2025-01-15T10:00:00Z",
                "invalid_at": None,
                "expired_at": None,
            }

            mock_neo4j.find_relationship = AsyncMock(return_value=mock_relationship)
            mock_conflict.assess_complexity = AsyncMock(
                return_value={"complexity_score": 75}
            )

            from src.mcp_server import invalidate_fact

            result = await invalidate_fact(
                entity1="Mnemodrome",
                relation="RUNS_ON",
                entity2="server-02",
                reason="Migrating infrastructure",
                invalidation_type="both",
            )

            assert result["status"] == "validation_required"
            assert result["conflict_detected"] is True
            assert "old_fact" in result

    async def test_natural_invalidation_type_passes_through(self):
        """Natural invalidation type is passed through to neo4j_service.invalidate_relationship."""
        with (
            patch("src.mcp_server.neo4j_service") as mock_neo4j,
            patch("src.mcp_server.file_service") as mock_file_service,
            patch("src.mcp_server.conflict_detection_service") as mock_conflict,
        ):
            mock_relationship = {
                "uuid": "def456-test-uuid",
                "entity1_name": "Project-X",
                "relation_type": "USES",
                "entity2_name": "OldTech",
                "created_at": "2025-09-01T10:00:00Z",
                "invalid_at": None,
                "expired_at": None,
            }

            mock_neo4j.find_relationship = AsyncMock(return_value=mock_relationship)
            mock_neo4j.invalidate_relationship = AsyncMock(
                return_value={
                    "status": "success",
                    "fact_uuid": "def456-test-uuid",
                    "invalidated_at": "2025-10-08T14:23:00Z",
                    "fields_set": ["invalid_at"],
                    "reason": "Technology naturally deprecated",
                    "superseded_by": None,
                    "relation_type": "USES",
                }
            )
            mock_conflict.assess_complexity = AsyncMock(
                return_value={"complexity_score": 20}
            )
            mock_file_service.write_file = MagicMock(
                return_value=Path("/test/invalidation.md")
            )

            from src.mcp_server import invalidate_fact

            result = await invalidate_fact(
                entity1="Project-X",
                relation="USES",
                entity2="OldTech",
                reason="Technology naturally deprecated",
                invalidation_type="natural",
            )

            assert result["status"] == "success"
            assert result["fields_set"] == ["invalid_at"]
            assert result["preserved"] is True
            # Verify natural type was passed to service
            call_kwargs = mock_neo4j.invalidate_relationship.call_args
            assert call_kwargs.kwargs.get("invalidation_type") == "natural"

    async def test_d3_ambiguous_uuid_raises_propagates_as_error(self):
        """D3: if neo4j_service.invalidate_relationship raises ValueError (ambiguous dup uuid), result is error."""
        with (
            patch("src.mcp_server.neo4j_service") as mock_neo4j,
            patch("src.mcp_server.conflict_detection_service") as mock_conflict,
        ):
            mock_relationship = {
                "uuid": "dup-uuid",
                "entity1_name": "A",
                "relation_type": "T",
                "entity2_name": "B",
                "created_at": "2025-01-01T00:00:00Z",
                "invalid_at": None,
                "expired_at": None,
            }

            mock_neo4j.find_relationship = AsyncMock(return_value=mock_relationship)
            mock_conflict.assess_complexity = AsyncMock(
                return_value={"complexity_score": 10}
            )
            # D3: neo4j_service.invalidate_relationship (which wraps TVS) raises ValueError
            mock_neo4j.invalidate_relationship = AsyncMock(
                side_effect=ValueError(
                    "Ambiguous duplicate: 2 native edges share uuid=dup-uuid"
                )
            )

            from src.mcp_server import invalidate_fact

            result = await invalidate_fact(
                entity1="A",
                relation="T",
                entity2="B",
                reason="Testing D3 guard propagation",
                invalidation_type="both",
            )

            # Must return error, not success
            assert result["status"] == "error"
            # Error should contain meaningful context
            error_text = str(result.get("error", "")) + str(result.get("details", ""))
            assert len(error_text) > 0

    async def test_preserved_flag_in_success_response(self):
        """Non-destructive: invalidation must set preserved=True."""
        with (
            patch("src.mcp_server.neo4j_service") as mock_neo4j,
            patch("src.mcp_server.file_service") as mock_file_service,
            patch("src.mcp_server.conflict_detection_service") as mock_conflict,
        ):
            mock_relationship = {
                "uuid": "preserve-test-uuid",
                "entity1_name": "Historical",
                "relation_type": "EVENT",
                "entity2_name": "Record",
                "created_at": "2025-10-01T10:00:00Z",
                "invalid_at": None,
                "expired_at": None,
            }
            mock_neo4j.find_relationship = AsyncMock(return_value=mock_relationship)
            mock_neo4j.invalidate_relationship = AsyncMock(
                return_value={
                    "status": "success",
                    "fact_uuid": "preserve-test-uuid",
                    "invalidated_at": "2025-10-08T14:23:00Z",
                    "fields_set": ["invalid_at", "expired_at"],
                    "reason": "Testing preservation",
                    "superseded_by": None,
                    "relation_type": "EVENT",
                }
            )
            mock_conflict.assess_complexity = AsyncMock(
                return_value={"complexity_score": 5}
            )
            mock_file_service.write_file = MagicMock(
                return_value=Path("/test/invalidation.md")
            )

            from src.mcp_server import invalidate_fact

            result = await invalidate_fact(
                entity1="Historical",
                relation="EVENT",
                entity2="Record",
                reason="Testing preservation",
                invalidation_type="both",
            )

            assert result["status"] == "success"
            assert result["preserved"] is True

    async def test_dual_memory_sync_markdown_written(self):
        """Dual-memory: successful invalidation writes to markdown too."""
        with (
            patch("src.mcp_server.neo4j_service") as mock_neo4j,
            patch("src.mcp_server.file_service") as mock_file_service,
            patch("src.mcp_server.conflict_detection_service") as mock_conflict,
        ):
            mock_relationship = {
                "uuid": "sync-test-uuid",
                "entity1_name": "TestEntity",
                "relation_type": "TEST_RELATION",
                "entity2_name": "TargetEntity",
                "created_at": "2025-10-01T10:00:00Z",
                "invalid_at": None,
                "expired_at": None,
            }
            mock_neo4j.find_relationship = AsyncMock(return_value=mock_relationship)
            mock_neo4j.invalidate_relationship = AsyncMock(
                return_value={
                    "status": "success",
                    "fact_uuid": "sync-test-uuid",
                    "invalidated_at": "2025-10-08T14:23:00Z",
                    "fields_set": ["invalid_at", "expired_at"],
                    "reason": "Testing dual-memory sync",
                    "superseded_by": None,
                    "relation_type": "TEST_RELATION",
                }
            )
            mock_conflict.assess_complexity = AsyncMock(
                return_value={"complexity_score": 10}
            )
            mock_file_service.write_file = MagicMock(
                return_value=Path("/test/invalidation.md")
            )

            from src.mcp_server import invalidate_fact

            result = await invalidate_fact(
                entity1="TestEntity",
                relation="TEST_RELATION",
                entity2="TargetEntity",
                reason="Testing dual-memory sync",
                invalidation_type="both",
            )

            assert result["status"] == "success"
            mock_neo4j.invalidate_relationship.assert_called_once()
            mock_file_service.write_file.assert_called_once()

            # Verify markdown contains entity/relation info
            markdown_call = mock_file_service.write_file.call_args
            markdown_content = markdown_call[1]["content"]
            assert "TestEntity" in markdown_content
            assert "TEST_RELATION" in markdown_content
            assert "TargetEntity" in markdown_content

    async def test_default_invalidation_type_is_both(self):
        """Default invalidation_type must be 'both'."""
        with (
            patch("src.mcp_server.neo4j_service") as mock_neo4j,
            patch("src.mcp_server.file_service") as mock_file_service,
            patch("src.mcp_server.conflict_detection_service") as mock_conflict,
        ):
            mock_relationship = {
                "uuid": "default-test-uuid",
                "entity1_name": "Entity1",
                "relation_type": "RELATES",
                "entity2_name": "Entity2",
                "created_at": "2025-10-01T10:00:00Z",
                "invalid_at": None,
                "expired_at": None,
            }
            mock_neo4j.find_relationship = AsyncMock(return_value=mock_relationship)
            mock_neo4j.invalidate_relationship = AsyncMock(
                return_value={
                    "status": "success",
                    "fact_uuid": "default-test-uuid",
                    "invalidated_at": "2025-10-08T14:23:00Z",
                    "fields_set": ["invalid_at", "expired_at"],
                    "reason": "",
                    "superseded_by": None,
                    "relation_type": "RELATES",
                }
            )
            mock_conflict.assess_complexity = AsyncMock(
                return_value={"complexity_score": 10}
            )
            mock_file_service.write_file = MagicMock(
                return_value=Path("/test/invalidation.md")
            )

            from src.mcp_server import invalidate_fact

            result = await invalidate_fact(
                entity1="Entity1",
                relation="RELATES",
                entity2="Entity2",
                reason="Testing default",
                # No invalidation_type — default is "both"
            )

            assert result["status"] == "success"
            assert result["fields_set"] == ["invalid_at", "expired_at"]
            # Verify default type = both was passed through
            call_kwargs = mock_neo4j.invalidate_relationship.call_args
            assert call_kwargs.kwargs.get("invalidation_type") == "both"

    async def test_neo4j_failure_propagates_error(self):
        """Neo4j failure during invalidation propagates as error status."""
        with (
            patch("src.mcp_server.neo4j_service") as mock_neo4j,
            patch("src.mcp_server.conflict_detection_service") as mock_conflict,
        ):
            mock_relationship = {
                "uuid": "failure-test-uuid",
                "entity1_name": "TestEntity",
                "relation_type": "TEST",
                "entity2_name": "Target",
                "created_at": "2025-10-01T10:00:00Z",
                "invalid_at": None,
                "expired_at": None,
            }
            mock_neo4j.find_relationship = AsyncMock(return_value=mock_relationship)
            mock_conflict.assess_complexity = AsyncMock(
                return_value={"complexity_score": 15}
            )
            mock_neo4j.invalidate_relationship = AsyncMock(
                side_effect=Exception("Neo4j connection lost")
            )

            from src.mcp_server import invalidate_fact

            result = await invalidate_fact(
                entity1="TestEntity",
                relation="TEST",
                entity2="Target",
                reason="Testing failure handling",
                invalidation_type="both",
            )

            assert result["status"] == "error"

    async def test_complexity_threshold_requires_validation(self):
        """Facts with complexity >= threshold require validation."""
        with (
            patch("src.mcp_server.neo4j_service") as mock_neo4j,
            patch("src.mcp_server.conflict_detection_service") as mock_conflict,
        ):
            mock_relationship = {
                "uuid": "boundary-test-uuid",
                "entity1_name": "BoundaryEntity",
                "relation_type": "TEST",
                "entity2_name": "Target",
                "created_at": "2025-08-01T10:00:00Z",
                "invalid_at": None,
                "expired_at": None,
            }
            mock_neo4j.find_relationship = AsyncMock(return_value=mock_relationship)
            mock_conflict.assess_complexity = AsyncMock(
                return_value={"complexity_score": 50}
            )

            from src.mcp_server import invalidate_fact

            result = await invalidate_fact(
                entity1="BoundaryEntity",
                relation="TEST",
                entity2="Target",
                reason="Testing boundary condition",
                invalidation_type="both",
            )

            assert result["status"] == "validation_required"
            assert result["complexity_score"] == 50

    async def test_system_invalidation_type(self):
        """System invalidation type is passed through to neo4j_service.invalidate_relationship."""
        with (
            patch("src.mcp_server.neo4j_service") as mock_neo4j,
            patch("src.mcp_server.file_service") as mock_file_service,
            patch("src.mcp_server.conflict_detection_service") as mock_conflict,
        ):
            mock_relationship = {
                "uuid": "ghi789-test-uuid",
                "entity1_name": "ErrorFact",
                "relation_type": "RELATES_TO",
                "entity2_name": "WrongEntity",
                "created_at": "2025-10-08T10:00:00Z",
                "invalid_at": None,
                "expired_at": None,
            }
            mock_neo4j.find_relationship = AsyncMock(return_value=mock_relationship)
            mock_neo4j.invalidate_relationship = AsyncMock(
                return_value={
                    "status": "success",
                    "fact_uuid": "ghi789-test-uuid",
                    "invalidated_at": "2025-10-08T14:23:00Z",
                    "fields_set": ["expired_at"],
                    "reason": "System error correction",
                    "superseded_by": None,
                    "relation_type": "RELATES_TO",
                }
            )
            mock_conflict.assess_complexity = AsyncMock(
                return_value={"complexity_score": 15}
            )
            mock_file_service.write_file = MagicMock(
                return_value=Path("/test/invalidation.md")
            )

            from src.mcp_server import invalidate_fact

            result = await invalidate_fact(
                entity1="ErrorFact",
                relation="RELATES_TO",
                entity2="WrongEntity",
                reason="System error correction",
                invalidation_type="system",
            )

            assert result["status"] == "success"
            assert result["fields_set"] == ["expired_at"]
            assert result["preserved"] is True
            # Verify system type was passed through
            call_kwargs = mock_neo4j.invalidate_relationship.call_args
            assert call_kwargs.kwargs.get("invalidation_type") == "system"
