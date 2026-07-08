"""Contract tests for enhanced capture_fact MCP tool with conflict detection.

Feature 006: Fact Invalidation and Temporal Validity
Contract: capture_fact tool enhancement with automatic conflict detection

These tests define the contract for conflict detection behavior BEFORE implementation.
Tests MUST FAIL initially (Spec-Driven Development).

Contract Specification:
1. Before creating relationship, detect existing facts with same (entity1, relation)
2. Calculate complexity score for conflicts
3. Auto-resolve simple conflicts if enabled (score < 30)
4. Return validation_required for complex conflicts (score >= 70)
5. Preserve existing behavior when no conflicts exist
"""

import pytest

pytestmark = pytest.mark.xfail(
    reason="Spec-driven contract: auto_resolve_conflicts not yet implemented"
)
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
class TestEnhancedCaptureFactConflictDetection:
    """Test conflict detection in enhanced capture_fact tool."""

    async def test_no_conflict_preserves_existing_behavior(self):
        """Test that capture_fact works normally when no conflicts exist."""
        # ARRANGE: Mock services to return no existing facts
        with (
            patch("src.mcp_server.classification_service") as mock_classification,
            patch("src.mcp_server.entity_factory") as mock_entity_factory,
            patch("src.mcp_server.relationship_factory") as mock_relationship_factory,
            patch("src.mcp_server.file_service") as mock_file_service,
            patch("src.services.neo4j_service.neo4j_service") as mock_neo4j,
        ):
            # Mock entity classification
            mock_classification.classify_entity = AsyncMock(
                side_effect=["Project", "Technology"]
            )

            # Mock entity resolution
            mock_entity_factory.create_or_resolve = AsyncMock(
                side_effect=["uuid-entity1", "uuid-entity2"]
            )

            # Mock conflict detection query - NO CONFLICTS
            mock_neo4j.query = AsyncMock(return_value=[])

            # Mock relationship creation
            mock_relationship_factory.create = AsyncMock(
                return_value={
                    "source_uuid": "uuid-entity1",
                    "source_name": "Mnemodrome",
                    "relation_type": "RUNS_ON",
                    "target_uuid": "uuid-entity2",
                    "target_name": "server-01",
                    "created_at": datetime.now(),
                }
            )

            # Mock file service write
            from pathlib import Path

            mock_file_service.write_file = MagicMock(return_value=Path("/test/fact.md"))

            # ACT: Import and call enhanced capture_fact
            from src.mcp_server import capture_fact

            result = await capture_fact(
                entity1="Mnemodrome",
                relation="RUNS_ON",
                entity2="server-01",
                context="Production deployment",
                auto_resolve_conflicts=False,
            )

            # ASSERT: No conflict detected, normal flow
            assert "✅ Fact captured successfully" in result
            assert "uuid-entity1" in result
            assert "uuid-entity2" in result
            assert mock_relationship_factory.create.called

    async def test_conflict_detection_before_relationship_creation(self):
        """Test that conflicts are detected BEFORE creating new relationship."""
        # ARRANGE: Mock services with existing conflicting fact
        with (
            patch("src.mcp_server.classification_service") as mock_classification,
            patch("src.mcp_server.entity_factory") as mock_entity_factory,
            patch("src.mcp_server.relationship_factory") as mock_relationship_factory,
            patch("src.services.neo4j_service.neo4j_service") as mock_neo4j,
            patch(
                "src.services.conflict_detection_service.conflict_detection_service"
            ) as mock_conflict,
        ):
            # Mock entity classification and resolution
            mock_classification.classify_entity = AsyncMock(
                side_effect=["Project", "Technology"]
            )
            mock_entity_factory.create_or_resolve = AsyncMock(
                side_effect=["uuid-entity1", "uuid-entity2-new"]
            )

            # Mock conflict detection query - CONFLICT EXISTS
            existing_fact = {
                "r": {
                    "uuid": "abc123-old-fact",
                    "entity2_uuid": "uuid-entity2-old",
                    "entity2_name": "server-02",
                    "created_at": (datetime.now() - timedelta(days=90)).isoformat(),
                    "valid_at": (datetime.now() - timedelta(days=90)).isoformat(),
                    "invalid_at": None,
                    "expired_at": None,
                }
            }
            mock_neo4j.query = AsyncMock(return_value=[existing_fact])

            # Mock complexity score calculation - COMPLEX CONFLICT
            mock_conflict.calculate_complexity_score = AsyncMock(return_value=75)

            # ACT: Call enhanced capture_fact
            from src.mcp_server import capture_fact

            result = await capture_fact(
                entity1="Mnemodrome",
                relation="RUNS_ON",
                entity2="server-01",
                context="Infrastructure migration",
                auto_resolve_conflicts=False,
            )

            # ASSERT: Conflict detected, validation required
            assert "conflict_detected" in result or "validation_required" in result
            assert "75" in result  # Complexity score
            assert "server-02" in result  # Old conflicting value
            # Relationship should NOT be created
            assert not mock_relationship_factory.create.called

    async def test_conflict_detection_query_uses_temporal_filtering(self):
        """Test that conflict detection only queries VALID facts (invalid_at IS NULL)."""
        # ARRANGE: Mock neo4j query to capture the Cypher
        with (
            patch("src.mcp_server.classification_service") as mock_classification,
            patch("src.mcp_server.entity_factory") as mock_entity_factory,
            patch("src.services.neo4j_service.neo4j_service") as mock_neo4j,
        ):
            mock_classification.classify_entity = AsyncMock(
                side_effect=["Project", "Technology"]
            )
            mock_entity_factory.create_or_resolve = AsyncMock(
                side_effect=["uuid-entity1", "uuid-entity2"]
            )

            # Track query calls
            mock_neo4j.query = AsyncMock(return_value=[])

            # ACT: Call enhanced capture_fact
            from src.mcp_server import capture_fact

            await capture_fact(
                entity1="Mnemodrome",
                relation="RUNS_ON",
                entity2="server-01",
                auto_resolve_conflicts=False,
            )

            # ASSERT: Conflict detection query was made with temporal filtering
            query_calls = [call.args for call in mock_neo4j.query.call_args_list]
            conflict_query = None
            for call in query_calls:
                if len(call) > 0 and "invalid_at IS NULL" in call[0]:
                    conflict_query = call[0]
                    break

            # The query MUST filter by invalid_at and expired_at
            assert conflict_query is not None, "Conflict detection query not found"
            assert "invalid_at IS NULL" in conflict_query
            assert "expired_at IS NULL" in conflict_query

    async def test_auto_resolve_simple_conflicts(self):
        """Test auto-resolution of simple conflicts (score < 30)."""
        # ARRANGE: Mock services with simple conflict
        with (
            patch("src.mcp_server.classification_service") as mock_classification,
            patch("src.mcp_server.entity_factory") as mock_entity_factory,
            patch("src.mcp_server.relationship_factory") as mock_relationship_factory,
            patch("src.mcp_server.file_service") as mock_file_service,
            patch("src.services.neo4j_service.neo4j_service") as mock_neo4j,
            patch(
                "src.services.conflict_detection_service.conflict_detection_service"
            ) as mock_conflict,
        ):
            # Mock entity classification and resolution
            mock_classification.classify_entity = AsyncMock(
                side_effect=["Project", "Technology"]
            )
            mock_entity_factory.create_or_resolve = AsyncMock(
                side_effect=["uuid-entity1", "uuid-entity2-new"]
            )

            # Mock conflict detection - SIMPLE CONFLICT (recent, same pattern)
            existing_fact = {
                "r": {
                    "uuid": "abc123-old-fact",
                    "entity2_uuid": "uuid-entity2-old",
                    "entity2_name": "server-02",
                    "created_at": (datetime.now() - timedelta(hours=12)).isoformat(),
                    "valid_at": (datetime.now() - timedelta(hours=12)).isoformat(),
                    "invalid_at": None,
                    "expired_at": None,
                }
            }

            # Setup query mock to return conflict, then empty after invalidation
            mock_neo4j.query = AsyncMock(
                side_effect=[
                    [existing_fact],  # First call: conflict detection
                    [{"uuid": "abc123-old-fact"}],  # Second call: invalidation
                ]
            )

            # Mock complexity score - SIMPLE (< 30)
            mock_conflict.calculate_complexity_score = AsyncMock(return_value=25)

            # Mock relationship creation (after auto-resolution)
            mock_relationship_factory.create = AsyncMock(
                return_value={
                    "source_uuid": "uuid-entity1",
                    "source_name": "Mnemodrome",
                    "relation_type": "RUNS_ON",
                    "target_uuid": "uuid-entity2-new",
                    "target_name": "server-01",
                    "created_at": datetime.now(),
                }
            )

            from pathlib import Path

            mock_file_service.write_file = MagicMock(return_value=Path("/test/fact.md"))

            # ACT: Call with auto_resolve_conflicts=True
            from src.mcp_server import capture_fact

            result = await capture_fact(
                entity1="Mnemodrome",
                relation="RUNS_ON",
                entity2="server-01",
                context="Infrastructure update",
                auto_resolve_conflicts=True,
            )

            # ASSERT: Conflict auto-resolved, new fact created
            assert "success" in result.lower() or "✅" in result
            assert "conflict_auto_resolved" in result or "auto" in result.lower()
            assert "abc123-old-fact" in result  # Invalidated fact UUID
            assert "25" in result or "complexity" in result.lower()  # Complexity score
            # Relationship SHOULD be created after auto-resolution
            assert mock_relationship_factory.create.called

    async def test_validation_required_for_complex_conflicts(self):
        """Test that complex conflicts (score >= 70) always require validation."""
        # ARRANGE: Mock services with complex conflict
        with (
            patch("src.mcp_server.classification_service") as mock_classification,
            patch("src.mcp_server.entity_factory") as mock_entity_factory,
            patch("src.mcp_server.relationship_factory") as mock_relationship_factory,
            patch("src.services.neo4j_service.neo4j_service") as mock_neo4j,
            patch(
                "src.services.conflict_detection_service.conflict_detection_service"
            ) as mock_conflict,
        ):
            # Mock entity classification and resolution
            mock_classification.classify_entity = AsyncMock(
                side_effect=["Project", "Technology"]
            )
            mock_entity_factory.create_or_resolve = AsyncMock(
                side_effect=["uuid-entity1", "uuid-entity2-new"]
            )

            # Mock conflict detection - COMPLEX CONFLICT (old, dependencies)
            existing_fact = {
                "r": {
                    "uuid": "abc123-old-fact",
                    "entity2_uuid": "uuid-entity2-old",
                    "entity2_name": "server-02",
                    "created_at": (datetime.now() - timedelta(days=120)).isoformat(),
                    "valid_at": (datetime.now() - timedelta(days=120)).isoformat(),
                    "invalid_at": None,
                    "expired_at": None,
                    "has_dependencies": True,
                }
            }
            mock_neo4j.query = AsyncMock(return_value=[existing_fact])

            # Mock complexity score - COMPLEX (>= 70)
            mock_conflict.calculate_complexity_score = AsyncMock(return_value=85)

            # ACT: Call with auto_resolve_conflicts=True (should be ignored for complex)
            from src.mcp_server import capture_fact

            result = await capture_fact(
                entity1="Mnemodrome",
                relation="RUNS_ON",
                entity2="server-01",
                context="Major infrastructure change",
                auto_resolve_conflicts=True,  # Ignored for complex conflicts
            )

            # ASSERT: Validation required, NO auto-resolution
            assert "validation_required" in result or "conflict_detected" in result
            assert "85" in result  # Complexity score
            assert "server-02" in result  # Old value shown
            assert "action_required" in result.lower() or "approve" in result.lower()
            # Relationship should NOT be created
            assert not mock_relationship_factory.create.called

    async def test_auto_resolve_parameter_default_false(self):
        """Test that auto_resolve_conflicts defaults to False."""
        # ARRANGE: Mock services with simple conflict
        with (
            patch("src.mcp_server.classification_service") as mock_classification,
            patch("src.mcp_server.entity_factory") as mock_entity_factory,
            patch("src.mcp_server.relationship_factory") as mock_relationship_factory,
            patch("src.services.neo4j_service.neo4j_service") as mock_neo4j,
            patch(
                "src.services.conflict_detection_service.conflict_detection_service"
            ) as mock_conflict,
        ):
            # Mock entity classification and resolution
            mock_classification.classify_entity = AsyncMock(
                side_effect=["Project", "Technology"]
            )
            mock_entity_factory.create_or_resolve = AsyncMock(
                side_effect=["uuid-entity1", "uuid-entity2"]
            )

            # Mock simple conflict
            existing_fact = {
                "r": {
                    "uuid": "abc123",
                    "entity2_name": "server-02",
                    "created_at": (datetime.now() - timedelta(hours=6)).isoformat(),
                    "invalid_at": None,
                    "expired_at": None,
                }
            }
            mock_neo4j.query = AsyncMock(return_value=[existing_fact])
            mock_conflict.calculate_complexity_score = AsyncMock(return_value=20)

            # ACT: Call WITHOUT auto_resolve_conflicts parameter
            from src.mcp_server import capture_fact

            result = await capture_fact(
                entity1="Mnemodrome",
                relation="RUNS_ON",
                entity2="server-01",
                context="Test",
                # auto_resolve_conflicts NOT provided (should default to False)
            )

            # ASSERT: Even though score is low, no auto-resolution (default False)
            assert "validation_required" in result or "conflict_detected" in result
            assert not mock_relationship_factory.create.called

    async def test_conflict_response_includes_existing_fact_details(self):
        """Test that conflict response includes all details of existing fact."""
        # ARRANGE: Mock services with conflict
        with (
            patch("src.mcp_server.classification_service") as mock_classification,
            patch("src.mcp_server.entity_factory") as mock_entity_factory,
            patch("src.services.neo4j_service.neo4j_service") as mock_neo4j,
            patch(
                "src.services.conflict_detection_service.conflict_detection_service"
            ) as mock_conflict,
        ):
            # Mock entity classification and resolution
            mock_classification.classify_entity = AsyncMock(
                side_effect=["Project", "Technology"]
            )
            mock_entity_factory.create_or_resolve = AsyncMock(
                side_effect=["uuid-entity1", "uuid-entity2-new"]
            )

            # Mock conflict with full details
            created_timestamp = "2025-01-15T10:00:00Z"
            existing_fact = {
                "r": {
                    "uuid": "abc123-detailed",
                    "entity2_uuid": "uuid-old",
                    "entity2_name": "server-02",
                    "created_at": created_timestamp,
                    "valid_at": created_timestamp,
                    "invalid_at": None,
                    "expired_at": None,
                }
            }
            mock_neo4j.query = AsyncMock(return_value=[existing_fact])
            mock_conflict.calculate_complexity_score = AsyncMock(return_value=75)

            # ACT: Call enhanced capture_fact
            from src.mcp_server import capture_fact

            result = await capture_fact(
                entity1="Mnemodrome",
                relation="RUNS_ON",
                entity2="server-01",
                auto_resolve_conflicts=False,
            )

            # ASSERT: Response includes all existing fact details
            assert "abc123-detailed" in result or "uuid" in result.lower()
            assert "server-02" in result  # Old target entity
            assert (
                "2025-01-15" in result or created_timestamp in result
            )  # Created timestamp
            assert "75" in result  # Complexity score

    async def test_multiple_conflicts_all_listed(self):
        """Test that all conflicting facts are listed when multiple exist."""
        # ARRANGE: Mock services with MULTIPLE conflicts
        with (
            patch("src.mcp_server.classification_service") as mock_classification,
            patch("src.mcp_server.entity_factory") as mock_entity_factory,
            patch("src.services.neo4j_service.neo4j_service") as mock_neo4j,
            patch(
                "src.services.conflict_detection_service.conflict_detection_service"
            ) as mock_conflict,
        ):
            # Mock entity classification and resolution
            mock_classification.classify_entity = AsyncMock(
                side_effect=["Project", "Technology"]
            )
            mock_entity_factory.create_or_resolve = AsyncMock(
                side_effect=["uuid-entity1", "uuid-entity2-new"]
            )

            # Mock MULTIPLE conflicting facts
            existing_facts = [
                {
                    "r": {
                        "uuid": "fact-1",
                        "entity2_name": "server-02",
                        "created_at": (datetime.now() - timedelta(days=60)).isoformat(),
                        "invalid_at": None,
                        "expired_at": None,
                    }
                },
                {
                    "r": {
                        "uuid": "fact-2",
                        "entity2_name": "inference03",
                        "created_at": (datetime.now() - timedelta(days=30)).isoformat(),
                        "invalid_at": None,
                        "expired_at": None,
                    }
                },
            ]
            mock_neo4j.query = AsyncMock(return_value=existing_facts)
            # Multiple conflicts = higher complexity
            mock_conflict.calculate_complexity_score = AsyncMock(return_value=90)

            # ACT: Call enhanced capture_fact
            from src.mcp_server import capture_fact

            result = await capture_fact(
                entity1="Mnemodrome",
                relation="RUNS_ON",
                entity2="server-01",
                auto_resolve_conflicts=False,
            )

            # ASSERT: All conflicts listed
            assert "fact-1" in result or "server-02" in result
            assert "fact-2" in result or "inference03" in result
            assert "90" in result  # High complexity due to multiple conflicts

    async def test_auto_resolution_creates_supersession_chain(self):
        """Test that auto-resolved conflicts create proper supersession chain."""
        # ARRANGE: Mock services with simple conflict
        with (
            patch("src.mcp_server.classification_service") as mock_classification,
            patch("src.mcp_server.entity_factory") as mock_entity_factory,
            patch("src.mcp_server.relationship_factory") as mock_relationship_factory,
            patch("src.mcp_server.file_service") as mock_file_service,
            patch("src.services.neo4j_service.neo4j_service") as mock_neo4j,
            patch(
                "src.services.conflict_detection_service.conflict_detection_service"
            ) as mock_conflict,
        ):
            # Mock entity classification and resolution
            mock_classification.classify_entity = AsyncMock(
                side_effect=["Project", "Technology"]
            )
            mock_entity_factory.create_or_resolve = AsyncMock(
                side_effect=["uuid-entity1", "uuid-entity2-new"]
            )

            # Mock simple conflict
            existing_fact = {
                "r": {
                    "uuid": "old-fact-uuid",
                    "entity2_name": "server-02",
                    "created_at": (datetime.now() - timedelta(hours=6)).isoformat(),
                    "invalid_at": None,
                    "expired_at": None,
                }
            }

            # Setup query responses
            mock_neo4j.query = AsyncMock(
                side_effect=[
                    [existing_fact],  # Conflict detection
                    [{"uuid": "old-fact-uuid"}],  # Invalidation
                ]
            )

            mock_conflict.calculate_complexity_score = AsyncMock(return_value=25)

            # Mock relationship creation
            new_fact_uuid = "new-fact-uuid"
            mock_relationship_factory.create = AsyncMock(
                return_value={
                    "uuid": new_fact_uuid,
                    "source_uuid": "uuid-entity1",
                    "source_name": "Mnemodrome",
                    "relation_type": "RUNS_ON",
                    "target_uuid": "uuid-entity2-new",
                    "target_name": "server-01",
                    "created_at": datetime.now(),
                }
            )

            from pathlib import Path

            mock_file_service.write_file = MagicMock(return_value=Path("/test/fact.md"))

            # ACT: Call with auto-resolution
            from src.mcp_server import capture_fact

            result = await capture_fact(
                entity1="Mnemodrome",
                relation="RUNS_ON",
                entity2="server-01",
                auto_resolve_conflicts=True,
            )

            # ASSERT: Supersession chain created
            assert "old-fact-uuid" in result  # Invalidated fact
            # New fact created (relationship factory called)
            assert mock_relationship_factory.create.called

            # Check that invalidation query was called
            invalidation_query_made = any(
                "SET" in str(call.args[0]) and "invalid_at" in str(call.args[0])
                for call in mock_neo4j.query.call_args_list
                if len(call.args) > 0
            )
            assert invalidation_query_made, (
                "Invalidation query should set invalid_at field"
            )
