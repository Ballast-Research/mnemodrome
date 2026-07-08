"""
Contract tests for ActivationService async methods.

Tests update_access_tracking(), strengthen_hebbian_links(), and
update_confidence() against their contracts in activation-service.md.

Uses mocking for Neo4j and EventLog dependencies.
"""

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
class TestUpdateAccessTracking:
    """Contract tests for update_access_tracking()."""

    async def test_increment_existing_access_count(self):
        """Node with existing access_count → increments by 1."""
        with patch("src.services.activation_service.neo4j_service") as mock_neo4j:
            mock_neo4j.batch_update_access = AsyncMock()

            from src.services.activation_service import ActivationService

            service = ActivationService()

            await service.update_access_tracking(["uuid-1", "uuid-2"])

            mock_neo4j.batch_update_access.assert_called_once_with(["uuid-1", "uuid-2"])

    async def test_initialize_missing_access_count(self):
        """Node without access_count → handled by batch_update_access coalesce pattern."""
        with patch("src.services.activation_service.neo4j_service") as mock_neo4j:
            mock_neo4j.batch_update_access = AsyncMock()

            from src.services.activation_service import ActivationService

            service = ActivationService()

            await service.update_access_tracking(["uuid-new"])

            # batch_update_access internally uses coalesce for lazy init
            mock_neo4j.batch_update_access.assert_called_once_with(["uuid-new"])

    async def test_neo4j_failure_logs_warning(self):
        """Neo4j unavailable → logs warning, does not raise."""
        with (
            patch("src.services.activation_service.neo4j_service") as mock_neo4j,
            patch("src.services.activation_service.logger") as mock_logger,
        ):
            mock_neo4j.batch_update_access = AsyncMock(
                side_effect=Exception("Neo4j connection refused")
            )

            from src.services.activation_service import ActivationService

            service = ActivationService()

            # Should NOT raise
            await service.update_access_tracking(["uuid-1"])

            # Should log warning
            mock_logger.warning.assert_called()

    async def test_empty_uuids_no_op(self):
        """Empty UUID list → no query executed."""
        with patch("src.services.activation_service.neo4j_service") as mock_neo4j:
            from src.services.activation_service import ActivationService

            service = ActivationService()

            await service.update_access_tracking([])

            mock_neo4j.query.assert_not_called()


@pytest.mark.asyncio
class TestStrengthenHebbianLinks:
    """Contract tests for strengthen_hebbian_links()."""

    async def test_existing_edge_strengthened(self):
        """Two entities with existing edge → weight increases."""
        with patch("src.services.activation_service.neo4j_service") as mock_neo4j:
            mock_neo4j.query = AsyncMock(return_value=[{"updated": 1}])

            from src.services.activation_service import ActivationService

            service = ActivationService()

            await service.strengthen_hebbian_links(["uuid-a", "uuid-b"], increment=0.1)

            mock_neo4j.query.assert_called_once()
            cypher = mock_neo4j.query.call_args[0][0]
            assert "hebbian_weight" in cypher
            assert "last_coactivation" in cypher
            assert "invalid_at IS NULL" in cypher

    async def test_no_edge_no_op(self):
        """Two entities with no edge → no change (no edge creation)."""
        with patch("src.services.activation_service.neo4j_service") as mock_neo4j:
            # No matching edges found
            mock_neo4j.query = AsyncMock(return_value=[])

            from src.services.activation_service import ActivationService

            service = ActivationService()

            await service.strengthen_hebbian_links(["uuid-a", "uuid-b"])

            # Query still runs (to look for edges), but no CREATE
            mock_neo4j.query.assert_called_once()
            cypher = mock_neo4j.query.call_args[0][0]
            assert "CREATE" not in cypher

    async def test_invalidated_edge_skipped(self):
        """Edge already invalidated → no update (WHERE invalid_at IS NULL)."""
        with patch("src.services.activation_service.neo4j_service") as mock_neo4j:
            mock_neo4j.query = AsyncMock(return_value=[])

            from src.services.activation_service import ActivationService

            service = ActivationService()

            await service.strengthen_hebbian_links(["uuid-a", "uuid-b"])

            cypher = mock_neo4j.query.call_args[0][0]
            # Query must filter out invalidated edges
            assert "invalid_at IS NULL" in cypher

    async def test_three_entity_pairs(self):
        """Three entities activated → all 3 pairs checked."""
        with patch("src.services.activation_service.neo4j_service") as mock_neo4j:
            mock_neo4j.query = AsyncMock(return_value=[])

            from src.services.activation_service import ActivationService

            service = ActivationService()

            await service.strengthen_hebbian_links(["uuid-a", "uuid-b", "uuid-c"])

            # Should generate 3 pairs: (a,b), (a,c), (b,c)
            call_args = mock_neo4j.query.call_args
            params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
            pairs = params.get("pairs", [])
            assert len(pairs) == 3

    async def test_single_uuid_no_op(self):
        """Single UUID → no pairs to process."""
        with patch("src.services.activation_service.neo4j_service") as mock_neo4j:
            from src.services.activation_service import ActivationService

            service = ActivationService()

            await service.strengthen_hebbian_links(["uuid-only"])

            mock_neo4j.query.assert_not_called()

    async def test_neo4j_failure_silent(self):
        """Neo4j failure → logs warning, does not raise."""
        with (
            patch("src.services.activation_service.neo4j_service") as mock_neo4j,
            patch("src.services.activation_service.logger") as mock_logger,
        ):
            mock_neo4j.query = AsyncMock(side_effect=Exception("Connection lost"))

            from src.services.activation_service import ActivationService

            service = ActivationService()

            # Should NOT raise
            await service.strengthen_hebbian_links(["uuid-a", "uuid-b"])

            mock_logger.warning.assert_called()


@pytest.mark.asyncio
class TestUpdateConfidence:
    """Contract tests for update_confidence()."""

    async def test_returns_new_confidence_value(self):
        """Returns updated confidence after Bayesian update."""
        with (
            patch("src.services.activation_service.neo4j_service") as mock_neo4j,
            patch("src.services.activation_service.event_log_service") as mock_event,
        ):
            # Mock reading current confidence
            mock_neo4j.query = AsyncMock(
                side_effect=[
                    [{"confidence": 0.8}],  # Read current
                    [],  # Write new
                ]
            )
            mock_event.log_event = AsyncMock(return_value=1)

            from src.services.activation_service import ActivationService

            service = ActivationService()

            result = await service.update_confidence(
                relationship_uuid="rel-uuid-1",
                evidence_strength=2.0,
                is_supporting=True,
            )

            assert isinstance(result, float)
            assert result > 0.8  # Supporting evidence should increase

    async def test_logs_to_event_log(self):
        """Confidence adjustments are logged to event log (FR-012)."""
        with (
            patch("src.services.activation_service.neo4j_service") as mock_neo4j,
            patch("src.services.activation_service.event_log_service") as mock_event,
        ):
            mock_neo4j.query = AsyncMock(
                side_effect=[
                    [{"confidence": 0.5}],
                    [],
                ]
            )
            mock_event.log_event = AsyncMock(return_value=1)

            from src.services.activation_service import ActivationService

            service = ActivationService()

            await service.update_confidence(
                relationship_uuid="rel-uuid-1",
                evidence_strength=1.5,
                is_supporting=True,
            )

            # Must log the confidence adjustment
            mock_event.log_event.assert_called_once()
            log_call = mock_event.log_event.call_args
            assert (
                log_call[1].get("event_type") == "confidence_adjustment"
                or log_call[1].get("action") == "confidence_adjustment"
                or "confidence" in str(log_call)
            )

    async def test_contradiction_decreases_confidence(self):
        """Contradictory evidence decreases confidence."""
        with (
            patch("src.services.activation_service.neo4j_service") as mock_neo4j,
            patch("src.services.activation_service.event_log_service") as mock_event,
        ):
            mock_neo4j.query = AsyncMock(
                side_effect=[
                    [{"confidence": 0.8}],
                    [],
                ]
            )
            mock_event.log_event = AsyncMock(return_value=1)

            from src.services.activation_service import ActivationService

            service = ActivationService()

            result = await service.update_confidence(
                relationship_uuid="rel-uuid-1",
                evidence_strength=2.0,
                is_supporting=False,
            )

            assert result < 0.8

    async def test_missing_relationship_raises(self):
        """Non-existent relationship UUID → raises exception."""
        with (
            patch("src.services.activation_service.neo4j_service") as mock_neo4j,
            patch("src.services.activation_service.event_log_service") as mock_event,
        ):
            mock_neo4j.query = AsyncMock(return_value=[])  # No results

            from src.services.activation_service import ActivationService

            service = ActivationService()

            with pytest.raises(Exception):
                await service.update_confidence(
                    relationship_uuid="nonexistent-uuid",
                    evidence_strength=2.0,
                    is_supporting=True,
                )
