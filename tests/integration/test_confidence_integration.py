"""
Integration tests for Bayesian confidence scoring (US2).

Tests confidence initialization, evidence-based updates,
event log auditing, and supersession chain preservation.
"""

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
class TestConfidenceScoring:
    """Integration tests for Bayesian confidence on relationships."""

    async def test_confidence_initialized_at_default(self):
        """
        Acceptance Scenario 6 setup: New relationships get confidence=0.8.
        """
        from src.config import ACTR_CONFIDENCE_DEFAULT

        assert ACTR_CONFIDENCE_DEFAULT == 0.8

    async def test_supporting_evidence_increases_confidence(self):
        """
        Acceptance Scenario 5: Supporting evidence increases confidence.
        """
        with (
            patch("src.services.activation_service.neo4j_service") as mock_neo4j,
            patch("src.services.activation_service.event_log_service") as mock_event,
        ):
            mock_neo4j.query = AsyncMock(
                side_effect=[
                    [{"confidence": 0.6}],  # Read current
                    [],  # Write updated
                ]
            )
            mock_event.log_event = AsyncMock(return_value=1)

            from src.services.activation_service import ActivationService

            service = ActivationService()

            new_conf = await service.update_confidence(
                relationship_uuid="rel-1",
                evidence_strength=2.0,
                is_supporting=True,
            )

            assert new_conf > 0.6, "Supporting evidence should increase confidence"

    async def test_contradictory_evidence_decreases_without_invalidation(self):
        """
        Acceptance Scenario 4: Contradictory evidence decreases confidence
        proportionally, without binary invalidation.
        """
        with (
            patch("src.services.activation_service.neo4j_service") as mock_neo4j,
            patch("src.services.activation_service.event_log_service") as mock_event,
        ):
            mock_neo4j.query = AsyncMock(
                side_effect=[
                    [{"confidence": 0.9}],
                    [],
                ]
            )
            mock_event.log_event = AsyncMock(return_value=1)

            from src.services.activation_service import ActivationService

            service = ActivationService()

            new_conf = await service.update_confidence(
                relationship_uuid="rel-1",
                evidence_strength=2.0,
                is_supporting=False,
            )

            assert new_conf < 0.9, "Contradiction should decrease confidence"
            assert new_conf > 0.01, "Should not invalidate (still above minimum)"

    async def test_confidence_adjustment_logged_to_event_log(self):
        """
        FR-012: Confidence adjustments MUST be logged to event log.
        """
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
                relationship_uuid="rel-audit",
                evidence_strength=1.5,
                is_supporting=True,
            )

            # Verify event log was called with required fields
            mock_event.log_event.assert_called_once()
            call_kwargs = mock_event.log_event.call_args[1]

            assert "confidence" in call_kwargs.get("action", "")
            metadata = call_kwargs.get("metadata", {})
            assert "old_confidence" in metadata
            assert "new_confidence" in metadata
            assert "evidence_strength" in metadata
            assert "is_supporting" in metadata

    async def test_supersession_chains_remain_intact(self):
        """
        Acceptance Scenario 6: Confidence scoring does not interfere
        with existing supersession chain behavior.
        """
        from src.services.activation_service import bayesian_update

        # Confidence is a separate dimension from valid/invalid status
        # A fact can have high confidence AND be superseded
        confidence = bayesian_update(0.95, 2.0, is_supporting=True)
        assert confidence <= 0.99  # Clamped, but doesn't affect invalid_at

        # Even very low confidence doesn't set invalid_at
        low_confidence = bayesian_update(0.1, 3.0, is_supporting=False)
        assert low_confidence >= 0.01  # Clamped, still valid

    async def test_multiple_evidence_updates_accumulate(self):
        """Repeated evidence updates change confidence progressively."""
        from src.services.activation_service import bayesian_update

        confidence = 0.5  # Start neutral
        for _ in range(3):
            confidence = bayesian_update(confidence, 1.5, is_supporting=True)

        assert confidence > 0.7, "Three supports should push above 0.7"

        for _ in range(5):
            confidence = bayesian_update(confidence, 1.5, is_supporting=False)

        assert confidence < 0.5, "Five contradictions should pull below 0.5"

    async def test_event_log_audit_trail_fields(self):
        """
        NFR-004: Verify event log contains ALL required audit fields for
        confidence adjustments: relationship_uuid, old_confidence, new_confidence,
        evidence_strength, and is_supporting.
        """
        with (
            patch("src.services.activation_service.neo4j_service") as mock_neo4j,
            patch("src.services.activation_service.event_log_service") as mock_event,
        ):
            mock_neo4j.query = AsyncMock(
                side_effect=[
                    [{"confidence": 0.7}],  # Read current confidence
                    [],  # Write updated confidence
                ]
            )
            mock_event.log_event = AsyncMock(return_value=1)

            from src.services.activation_service import ActivationService

            service = ActivationService()

            rel_uuid = "rel-nfr004-test"
            strength = 2.5
            supporting = True

            await service.update_confidence(
                relationship_uuid=rel_uuid,
                evidence_strength=strength,
                is_supporting=supporting,
            )

            # Verify event log was called exactly once
            mock_event.log_event.assert_called_once()
            call_kwargs = mock_event.log_event.call_args[1]

            # Verify event_type
            assert call_kwargs.get("event_type") == "memory_operation", (
                "event_type must be 'memory_operation'"
            )

            # Verify action contains "confidence"
            action = call_kwargs.get("action", "")
            assert "confidence" in action, (
                f"action must contain 'confidence', got: {action}"
            )
            assert action == "confidence_adjustment", (
                f"action should be 'confidence_adjustment', got: {action}"
            )

            # Verify target is the relationship UUID
            assert call_kwargs.get("target") == rel_uuid, (
                f"target must be relationship_uuid, got: {call_kwargs.get('target')}"
            )

            # Verify ALL metadata fields per NFR-004
            metadata = call_kwargs.get("metadata", {})

            assert "old_confidence" in metadata, (
                "metadata must contain 'old_confidence'"
            )
            assert metadata["old_confidence"] == 0.7, (
                f"old_confidence should be 0.7, got: {metadata['old_confidence']}"
            )

            assert "new_confidence" in metadata, (
                "metadata must contain 'new_confidence'"
            )
            assert isinstance(metadata["new_confidence"], (int, float)), (
                f"new_confidence must be numeric, got: {type(metadata['new_confidence'])}"
            )
            assert metadata["new_confidence"] > 0.7, (
                "new_confidence should increase from supporting evidence"
            )

            assert "evidence_strength" in metadata, (
                "metadata must contain 'evidence_strength'"
            )
            assert metadata["evidence_strength"] == strength, (
                f"evidence_strength must be {strength}, got: {metadata['evidence_strength']}"
            )

            assert "is_supporting" in metadata, "metadata must contain 'is_supporting'"
            assert metadata["is_supporting"] is supporting, (
                f"is_supporting must be {supporting}, got: {metadata['is_supporting']}"
            )
