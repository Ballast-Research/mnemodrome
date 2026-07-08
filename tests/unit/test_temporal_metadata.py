"""Unit tests for TemporalMetadata Pydantic model."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from src.models.temporal_metadata import TemporalMetadata


@pytest.fixture
def now():
    """Current datetime in UTC."""
    return datetime.now(UTC)


class TestTemporalMetadataCreation:
    """Test TemporalMetadata creation and validation."""

    def test_creates_with_valid_at_only(self, now):
        """Test creates with only valid_at (currently valid)."""
        metadata = TemporalMetadata(valid_at=now)

        assert metadata.valid_at == now
        assert metadata.invalid_at is None
        assert metadata.expired_at is None

    def test_creates_with_valid_and_invalid(self, now):
        """Test creates with valid_at and invalid_at."""
        valid_at = now - timedelta(days=10)
        invalid_at = now - timedelta(days=1)

        metadata = TemporalMetadata(valid_at=valid_at, invalid_at=invalid_at)

        assert metadata.valid_at == valid_at
        assert metadata.invalid_at == invalid_at
        assert metadata.expired_at is None

    def test_creates_with_valid_and_expired(self, now):
        """Test creates with valid_at and expired_at."""
        valid_at = now - timedelta(days=10)
        expired_at = now - timedelta(days=2)

        metadata = TemporalMetadata(valid_at=valid_at, expired_at=expired_at)

        assert metadata.valid_at == valid_at
        assert metadata.invalid_at is None
        assert metadata.expired_at == expired_at

    def test_creates_with_all_times(self, now):
        """Test creates with valid_at, invalid_at, and expired_at."""
        valid_at = now - timedelta(days=10)
        invalid_at = now - timedelta(days=5)
        expired_at = now - timedelta(days=1)

        metadata = TemporalMetadata(
            valid_at=valid_at, invalid_at=invalid_at, expired_at=expired_at
        )

        assert metadata.valid_at == valid_at
        assert metadata.invalid_at == invalid_at
        assert metadata.expired_at == expired_at

    def test_creates_with_invalidation_reason(self, now):
        """Test creates with invalidation reason."""
        metadata = TemporalMetadata(
            valid_at=now,
            expired_at=now,
            invalidation_reason="Infrastructure migrated to server-01",
        )

        assert metadata.invalidation_reason == "Infrastructure migrated to server-01"

    def test_creates_with_superseded_by(self, now):
        """Test creates with superseded_by UUID."""
        superseded_uuid = "abc123-def456-789xyz"
        metadata = TemporalMetadata(
            valid_at=now, expired_at=now, superseded_by=superseded_uuid
        )

        assert metadata.superseded_by == superseded_uuid

    def test_creates_with_complexity_score(self, now):
        """Test creates with complexity score."""
        metadata = TemporalMetadata(valid_at=now, complexity_score=45)

        assert metadata.complexity_score == 45

    def test_creates_with_validation_decision(self, now):
        """Test creates with validation decision."""
        metadata = TemporalMetadata(valid_at=now, validation_decision="approve")

        assert metadata.validation_decision == "approve"


class TestValidAtField:
    """Test valid_at field."""

    def test_requires_valid_at(self):
        """Test valid_at is required."""
        with pytest.raises(ValidationError):
            TemporalMetadata()

    def test_accepts_timezone_aware_datetime(self, now):
        """Test accepts timezone-aware datetime."""
        metadata = TemporalMetadata(valid_at=now)
        assert metadata.valid_at == now

    def test_accepts_iso_string(self):
        """Test accepts ISO format string."""
        iso_string = "2025-01-15T10:00:00+00:00"
        metadata = TemporalMetadata(valid_at=iso_string)
        assert isinstance(metadata.valid_at, datetime)


class TestOptionalDateTimeFields:
    """Test optional datetime fields."""

    def test_invalid_at_optional(self, now):
        """Test invalid_at is optional."""
        metadata = TemporalMetadata(valid_at=now, invalid_at=None)
        assert metadata.invalid_at is None

    def test_expired_at_optional(self, now):
        """Test expired_at is optional."""
        metadata = TemporalMetadata(valid_at=now, expired_at=None)
        assert metadata.expired_at is None

    def test_accepts_past_dates(self, now):
        """Test accepts past dates."""
        past = now - timedelta(days=30)
        metadata = TemporalMetadata(valid_at=past, invalid_at=now, expired_at=now)

        assert metadata.valid_at == past

    def test_accepts_future_dates(self, now):
        """Test accepts future dates."""
        future = now + timedelta(days=30)
        metadata = TemporalMetadata(valid_at=now, invalid_at=future)

        assert metadata.invalid_at == future


class TestComplexityScoreValidation:
    """Test complexity_score field validation."""

    def test_accepts_valid_complexity_scores(self, now):
        """Test accepts valid complexity scores (0-100)."""
        for score in [0, 25, 50, 75, 100]:
            metadata = TemporalMetadata(valid_at=now, complexity_score=score)
            assert metadata.complexity_score == score

    def test_rejects_negative_complexity_score(self, now):
        """Test rejects negative scores."""
        with pytest.raises(ValidationError):
            TemporalMetadata(valid_at=now, complexity_score=-10)

    def test_rejects_complexity_over_100(self, now):
        """Test rejects scores over 100."""
        with pytest.raises(ValidationError):
            TemporalMetadata(valid_at=now, complexity_score=150)

    def test_accepts_none_complexity_score(self, now):
        """Test accepts None for complexity_score."""
        metadata = TemporalMetadata(valid_at=now, complexity_score=None)
        assert metadata.complexity_score is None


class TestValidationDecisionField:
    """Test validation_decision field."""

    def test_accepts_approve_decision(self, now):
        """Test accepts 'approve' decision."""
        metadata = TemporalMetadata(valid_at=now, validation_decision="approve")

        assert metadata.validation_decision == "approve"

    def test_accepts_reject_decision(self, now):
        """Test accepts 'reject' decision."""
        metadata = TemporalMetadata(valid_at=now, validation_decision="reject")

        assert metadata.validation_decision == "reject"

    def test_accepts_modify_decision(self, now):
        """Test accepts 'modify' decision."""
        metadata = TemporalMetadata(valid_at=now, validation_decision="modify")

        assert metadata.validation_decision == "modify"

    def test_accepts_none_validation_decision(self, now):
        """Test accepts None for validation_decision."""
        metadata = TemporalMetadata(valid_at=now, validation_decision=None)
        assert metadata.validation_decision is None


class TestIsCurrentProperty:
    """Test is_current property."""

    def test_is_current_true_when_both_none(self, now):
        """Test is_current=True when both invalid_at and expired_at are None."""
        metadata = TemporalMetadata(valid_at=now)

        assert metadata.is_current is True

    def test_is_current_false_with_invalid_at(self, now):
        """Test is_current=False when invalid_at is set."""
        metadata = TemporalMetadata(
            valid_at=now - timedelta(days=10), invalid_at=now - timedelta(days=1)
        )

        assert metadata.is_current is False

    def test_is_current_false_with_expired_at(self, now):
        """Test is_current=False when expired_at is set."""
        metadata = TemporalMetadata(
            valid_at=now - timedelta(days=10), expired_at=now - timedelta(days=1)
        )

        assert metadata.is_current is False

    def test_is_current_false_with_both(self, now):
        """Test is_current=False when both are set."""
        metadata = TemporalMetadata(
            valid_at=now - timedelta(days=10),
            invalid_at=now - timedelta(days=5),
            expired_at=now - timedelta(days=1),
        )

        assert metadata.is_current is False


class TestInvalidationTypeProperty:
    """Test invalidation_type property."""

    def test_returns_none_for_current_fact(self, now):
        """Test returns None for currently valid facts."""
        metadata = TemporalMetadata(valid_at=now)

        assert metadata.invalidation_type is None

    def test_returns_natural_for_invalid_at_only(self, now):
        """Test returns 'natural' when only invalid_at set."""
        metadata = TemporalMetadata(
            valid_at=now - timedelta(days=10), invalid_at=now - timedelta(days=1)
        )

        assert metadata.invalidation_type == "natural"

    def test_returns_system_for_expired_at_only(self, now):
        """Test returns 'system' when only expired_at set."""
        metadata = TemporalMetadata(
            valid_at=now - timedelta(days=10), expired_at=now - timedelta(days=1)
        )

        assert metadata.invalidation_type == "system"

    def test_returns_both_when_both_set(self, now):
        """Test returns 'both' when both invalid_at and expired_at set."""
        metadata = TemporalMetadata(
            valid_at=now - timedelta(days=10),
            invalid_at=now - timedelta(days=5),
            expired_at=now - timedelta(days=1),
        )

        assert metadata.invalidation_type == "both"


class TestTemporalMetadataEquality:
    """Test TemporalMetadata equality."""

    def test_equals_with_same_values(self, now):
        """Test equality with same values."""
        metadata1 = TemporalMetadata(valid_at=now)
        metadata2 = TemporalMetadata(valid_at=now)

        assert metadata1 == metadata2

    def test_differs_with_different_valid_at(self, now):
        """Test differs with different valid_at."""
        past = now - timedelta(days=1)
        metadata1 = TemporalMetadata(valid_at=now)
        metadata2 = TemporalMetadata(valid_at=past)

        assert metadata1 != metadata2

    def test_differs_with_different_complexity_score(self, now):
        """Test differs with different complexity scores."""
        metadata1 = TemporalMetadata(valid_at=now, complexity_score=50)
        metadata2 = TemporalMetadata(valid_at=now, complexity_score=75)

        assert metadata1 != metadata2


class TestTemporalMetadataSerialization:
    """Test TemporalMetadata serialization."""

    def test_converts_to_dict(self, now):
        """Test converts to dictionary."""
        metadata = TemporalMetadata(
            valid_at=now, complexity_score=45, validation_decision="approve"
        )

        metadata_dict = metadata.model_dump()

        assert isinstance(metadata_dict, dict)
        assert "valid_at" in metadata_dict
        assert metadata_dict["complexity_score"] == 45
        assert metadata_dict["validation_decision"] == "approve"

    def test_converts_to_json(self, now):
        """Test converts to JSON string."""
        metadata = TemporalMetadata(valid_at=now, complexity_score=50)

        json_str = metadata.model_dump_json()

        assert isinstance(json_str, str)
        assert "valid_at" in json_str
        assert "complexity_score" in json_str

    def test_reconstructs_from_dict(self, now):
        """Test reconstructs from dictionary."""
        original = TemporalMetadata(
            valid_at=now, invalid_at=now - timedelta(days=1), complexity_score=50
        )

        metadata_dict = original.model_dump()
        reconstructed = TemporalMetadata(**metadata_dict)

        assert reconstructed.valid_at == original.valid_at
        assert reconstructed.complexity_score == original.complexity_score
        assert reconstructed.is_current == original.is_current


class TestTemporalMetadataRealWorldScenarios:
    """Test real-world temporal validity scenarios."""

    def test_models_natural_expiration(self, now):
        """Test modeling natural fact expiration over time."""
        deployment_start = now - timedelta(days=30)
        migration_date = now - timedelta(days=5)

        # Fact: Service ran on old host, then migrated
        metadata = TemporalMetadata(
            valid_at=deployment_start,
            invalid_at=migration_date,
            invalidation_reason="Migrated to new infrastructure",
        )

        assert metadata.invalidation_type == "natural"
        assert not metadata.is_current

    def test_models_system_invalidation(self, now):
        """Test modeling system-level fact invalidation."""
        fact_created = now - timedelta(days=60)
        system_update = now - timedelta(days=10)

        # Fact was invalid due to system correction
        metadata = TemporalMetadata(
            valid_at=fact_created,
            expired_at=system_update,
            invalidation_reason="Corrected duplicate fact",
            superseded_by="fact-uuid-new",
        )

        assert metadata.invalidation_type == "system"
        assert metadata.superseded_by == "fact-uuid-new"

    def test_models_pending_validation(self, now):
        """Test modeling facts pending human validation."""
        candidate_fact = now - timedelta(hours=1)

        metadata = TemporalMetadata(
            valid_at=candidate_fact,
            complexity_score=75,
            validation_decision=None,  # Not yet decided
        )

        assert metadata.is_current
        assert metadata.invalidation_type is None
        assert metadata.complexity_score == 75

    def test_models_approved_conflict_resolution(self, now):
        """Test modeling approved conflict resolution."""
        old_fact = now - timedelta(days=90)
        conflict_detected = now - timedelta(days=10)
        human_approval = now - timedelta(days=1)

        metadata = TemporalMetadata(
            valid_at=old_fact,
            invalid_at=conflict_detected,
            expired_at=human_approval,
            invalidation_reason="Conflict with newer information",
            validation_decision="approve",
            complexity_score=85,
        )

        assert metadata.invalidation_type == "both"
        assert metadata.validation_decision == "approve"
        assert not metadata.is_current
