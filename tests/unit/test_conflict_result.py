"""Unit tests for ConflictResult Pydantic model."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.models.conflict_detection_models import (
    ConflictResult,
    ExtractedClaim,
    StoredFact,
)


@pytest.fixture
def stored_fact():
    """Provide test StoredFact."""
    return StoredFact(
        entity1_uuid="e1-uuid",
        entity1_name="server-01",
        relation="RUNS",
        entity2_uuid="e2-uuid",
        entity2_name="Mnemodrome",
        created_at=datetime.now(UTC),
        fact_uuid="fact-uuid",
        is_temporal=False,
    )


@pytest.fixture
def extracted_claim():
    """Provide test ExtractedClaim."""
    return ExtractedClaim(
        claim_type="assertion",
        entity1="server-01",
        relation="RUNS",
        entity2="ProxyVM",
        confidence=0.95,
        context="Context here",
        sentence="server-01 runs ProxyVM",
    )


class TestConflictResultCreation:
    """Test ConflictResult creation and validation."""

    def test_creates_with_required_fields(self, stored_fact, extracted_claim):
        """Test creates ConflictResult with required fields."""
        result = ConflictResult(
            conflict_detected=True,
            conflict_type="exact_contradiction",
            severity=75,
            recommendation="warn",
            stored_fact=stored_fact,
            claim=extracted_claim,
        )

        assert result.conflict_detected is True
        assert result.severity == 75
        assert result.recommendation == "warn"

    def test_creates_with_optional_fields(self, stored_fact, extracted_claim):
        """Test creates with optional fields."""
        result = ConflictResult(
            conflict_detected=True,
            conflict_type="temporal_conflict",
            severity=50,
            recommendation="auto_correct",
            correction_suggestion="server-01 RUNS Mnemodrome",
            stored_fact=stored_fact,
            claim=extracted_claim,
        )

        assert result.correction_suggestion == "server-01 RUNS Mnemodrome"
        assert result.conflict_type == "temporal_conflict"

    def test_creates_without_optional_fact_claim(self):
        """Test creates without optional fact and claim."""
        result = ConflictResult(
            conflict_detected=False,
            conflict_type=None,
            severity=0,
            recommendation="skip",
            stored_fact=None,
            claim=None,
        )

        assert result.conflict_detected is False
        assert result.stored_fact is None
        assert result.claim is None


class TestSeverityValidation:
    """Test severity field validation."""

    def test_validates_severity_bounds(self, stored_fact, extracted_claim):
        """Test severity must be 0-100."""
        with pytest.raises(ValidationError):
            ConflictResult(
                conflict_detected=True,
                conflict_type="exact_contradiction",
                severity=150,  # Invalid: > 100
                recommendation="warn",
                stored_fact=stored_fact,
                claim=extracted_claim,
            )

    def test_accepts_zero_severity(self, stored_fact, extracted_claim):
        """Test accepts severity of 0."""
        result = ConflictResult(
            conflict_detected=False,
            conflict_type=None,
            severity=0,
            recommendation="skip",
            stored_fact=stored_fact,
            claim=extracted_claim,
        )

        assert result.severity == 0

    def test_accepts_hundred_severity(self, stored_fact, extracted_claim):
        """Test accepts severity of 100."""
        result = ConflictResult(
            conflict_detected=True,
            conflict_type="exact_contradiction",
            severity=100,
            recommendation="warn",
            stored_fact=stored_fact,
            claim=extracted_claim,
        )

        assert result.severity == 100

    def test_rejects_negative_severity(self, stored_fact, extracted_claim):
        """Test rejects negative severity."""
        with pytest.raises(ValidationError):
            ConflictResult(
                conflict_detected=True,
                conflict_type="exact_contradiction",
                severity=-10,
                recommendation="warn",
                stored_fact=stored_fact,
                claim=extracted_claim,
            )


class TestConflictTypeValidation:
    """Test conflict_type field validation."""

    def test_accepts_valid_conflict_types(self, stored_fact, extracted_claim):
        """Test accepts all valid conflict types."""
        valid_types = [
            "exact_contradiction",
            "partial_mismatch",
            "temporal_conflict",
            "stale_fact",
        ]

        for conflict_type in valid_types:
            result = ConflictResult(
                conflict_detected=True,
                conflict_type=conflict_type,
                severity=50,
                recommendation="warn",
                stored_fact=stored_fact,
                claim=extracted_claim,
            )
            assert result.conflict_type == conflict_type

    def test_accepts_none_conflict_type(self, stored_fact, extracted_claim):
        """Test accepts None for conflict_type."""
        result = ConflictResult(
            conflict_detected=False,
            conflict_type=None,
            severity=0,
            recommendation="skip",
            stored_fact=stored_fact,
            claim=extracted_claim,
        )

        assert result.conflict_type is None

    def test_rejects_invalid_conflict_type(self, stored_fact, extracted_claim):
        """Test rejects invalid conflict types."""
        with pytest.raises(ValidationError):
            ConflictResult(
                conflict_detected=True,
                conflict_type="invalid_type",
                severity=50,
                recommendation="warn",
                stored_fact=stored_fact,
                claim=extracted_claim,
            )


class TestRecommendationValidation:
    """Test recommendation field validation."""

    def test_requires_recommendation(self, stored_fact, extracted_claim):
        """Test recommendation is required."""
        with pytest.raises(ValidationError):
            ConflictResult(
                conflict_detected=True,
                conflict_type="exact_contradiction",
                severity=50,
                stored_fact=stored_fact,
                claim=extracted_claim,
            )

    def test_accepts_any_recommendation_string(self, stored_fact, extracted_claim):
        """Test accepts any string for recommendation (for flexibility)."""
        result = ConflictResult(
            conflict_detected=True,
            conflict_type="exact_contradiction",
            severity=50,
            recommendation="custom_action",
            stored_fact=stored_fact,
            claim=extracted_claim,
        )

        assert result.recommendation == "custom_action"


class TestConflictDetectedField:
    """Test conflict_detected field."""

    def test_true_value(self, stored_fact, extracted_claim):
        """Test conflict_detected=True."""
        result = ConflictResult(
            conflict_detected=True,
            conflict_type="exact_contradiction",
            severity=50,
            recommendation="warn",
            stored_fact=stored_fact,
            claim=extracted_claim,
        )

        assert result.conflict_detected is True

    def test_false_value(self):
        """Test conflict_detected=False."""
        result = ConflictResult(
            conflict_detected=False,
            conflict_type=None,
            severity=0,
            recommendation="skip",
        )

        assert result.conflict_detected is False


class TestStoredFactField:
    """Test stored_fact field."""

    def test_accepts_stored_fact_object(self, stored_fact, extracted_claim):
        """Test accepts StoredFact object."""
        result = ConflictResult(
            conflict_detected=True,
            conflict_type="exact_contradiction",
            severity=50,
            recommendation="warn",
            stored_fact=stored_fact,
            claim=extracted_claim,
        )

        assert result.stored_fact == stored_fact
        assert result.stored_fact.entity1_name == "server-01"

    def test_accepts_none_stored_fact(self, extracted_claim):
        """Test accepts None for stored_fact."""
        result = ConflictResult(
            conflict_detected=False,
            conflict_type=None,
            severity=0,
            recommendation="skip",
            stored_fact=None,
            claim=extracted_claim,
        )

        assert result.stored_fact is None


class TestClaimField:
    """Test claim field."""

    def test_accepts_extracted_claim_object(self, stored_fact, extracted_claim):
        """Test accepts ExtractedClaim object."""
        result = ConflictResult(
            conflict_detected=True,
            conflict_type="exact_contradiction",
            severity=50,
            recommendation="warn",
            stored_fact=stored_fact,
            claim=extracted_claim,
        )

        assert result.claim == extracted_claim
        assert result.claim.entity1 == "server-01"

    def test_accepts_none_claim(self, stored_fact):
        """Test accepts None for claim."""
        result = ConflictResult(
            conflict_detected=False,
            conflict_type=None,
            severity=0,
            recommendation="skip",
            stored_fact=stored_fact,
            claim=None,
        )

        assert result.claim is None


class TestCorrectionSuggestionField:
    """Test correction_suggestion field."""

    def test_stores_correction_suggestion(self, stored_fact, extracted_claim):
        """Test stores correction suggestion."""
        suggestion = "server-01 RUNS Mnemodrome"
        result = ConflictResult(
            conflict_detected=True,
            conflict_type="exact_contradiction",
            severity=50,
            recommendation="auto_correct",
            correction_suggestion=suggestion,
            stored_fact=stored_fact,
            claim=extracted_claim,
        )

        assert result.correction_suggestion == suggestion

    def test_accepts_none_correction_suggestion(self, stored_fact, extracted_claim):
        """Test accepts None for correction_suggestion."""
        result = ConflictResult(
            conflict_detected=True,
            conflict_type="exact_contradiction",
            severity=80,
            recommendation="warn",
            correction_suggestion=None,
            stored_fact=stored_fact,
            claim=extracted_claim,
        )

        assert result.correction_suggestion is None


class TestConflictResultEquality:
    """Test ConflictResult equality and comparison."""

    def test_equals_with_same_values(self, stored_fact, extracted_claim):
        """Test equality with same values."""
        result1 = ConflictResult(
            conflict_detected=True,
            conflict_type="exact_contradiction",
            severity=50,
            recommendation="warn",
            stored_fact=stored_fact,
            claim=extracted_claim,
        )

        result2 = ConflictResult(
            conflict_detected=True,
            conflict_type="exact_contradiction",
            severity=50,
            recommendation="warn",
            stored_fact=stored_fact,
            claim=extracted_claim,
        )

        assert result1 == result2

    def test_differs_with_different_severity(self, stored_fact, extracted_claim):
        """Test differs with different severity."""
        result1 = ConflictResult(
            conflict_detected=True,
            conflict_type="exact_contradiction",
            severity=50,
            recommendation="warn",
            stored_fact=stored_fact,
            claim=extracted_claim,
        )

        result2 = ConflictResult(
            conflict_detected=True,
            conflict_type="exact_contradiction",
            severity=75,
            recommendation="warn",
            stored_fact=stored_fact,
            claim=extracted_claim,
        )

        assert result1 != result2


class TestConflictResultSerialization:
    """Test ConflictResult serialization."""

    def test_converts_to_dict(self, stored_fact, extracted_claim):
        """Test converts to dictionary."""
        result = ConflictResult(
            conflict_detected=True,
            conflict_type="exact_contradiction",
            severity=50,
            recommendation="warn",
            stored_fact=stored_fact,
            claim=extracted_claim,
        )

        result_dict = result.model_dump()

        assert isinstance(result_dict, dict)
        assert result_dict["conflict_detected"] is True
        assert result_dict["severity"] == 50
        assert result_dict["stored_fact"] is not None

    def test_converts_to_json(self, stored_fact, extracted_claim):
        """Test converts to JSON string."""
        result = ConflictResult(
            conflict_detected=True,
            conflict_type="exact_contradiction",
            severity=50,
            recommendation="warn",
            stored_fact=stored_fact,
            claim=extracted_claim,
        )

        json_str = result.model_dump_json()

        assert isinstance(json_str, str)
        assert "conflict_detected" in json_str
        assert "severity" in json_str

    def test_reconstructs_from_dict(self, stored_fact, extracted_claim):
        """Test reconstructs from dictionary."""
        original = ConflictResult(
            conflict_detected=True,
            conflict_type="exact_contradiction",
            severity=50,
            recommendation="warn",
            stored_fact=stored_fact,
            claim=extracted_claim,
        )

        result_dict = original.model_dump()
        reconstructed = ConflictResult(**result_dict)

        assert reconstructed.conflict_detected == original.conflict_detected
        assert reconstructed.severity == original.severity
