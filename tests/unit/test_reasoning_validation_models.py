"""
Unit tests for reasoning validation Pydantic models (Feature 028).

Tests model construction, validation, and serialization.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.models.reasoning_validation_models import (
    ConflictSummary,
    LearningReference,
    PatternReference,
    ReasoningValidationResult,
)


class TestLearningReference:
    """Tests for LearningReference model."""

    def test_learning_reference_required_fields(self):
        """Test that uuid and title are required."""
        with pytest.raises(ValidationError):
            LearningReference(similarity_score=0.9)

    def test_learning_reference_optional_created_at(self):
        """Test that created_at is optional."""
        ref = LearningReference(
            uuid="abc-123",
            title="Test Learning",
            similarity_score=0.85,
            created_at=None,
        )
        assert ref.created_at is None

    def test_learning_reference_with_created_at(self):
        """Test LearningReference with created_at datetime."""
        now = datetime.now()
        ref = LearningReference(
            uuid="abc-123",
            title="Test Learning",
            similarity_score=0.85,
            created_at=now,
            excerpt="Sample excerpt",
        )
        assert ref.created_at == now
        assert ref.excerpt == "Sample excerpt"

    def test_learning_reference_default_excerpt(self):
        """Test that excerpt defaults to empty string."""
        ref = LearningReference(
            uuid="abc-123",
            title="Test Learning",
            similarity_score=0.85,
        )
        assert ref.excerpt == ""


class TestPatternReference:
    """Tests for PatternReference model."""

    def test_pattern_reference_all_fields_required(self):
        """Test that all fields are required."""
        with pytest.raises(ValidationError):
            PatternReference(
                pattern_uuid="p-123",
                name="Test Pattern",
                pattern_type="diagnostic",
            )

    def test_pattern_reference_valid(self):
        """Test valid PatternReference construction."""
        ref = PatternReference(
            pattern_uuid="p-123",
            name="Test Pattern",
            pattern_type="diagnostic",
            similarity_score=0.92,
            success_rate=0.85,
            occurrence_count=5,
            relevance_explanation="Pattern matches query well",
        )
        assert ref.pattern_uuid == "p-123"
        assert ref.success_rate == 0.85
        assert ref.occurrence_count == 5


class TestConflictSummary:
    """Tests for ConflictSummary model."""

    def test_conflict_summary_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError):
            ConflictSummary(
                conflict_type="exact_contradiction",
                severity=75,
            )

    def test_conflict_summary_optional_correction(self):
        """Test that correction_suggestion is optional."""
        summary = ConflictSummary(
            conflict_type="exact_contradiction",
            severity=75,
            claim="Mnemodrome RUNS_ON server-02",
            stored_fact="Mnemodrome RUNS_ON hypervisor-01",
            recommendation="auto_correct",
            correction_suggestion=None,
        )
        assert summary.correction_suggestion is None

    def test_conflict_summary_with_correction(self):
        """Test ConflictSummary with correction_suggestion."""
        summary = ConflictSummary(
            conflict_type="exact_contradiction",
            severity=75,
            claim="Mnemodrome RUNS_ON server-02",
            stored_fact="Mnemodrome RUNS_ON hypervisor-01",
            recommendation="auto_correct",
            correction_suggestion="Mnemodrome RUNS_ON hypervisor-01",
        )
        assert summary.correction_suggestion == "Mnemodrome RUNS_ON hypervisor-01"


class TestReasoningValidationResult:
    """Tests for ReasoningValidationResult model."""

    def test_result_model_defaults(self):
        """Test that all fields have correct defaults."""
        result = ReasoningValidationResult(validation_status="clean")
        assert result.validation_status == "clean"
        assert result.conflicts == []
        assert result.relevant_learnings == []
        assert result.relevant_patterns == []
        assert result.claims_analyzed == 0
        assert result.entities_checked == 0
        assert result.elapsed_ms == 0.0
        assert result.degraded_queries == []
        assert result.summary == ""

    def test_result_model_serialization(self):
        """Test that model serializes to dict and JSON cleanly."""
        result = ReasoningValidationResult(
            validation_status="reinforced",
            claims_analyzed=3,
            entities_checked=2,
            elapsed_ms=142.5,
            summary="Test summary",
        )

        # Test model_dump
        data = result.model_dump()
        assert isinstance(data, dict)
        assert data["validation_status"] == "reinforced"
        assert data["claims_analyzed"] == 3

        # Test model_dump_json
        json_str = result.model_dump_json()
        assert isinstance(json_str, str)
        assert "reinforced" in json_str

    def test_validation_status_accepts_all_four_literals(self):
        """Test that all four valid literals are accepted."""
        statuses = ["clean", "reinforced", "conflicts_detected", "degraded"]
        for status in statuses:
            result = ReasoningValidationResult(validation_status=status)
            assert result.validation_status == status

    def test_validation_status_rejects_invalid(self):
        """Test that invalid status values are rejected."""
        with pytest.raises(ValidationError):
            ReasoningValidationResult(validation_status="unknown")

    def test_result_with_conflicts(self):
        """Test ReasoningValidationResult with conflicts."""
        conflict = ConflictSummary(
            conflict_type="exact_contradiction",
            severity=80,
            claim="Test claim",
            stored_fact="Stored fact",
            recommendation="warn",
        )
        result = ReasoningValidationResult(
            validation_status="conflicts_detected",
            conflicts=[conflict],
            claims_analyzed=1,
        )
        assert len(result.conflicts) == 1
        assert result.conflicts[0].severity == 80

    def test_result_with_learnings(self):
        """Test ReasoningValidationResult with learnings."""
        learning = LearningReference(
            uuid="l-123",
            title="Test Learning",
            similarity_score=0.9,
        )
        result = ReasoningValidationResult(
            validation_status="reinforced",
            relevant_learnings=[learning],
        )
        assert len(result.relevant_learnings) == 1
        assert result.relevant_learnings[0].uuid == "l-123"

    def test_result_with_patterns(self):
        """Test ReasoningValidationResult with patterns."""
        pattern = PatternReference(
            pattern_uuid="p-123",
            name="Test Pattern",
            pattern_type="diagnostic",
            similarity_score=0.85,
            success_rate=0.9,
            occurrence_count=3,
            relevance_explanation="Test explanation",
        )
        result = ReasoningValidationResult(
            validation_status="reinforced",
            relevant_patterns=[pattern],
        )
        assert len(result.relevant_patterns) == 1
        assert result.relevant_patterns[0].name == "Test Pattern"

    def test_result_with_degraded_queries(self):
        """Test ReasoningValidationResult with degraded_queries."""
        result = ReasoningValidationResult(
            validation_status="clean",
            degraded_queries=["fact_retrieval", "pattern_search"],
        )
        assert len(result.degraded_queries) == 2
        assert "fact_retrieval" in result.degraded_queries
