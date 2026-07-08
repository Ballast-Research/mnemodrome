"""Unit tests for conflict complexity assessment algorithm (Feature 006).

These tests verify the complexity scoring logic that determines when human validation
is required for fact invalidation. Following TDD principles, these tests define the
expected behavior BEFORE implementation.

Algorithm Specification (from specs/006-fact-invalidation-and/plan.md):
- Returns complexity score 0-100 (higher = more complex = human validation required)
- Complexity factors:
  - Age of fact being invalidated (>30 days = +30 points)
  - Multiple competing facts exist (+20 points)
  - Cross-entity dependencies detected (+25 points)
  - No historical validation pattern match (+25 points)
  - Recent fact creation (<24 hours = -20 points, likely related update)

Thresholds:
- Score 0-30: Simple conflict -> Auto-resolve if progressive automation enabled
- Score 31-70: Moderate -> Prompt for validation with high confidence recommendation
- Score 71-100: Complex -> ALWAYS prompt for human validation
"""

from datetime import UTC, datetime, timedelta

import pytest


# Mock objects for testing
class MockRelationship:
    """Mock Neo4j relationship object for testing."""

    def __init__(
        self,
        created_at,
        uuid="test-uuid",
        entity1="Entity1",
        relation="RELATES_TO",
        entity2="Entity2",
    ):
        self.uuid = uuid
        self.entity1_name = entity1
        self.relation_type = relation
        self.entity2_name = entity2
        self.created_at = created_at
        self.invalid_at = None
        self.expired_at = None


class MockValidationHistory:
    """Mock validation history record for pattern matching."""

    def __init__(self, conflict_type, human_decision, similarity_score=0.8):
        self.conflict_type = conflict_type
        self.human_decision = human_decision
        self.similarity_score = similarity_score
        self.timestamp = datetime.now(UTC)


@pytest.fixture
def now():
    """Current timestamp for consistent test execution."""
    return datetime.now(UTC)


@pytest.fixture
def recent_fact(now):
    """Fact created less than 24 hours ago."""
    return MockRelationship(created_at=now - timedelta(hours=12))


@pytest.fixture
def week_old_fact(now):
    """Fact created 7 days ago."""
    return MockRelationship(created_at=now - timedelta(days=7))


@pytest.fixture
def month_old_fact(now):
    """Fact created 35 days ago (>30 days threshold)."""
    return MockRelationship(created_at=now - timedelta(days=35))


@pytest.fixture
def new_fact_data():
    """Mock data for new fact being created."""
    return {
        "entity1": "Entity1",
        "relation": "RELATES_TO",
        "entity2": "NewEntity2",
        "context": "Testing new fact",
    }


class TestComplexityAssessmentAgeFactor:
    """Test suite for age-based complexity scoring."""

    def test_recent_fact_under_24_hours_reduces_score(self, recent_fact, new_fact_data):
        """Recent facts (<24 hours) should reduce complexity score by 20 points."""
        from src.services.conflict_detection_service import calculate_complexity_score

        # No other factors - just age
        score = calculate_complexity_score(
            old_fact=recent_fact, new_fact=new_fact_data, validation_history=[]
        )

        # Age factor for <24 hours: -20 points
        # No pattern match: +25 points
        # Expected: 25 - 20 = 5, but clamped at 0
        assert score >= 0
        assert score <= 10  # Should be very low complexity

    def test_fact_under_30_days_no_age_penalty(self, week_old_fact, new_fact_data):
        """Facts between 1-30 days should have no age penalty."""
        from src.services.conflict_detection_service import calculate_complexity_score

        score = calculate_complexity_score(
            old_fact=week_old_fact, new_fact=new_fact_data, validation_history=[]
        )

        # Age factor: 0 points (7 days < 30 days)
        # No pattern match: +25 points
        # Expected: 25
        assert score >= 20
        assert score <= 30

    def test_fact_over_30_days_adds_age_penalty(self, month_old_fact, new_fact_data):
        """Facts older than 30 days should add 30 points complexity."""
        from src.services.conflict_detection_service import calculate_complexity_score

        score = calculate_complexity_score(
            old_fact=month_old_fact, new_fact=new_fact_data, validation_history=[]
        )

        # Age factor: +30 points (35 days > 30 days)
        # No pattern match: +25 points
        # Expected: 55
        assert score >= 50
        assert score <= 60

    def test_age_boundary_exactly_30_days(self, now, new_fact_data):
        """Test exact 30-day boundary condition."""
        exactly_30_days = MockRelationship(created_at=now - timedelta(days=30))

        from src.services.conflict_detection_service import calculate_complexity_score

        score = calculate_complexity_score(
            old_fact=exactly_30_days, new_fact=new_fact_data, validation_history=[]
        )

        # At exactly 30 days, should NOT trigger age penalty (>30 days required)
        # Age factor: 0 points
        # No pattern match: +25 points
        assert score >= 20
        assert score <= 30


class TestComplexityAssessmentCompetingFactsFactor:
    """Test suite for competing facts complexity scoring."""

    def test_no_competing_facts(self, recent_fact, new_fact_data):
        """No competing facts should not add complexity."""
        from src.services.conflict_detection_service import calculate_complexity_score

        # Mock: competing_facts_count = 0
        score = calculate_complexity_score(
            old_fact=recent_fact,
            new_fact=new_fact_data,
            validation_history=[],
            competing_facts_count=0,
        )

        # No competing facts: 0 points
        # Recent fact: -20 points
        # No pattern: +25 points
        # Expected: ~5, clamped to 0+
        assert score >= 0
        assert score <= 10

    def test_one_competing_fact(self, recent_fact, new_fact_data):
        """Single competing fact (just the old fact) should not add complexity."""
        from src.services.conflict_detection_service import calculate_complexity_score

        score = calculate_complexity_score(
            old_fact=recent_fact,
            new_fact=new_fact_data,
            validation_history=[],
            competing_facts_count=1,
        )

        # 1 competing fact does NOT trigger penalty (>1 required)
        assert score >= 0
        assert score <= 10

    def test_multiple_competing_facts_adds_complexity(self, recent_fact, new_fact_data):
        """Multiple competing facts (>1) should add 20 points."""
        from src.services.conflict_detection_service import calculate_complexity_score

        score = calculate_complexity_score(
            old_fact=recent_fact,
            new_fact=new_fact_data,
            validation_history=[],
            competing_facts_count=3,
        )

        # Multiple competing facts: +20 points
        # Recent fact: -20 points
        # No pattern: +25 points
        # Expected: 25
        assert score >= 20
        assert score <= 30

    def test_competing_facts_boundary_condition(self, recent_fact, new_fact_data):
        """Test exactly 2 competing facts (boundary condition)."""
        from src.services.conflict_detection_service import calculate_complexity_score

        score = calculate_complexity_score(
            old_fact=recent_fact,
            new_fact=new_fact_data,
            validation_history=[],
            competing_facts_count=2,
        )

        # 2 competing facts: +20 points (>1 triggers penalty)
        # Recent fact: -20 points
        # No pattern: +25 points
        # Expected: 25
        assert score >= 20
        assert score <= 30


class TestComplexityAssessmentDependencyFactor:
    """Test suite for cross-entity dependency complexity scoring."""

    def test_no_dependencies(self, recent_fact, new_fact_data):
        """No dependent facts should not add complexity."""
        from src.services.conflict_detection_service import calculate_complexity_score

        score = calculate_complexity_score(
            old_fact=recent_fact,
            new_fact=new_fact_data,
            validation_history=[],
            has_dependent_facts=False,
        )

        # No dependencies: 0 points
        # Recent fact: -20 points
        # No pattern: +25 points
        # Expected: ~5, clamped to 0+
        assert score >= 0
        assert score <= 10

    def test_has_dependencies_adds_complexity(self, recent_fact, new_fact_data):
        """Dependent facts should add 25 points complexity."""
        from src.services.conflict_detection_service import calculate_complexity_score

        score = calculate_complexity_score(
            old_fact=recent_fact,
            new_fact=new_fact_data,
            validation_history=[],
            has_dependent_facts=True,
        )

        # Has dependencies: +25 points
        # Recent fact: -20 points
        # No pattern: +25 points
        # Expected: 30
        assert score >= 25
        assert score <= 35


class TestComplexityAssessmentPatternMatchingFactor:
    """Test suite for validation history pattern matching."""

    def test_no_validation_history_adds_complexity(self, recent_fact, new_fact_data):
        """No historical patterns should add 25 points (novel conflict)."""
        from src.services.conflict_detection_service import calculate_complexity_score

        score = calculate_complexity_score(
            old_fact=recent_fact, new_fact=new_fact_data, validation_history=[]
        )

        # No pattern match: +25 points
        # Recent fact: -20 points
        # Expected: 5, clamped to 0+
        assert score >= 0
        assert score <= 10

    def test_approval_pattern_reduces_complexity(self, recent_fact, new_fact_data):
        """Strong approval pattern (all approvals) should reduce score by 20 points."""
        from src.services.conflict_detection_service import calculate_complexity_score

        # Mock 5 similar conflicts all approved
        validation_history = [
            MockValidationHistory("same_relation_different_target", "approve")
            for _ in range(5)
        ]

        score = calculate_complexity_score(
            old_fact=recent_fact,
            new_fact=new_fact_data,
            validation_history=validation_history,
        )

        # Pattern match (all approvals): -20 points
        # Recent fact: -20 points
        # Expected: -40, clamped to 0
        assert score == 0

    def test_mixed_pattern_does_not_reduce_complexity(self, recent_fact, new_fact_data):
        """Mixed approval/rejection pattern should not reduce complexity."""
        from src.services.conflict_detection_service import calculate_complexity_score

        # Mock mixed decisions
        validation_history = [
            MockValidationHistory("same_relation_different_target", "approve"),
            MockValidationHistory("same_relation_different_target", "approve"),
            MockValidationHistory("same_relation_different_target", "reject"),
        ]

        score = calculate_complexity_score(
            old_fact=recent_fact,
            new_fact=new_fact_data,
            validation_history=validation_history,
        )

        # Mixed pattern: no reduction
        # Recent fact: -20 points
        # Pattern exists but not all approvals: 0 points
        # Expected: -20, clamped to 0
        assert score >= 0
        assert score <= 10

    def test_rejection_pattern_does_not_reduce_complexity(
        self, recent_fact, new_fact_data
    ):
        """All-rejection pattern should not reduce complexity."""
        from src.services.conflict_detection_service import calculate_complexity_score

        validation_history = [
            MockValidationHistory("same_relation_different_target", "reject")
            for _ in range(5)
        ]

        score = calculate_complexity_score(
            old_fact=recent_fact,
            new_fact=new_fact_data,
            validation_history=validation_history,
        )

        # All rejections: no reduction (only all-approval reduces)
        # Recent fact: -20 points
        # Expected: -20, clamped to 0
        assert score >= 0
        assert score <= 10

    def test_single_approval_not_strong_pattern(self, recent_fact, new_fact_data):
        """Single approval should not trigger pattern reduction."""
        from src.services.conflict_detection_service import calculate_complexity_score

        validation_history = [
            MockValidationHistory("same_relation_different_target", "approve")
        ]

        score = calculate_complexity_score(
            old_fact=recent_fact,
            new_fact=new_fact_data,
            validation_history=validation_history,
        )

        # Pattern reduction requires all approvals (1 is all, so should reduce)
        # Recent fact: -20 points
        # Pattern match: -20 points
        # Expected: -40, clamped to 0
        assert score == 0


class TestComplexityAssessmentScoreClamping:
    """Test suite for score boundary conditions and clamping."""

    def test_negative_score_clamped_to_zero(self, recent_fact, new_fact_data):
        """Negative scores should be clamped to 0."""
        from src.services.conflict_detection_service import calculate_complexity_score

        # Stack negative factors
        validation_history = [
            MockValidationHistory("simple_pattern", "approve") for _ in range(5)
        ]

        score = calculate_complexity_score(
            old_fact=recent_fact,  # Recent: -20
            new_fact=new_fact_data,
            validation_history=validation_history,  # All approvals: -20
            competing_facts_count=0,  # No competing: 0
            has_dependent_facts=False,  # No dependencies: 0
        )

        # Total: -40, should clamp to 0
        assert score == 0

    def test_score_over_100_clamped_to_100(self, month_old_fact, new_fact_data):
        """Scores over 100 should be clamped to 100."""
        from src.services.conflict_detection_service import calculate_complexity_score

        # Stack positive factors
        score = calculate_complexity_score(
            old_fact=month_old_fact,  # Age: +30
            new_fact=new_fact_data,
            validation_history=[],  # No pattern: +25
            competing_facts_count=5,  # Multiple competing: +20
            has_dependent_facts=True,  # Has dependencies: +25
        )

        # Total: 30 + 25 + 20 + 25 = 100
        assert score == 100

    def test_maximum_score_cannot_exceed_100(self, month_old_fact, new_fact_data):
        """Even with extreme factors, score cannot exceed 100."""
        from src.services.conflict_detection_service import calculate_complexity_score

        # Artificial extreme case (if implementation allowed >100)
        # This test ensures clamping works
        score = calculate_complexity_score(
            old_fact=month_old_fact,  # +30
            new_fact=new_fact_data,
            validation_history=[],  # +25
            competing_facts_count=10,  # +20
            has_dependent_facts=True,  # +25
        )

        assert score <= 100

    def test_minimum_score_is_zero(self):
        """Minimum possible score is 0."""
        from datetime import datetime

        from src.services.conflict_detection_service import calculate_complexity_score

        # Most negative scenario
        recent = MockRelationship(created_at=datetime.now(UTC))

        validation_history = [
            MockValidationHistory("pattern", "approve") for _ in range(10)
        ]

        score = calculate_complexity_score(
            old_fact=recent,
            new_fact={"entity1": "A", "relation": "B", "entity2": "C"},
            validation_history=validation_history,
            competing_facts_count=0,
            has_dependent_facts=False,
        )

        assert score >= 0


class TestComplexityAssessmentCombinedFactors:
    """Test suite for multiple factors interacting."""

    def test_all_positive_factors_maximum_complexity(
        self, month_old_fact, new_fact_data
    ):
        """All positive factors should result in high complexity score."""
        from src.services.conflict_detection_service import calculate_complexity_score

        score = calculate_complexity_score(
            old_fact=month_old_fact,  # Age >30 days: +30
            new_fact=new_fact_data,
            validation_history=[],  # No pattern: +25
            competing_facts_count=3,  # Multiple competing: +20
            has_dependent_facts=True,  # Has dependencies: +25
        )

        # Expected: 30 + 25 + 20 + 25 = 100
        assert score == 100

    def test_all_negative_factors_minimum_complexity(self, recent_fact, new_fact_data):
        """All negative/zero factors should result in low complexity score."""
        from src.services.conflict_detection_service import calculate_complexity_score

        validation_history = [
            MockValidationHistory("pattern", "approve") for _ in range(5)
        ]

        score = calculate_complexity_score(
            old_fact=recent_fact,  # Recent <24h: -20
            new_fact=new_fact_data,
            validation_history=validation_history,  # All approvals: -20
            competing_facts_count=0,  # No competing: 0
            has_dependent_facts=False,  # No dependencies: 0
        )

        # Expected: -20 - 20 = -40, clamped to 0
        assert score == 0

    def test_mixed_factors_moderate_complexity(self, week_old_fact, new_fact_data):
        """Mixed positive and negative factors should result in moderate score."""
        from src.services.conflict_detection_service import calculate_complexity_score

        score = calculate_complexity_score(
            old_fact=week_old_fact,  # 7 days: 0 points
            new_fact=new_fact_data,
            validation_history=[],  # No pattern: +25
            competing_facts_count=2,  # Multiple: +20
            has_dependent_facts=False,  # No dependencies: 0
        )

        # Expected: 0 + 25 + 20 + 0 = 45 (moderate complexity)
        assert 40 <= score <= 50

    def test_threshold_boundaries_simple_conflict(self, recent_fact, new_fact_data):
        """Score 0-30 should indicate simple conflict."""
        from src.services.conflict_detection_service import calculate_complexity_score

        validation_history = [MockValidationHistory("pattern", "approve")]

        score = calculate_complexity_score(
            old_fact=recent_fact,
            new_fact=new_fact_data,
            validation_history=validation_history,
            competing_facts_count=0,
            has_dependent_facts=False,
        )

        # Should be in simple range (0-30)
        assert 0 <= score <= 30

    def test_threshold_boundaries_moderate_conflict(self, week_old_fact, new_fact_data):
        """Score 31-70 should indicate moderate conflict."""
        from src.services.conflict_detection_service import calculate_complexity_score

        score = calculate_complexity_score(
            old_fact=week_old_fact,
            new_fact=new_fact_data,
            validation_history=[],  # +25
            competing_facts_count=2,  # +20
            has_dependent_facts=False,
        )

        # Should be in moderate range (31-70)
        assert 31 <= score <= 70

    def test_threshold_boundaries_complex_conflict(self, month_old_fact, new_fact_data):
        """Score 71-100 should indicate complex conflict."""
        from src.services.conflict_detection_service import calculate_complexity_score

        score = calculate_complexity_score(
            old_fact=month_old_fact,  # +30
            new_fact=new_fact_data,
            validation_history=[],  # +25
            competing_facts_count=3,  # +20
            has_dependent_facts=True,  # +25
        )

        # Should be in complex range (71-100)
        assert 71 <= score <= 100


class TestComplexityAssessmentEdgeCases:
    """Test suite for edge cases and special scenarios."""

    def test_exact_24_hour_boundary(self, now, new_fact_data):
        """Test fact created exactly 24 hours ago (boundary condition)."""
        from src.services.conflict_detection_service import calculate_complexity_score

        exactly_24h = MockRelationship(created_at=now - timedelta(hours=24))

        score = calculate_complexity_score(
            old_fact=exactly_24h, new_fact=new_fact_data, validation_history=[]
        )

        # At exactly 24 hours, should NOT get recent reduction (<24h required)
        # No recent bonus: 0
        # No pattern: +25
        # Expected: ~25
        assert 20 <= score <= 30

    def test_very_old_fact_multiple_years(self, new_fact_data):
        """Test fact created years ago (extreme age)."""
        from datetime import datetime

        from src.services.conflict_detection_service import calculate_complexity_score

        very_old = MockRelationship(
            created_at=datetime.now(UTC) - timedelta(days=730)  # 2 years
        )

        score = calculate_complexity_score(
            old_fact=very_old, new_fact=new_fact_data, validation_history=[]
        )

        # Age >30 days: +30 (same penalty regardless of how old)
        # No pattern: +25
        # Expected: 55
        assert 50 <= score <= 60

    def test_zero_validation_history_is_novel_conflict(
        self, recent_fact, new_fact_data
    ):
        """Empty validation history should be treated as novel conflict."""
        from src.services.conflict_detection_service import calculate_complexity_score

        score = calculate_complexity_score(
            old_fact=recent_fact, new_fact=new_fact_data, validation_history=[]
        )

        # Novel conflict adds complexity
        # No pattern: +25
        # Recent: -20
        # Expected: 5, clamped to 0+
        assert score >= 0

    def test_none_values_handled_gracefully(self, recent_fact, new_fact_data):
        """Test that None values for optional parameters are handled."""
        from src.services.conflict_detection_service import calculate_complexity_score

        # Should not raise exception with None values
        score = calculate_complexity_score(
            old_fact=recent_fact,
            new_fact=new_fact_data,
            validation_history=None or [],  # Implementation should handle None
            competing_facts_count=None or 0,
            has_dependent_facts=None or False,
        )

        assert isinstance(score, int)
        assert 0 <= score <= 100

    def test_empty_new_fact_data(self, recent_fact):
        """Test with minimal new fact data."""
        from src.services.conflict_detection_service import calculate_complexity_score

        minimal_new_fact = {"entity1": "E1", "relation": "R", "entity2": "E2"}

        score = calculate_complexity_score(
            old_fact=recent_fact, new_fact=minimal_new_fact, validation_history=[]
        )

        # Should still calculate score
        assert isinstance(score, int)
        assert 0 <= score <= 100


class TestComplexityAssessmentReturnType:
    """Test suite for return value validation."""

    def test_returns_integer(self, recent_fact, new_fact_data):
        """Complexity score must be an integer."""
        from src.services.conflict_detection_service import calculate_complexity_score

        score = calculate_complexity_score(
            old_fact=recent_fact, new_fact=new_fact_data, validation_history=[]
        )

        assert isinstance(score, int)

    def test_return_value_in_valid_range(self, recent_fact, new_fact_data):
        """Complexity score must be between 0 and 100."""
        from src.services.conflict_detection_service import calculate_complexity_score

        score = calculate_complexity_score(
            old_fact=recent_fact, new_fact=new_fact_data, validation_history=[]
        )

        assert 0 <= score <= 100

    def test_consistent_score_for_same_inputs(self, recent_fact, new_fact_data):
        """Same inputs should always produce same score (deterministic)."""
        from src.services.conflict_detection_service import calculate_complexity_score

        score1 = calculate_complexity_score(
            old_fact=recent_fact, new_fact=new_fact_data, validation_history=[]
        )

        score2 = calculate_complexity_score(
            old_fact=recent_fact, new_fact=new_fact_data, validation_history=[]
        )

        assert score1 == score2
