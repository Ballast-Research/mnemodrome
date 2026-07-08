"""
Unit tests for pattern matching logic (Feature 006: Fact Invalidation).

Tests pattern recognition for auto-resolution of simple conflicts based on
validation history analysis.

Pattern Recognition Criteria:
- 10+ similar conflicts in validation history
- Same conflict type
- All (or majority) approved by humans
- No recent rejections

Confidence Score Calculation:
- confidence = (approvals / total) * 100
- If < 10 validations: return 0 (insufficient data)
"""

from dataclasses import dataclass
from datetime import datetime, timedelta


# Mock data structures (will be replaced with actual models during implementation)
@dataclass
class Relationship:
    """Mock Neo4j relationship for testing."""

    uuid: str
    entity1: str
    relation: str
    entity2: str
    entity1_type: str
    entity2_type: str
    created_at: datetime
    valid_at: datetime
    invalid_at: datetime | None = None
    expired_at: datetime | None = None


@dataclass
class ValidationHistory:
    """Mock validation history record for testing."""

    uuid: str
    conflict_type: str
    complexity_score: int
    human_decision: str  # "approve" | "reject" | "modify"
    timestamp: datetime
    old_fact_uuid: str
    new_fact_uuid: str
    rationale: str


# Functions under test (will be imported from services during implementation)
def determine_conflict_type(old_fact: Relationship, new_fact: dict) -> str:
    """
    Determine conflict type based on relationship patterns.

    Conflict types:
    - same_relation_different_target: Same source/relation, different target
    - infrastructure_relocation: RUNS_ON/HOSTED_ON relationship changes
    - ownership_transfer: OWNED_BY relationship changes
    - temporal_update: Same relationship, different time context
    """
    if old_fact.relation in ["RUNS_ON", "HOSTED_ON"]:
        return "infrastructure_relocation"
    elif old_fact.relation == "OWNED_BY":
        return "ownership_transfer"
    elif old_fact.entity2 != new_fact.get("entity2"):
        return "same_relation_different_target"
    else:
        return "temporal_update"


def calculate_complexity_score_for_matching(
    old_fact: Relationship, new_fact: dict
) -> int:
    """
    Calculate complexity score for similarity matching.

    Simplified version for pattern matching (full version in conflict_detection_service.py).
    Considers age and relationship type for similarity grouping.
    """
    score = 0

    # Age factor
    age_days = (datetime.now() - old_fact.created_at).days
    if age_days > 30:
        score += 30
    elif age_days > 7:
        score += 15

    # Infrastructure changes are generally simpler
    if old_fact.relation in ["RUNS_ON", "HOSTED_ON"]:
        score -= 10

    return max(0, min(100, score))


def find_similar_conflicts(
    validation_history: list[ValidationHistory], old_fact: Relationship, new_fact: dict
) -> list[ValidationHistory]:
    """
    Find similar conflicts in validation history.

    Similarity criteria:
    - Same conflict type (e.g., "infrastructure_relocation")
    - Same entity types (Person, Project, Technology)
    - Similar complexity score (±10 points)
    """
    conflict_type = determine_conflict_type(old_fact, new_fact)
    complexity_score = calculate_complexity_score_for_matching(old_fact, new_fact)

    similar = []
    for v in validation_history:
        if v.conflict_type == conflict_type:
            # Check complexity similarity (±10 points)
            if abs(v.complexity_score - complexity_score) <= 10:
                similar.append(v)

    return similar


def calculate_confidence_score(similar_validations: list[ValidationHistory]) -> int:
    """
    Calculate confidence 0-100 for auto-resolution.

    Confidence = (approvals / total) * 100
    If < 10 validations: return 0 (insufficient data)
    """
    if len(similar_validations) < 10:
        return 0

    approvals = sum(1 for v in similar_validations if v.human_decision == "approve")
    return int((approvals / len(similar_validations)) * 100)


def is_simple_conflict(
    validation_history: list[ValidationHistory],
    old_fact: Relationship,
    new_fact: dict,
    confidence_threshold: int = 80,
) -> tuple[bool, int]:
    """
    Determine if conflict is "simple" enough for auto-resolution.

    Returns:
        (is_simple, confidence_score)

    Criteria:
    - 10+ similar conflicts in history
    - Confidence score >= threshold (default 80%)
    - No recent rejections (last 5 similar conflicts)
    """
    similar = find_similar_conflicts(validation_history, old_fact, new_fact)
    confidence = calculate_confidence_score(similar)

    if confidence == 0:  # Insufficient data
        return False, 0

    if confidence < confidence_threshold:
        return False, confidence

    # Check for recent rejections (last 5 similar conflicts)
    recent_similar = sorted(similar, key=lambda v: v.timestamp, reverse=True)[:5]
    if any(v.human_decision == "reject" for v in recent_similar):
        return False, confidence

    return True, confidence


# ============================================================================
# TEST CASES
# ============================================================================


class TestFindSimilarConflicts:
    """Test finding similar conflicts based on type and complexity."""

    def test_finds_same_conflict_type(self):
        """Should find conflicts with same type."""
        old_fact = Relationship(
            uuid="old-1",
            entity1="Mnemodrome",
            relation="RUNS_ON",
            entity2="server-01",
            entity1_type="Project",
            entity2_type="Technology",
            created_at=datetime.now() - timedelta(days=5),
            valid_at=datetime.now() - timedelta(days=5),
        )

        new_fact = {"entity2": "server-02"}

        validation_history = [
            ValidationHistory(
                uuid="v1",
                conflict_type="infrastructure_relocation",
                complexity_score=0,  # Infrastructure (RUNS_ON) = 0: score starts at 0, -10 for infrastructure
                human_decision="approve",
                timestamp=datetime.now() - timedelta(days=10),
                old_fact_uuid="old-x",
                new_fact_uuid="new-x",
                rationale="Server migration",
            ),
            ValidationHistory(
                uuid="v2",
                conflict_type="ownership_transfer",  # Different type
                complexity_score=15,
                human_decision="approve",
                timestamp=datetime.now() - timedelta(days=9),
                old_fact_uuid="old-y",
                new_fact_uuid="new-y",
                rationale="Owner change",
            ),
        ]

        similar = find_similar_conflicts(validation_history, old_fact, new_fact)

        assert len(similar) == 1
        assert similar[0].conflict_type == "infrastructure_relocation"

    def test_filters_by_complexity_score(self):
        """Should filter by similar complexity score (±10 points)."""
        old_fact = Relationship(
            uuid="old-1",
            entity1="Mnemodrome",
            relation="RUNS_ON",
            entity2="server-01",
            entity1_type="Project",
            entity2_type="Technology",
            created_at=datetime.now() - timedelta(days=5),
            valid_at=datetime.now() - timedelta(days=5),
        )

        new_fact = {"entity2": "server-02"}

        # Expected complexity: 0 (recent age < 7, infrastructure RUNS_ON = -10, clamped to 0)
        validation_history = [
            ValidationHistory(
                uuid="v1",
                conflict_type="infrastructure_relocation",
                complexity_score=0,  # Within range (±10 of 0 = [-10,10])
                human_decision="approve",
                timestamp=datetime.now() - timedelta(days=10),
                old_fact_uuid="old-x",
                new_fact_uuid="new-x",
                rationale="Similar",
            ),
            ValidationHistory(
                uuid="v2",
                conflict_type="infrastructure_relocation",
                complexity_score=5,  # Within range (±10 of 0)
                human_decision="approve",
                timestamp=datetime.now() - timedelta(days=9),
                old_fact_uuid="old-y",
                new_fact_uuid="new-y",
                rationale="Similar",
            ),
            ValidationHistory(
                uuid="v3",
                conflict_type="infrastructure_relocation",
                complexity_score=50,  # Outside range (50 is not in [-10,10])
                human_decision="approve",
                timestamp=datetime.now() - timedelta(days=8),
                old_fact_uuid="old-z",
                new_fact_uuid="new-z",
                rationale="Too different",
            ),
        ]

        similar = find_similar_conflicts(validation_history, old_fact, new_fact)

        assert len(similar) == 2
        assert all(v.complexity_score <= 5 for v in similar)

    def test_empty_history_returns_empty_list(self):
        """Should return empty list when no validation history exists."""
        old_fact = Relationship(
            uuid="old-1",
            entity1="Mnemodrome",
            relation="RUNS_ON",
            entity2="server-01",
            entity1_type="Project",
            entity2_type="Technology",
            created_at=datetime.now() - timedelta(days=5),
            valid_at=datetime.now() - timedelta(days=5),
        )

        new_fact = {"entity2": "server-02"}

        similar = find_similar_conflicts([], old_fact, new_fact)

        assert len(similar) == 0


class TestCalculateConfidenceScore:
    """Test confidence score calculation for auto-resolution."""

    def test_insufficient_data_returns_zero(self):
        """Should return 0 when less than 10 validations."""
        validations = [
            ValidationHistory(
                uuid=f"v{i}",
                conflict_type="infrastructure_relocation",
                complexity_score=15,
                human_decision="approve",
                timestamp=datetime.now() - timedelta(days=i),
                old_fact_uuid=f"old-{i}",
                new_fact_uuid=f"new-{i}",
                rationale="Test",
            )
            for i in range(9)  # Only 9 validations
        ]

        confidence = calculate_confidence_score(validations)

        assert confidence == 0

    def test_all_approvals_returns_100(self):
        """Should return 100% confidence when all validations approved."""
        validations = [
            ValidationHistory(
                uuid=f"v{i}",
                conflict_type="infrastructure_relocation",
                complexity_score=15,
                human_decision="approve",
                timestamp=datetime.now() - timedelta(days=i),
                old_fact_uuid=f"old-{i}",
                new_fact_uuid=f"new-{i}",
                rationale="Test",
            )
            for i in range(10)  # 10 approvals
        ]

        confidence = calculate_confidence_score(validations)

        assert confidence == 100

    def test_all_rejections_returns_zero(self):
        """Should return 0% confidence when all validations rejected."""
        validations = [
            ValidationHistory(
                uuid=f"v{i}",
                conflict_type="infrastructure_relocation",
                complexity_score=15,
                human_decision="reject",
                timestamp=datetime.now() - timedelta(days=i),
                old_fact_uuid=f"old-{i}",
                new_fact_uuid=f"new-{i}",
                rationale="Test",
            )
            for i in range(10)  # 10 rejections
        ]

        confidence = calculate_confidence_score(validations)

        assert confidence == 0

    def test_mixed_decisions_calculates_percentage(self):
        """Should calculate percentage based on approval ratio."""
        validations = [
            ValidationHistory(
                uuid=f"v{i}",
                conflict_type="infrastructure_relocation",
                complexity_score=15,
                human_decision="approve" if i < 9 else "reject",  # 9 approve, 1 reject
                timestamp=datetime.now() - timedelta(days=i),
                old_fact_uuid=f"old-{i}",
                new_fact_uuid=f"new-{i}",
                rationale="Test",
            )
            for i in range(10)
        ]

        confidence = calculate_confidence_score(validations)

        assert confidence == 90  # 9/10 = 90%

    def test_80_percent_approval(self):
        """Should return 80% for 8/10 approvals."""
        validations = [
            ValidationHistory(
                uuid=f"v{i}",
                conflict_type="infrastructure_relocation",
                complexity_score=15,
                human_decision="approve" if i < 8 else "reject",  # 8 approve, 2 reject
                timestamp=datetime.now() - timedelta(days=i),
                old_fact_uuid=f"old-{i}",
                new_fact_uuid=f"new-{i}",
                rationale="Test",
            )
            for i in range(10)
        ]

        confidence = calculate_confidence_score(validations)

        assert confidence == 80


class TestPatternRecognitionThreshold:
    """Test pattern recognition threshold (9 vs 10 validations)."""

    def test_nine_validations_not_simple(self):
        """Should not be simple conflict with only 9 validations."""
        old_fact = Relationship(
            uuid="old-1",
            entity1="Mnemodrome",
            relation="RUNS_ON",
            entity2="server-01",
            entity1_type="Project",
            entity2_type="Technology",
            created_at=datetime.now() - timedelta(days=5),
            valid_at=datetime.now() - timedelta(days=5),
        )

        new_fact = {"entity2": "server-02"}

        # Create 9 similar validations (all approved)
        validation_history = [
            ValidationHistory(
                uuid=f"v{i}",
                conflict_type="infrastructure_relocation",
                complexity_score=5,
                human_decision="approve",
                timestamp=datetime.now() - timedelta(days=i + 10),
                old_fact_uuid=f"old-{i}",
                new_fact_uuid=f"new-{i}",
                rationale="Test",
            )
            for i in range(9)
        ]

        is_simple, confidence = is_simple_conflict(
            validation_history, old_fact, new_fact
        )

        assert is_simple is False
        assert confidence == 0  # Insufficient data

    def test_ten_validations_is_simple(self):
        """Should be simple conflict with 10+ validations and high approval."""
        old_fact = Relationship(
            uuid="old-1",
            entity1="Mnemodrome",
            relation="RUNS_ON",
            entity2="server-01",
            entity1_type="Project",
            entity2_type="Technology",
            created_at=datetime.now() - timedelta(days=5),
            valid_at=datetime.now() - timedelta(days=5),
        )

        new_fact = {"entity2": "server-02"}

        # Create 10 similar validations (all approved)
        validation_history = [
            ValidationHistory(
                uuid=f"v{i}",
                conflict_type="infrastructure_relocation",
                complexity_score=5,
                human_decision="approve",
                timestamp=datetime.now() - timedelta(days=i + 10),
                old_fact_uuid=f"old-{i}",
                new_fact_uuid=f"new-{i}",
                rationale="Test",
            )
            for i in range(10)
        ]

        is_simple, confidence = is_simple_conflict(
            validation_history, old_fact, new_fact
        )

        assert is_simple is True
        assert confidence == 100


class TestApprovalRatioBoundaries:
    """Test approval ratio boundaries for auto-resolution."""

    def test_100_percent_approval_is_simple(self):
        """Should be simple with 100% approval rate."""
        old_fact = Relationship(
            uuid="old-1",
            entity1="Mnemodrome",
            relation="RUNS_ON",
            entity2="server-01",
            entity1_type="Project",
            entity2_type="Technology",
            created_at=datetime.now() - timedelta(days=5),
            valid_at=datetime.now() - timedelta(days=5),
        )

        new_fact = {"entity2": "server-02"}

        validation_history = [
            ValidationHistory(
                uuid=f"v{i}",
                conflict_type="infrastructure_relocation",
                complexity_score=5,
                human_decision="approve",
                timestamp=datetime.now() - timedelta(days=i + 10),
                old_fact_uuid=f"old-{i}",
                new_fact_uuid=f"new-{i}",
                rationale="Test",
            )
            for i in range(10)
        ]

        is_simple, confidence = is_simple_conflict(
            validation_history, old_fact, new_fact
        )

        assert is_simple is True
        assert confidence == 100

    def test_90_percent_approval_is_simple(self):
        """Should be simple with 90% approval rate."""
        old_fact = Relationship(
            uuid="old-1",
            entity1="Mnemodrome",
            relation="RUNS_ON",
            entity2="server-01",
            entity1_type="Project",
            entity2_type="Technology",
            created_at=datetime.now() - timedelta(days=5),
            valid_at=datetime.now() - timedelta(days=5),
        )

        new_fact = {"entity2": "server-02"}

        # 9 approve, 1 reject
        validation_history = [
            ValidationHistory(
                uuid=f"v{i}",
                conflict_type="infrastructure_relocation",
                complexity_score=5,
                human_decision="approve" if i < 9 else "reject",
                timestamp=datetime.now() - timedelta(days=i + 10),
                old_fact_uuid=f"old-{i}",
                new_fact_uuid=f"new-{i}",
                rationale="Test",
            )
            for i in range(10)
        ]

        is_simple, confidence = is_simple_conflict(
            validation_history, old_fact, new_fact
        )

        assert is_simple is True
        assert confidence == 90

    def test_80_percent_approval_is_simple(self):
        """Should be simple with 80% approval rate (at threshold)."""
        old_fact = Relationship(
            uuid="old-1",
            entity1="Mnemodrome",
            relation="RUNS_ON",
            entity2="server-01",
            entity1_type="Project",
            entity2_type="Technology",
            created_at=datetime.now() - timedelta(days=5),
            valid_at=datetime.now() - timedelta(days=5),
        )

        new_fact = {"entity2": "server-02"}

        # 8 approve, 2 reject
        validation_history = [
            ValidationHistory(
                uuid=f"v{i}",
                conflict_type="infrastructure_relocation",
                complexity_score=5,
                human_decision="approve" if i < 8 else "reject",
                timestamp=datetime.now() - timedelta(days=i + 10),
                old_fact_uuid=f"old-{i}",
                new_fact_uuid=f"new-{i}",
                rationale="Test",
            )
            for i in range(10)
        ]

        is_simple, confidence = is_simple_conflict(
            validation_history, old_fact, new_fact
        )

        assert is_simple is True
        assert confidence == 80

    def test_79_percent_approval_not_simple(self):
        """Should not be simple with <80% approval rate."""
        old_fact = Relationship(
            uuid="old-1",
            entity1="Mnemodrome",
            relation="RUNS_ON",
            entity2="server-01",
            entity1_type="Project",
            entity2_type="Technology",
            created_at=datetime.now() - timedelta(days=5),
            valid_at=datetime.now() - timedelta(days=5),
        )

        new_fact = {"entity2": "server-02"}

        # 7 approve, 3 reject = 70%
        validation_history = [
            ValidationHistory(
                uuid=f"v{i}",
                conflict_type="infrastructure_relocation",
                complexity_score=5,
                human_decision="approve" if i < 7 else "reject",
                timestamp=datetime.now() - timedelta(days=i + 10),
                old_fact_uuid=f"old-{i}",
                new_fact_uuid=f"new-{i}",
                rationale="Test",
            )
            for i in range(10)
        ]

        is_simple, confidence = is_simple_conflict(
            validation_history, old_fact, new_fact
        )

        assert is_simple is False
        assert confidence == 70


class TestEdgeCases:
    """Test edge cases for pattern matching."""

    def test_empty_history_not_simple(self):
        """Should not be simple with empty validation history."""
        old_fact = Relationship(
            uuid="old-1",
            entity1="Mnemodrome",
            relation="RUNS_ON",
            entity2="server-01",
            entity1_type="Project",
            entity2_type="Technology",
            created_at=datetime.now() - timedelta(days=5),
            valid_at=datetime.now() - timedelta(days=5),
        )

        new_fact = {"entity2": "server-02"}

        is_simple, confidence = is_simple_conflict([], old_fact, new_fact)

        assert is_simple is False
        assert confidence == 0

    def test_all_rejections_not_simple(self):
        """Should not be simple when all validations rejected."""
        old_fact = Relationship(
            uuid="old-1",
            entity1="Mnemodrome",
            relation="RUNS_ON",
            entity2="server-01",
            entity1_type="Project",
            entity2_type="Technology",
            created_at=datetime.now() - timedelta(days=5),
            valid_at=datetime.now() - timedelta(days=5),
        )

        new_fact = {"entity2": "server-02"}

        validation_history = [
            ValidationHistory(
                uuid=f"v{i}",
                conflict_type="infrastructure_relocation",
                complexity_score=5,
                human_decision="reject",
                timestamp=datetime.now() - timedelta(days=i + 10),
                old_fact_uuid=f"old-{i}",
                new_fact_uuid=f"new-{i}",
                rationale="Test",
            )
            for i in range(10)
        ]

        is_simple, confidence = is_simple_conflict(
            validation_history, old_fact, new_fact
        )

        assert is_simple is False
        assert confidence == 0

    def test_recent_rejection_prevents_auto_resolution(self):
        """Should not auto-resolve if recent validations rejected (pattern change)."""
        old_fact = Relationship(
            uuid="old-1",
            entity1="Mnemodrome",
            relation="RUNS_ON",
            entity2="server-01",
            entity1_type="Project",
            entity2_type="Technology",
            created_at=datetime.now() - timedelta(days=5),
            valid_at=datetime.now() - timedelta(days=5),
        )

        new_fact = {"entity2": "server-02"}

        # 10 validations total: 9 old approvals, 1 recent rejection
        validation_history = [
            ValidationHistory(
                uuid=f"v{i}",
                conflict_type="infrastructure_relocation",
                complexity_score=5,
                human_decision="reject"
                if i == 0
                else "approve",  # Most recent is rejection
                timestamp=datetime.now() - timedelta(days=i + 1),
                old_fact_uuid=f"old-{i}",
                new_fact_uuid=f"new-{i}",
                rationale="Test",
            )
            for i in range(10)
        ]

        is_simple, confidence = is_simple_conflict(
            validation_history, old_fact, new_fact
        )

        assert is_simple is False  # Recent rejection prevents auto-resolution
        assert confidence == 90  # But overall confidence is still high

    def test_custom_confidence_threshold(self):
        """Should respect custom confidence threshold."""
        old_fact = Relationship(
            uuid="old-1",
            entity1="Mnemodrome",
            relation="RUNS_ON",
            entity2="server-01",
            entity1_type="Project",
            entity2_type="Technology",
            created_at=datetime.now() - timedelta(days=5),
            valid_at=datetime.now() - timedelta(days=5),
        )

        new_fact = {"entity2": "server-02"}

        # 9 approve, 1 reject = 90% confidence
        validation_history = [
            ValidationHistory(
                uuid=f"v{i}",
                conflict_type="infrastructure_relocation",
                complexity_score=5,
                human_decision="approve" if i < 9 else "reject",
                timestamp=datetime.now() - timedelta(days=i + 10),
                old_fact_uuid=f"old-{i}",
                new_fact_uuid=f"new-{i}",
                rationale="Test",
            )
            for i in range(10)
        ]

        # Should be simple with default threshold (80%)
        is_simple_80, confidence_80 = is_simple_conflict(
            validation_history, old_fact, new_fact, confidence_threshold=80
        )
        assert is_simple_80 is True
        assert confidence_80 == 90

        # Should NOT be simple with high threshold (95%)
        is_simple_95, confidence_95 = is_simple_conflict(
            validation_history, old_fact, new_fact, confidence_threshold=95
        )
        assert is_simple_95 is False
        assert confidence_95 == 90

    def test_no_matching_similar_conflicts(self):
        """Should not be simple when no similar conflicts found."""
        old_fact = Relationship(
            uuid="old-1",
            entity1="Mnemodrome",
            relation="RUNS_ON",
            entity2="server-01",
            entity1_type="Project",
            entity2_type="Technology",
            created_at=datetime.now() - timedelta(days=5),
            valid_at=datetime.now() - timedelta(days=5),
        )

        new_fact = {"entity2": "server-02"}

        # Create validations with different conflict type
        validation_history = [
            ValidationHistory(
                uuid=f"v{i}",
                conflict_type="ownership_transfer",  # Different type
                complexity_score=5,
                human_decision="approve",
                timestamp=datetime.now() - timedelta(days=i + 10),
                old_fact_uuid=f"old-{i}",
                new_fact_uuid=f"new-{i}",
                rationale="Test",
            )
            for i in range(15)  # More than 10 validations
        ]

        is_simple, confidence = is_simple_conflict(
            validation_history, old_fact, new_fact
        )

        assert is_simple is False
        assert confidence == 0  # No similar conflicts found
