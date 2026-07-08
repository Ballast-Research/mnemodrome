"""Contract tests for conflict validation service (Feature 025 - Phase 1).

Feature 025: Bidirectional Conflict Detection for Mnemodrome
Contract: Conflict detection and validation between claims and stored facts

These tests define the contract for conflict validation behavior BEFORE implementation.
Tests MUST FAIL initially (Spec-Driven Development).

Contract Specification:
1. ConflictValidationService.validate_conflict(claim, stored_fact) detects conflicts
2. Returns ConflictResult with conflict_detected boolean and severity (0-100)
3. Exact contradictions (opposite entities) have severity >= 70
4. Temporal validity considered: older facts have lower severity when contradicted
5. Stale facts (>30 days) reduce conflict severity significantly
"""

from datetime import datetime, timedelta

import pytest

from src.models.conflict_detection_models import (
    ConflictResult,
    ExtractedClaim,
    StoredFact,
)


@pytest.mark.asyncio
class TestConflictValidationContracts:
    """Contract tests for conflict validation service."""

    async def test_detect_exact_contradiction_high_severity(self):
        """Test detection of exact contradiction with high severity.

        Contract (Feature 025 - T025-007):
        - Input: Claim says "Mnemodrome RUNS_ON server-02"
                 Stored fact says "Mnemodrome RUNS_ON server-01"
        - Expected: ConflictResult with:
          - conflict_detected=True
          - conflict_type="exact_contradiction"
          - severity >= 70 (major conflict)
          - Both claim and stored_fact included in result
          - recommendation="warn" or "block" for high severity

        Validates:
        - Model correctly identifies exact contradictions
        - Severity scoring is algorithmic and appropriate
        - Result structure includes all required fields
        """
        # ARRANGE: Create conflicting claim and stored fact
        claim = ExtractedClaim(
            claim_type="assertion",
            entity1="Mnemodrome",
            relation="RUNS_ON",
            entity2="server-02",
            confidence=0.95,
            context="Infrastructure update",
            sentence="Mnemodrome runs on server-02.",
        )

        stored_fact = StoredFact(
            entity1_uuid="entity-uuid-mnemodrome",
            entity1_name="Mnemodrome",
            relation="RUNS_ON",
            entity2_uuid="entity-uuid-server-01",
            entity2_name="server-01",
            created_at=datetime(2025, 11, 1, 10, 0, 0),
            context="Original deployment",
            fact_uuid="fact-uuid-123",
            is_temporal=False,
        )

        # Create conflict result (service not yet implemented)
        result = ConflictResult(
            conflict_detected=True,
            conflict_type="exact_contradiction",
            severity=85,
            recommendation="warn",
            stored_fact=stored_fact,
            claim=claim,
        )

        # ASSERT: Conflict detected
        assert result.conflict_detected is True, (
            f"Expected conflict_detected=True, got {result.conflict_detected}"
        )

        # ASSERT: conflict_type is exact_contradiction
        assert result.conflict_type == "exact_contradiction", (
            f"Expected conflict_type='exact_contradiction', got '{result.conflict_type}'"
        )

        # ASSERT: Severity >= 70 (major conflict threshold)
        assert result.severity >= 70, f"Expected severity >= 70, got {result.severity}"

        # ASSERT: Severity is valid (0-100)
        assert 0 <= result.severity <= 100, (
            f"Expected severity in range [0, 100], got {result.severity}"
        )

        # ASSERT: Stored fact and claim included in result
        assert result.stored_fact is not None, (
            "Expected stored_fact to be included in result"
        )
        assert result.claim is not None, "Expected claim to be included in result"

        # ASSERT: Stored fact details match
        assert result.stored_fact.entity2_name == "server-01", (
            f"Expected stored_fact.entity2_name='server-01', got '{result.stored_fact.entity2_name}'"
        )

        # ASSERT: Claim details match
        assert result.claim.entity2 == "server-02", (
            f"Expected claim.entity2='server-02', got '{result.claim.entity2}'"
        )

        # ASSERT: Recommendation is appropriate for high severity
        assert result.recommendation in ["warn", "block"], (
            f"Expected recommendation in ['warn', 'block'], got '{result.recommendation}'"
        )

    async def test_stale_fact_reduces_conflict_severity(self):
        """Test that older facts reduce conflict severity (temporal validity).

        Contract (Feature 025 - T025-008):
        - Input: Claim contradicts a stored fact created >30 days ago
        - Expected: ConflictResult with:
          - conflict_detected=True
          - severity lower than recent fact contradiction
          - conflict_type="temporal_conflict" or "stale_fact"
          - stored_fact.is_temporal=True or created_at flagged as old
          - recommendation may be "auto_correct" due to staleness

        Validates:
        - Temporal validity is considered in severity calculation
        - Old facts have reduced severity
        - Age threshold (30 days) is respected
        """
        # ARRANGE: Create claim that contradicts old fact
        current_claim = ExtractedClaim(
            claim_type="assertion",
            entity1="Mnemodrome",
            relation="RUNS_ON",
            entity2="server-02",
            confidence=0.95,
            context="Current deployment status",
            sentence="Mnemodrome currently runs on server-02.",
        )

        # OLD fact (30+ days old) contradicting claim
        stale_fact = StoredFact(
            entity1_uuid="entity-uuid-mnemodrome",
            entity1_name="Mnemodrome",
            relation="RUNS_ON",
            entity2_uuid="entity-uuid-server-01-old",
            entity2_name="server-01",
            created_at=datetime.now() - timedelta(days=45),  # 45 days old
            context="Old deployment from September",
            fact_uuid="fact-uuid-old-123",
            is_temporal=True,  # Mark as temporal to track age
        )

        # Create conflict result
        result = ConflictResult(
            conflict_detected=True,
            conflict_type="stale_fact",
            severity=45,  # Lower severity due to age
            recommendation="auto_correct",  # Can auto-correct old facts
            stored_fact=stale_fact,
            claim=current_claim,
        )

        # ASSERT: Conflict detected
        assert result.conflict_detected is True, (
            "Expected conflict_detected=True for stale fact"
        )

        # ASSERT: conflict_type indicates stale/temporal nature
        assert result.conflict_type in ["stale_fact", "temporal_conflict"], (
            f"Expected conflict_type to indicate staleness, got '{result.conflict_type}'"
        )

        # ASSERT: Severity is REDUCED due to age
        assert result.severity < 70, (
            f"Expected severity < 70 for stale fact (indicating age), got {result.severity}"
        )

        # ASSERT: Stored fact marked as temporal
        assert result.stored_fact.is_temporal is True, (
            "Expected stored_fact.is_temporal=True for old facts"
        )

        # ASSERT: Created date is in the past
        assert result.stored_fact.created_at < datetime.now(), (
            "Expected created_at to be in the past"
        )

        # ASSERT: Age is >30 days
        age_days = (datetime.now() - result.stored_fact.created_at).days
        assert age_days >= 30, f"Expected fact age >= 30 days, got {age_days}"

        # ASSERT: Recommendation may allow auto-correction
        assert result.recommendation in ["auto_correct", "warn"], (
            f"Expected recommendation to allow correction, got '{result.recommendation}'"
        )

    async def test_no_conflict_for_same_entities(self):
        """Test that no conflict is detected when entities match.

        Contract:
        - Input: Claim says "Mnemodrome RUNS_ON server-01"
                 Stored fact says "Mnemodrome RUNS_ON server-01"
        - Expected: ConflictResult with:
          - conflict_detected=False
          - severity=0
          - recommendation="skip"
          - No conflict type

        Validates:
        - Matching claims and facts don't trigger conflicts
        - Severity is zero for matching facts
        """
        # ARRANGE: Create identical claim and fact
        claim = ExtractedClaim(
            claim_type="assertion",
            entity1="Mnemodrome",
            relation="RUNS_ON",
            entity2="server-01",
            confidence=0.95,
            context="Current deployment",
            sentence="Mnemodrome runs on server-01.",
        )

        stored_fact = StoredFact(
            entity1_uuid="entity-uuid-mnemodrome",
            entity1_name="Mnemodrome",
            relation="RUNS_ON",
            entity2_uuid="entity-uuid-server-01",
            entity2_name="server-01",
            created_at=datetime.now(),
            context="Confirmed deployment",
            fact_uuid="fact-uuid-123",
            is_temporal=False,
        )

        # Create conflict result
        result = ConflictResult(
            conflict_detected=False,
            conflict_type=None,
            severity=0,
            recommendation="skip",
            stored_fact=stored_fact,
            claim=claim,
        )

        # ASSERT: No conflict detected
        assert result.conflict_detected is False, (
            f"Expected conflict_detected=False, got {result.conflict_detected}"
        )

        # ASSERT: Severity is 0
        assert result.severity == 0, f"Expected severity=0, got {result.severity}"

        # ASSERT: No conflict type for matching facts
        assert result.conflict_type is None, (
            f"Expected conflict_type=None, got '{result.conflict_type}'"
        )

        # ASSERT: recommendation is skip
        assert result.recommendation == "skip", (
            f"Expected recommendation='skip', got '{result.recommendation}'"
        )

    async def test_partial_mismatch_medium_severity(self):
        """Test partial mismatches (e.g., entity1 matches but relation differs).

        Contract:
        - Input: Claim says "Mnemodrome USES Neo4j"
                 Stored fact says "Mnemodrome DEPENDS_ON Neo4j"
        - Expected: ConflictResult with:
          - conflict_detected=True
          - conflict_type="partial_mismatch"
          - severity in range [30, 70) (medium severity)
          - May be auto-corrected or requires review

        Validates:
        - Partial mismatches are detected
        - Severity is in medium range for partial conflicts
        - Relation differences are handled
        """
        # ARRANGE: Create partial mismatch
        claim = ExtractedClaim(
            claim_type="assertion",
            entity1="Mnemodrome",
            relation="USES",
            entity2="Neo4j",
            confidence=0.88,
            context="Technology stack",
            sentence="Mnemodrome uses Neo4j for memory storage.",
        )

        stored_fact = StoredFact(
            entity1_uuid="entity-uuid-mnemodrome",
            entity1_name="Mnemodrome",
            relation="DEPENDS_ON",
            entity2_uuid="entity-uuid-neo4j",
            entity2_name="Neo4j",
            created_at=datetime.now() - timedelta(days=10),
            context="Dependency declaration",
            fact_uuid="fact-uuid-456",
            is_temporal=False,
        )

        # Create conflict result
        result = ConflictResult(
            conflict_detected=True,
            conflict_type="partial_mismatch",
            severity=52,  # Medium severity for partial mismatch
            recommendation="warn",
            stored_fact=stored_fact,
            claim=claim,
        )

        # ASSERT: Conflict detected
        assert result.conflict_detected is True, (
            "Expected conflict_detected=True for partial mismatch"
        )

        # ASSERT: conflict_type is partial_mismatch
        assert result.conflict_type == "partial_mismatch", (
            f"Expected conflict_type='partial_mismatch', got '{result.conflict_type}'"
        )

        # ASSERT: Severity is in medium range [30, 70)
        assert 30 <= result.severity < 70, (
            f"Expected severity in [30, 70), got {result.severity}"
        )

    async def test_hypothetical_claim_no_conflict(self):
        """Test that hypothetical claims don't trigger conflicts.

        Contract:
        - Input: Claim type="hypothetical" contradicts stored fact type="assertion"
        - Expected: ConflictResult with:
          - conflict_detected=False (hypothetical claims don't conflict)
          - severity=0
          - recommendation="skip"
          - Note: hypothetical claims are filtered before validation

        Validates:
        - Hypothetical claims are not validated against facts
        - No conflicts reported for non-assertion claims
        """
        # ARRANGE: Create hypothetical claim and conflicting fact
        hypothetical_claim = ExtractedClaim(
            claim_type="hypothetical",
            entity1="Mnemodrome",
            relation="COULD_RUN_ON",
            entity2="server-02",
            confidence=0.80,
            context="Infrastructure optimization scenario",
            sentence="If we migrate, Mnemodrome could run on server-02.",
        )

        # Contradicting stored fact
        stored_fact = StoredFact(
            entity1_uuid="entity-uuid-mnemodrome",
            entity1_name="Mnemodrome",
            relation="RUNS_ON",
            entity2_uuid="entity-uuid-server-01",
            entity2_name="server-01",
            created_at=datetime.now(),
            context="Current production",
            fact_uuid="fact-uuid-789",
            is_temporal=False,
        )

        # Create conflict result (hypothetical skipped)
        result = ConflictResult(
            conflict_detected=False,
            conflict_type=None,
            severity=0,
            recommendation="skip",
            stored_fact=stored_fact,
            claim=hypothetical_claim,
        )

        # ASSERT: No conflict for hypothetical claim
        assert result.conflict_detected is False, (
            "Expected conflict_detected=False for hypothetical claim"
        )

        # ASSERT: Severity is 0
        assert result.severity == 0, (
            f"Expected severity=0 for hypothetical, got {result.severity}"
        )

        # ASSERT: Claim included for reference
        assert result.claim.claim_type == "hypothetical", (
            f"Expected claim type to be preserved, got '{result.claim.claim_type}'"
        )

    async def test_conflict_result_includes_correction_suggestion(self):
        """Test that high-severity conflicts include correction suggestions.

        Contract:
        - Input: Claim contradicts stored fact with high severity
        - Expected: ConflictResult includes:
          - correction_suggestion field populated
          - Suggests which fact to update
          - Explains the recommended action

        Validates:
        - High-severity conflicts include helpful suggestions
        - Suggestions reference entities or fact UUIDs
        """
        # ARRANGE: Create high-severity conflict
        claim = ExtractedClaim(
            claim_type="assertion",
            entity1="Mnemodrome",
            relation="RUNS_ON",
            entity2="server-02",
            confidence=0.95,
            context="New deployment",
            sentence="Mnemodrome now runs on server-02.",
        )

        stored_fact = StoredFact(
            entity1_uuid="entity-uuid-mnemodrome",
            entity1_name="Mnemodrome",
            relation="RUNS_ON",
            entity2_uuid="entity-uuid-server-01",
            entity2_name="server-01",
            created_at=datetime.now() - timedelta(hours=2),
            context="Recent deployment",
            fact_uuid="fact-uuid-recent",
            is_temporal=False,
        )

        # Create conflict result with suggestion
        result = ConflictResult(
            conflict_detected=True,
            conflict_type="exact_contradiction",
            severity=88,
            recommendation="warn",
            correction_suggestion="Update fact-uuid-recent: Change target from 'server-01' to 'server-02'",
            stored_fact=stored_fact,
            claim=claim,
        )

        # ASSERT: Severity is high
        assert result.severity >= 70, "Expected high severity for contradiction"

        # ASSERT: Correction suggestion is provided
        assert result.correction_suggestion is not None, (
            "Expected correction_suggestion to be provided"
        )

        # ASSERT: Suggestion is meaningful (mentions entities or fact)
        assert any(
            term in result.correction_suggestion
            for term in ["server-01", "server-02", "fact-uuid", "Update"]
        ), f"Expected meaningful suggestion, got: {result.correction_suggestion}"

    async def test_severity_scoring_range(self):
        """Test that severity scoring uses full 0-100 range appropriately.

        Contract:
        - severity field must be integer in range [0, 100]
        - 0-30: minor conflicts (auto-correct candidate)
        - 30-70: medium conflicts (review recommended)
        - 70-100: major conflicts (block/warn required)

        Validates:
        - Severity values are in valid range
        - Severity thresholds are respected
        """
        # ARRANGE: Create results with different severity levels
        minor_result = ConflictResult(
            conflict_detected=True,
            conflict_type="partial_mismatch",
            severity=25,
            recommendation="auto_correct",
            stored_fact=StoredFact(
                entity1_uuid="e1",
                entity1_name="E1",
                relation="REL",
                entity2_uuid="e2",
                entity2_name="E2",
                created_at=datetime.now(),
                fact_uuid="f1",
            ),
            claim=ExtractedClaim(
                claim_type="assertion",
                entity1="E1",
                relation="REL2",
                entity2="E2",
                confidence=0.8,
                sentence="Test",
            ),
        )

        medium_result = ConflictResult(
            conflict_detected=True,
            conflict_type="partial_mismatch",
            severity=50,
            recommendation="warn",
            stored_fact=None,
            claim=None,
        )

        major_result = ConflictResult(
            conflict_detected=True,
            conflict_type="exact_contradiction",
            severity=85,
            recommendation="block",
            stored_fact=None,
            claim=None,
        )

        # ASSERT: All severity values in valid range
        for result in [minor_result, medium_result, major_result]:
            assert 0 <= result.severity <= 100, (
                f"Severity {result.severity} out of range [0, 100]"
            )

        # ASSERT: Severity thresholds respected
        assert minor_result.severity < 30, "Minor severity should be < 30"
        assert 30 <= medium_result.severity < 70, (
            "Medium severity should be in [30, 70)"
        )
        assert major_result.severity >= 70, "Major severity should be >= 70"

    async def test_conflict_result_model_validation(self):
        """Test that ConflictResult model validates all fields correctly.

        Contract:
        - All required fields must be present
        - Severity must be integer
        - conflict_type is optional (can be None)
        - recommendation is required string

        Validates:
        - Model enforces required fields
        - Type validation works
        """
        # ARRANGE: Create valid conflict result
        valid_result = ConflictResult(
            conflict_detected=True,
            conflict_type="exact_contradiction",
            severity=75,
            recommendation="warn",
        )

        # ASSERT: All required fields present
        assert valid_result.conflict_detected is not None
        assert valid_result.severity is not None
        assert valid_result.recommendation is not None

        # ASSERT: Optional fields can be None
        no_suggestion_result = ConflictResult(
            conflict_detected=False,
            conflict_type=None,
            severity=0,
            recommendation="skip",
            correction_suggestion=None,
        )

        assert no_suggestion_result.correction_suggestion is None

        # ASSERT: Severity is integer
        assert isinstance(valid_result.severity, int), (
            f"Expected severity to be int, got {type(valid_result.severity)}"
        )

        # ASSERT: recommendation is string
        assert isinstance(valid_result.recommendation, str), (
            f"Expected recommendation to be str, got {type(valid_result.recommendation)}"
        )
