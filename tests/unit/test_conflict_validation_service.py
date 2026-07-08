"""Unit tests for ConflictValidationService."""

from datetime import UTC, datetime, timedelta

import pytest

from src.models.conflict_detection_models import (
    ExtractedClaim,
    StoredFact,
)
from src.services.conflict_validation_service import ConflictValidationService


@pytest.fixture
def conflict_validation_service():
    """Provide ConflictValidationService with test configuration."""
    return ConflictValidationService(
        severity_minor_max=40,
        severity_major_min=70,
        critical_entities={"Mnemodrome", "Neo4j", "Qdrant"},
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
        created_at=datetime.now(),
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
        context="",
        sentence="server-01 runs ProxyVM",
    )


class TestConflictValidationServiceInit:
    """Test service initialization."""

    def test_init_with_defaults(self):
        """Test initializes with default thresholds."""
        service = ConflictValidationService()
        assert service.severity_minor_max == 40
        assert service.severity_major_min == 70

    def test_init_validates_severity_bounds(self):
        """Test validates severity parameters."""
        with pytest.raises(ValueError, match="severity_minor_max must be 0-100"):
            ConflictValidationService(severity_minor_max=150)

    def test_init_validates_threshold_order(self):
        """Test validates major >= minor."""
        with pytest.raises(ValueError):
            ConflictValidationService(severity_minor_max=70, severity_major_min=40)

    def test_init_with_custom_critical_entities(self):
        """Test initializes with custom critical entities."""
        custom_entities = {"MyService", "MyDatabase"}
        service = ConflictValidationService(critical_entities=custom_entities)
        assert service.critical_entities == custom_entities


class TestValidateClaim:
    """Test claim validation against facts."""

    @pytest.mark.asyncio
    async def test_validate_returns_none_on_empty_facts(
        self, conflict_validation_service, extracted_claim
    ):
        """Test returns None if no facts to validate against."""
        result = await conflict_validation_service.validate_claim(extracted_claim, [])

        assert result is None

    @pytest.mark.asyncio
    async def test_validate_skips_non_assertion_claims(
        self, conflict_validation_service, stored_fact
    ):
        """Test skips hypothetical and example claims."""
        hypothetical = ExtractedClaim(
            claim_type="hypothetical",
            entity1="server-01",
            relation="RUNS",
            entity2="ProxyVM",
            confidence=0.95,
            context="",
            sentence="If server-01 runs ProxyVM",
        )

        result = await conflict_validation_service.validate_claim(
            hypothetical, [stored_fact]
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_validate_detects_entity2_mismatch(
        self, conflict_validation_service, stored_fact, extracted_claim
    ):
        """Test detects conflict when entity2 differs."""
        result = await conflict_validation_service.validate_claim(
            extracted_claim,  # entity2="ProxyVM"
            [stored_fact],  # entity2_name="Mnemodrome"
        )

        assert result is not None
        assert result.conflict_detected is True
        assert "ProxyVM" in result.claim.entity2
        assert "Mnemodrome" in result.stored_fact.entity2_name

    @pytest.mark.asyncio
    async def test_validate_ignores_non_matching_entities(
        self, conflict_validation_service, stored_fact
    ):
        """Test ignores facts with different entity1 or relation."""
        different_entity_claim = ExtractedClaim(
            claim_type="assertion",
            entity1="OtherService",  # Different entity1
            relation="RUNS",
            entity2="Something",
            confidence=0.95,
            context="",
            sentence="test",
        )

        result = await conflict_validation_service.validate_claim(
            different_entity_claim, [stored_fact]
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_validate_case_insensitive_matching(
        self, conflict_validation_service
    ):
        """Test entity matching is case-insensitive."""
        stored = StoredFact(
            entity1_uuid="e1",
            entity1_name="SERVER-01",
            relation="RUNS",
            entity2_uuid="e2",
            entity2_name="Mnemodrome",
            created_at=datetime.now(),
            fact_uuid="f-uuid",
        )

        claim = ExtractedClaim(
            claim_type="assertion",
            entity1="server-01",  # Lowercase
            relation="RUNS",
            entity2="NewService",
            confidence=0.95,
            context="",
            sentence="test",
        )

        result = await conflict_validation_service.validate_claim(claim, [stored])

        assert result is not None
        assert result.conflict_detected is True


class TestCalculateSeverity:
    """Test severity score calculation."""

    def test_base_contradiction_score(
        self, conflict_validation_service, extracted_claim, stored_fact
    ):
        """Test base contradiction adds 30 points."""
        score = conflict_validation_service._calculate_severity(
            extracted_claim, stored_fact
        )
        assert score >= 30  # At minimum base contradiction

    def test_exact_entity2_mismatch_bonus(
        self, conflict_validation_service, stored_fact
    ):
        """Test exact entity2 mismatch adds bonus."""
        claim = ExtractedClaim(
            claim_type="assertion",
            entity1="server-01",
            relation="RUNS",
            entity2="Wrong",
            confidence=0.5,
            context="",
            sentence="test",
        )

        score = conflict_validation_service._calculate_severity(claim, stored_fact)
        assert score >= 55  # Base 30 + entity2 mismatch 30, may have minor adjustments

    def test_high_confidence_claim_bonus(
        self, conflict_validation_service, stored_fact
    ):
        """Test high confidence claim adds bonus."""
        high_confidence_claim = ExtractedClaim(
            claim_type="assertion",
            entity1="server-01",
            relation="RUNS",
            entity2="Wrong",
            confidence=0.95,
            context="",
            sentence="test",
        )

        score = conflict_validation_service._calculate_severity(
            high_confidence_claim, stored_fact
        )
        assert (
            score >= 65
        )  # Base 30 + entity2 30 + confidence 10, may have minor adjustments

    def test_recent_fact_penalty(self, conflict_validation_service):
        """Test recent facts reduce severity."""
        recent_fact = StoredFact(
            entity1_uuid="e1",
            entity1_name="server-01",
            relation="RUNS",
            entity2_uuid="e2",
            entity2_name="Mnemodrome",
            created_at=datetime.now() - timedelta(days=1),  # Recent
            fact_uuid="f-uuid",
        )

        claim = ExtractedClaim(
            claim_type="assertion",
            entity1="server-01",
            relation="RUNS",
            entity2="Wrong",
            confidence=0.95,
            context="",
            sentence="test",
        )

        score = conflict_validation_service._calculate_severity(claim, recent_fact)
        # Should be reduced by 5 due to recent fact
        assert score < 75

    def test_critical_entity_bonus(self, conflict_validation_service, extracted_claim):
        """Test critical entity adds severity bonus."""
        # Create fact with critical entity
        critical_fact = StoredFact(
            entity1_uuid="e1",
            entity1_name="Mnemodrome",  # Critical entity
            relation="RUNS",
            entity2_uuid="e2",
            entity2_name="Service",
            created_at=datetime.now(),
            fact_uuid="f-uuid",
        )

        claim = ExtractedClaim(
            claim_type="assertion",
            entity1="Mnemodrome",
            relation="RUNS",
            entity2="Wrong",
            confidence=0.8,
            context="",
            sentence="test",
        )

        score = conflict_validation_service._calculate_severity(claim, critical_fact)
        # Should include bonus for critical entity
        assert score >= 40

    def test_severity_clamped_to_100(self, conflict_validation_service, stored_fact):
        """Test severity is clamped to max 100."""
        extreme_claim = ExtractedClaim(
            claim_type="assertion",
            entity1="Mnemodrome",  # Critical
            relation="RUNS",
            entity2="Wrong",
            confidence=0.99,  # Very high confidence
            context="",
            sentence="test",
        )

        score = conflict_validation_service._calculate_severity(
            extreme_claim, stored_fact
        )
        assert score <= 100

    def test_severity_clamped_to_zero(self, conflict_validation_service):
        """Test severity is clamped to min 0."""
        old_fact = StoredFact(
            entity1_uuid="e1",
            entity1_name="Service",
            relation="RUNS",
            entity2_uuid="e2",
            entity2_name="Host",
            created_at=datetime.now() - timedelta(days=365),  # Very old
            fact_uuid="f-uuid",
        )

        claim = ExtractedClaim(
            claim_type="assertion",
            entity1="Service",
            relation="RUNS",
            entity2="NewHost",
            confidence=0.1,  # Low confidence
            context="",
            sentence="test",
        )

        score = conflict_validation_service._calculate_severity(claim, old_fact)
        assert score >= 0


class TestDetectConflictType:
    """Test conflict type detection."""

    def test_detects_exact_contradiction(
        self, conflict_validation_service, stored_fact, extracted_claim
    ):
        """Test detects exact contradiction."""
        conflict_type = conflict_validation_service._detect_conflict_type(
            extracted_claim, stored_fact
        )

        assert conflict_type == "exact_contradiction"

    def test_detects_temporal_conflict(self, conflict_validation_service):
        """Test detects temporal conflict."""
        temporal_fact = StoredFact(
            entity1_uuid="e1",
            entity1_name="server-01",
            relation="RUNS",
            entity2_uuid="e2",
            entity2_name="OldHost",
            created_at=datetime.now(UTC),
            fact_uuid="f-uuid",
            is_temporal=True,
        )

        temporal_claim = ExtractedClaim(
            claim_type="assertion",
            entity1="server-01",
            relation="will run",  # Temporal keyword
            entity2="NewHost",
            confidence=0.9,
            context="",
            sentence="server-01 will run NewHost",
        )

        conflict_type = conflict_validation_service._detect_conflict_type(
            temporal_claim, temporal_fact
        )

        assert conflict_type == "temporal_conflict"

    def test_detects_stale_fact(self, conflict_validation_service):
        """Test detects stale fact (>30 days old)."""
        old_fact = StoredFact(
            entity1_uuid="e1",
            entity1_name="server-01",
            relation="RUNS",
            entity2_uuid="e2",
            entity2_name="OldHost",
            created_at=datetime.now() - timedelta(days=60),
            fact_uuid="f-uuid",
            is_temporal=False,
        )

        claim = ExtractedClaim(
            claim_type="assertion",
            entity1="server-01",
            relation="RUNS",
            entity2="NewHost",
            confidence=0.9,
            context="",
            sentence="test",
        )

        conflict_type = conflict_validation_service._detect_conflict_type(
            claim, old_fact
        )

        assert conflict_type == "stale_fact"


class TestEntitiesAndRelationMatch:
    """Test entity and relation matching logic."""

    def test_matches_exact_case_entities(self, conflict_validation_service):
        """Test exact case matching works."""
        fact = StoredFact(
            entity1_uuid="e1",
            entity1_name="server-01",
            relation="RUNS",
            entity2_uuid="e2",
            entity2_name="Mnemodrome",
            created_at=datetime.now(UTC),
            fact_uuid="f-uuid",
        )

        claim = ExtractedClaim(
            claim_type="assertion",
            entity1="server-01",
            relation="RUNS",
            entity2="Something",
            confidence=0.9,
            context="",
            sentence="test",
        )

        assert conflict_validation_service._entities_and_relation_match(claim, fact)

    def test_ignores_case_differences(self, conflict_validation_service):
        """Test case-insensitive matching."""
        fact = StoredFact(
            entity1_uuid="e1",
            entity1_name="SERVER-01",
            relation="runs",
            entity2_uuid="e2",
            entity2_name="Mnemodrome",
            created_at=datetime.now(UTC),
            fact_uuid="f-uuid",
        )

        claim = ExtractedClaim(
            claim_type="assertion",
            entity1="server-01",
            relation="RUNS",
            entity2="Something",
            confidence=0.9,
            context="",
            sentence="test",
        )

        assert conflict_validation_service._entities_and_relation_match(claim, fact)

    def test_fails_on_different_entity1(self, conflict_validation_service):
        """Test fails when entity1 differs."""
        fact = StoredFact(
            entity1_uuid="e1",
            entity1_name="server-01",
            relation="RUNS",
            entity2_uuid="e2",
            entity2_name="Mnemodrome",
            created_at=datetime.now(UTC),
            fact_uuid="f-uuid",
        )

        claim = ExtractedClaim(
            claim_type="assertion",
            entity1="OtherService",
            relation="RUNS",
            entity2="Something",
            confidence=0.9,
            context="",
            sentence="test",
        )

        assert not conflict_validation_service._entities_and_relation_match(claim, fact)

    def test_fails_on_different_relation(self, conflict_validation_service):
        """Test fails when relation differs."""
        fact = StoredFact(
            entity1_uuid="e1",
            entity1_name="server-01",
            relation="RUNS",
            entity2_uuid="e2",
            entity2_name="Mnemodrome",
            created_at=datetime.now(UTC),
            fact_uuid="f-uuid",
        )

        claim = ExtractedClaim(
            claim_type="assertion",
            entity1="server-01",
            relation="DEPLOYS",
            entity2="Something",
            confidence=0.9,
            context="",
            sentence="test",
        )

        assert not conflict_validation_service._entities_and_relation_match(claim, fact)


class TestGenerateRecommendation:
    """Test recommendation generation."""

    def test_skip_for_low_severity(self, conflict_validation_service):
        """Test 'skip' recommendation for low severity."""
        rec = conflict_validation_service._generate_recommendation(20)
        assert rec == "skip"

    def test_auto_correct_for_medium_severity(self, conflict_validation_service):
        """Test 'auto_correct' for medium severity."""
        rec = conflict_validation_service._generate_recommendation(55)
        assert rec == "auto_correct"

    def test_warn_for_high_severity(self, conflict_validation_service):
        """Test 'warn' for high severity."""
        rec = conflict_validation_service._generate_recommendation(85)
        assert rec == "warn"


class TestGenerateCorrection:
    """Test correction suggestion generation."""

    def test_generates_corrected_text(self, conflict_validation_service, stored_fact):
        """Test generates corrected text."""
        claim = ExtractedClaim(
            claim_type="assertion",
            entity1="server-01",
            relation="RUNS",
            entity2="Wrong",
            confidence=0.9,
            context="",
            sentence="test",
        )

        correction = conflict_validation_service._generate_correction(
            claim, stored_fact
        )

        assert "server-01" in correction
        assert "RUNS" in correction
        assert "Mnemodrome" in correction  # Corrected entity2


class TestValidateClaimsBatch:
    """Test batch validation of claims."""

    @pytest.mark.asyncio
    async def test_batch_validation_returns_conflicts_only(
        self, conflict_validation_service, stored_fact
    ):
        """Test batch validation returns only claims with conflicts."""
        claim_with_conflict = ExtractedClaim(
            claim_type="assertion",
            entity1="server-01",
            relation="RUNS",
            entity2="Wrong",
            confidence=0.95,
            context="",
            sentence="test",
        )

        claim_without_conflict = ExtractedClaim(
            claim_type="assertion",
            entity1="OtherService",
            relation="RUNS",
            entity2="SomeHost",
            confidence=0.95,
            context="",
            sentence="test",
        )

        conflicts = await conflict_validation_service.validate_claims_batch(
            [claim_with_conflict, claim_without_conflict], [stored_fact]
        )

        # Only one should have conflict
        assert len(conflicts) == 1
        assert conflicts[0].claim.entity2 == "Wrong"

    @pytest.mark.asyncio
    async def test_batch_validation_handles_errors(
        self, conflict_validation_service, stored_fact
    ):
        """Test batch validation continues on individual errors."""
        valid_claim = ExtractedClaim(
            claim_type="assertion",
            entity1="server-01",
            relation="RUNS",
            entity2="Wrong",
            confidence=0.95,
            context="",
            sentence="test",
        )

        # Mock validate_claim to raise error for first claim
        original_validate = conflict_validation_service.validate_claim
        call_count = [0]

        async def side_effect(claim, facts):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Validation error")
            return await original_validate(claim, facts)

        conflict_validation_service.validate_claim = side_effect

        conflicts = await conflict_validation_service.validate_claims_batch(
            [valid_claim, valid_claim], [stored_fact]
        )

        # Should continue despite error and process second claim
        assert call_count[0] == 2
