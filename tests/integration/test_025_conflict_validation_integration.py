"""
Integration tests for Feature 025: Bidirectional Conflict Detection

Tests end-to-end conflict detection and validation:
- Claim extraction from various prompt types
- Conflict validation against stored facts
- Severity scoring with edge cases
- Auto-correction of minor conflicts
- Warning generation for major conflicts
- Timeout handling and graceful degradation

All test entities tagged with _test_025_ for cleanup.
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from src.models.conflict_detection_models import (
    ConflictResult,
    ExtractedClaim,
    StoredFact,
)
from src.services.auto_correction_service import AutoCorrectionService
from src.services.claim_extraction_service import ClaimExtractionService
from src.services.conflict_validation_service import ConflictValidationService


@pytest.mark.asyncio
class TestClaimExtractionIntegration:
    """Test claim extraction from various prompt types."""

    async def test_extract_claims_from_assertion(self):
        """Extract claims from assertive statement."""
        mock_llm = AsyncMock()
        mock_llm.provider = "anthropic"
        mock_llm._call_llm = AsyncMock(
            return_value="""
[
  {
    "claim_type": "assertion",
    "entity1": "Mnemodrome",
    "relation": "RUNS_ON",
    "entity2": "server-01",
    "confidence": 0.95,
    "sentence": "Mnemodrome runs on server-01 server.",
    "context": "Production deployment"
  }
]
"""
        )

        service = ClaimExtractionService(llm_service=mock_llm)
        claims = await service.extract_claims(
            "Mnemodrome runs on server-01 server for production workloads."
        )

        assert len(claims) == 1
        assert claims[0].claim_type == "assertion"
        assert claims[0].entity1 == "Mnemodrome"
        assert claims[0].entity2 == "server-01"
        assert claims[0].confidence == 0.95

    async def test_extract_claims_filters_hypothetical(self):
        """Test that hypothetical claims are extracted but can be filtered."""
        mock_llm = AsyncMock()
        mock_llm.provider = "anthropic"
        mock_llm._call_llm = AsyncMock(
            return_value="""
[
  {
    "claim_type": "assertion",
    "entity1": "Mnemodrome",
    "relation": "RUNS_ON",
    "entity2": "server-01",
    "confidence": 0.95,
    "sentence": "Mnemodrome runs on server-01.",
    "context": "Current"
  },
  {
    "claim_type": "hypothetical",
    "entity1": "Mnemodrome",
    "relation": "COULD_RUN_ON",
    "entity2": "server-02",
    "confidence": 0.80,
    "sentence": "If we migrate, Mnemodrome could run on server-02.",
    "context": "Future planning"
  }
]
"""
        )

        service = ClaimExtractionService(llm_service=mock_llm)
        all_claims = await service.extract_claims(
            "Mnemodrome runs on server-01. If we migrate, it could run on server-02."
        )

        assert len(all_claims) == 2

        # Filter assertions only
        assertions = await service.filter_assertions_only(all_claims)
        assert len(assertions) == 1
        assert assertions[0].claim_type == "assertion"

    async def test_extract_claims_empty_text_raises_error(self):
        """Test that empty text raises ValueError."""
        mock_llm = AsyncMock()
        mock_llm.provider = "anthropic"

        service = ClaimExtractionService(llm_service=mock_llm)

        with pytest.raises(ValueError, match="Input text cannot be empty"):
            await service.extract_claims("")

    async def test_extract_multiple_claims_from_text(self):
        """Extract multiple claims from single text."""
        mock_llm = AsyncMock()
        mock_llm.provider = "anthropic"
        mock_llm._call_llm = AsyncMock(
            return_value="""
[
  {
    "claim_type": "assertion",
    "entity1": "Mnemodrome",
    "relation": "RUNS_ON",
    "entity2": "server-01",
    "confidence": 0.95,
    "sentence": "Mnemodrome runs on server-01.",
    "context": "Deployment"
  },
  {
    "claim_type": "assertion",
    "entity1": "Mnemodrome",
    "relation": "USES",
    "entity2": "Neo4j",
    "confidence": 0.92,
    "sentence": "Mnemodrome uses Neo4j for storage.",
    "context": "Architecture"
  }
]
"""
        )

        service = ClaimExtractionService(llm_service=mock_llm)
        claims = await service.extract_claims(
            "Mnemodrome runs on server-01 and uses Neo4j for persistent storage."
        )

        assert len(claims) == 2
        assert claims[0].relation == "RUNS_ON"
        assert claims[1].relation == "USES"


@pytest.mark.asyncio
class TestConflictValidationIntegration:
    """Test conflict detection and validation against stored facts."""

    async def test_detect_conflict_entity2_mismatch(self):
        """Test detection of entity2 mismatch conflict."""
        # Create claim and stored fact with different entity2
        claim = ExtractedClaim(
            claim_type="assertion",
            entity1="Mnemodrome",
            relation="RUNS_ON",
            entity2="server-02",
            confidence=0.95,
            sentence="Mnemodrome runs on server-02.",
            context="Infrastructure update",
        )

        stored_fact = StoredFact(
            entity1_uuid="uuid-mnemodrome",
            entity1_name="Mnemodrome",
            relation="RUNS_ON",
            entity2_uuid="uuid-server-01",
            entity2_name="server-01",
            created_at=datetime.now() - timedelta(hours=12),
            context="Production deployment",
            fact_uuid="fact-uuid-1",
            is_temporal=False,
        )

        service = ConflictValidationService()
        result = await service.validate_claim(claim, [stored_fact])

        # Should detect conflict
        assert result is not None
        assert result.conflict_detected is True
        assert result.conflict_type in ["exact_contradiction", "temporal_conflict"]
        assert result.severity >= 70  # Major conflict

    async def test_no_conflict_matching_entities(self):
        """Test that matching claims don't trigger conflicts."""
        claim = ExtractedClaim(
            claim_type="assertion",
            entity1="Mnemodrome",
            relation="RUNS_ON",
            entity2="server-01",
            confidence=0.95,
            sentence="Mnemodrome runs on server-01.",
            context="Current deployment",
        )

        stored_fact = StoredFact(
            entity1_uuid="uuid-mnemodrome",
            entity1_name="Mnemodrome",
            relation="RUNS_ON",
            entity2_uuid="uuid-server-01",
            entity2_name="server-01",
            created_at=datetime.now(),
            context="Confirmed deployment",
            fact_uuid="fact-uuid-1",
            is_temporal=False,
        )

        service = ConflictValidationService()
        result = await service.validate_claim(claim, [stored_fact])

        # No conflict for matching facts
        assert result is None

    async def test_hypothetical_claims_skip_validation(self):
        """Test that hypothetical claims are not validated."""
        claim = ExtractedClaim(
            claim_type="hypothetical",
            entity1="Mnemodrome",
            relation="COULD_RUN_ON",
            entity2="server-02",
            confidence=0.80,
            sentence="Mnemodrome could run on server-02.",
            context="Future planning",
        )

        stored_fact = StoredFact(
            entity1_uuid="uuid-mnemodrome",
            entity1_name="Mnemodrome",
            relation="RUNS_ON",
            entity2_uuid="uuid-server-01",
            entity2_name="server-01",
            created_at=datetime.now(),
            context="Current",
            fact_uuid="fact-uuid-1",
            is_temporal=False,
        )

        service = ConflictValidationService()
        result = await service.validate_claim(claim, [stored_fact])

        # Hypothetical claims should be skipped
        assert result is None


@pytest.mark.asyncio
class TestSeverityScoringIntegration:
    """Test severity calculation with various edge cases."""

    async def test_severity_base_contradiction(self):
        """Test base contradiction score (30 points)."""
        claim = ExtractedClaim(
            claim_type="assertion",
            entity1="Service",
            relation="RUNS_ON",
            entity2="Server_A",
            confidence=0.50,  # Low confidence
            sentence="Service runs on Server A.",
            context="Test",
        )

        stored_fact = StoredFact(
            entity1_uuid="uuid-svc",
            entity1_name="Service",
            relation="RUNS_ON",
            entity2_uuid="uuid-srvb",
            entity2_name="Server_B",
            created_at=datetime.now() - timedelta(days=5),  # Recent (no age penalty)
            context="Test",
            fact_uuid="fact-uuid-1",
            is_temporal=False,
        )

        service = ConflictValidationService()
        result = await service.validate_claim(claim, [stored_fact])

        # Base contradiction: 30 points
        # Entity2 mismatch: +30 = 60 points
        # High confidence (<0.9): no bonus
        # Recent fact: no age penalty
        # Expected: ~60 points
        assert result is not None
        assert result.severity >= 50

    async def test_severity_high_confidence_bonus(self):
        """Test high confidence bonus (+10 points)."""
        claim = ExtractedClaim(
            claim_type="assertion",
            entity1="Service",
            relation="RUNS_ON",
            entity2="Server_A",
            confidence=0.95,  # High confidence
            sentence="Service runs on Server A.",
            context="Test",
        )

        stored_fact = StoredFact(
            entity1_uuid="uuid-svc",
            entity1_name="Service",
            relation="RUNS_ON",
            entity2_uuid="uuid-srvb",
            entity2_name="Server_B",
            created_at=datetime.now() - timedelta(days=5),
            context="Test",
            fact_uuid="fact-uuid-1",
            is_temporal=False,
        )

        service = ConflictValidationService()
        result = await service.validate_claim(claim, [stored_fact])

        # Base (30) + Entity2 mismatch (30) + High confidence (10) = 70
        assert result is not None
        assert result.severity >= 70

    async def test_severity_recent_fact_penalty(self):
        """Test recent fact penalty (-5 points)."""
        claim = ExtractedClaim(
            claim_type="assertion",
            entity1="Service",
            relation="RUNS_ON",
            entity2="Server_A",
            confidence=0.95,  # High confidence
            sentence="Service runs on Server A.",
            context="Test",
        )

        stored_fact = StoredFact(
            entity1_uuid="uuid-svc",
            entity1_name="Service",
            relation="RUNS_ON",
            entity2_uuid="uuid-srvb",
            entity2_name="Server_B",
            created_at=datetime.now() - timedelta(hours=2),  # Very recent
            context="Test",
            fact_uuid="fact-uuid-1",
            is_temporal=False,
        )

        service = ConflictValidationService()
        result = await service.validate_claim(claim, [stored_fact])

        # Base (30) + Entity2 (30) + Confidence (10) - Recent (-5) = 65
        assert result is not None
        assert result.severity >= 60 and result.severity <= 75

    async def test_severity_critical_entity_bonus(self):
        """Test critical entity severity boost (+10 points)."""
        claim = ExtractedClaim(
            claim_type="assertion",
            entity1="Mnemodrome",  # Critical entity
            relation="RUNS_ON",
            entity2="Server_A",
            confidence=0.95,
            sentence="Mnemodrome runs on Server A.",
            context="Test",
        )

        stored_fact = StoredFact(
            entity1_uuid="uuid-mnemodrome",
            entity1_name="Mnemodrome",
            relation="RUNS_ON",
            entity2_uuid="uuid-server-01",
            entity2_name="server-01",
            created_at=datetime.now() - timedelta(days=5),
            context="Test",
            fact_uuid="fact-uuid-1",
            is_temporal=False,
        )

        service = ConflictValidationService()
        result = await service.validate_claim(claim, [stored_fact])

        # Base (30) + Entity2 (30) + Confidence (10) + Critical (+10) = 80
        assert result is not None
        assert result.severity >= 75

    async def test_severity_clamped_0_to_100(self):
        """Test that severity is clamped to 0-100 range."""
        claim = ExtractedClaim(
            claim_type="assertion",
            entity1="Mnemodrome",
            relation="RUNS_ON",
            entity2="server-02",
            confidence=0.95,
            sentence="Mnemodrome runs on server-02.",
            context="Test",
        )

        stored_fact = StoredFact(
            entity1_uuid="uuid-mnemodrome",
            entity1_name="Mnemodrome",
            relation="RUNS_ON",
            entity2_uuid="uuid-server-01",
            entity2_name="server-01",
            created_at=datetime.now() - timedelta(days=90),  # Old fact
            context="Test",
            fact_uuid="fact-uuid-1",
            is_temporal=True,
        )

        service = ConflictValidationService()
        result = await service.validate_claim(claim, [stored_fact])

        # Score could exceed 100 with all factors, should be clamped
        assert result is not None
        assert 0 <= result.severity <= 100


@pytest.mark.asyncio
class TestAutoCorrectionIntegration:
    """Test auto-correction of conflicts based on severity."""

    async def test_auto_correct_minor_conflict(self):
        """Test auto-correction of minor conflicts (severity < 70)."""
        conflicts = [
            ConflictResult(
                conflict_detected=True,
                conflict_type="exact_contradiction",
                severity=50,  # Minor conflict
                recommendation="auto_correct",
                correction_suggestion="Service runs on Server_B",
                stored_fact=StoredFact(
                    entity1_uuid="uuid-svc",
                    entity1_name="Service",
                    relation="RUNS_ON",
                    entity2_uuid="uuid-srvb",
                    entity2_name="Server_B",
                    created_at=datetime.now(),
                    fact_uuid="fact-uuid-1",
                ),
                claim=ExtractedClaim(
                    claim_type="assertion",
                    entity1="Service",
                    relation="RUNS_ON",
                    entity2="Server_A",
                    confidence=0.95,
                    sentence="Service runs on Server A.",
                ),
            )
        ]

        service = AutoCorrectionService()
        result = await service.correct_text("Service runs on Server A", conflicts)

        # Should apply auto-correction
        assert result.action in ["corrected", "warned"]
        assert len(result.corrections_applied) > 0

    async def test_warn_major_conflict(self):
        """Test warning generation for major conflicts (severity >= 70)."""
        conflicts = [
            ConflictResult(
                conflict_detected=True,
                conflict_type="exact_contradiction",
                severity=85,  # Major conflict
                recommendation="warn",
                correction_suggestion="Mnemodrome runs on server-01",
                stored_fact=StoredFact(
                    entity1_uuid="uuid-mnemodrome",
                    entity1_name="Mnemodrome",
                    relation="RUNS_ON",
                    entity2_uuid="uuid-server-01",
                    entity2_name="server-01",
                    created_at=datetime.now(),
                    fact_uuid="fact-uuid-1",
                ),
                claim=ExtractedClaim(
                    claim_type="assertion",
                    entity1="Mnemodrome",
                    relation="RUNS_ON",
                    entity2="server-02",
                    confidence=0.95,
                    sentence="Mnemodrome runs on server-02.",
                ),
            )
        ]

        service = AutoCorrectionService()
        result = await service.correct_text("Mnemodrome runs on server-02", conflicts)

        # Should generate warning, not auto-correct
        assert result.action in ["warned"]
        assert result.warning_message is not None
        assert "WARNING" in result.warning_message

    async def test_skip_very_low_severity(self):
        """Test that very low severity conflicts are skipped."""
        conflicts = [
            ConflictResult(
                conflict_detected=True,
                conflict_type="partial_mismatch",
                severity=15,  # Below minor threshold
                recommendation="skip",
                correction_suggestion=None,
                stored_fact=StoredFact(
                    entity1_uuid="uuid-svc",
                    entity1_name="Service",
                    relation="USES",
                    entity2_uuid="uuid-lib",
                    entity2_name="Library_B",
                    created_at=datetime.now(),
                    fact_uuid="fact-uuid-1",
                ),
                claim=ExtractedClaim(
                    claim_type="assertion",
                    entity1="Service",
                    relation="USES",
                    entity2="Library_A",
                    confidence=0.50,
                    sentence="Service uses Library A.",
                ),
            )
        ]

        service = AutoCorrectionService()
        result = await service.correct_text("Service uses Library A", conflicts)

        # Should skip very low severity
        assert result.action == "skipped"
        assert len(result.corrections_applied) == 0

    async def test_empty_conflicts_skips_correction(self):
        """Test that empty conflict list returns original text."""
        service = AutoCorrectionService()
        result = await service.correct_text("Original text", [])

        assert result.action == "skipped"
        assert result.modified_text == "Original text"
        assert len(result.corrections_applied) == 0


@pytest.mark.asyncio
class TestTimeoutHandling:
    """Test timeout handling and graceful degradation."""

    async def test_claim_extraction_timeout_graceful(self):
        """Test graceful degradation when LLM times out."""
        mock_llm = AsyncMock()
        mock_llm.provider = "anthropic"
        mock_llm._call_llm = AsyncMock(side_effect=TimeoutError("LLM timeout"))

        service = ClaimExtractionService(llm_service=mock_llm)

        # Should raise exception (caller handles timeout)
        with pytest.raises(Exception):
            await service.extract_claims("Some text")

    async def test_validation_completes_within_timeout(self):
        """Test that validation completes within 1000ms target."""
        claim = ExtractedClaim(
            claim_type="assertion",
            entity1="Mnemodrome",
            relation="RUNS_ON",
            entity2="server-02",
            confidence=0.95,
            sentence="Mnemodrome runs on server-02.",
            context="Test",
        )

        stored_facts = [
            StoredFact(
                entity1_uuid="uuid-mnemodrome",
                entity1_name="Mnemodrome",
                relation="RUNS_ON",
                entity2_uuid=f"uuid-inf01-{i}",
                entity2_name="server-01",
                created_at=datetime.now() - timedelta(hours=i),
                context="Test",
                fact_uuid=f"fact-uuid-{i}",
                is_temporal=False,
            )
            for i in range(5)  # Multiple facts to validate
        ]

        service = ConflictValidationService()

        start = datetime.now()
        result = await service.validate_claim(claim, stored_facts)
        elapsed_ms = (datetime.now() - start).total_seconds() * 1000

        # Should complete in reasonable time
        assert elapsed_ms < 100  # Much faster than 1000ms target


@pytest.mark.asyncio
class TestEndToEndValidationFlow:
    """End-to-end validation flow tests."""

    async def test_full_validation_pipeline_no_conflict(self):
        """Complete pipeline: extract → validate → correct with no conflicts."""
        mock_llm = AsyncMock()
        mock_llm.provider = "anthropic"
        mock_llm._call_llm = AsyncMock(
            return_value="""
[
  {
    "claim_type": "assertion",
    "entity1": "Mnemodrome",
    "relation": "RUNS_ON",
    "entity2": "server-01",
    "confidence": 0.95,
    "sentence": "Mnemodrome runs on server-01.",
    "context": "Current"
  }
]
"""
        )

        # Step 1: Extract claims
        extraction_service = ClaimExtractionService(llm_service=mock_llm)
        claims = await extraction_service.extract_claims(
            "Mnemodrome runs on server-01."
        )
        assertions = await extraction_service.filter_assertions_only(claims)
        assert len(assertions) == 1

        # Step 2: Get stored facts (no conflict)
        stored_facts = [
            StoredFact(
                entity1_uuid="uuid-mnemodrome",
                entity1_name="Mnemodrome",
                relation="RUNS_ON",
                entity2_uuid="uuid-server-01",
                entity2_name="server-01",  # Same entity2
                created_at=datetime.now(),
                fact_uuid="fact-uuid-1",
            )
        ]

        # Step 3: Validate claims
        validation_service = ConflictValidationService()
        conflicts = await validation_service.validate_claims_batch(
            assertions, stored_facts
        )
        assert len(conflicts) == 0

        # Step 4: Correct (no corrections needed)
        correction_service = AutoCorrectionService()
        result = await correction_service.correct_text(
            "Mnemodrome runs on server-01.", conflicts
        )
        assert result.action == "skipped"

    async def test_full_validation_pipeline_with_major_conflict(self):
        """Complete pipeline: extract → validate → warn on major conflict."""
        mock_llm = AsyncMock()
        mock_llm.provider = "anthropic"
        mock_llm._call_llm = AsyncMock(
            return_value="""
[
  {
    "claim_type": "assertion",
    "entity1": "Mnemodrome",
    "relation": "RUNS_ON",
    "entity2": "server-02",
    "confidence": 0.95,
    "sentence": "Mnemodrome runs on server-02.",
    "context": "Update"
  }
]
"""
        )

        # Step 1: Extract claims
        extraction_service = ClaimExtractionService(llm_service=mock_llm)
        claims = await extraction_service.extract_claims(
            "Mnemodrome runs on server-02."
        )
        assertions = await extraction_service.filter_assertions_only(claims)

        # Step 2: Get conflicting stored facts
        stored_facts = [
            StoredFact(
                entity1_uuid="uuid-mnemodrome",
                entity1_name="Mnemodrome",
                relation="RUNS_ON",
                entity2_uuid="uuid-server-01",
                entity2_name="server-01",  # Different from claim
                created_at=datetime.now() - timedelta(hours=2),
                fact_uuid="fact-uuid-1",
            )
        ]

        # Step 3: Validate claims
        validation_service = ConflictValidationService()
        conflicts = await validation_service.validate_claims_batch(
            assertions, stored_facts
        )
        assert len(conflicts) > 0
        assert conflicts[0].severity >= 70

        # Step 4: Correct (warn on major conflict)
        correction_service = AutoCorrectionService()
        result = await correction_service.correct_text(
            "Mnemodrome runs on server-02.", conflicts
        )
        assert result.action == "warned"
        assert result.warning_message is not None


# Cleanup fixture for all test data
@pytest.fixture(autouse=True)
async def cleanup_test_025_data():
    """Clean up all _test_025_ tagged data after tests."""
    yield
    # Cleanup would go here if using real Neo4j
    pass
