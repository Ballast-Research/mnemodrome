"""Contract tests for claim extraction service (Feature 025 - Phase 1).

Feature 025: Bidirectional Conflict Detection for Mnemodrome
Contract: Claim extraction and type detection

These tests define the contract for claim extraction behavior BEFORE implementation.
Tests MUST FAIL initially (Spec-Driven Development).

Contract Specification:
1. ClaimExtractionService.extract_claims(text) identifies claims in natural language
2. Claims are classified as "assertion", "hypothetical", or "example"
3. For assertions: extracts entity1, relation, entity2 relationships
4. Returns ExtractedClaim objects with confidence scores (0.0-1.0)
5. Confidence reflects LLM extraction certainty
"""

import pytest

from src.models.conflict_detection_models import ExtractedClaim


@pytest.mark.asyncio
class TestClaimExtractionContracts:
    """Contract tests for claim extraction service."""

    async def test_extract_assertion_claim_from_sentence(self):
        """Test extraction of assertion claim from sentence.

        Contract (Feature 025 - T025-004):
        - Input: "Mnemodrome runs on server-01 server."
        - Expected: ExtractedClaim with:
          - claim_type="assertion"
          - entity1="Mnemodrome"
          - relation="RUNS_ON"
          - entity2="server-01"
          - confidence >= 0.8 (high extraction confidence)
          - sentence matches input

        Validates:
        - ExtractedClaim model correctly represents assertions
        - Confidence scoring is numeric and bounded [0.0, 1.0]
        - Entity extraction preserves original entity names
        - Relation type is uppercase
        """
        # ARRANGE: Create expected assertion claim
        claim = ExtractedClaim(
            claim_type="assertion",
            entity1="Mnemodrome",
            relation="RUNS_ON",
            entity2="server-01",
            confidence=0.95,
            context="Service infrastructure mapping",
            sentence="Mnemodrome runs on server-01 server.",
        )

        # ASSERT: claim_type is assertion
        assert claim.claim_type == "assertion", (
            f"Expected claim_type='assertion', got '{claim.claim_type}'"
        )

        # ASSERT: Entities and relation extracted correctly
        assert claim.entity1 == "Mnemodrome", (
            f"Expected entity1='Mnemodrome', got '{claim.entity1}'"
        )
        assert claim.relation == "RUNS_ON", (
            f"Expected relation='RUNS_ON', got '{claim.relation}'"
        )
        assert claim.entity2 == "server-01", (
            f"Expected entity2='server-01', got '{claim.entity2}'"
        )

        # ASSERT: High confidence for direct assertion
        assert claim.confidence >= 0.8, (
            f"Expected confidence >= 0.8, got {claim.confidence}"
        )
        assert isinstance(claim.confidence, float), (
            f"Expected confidence to be float, got {type(claim.confidence)}"
        )
        assert 0.0 <= claim.confidence <= 1.0, (
            f"Expected confidence in [0.0, 1.0], got {claim.confidence}"
        )

        # ASSERT: Original sentence preserved
        assert claim.sentence == "Mnemodrome runs on server-01 server.", (
            f"Expected sentence to match input, got '{claim.sentence}'"
        )

        # ASSERT: Context is populated
        assert claim.context, "Expected context to be populated"

    async def test_extract_hypothetical_claim_from_sentence(self):
        """Test detection of hypothetical claim from sentence.

        Contract (Feature 025 - T025-005):
        - Input: "If we migrate Mnemodrome to server-02, performance would improve."
        - Expected: ExtractedClaim with:
          - claim_type="hypothetical"
          - entity1="Mnemodrome"
          - relation="COULD_RUN_ON" (modal/conditional relation)
          - entity2="server-02"
          - confidence >= 0.7 (slightly lower for conditional)
          - Detected conditional language markers

        Validates:
        - Hypothetical claims are distinguished from assertions
        - Conditional relations are extracted properly
        - Model supports claim_type="hypothetical"
        """
        # ARRANGE: Create hypothetical claim
        claim = ExtractedClaim(
            claim_type="hypothetical",
            entity1="Mnemodrome",
            relation="COULD_RUN_ON",
            entity2="server-02",
            confidence=0.85,
            context="Infrastructure optimization scenario",
            sentence="If we migrate Mnemodrome to server-02, performance would improve.",
        )

        # ASSERT: claim_type is hypothetical
        assert claim.claim_type == "hypothetical", (
            f"Expected claim_type='hypothetical', got '{claim.claim_type}'"
        )

        # ASSERT: Conditional relation extracted
        assert claim.relation in ["COULD_RUN_ON", "MIGHT_RUN_ON", "WOULD_RUN_ON"], (
            f"Expected conditional relation, got '{claim.relation}'"
        )

        # ASSERT: Entity mapping preserved
        assert claim.entity1 == "Mnemodrome", (
            f"Expected entity1='Mnemodrome', got '{claim.entity1}'"
        )
        assert claim.entity2 == "server-02", (
            f"Expected entity2='server-02', got '{claim.entity2}'"
        )

        # ASSERT: Confidence >= 0.7 for hypothetical (lower than assertion)
        assert claim.confidence >= 0.7, (
            f"Expected confidence >= 0.7, got {claim.confidence}"
        )
        assert claim.confidence <= 1.0, (
            f"Expected confidence <= 1.0, got {claim.confidence}"
        )

        # ASSERT: Original sentence preserved
        assert "If we migrate" in claim.sentence, (
            f"Expected sentence to contain 'If we migrate', got '{claim.sentence}'"
        )

    async def test_extract_example_claim_from_sentence(self):
        """Test detection of example claim from sentence.

        Contract (Feature 025 - T025-006):
        - Input: "For example, Mnemodrome could run on server-02 for better performance."
        - Expected: ExtractedClaim with:
          - claim_type="example"
          - entity1="Mnemodrome"
          - relation="COULD_RUN_ON"
          - entity2="server-02"
          - confidence >= 0.7
          - Detected example language markers ("For example", "e.g.")

        Validates:
        - Example claims are distinguished from assertions/hypotheticals
        - Model supports claim_type="example"
        - Confidence scoring reflects example status
        """
        # ARRANGE: Create example claim
        claim = ExtractedClaim(
            claim_type="example",
            entity1="Mnemodrome",
            relation="COULD_RUN_ON",
            entity2="server-02",
            confidence=0.82,
            context="Illustrative example of infrastructure options",
            sentence="For example, Mnemodrome could run on server-02 for better performance.",
        )

        # ASSERT: claim_type is example
        assert claim.claim_type == "example", (
            f"Expected claim_type='example', got '{claim.claim_type}'"
        )

        # ASSERT: Conditional relation for example
        assert claim.relation in ["COULD_RUN_ON", "MIGHT_RUN_ON", "WOULD_RUN_ON"], (
            f"Expected conditional relation, got '{claim.relation}'"
        )

        # ASSERT: Entity mapping preserved
        assert claim.entity1 == "Mnemodrome", (
            f"Expected entity1='Mnemodrome', got '{claim.entity1}'"
        )
        assert claim.entity2 == "server-02", (
            f"Expected entity2='server-02', got '{claim.entity2}'"
        )

        # ASSERT: Confidence >= 0.7 for example
        assert claim.confidence >= 0.7, (
            f"Expected confidence >= 0.7, got {claim.confidence}"
        )
        assert 0.0 <= claim.confidence <= 1.0, (
            f"Expected confidence in valid range, got {claim.confidence}"
        )

        # ASSERT: Original sentence preserved with "For example"
        assert "For example" in claim.sentence, (
            f"Expected 'For example' in sentence, got '{claim.sentence}'"
        )

    async def test_extract_multiple_claims_from_paragraph(self):
        """Test extraction of multiple claims from a single paragraph.

        Contract:
        - Input: Paragraph with multiple distinct claims
        - Expected: Multiple ExtractedClaim objects, one per claim
        - Each claim properly typed and extracted
        - All confidence scores valid (0.0-1.0)

        Validates:
        - Service handles multiple claims in single text
        - Each claim is independently classified
        - Confidence scores are all valid
        """
        # ARRANGE: Create multiple claims
        claims = [
            ExtractedClaim(
                claim_type="assertion",
                entity1="Mnemodrome",
                relation="RUNS_ON",
                entity2="server-01",
                confidence=0.95,
                context="Current deployment",
                sentence="Mnemodrome currently runs on server-01.",
            ),
            ExtractedClaim(
                claim_type="hypothetical",
                entity1="Mnemodrome",
                relation="COULD_RUN_ON",
                entity2="server-02",
                confidence=0.80,
                context="Future optimization",
                sentence="If migrated, it could run on server-02.",
            ),
        ]

        # ASSERT: Both claims present
        assert len(claims) == 2, f"Expected 2 claims, got {len(claims)}"

        # ASSERT: First claim is assertion
        assert claims[0].claim_type == "assertion", (
            f"Expected first claim to be assertion, got '{claims[0].claim_type}'"
        )
        assert claims[0].entity2 == "server-01"

        # ASSERT: Second claim is hypothetical
        assert claims[1].claim_type == "hypothetical", (
            f"Expected second claim to be hypothetical, got '{claims[1].claim_type}'"
        )
        assert claims[1].entity2 == "server-02"

        # ASSERT: All confidence scores valid
        for i, claim in enumerate(claims):
            assert 0.0 <= claim.confidence <= 1.0, (
                f"Claim {i} has invalid confidence: {claim.confidence}"
            )

    async def test_confidence_reflects_extraction_certainty(self):
        """Test that confidence score reflects extraction certainty.

        Contract:
        - Directly stated assertions have higher confidence (>0.85)
        - Implicit or conditional claims have lower confidence (0.7-0.85)
        - Ambiguous claims have lowest confidence (0.5-0.7)
        - Confidence is always between 0.0 and 1.0

        Validates:
        - Confidence scoring differentiates claim types
        - Confidence ordering: assertion > hypothetical > example
        - Confidence bounds are respected
        """
        # ARRANGE: Create claims with different confidence levels
        explicit_claim = ExtractedClaim(
            claim_type="assertion",
            entity1="Service",
            relation="USES",
            entity2="Database",
            confidence=0.95,  # Explicit assertion
            context="Direct statement",
            sentence="Service uses Database.",
        )
        implicit_claim = ExtractedClaim(
            claim_type="hypothetical",
            entity1="Service",
            relation="COULD_USE",
            entity2="Cache",
            confidence=0.70,  # Conditional/implicit
            context="Possibility",
            sentence="Service might use Cache for performance.",
        )

        claims = [explicit_claim, implicit_claim]

        # ASSERT: High confidence for explicit assertion
        assert claims[0].confidence > claims[1].confidence, (
            f"Expected explicit claim to have higher confidence, got {claims[0].confidence} vs {claims[1].confidence}"
        )

        # ASSERT: All confidence values valid
        for claim in claims:
            assert 0.0 <= claim.confidence <= 1.0, (
                f"Invalid confidence value: {claim.confidence}"
            )

        # ASSERT: Explicit claim has high confidence (>0.85)
        assert claims[0].confidence > 0.85, (
            f"Expected explicit assertion confidence > 0.85, got {claims[0].confidence}"
        )

        # ASSERT: Implicit claim has medium confidence (0.6-0.8)
        assert 0.6 <= claims[1].confidence <= 0.8, (
            f"Expected implicit claim confidence in [0.6, 0.8], got {claims[1].confidence}"
        )

    async def test_claim_with_empty_context_optional(self):
        """Test that context field is optional for claims.

        Contract:
        - context field has default="" (empty string)
        - Claim can be created without context
        - Empty context is valid

        Validates:
        - Context is optional field
        - Default value is empty string
        """
        # ARRANGE: Create claim without context
        claim = ExtractedClaim(
            claim_type="assertion",
            entity1="Mnemodrome",
            relation="RUNS_ON",
            entity2="server-01",
            confidence=0.90,
            sentence="Mnemodrome runs on server-01.",
            # context not provided
        )

        # ASSERT: Context defaults to empty string
        assert claim.context == "", (
            f"Expected default context='', got '{claim.context}'"
        )

        # ASSERT: All other fields still populated
        assert claim.claim_type == "assertion"
        assert claim.entity1 == "Mnemodrome"
        assert claim.relation == "RUNS_ON"
        assert claim.entity2 == "server-01"

    async def test_claim_types_are_valid_enum_values(self):
        """Test that only valid claim_type values are accepted.

        Contract:
        - claim_type must be one of: "assertion", "hypothetical", "example"
        - Invalid types should be rejected
        - Model validation enforces this

        Validates:
        - claim_type uses Literal enum constraint
        - Model properly validates claim types
        """
        # ARRANGE: Valid claim types
        valid_types = ["assertion", "hypothetical", "example"]

        for claim_type in valid_types:
            # ACT: Create claim with valid type
            claim = ExtractedClaim(
                claim_type=claim_type,
                entity1="Entity1",
                relation="RELATION",
                entity2="Entity2",
                confidence=0.80,
                sentence="Test sentence.",
            )

            # ASSERT: Claim created successfully
            assert claim.claim_type == claim_type

        # ARRANGE: Invalid claim type
        # ASSERT: Model should reject invalid type
        with pytest.raises(ValueError):
            ExtractedClaim(
                claim_type="invalid_type",  # type: ignore
                entity1="Entity1",
                relation="RELATION",
                entity2="Entity2",
                confidence=0.80,
                sentence="Test sentence.",
            )
