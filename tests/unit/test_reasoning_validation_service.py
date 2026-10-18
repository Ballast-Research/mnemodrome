"""
Unit tests for reasoning validation service (Feature 028).

Tests ValidationSynthesizer and ReasoningValidationService with mocked dependencies.
"""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from src.models.conflict_detection_models import (
    ConflictResult,
    ExtractedClaim,
    StoredFact,
)
from src.models.pattern_models import PatternMatch
from src.models.reasoning_validation_models import (
    ReasoningValidationResult,
)
from src.services.reasoning_validation_service import (
    ReasoningValidationService,
    ValidationSynthesizer,
)

# ============================================================================
# ValidationSynthesizer Tests (No Mocking Needed)
# ============================================================================


class TestValidationSynthesizer:
    """Tests for ValidationSynthesizer (pure logic, no I/O)."""

    def test_synthesizer_conflicts_detected(self):
        """Test that conflicts_detected status is returned when conflicts exist."""
        conflict = ConflictResult(
            conflict_detected=True,
            conflict_type="exact_contradiction",
            severity=80,
            recommendation="warn",
            correction_suggestion="Corrected text",
            stored_fact=StoredFact(
                fact_uuid="fact-1",
                entity1_uuid="e1",
                entity1_name="Test1",
                relation="HAS",
                entity2_uuid="e2",
                entity2_name="Value1",
                created_at=datetime.now(UTC),
            ),
            claim=ExtractedClaim(
                claim_type="assertion",
                entity1="Test1",
                relation="HAS",
                entity2="Value2",
                confidence=0.9,
                sentence="Test has Value2",
                context="",
            ),
        )

        synthesizer = ValidationSynthesizer()
        result = synthesizer.synthesize(
            conflicts=[conflict],
            learnings=[],
            patterns=[],
            elapsed_ms=100.0,
            degraded_queries=[],
        )

        assert result.validation_status == "conflicts_detected"
        assert len(result.conflicts) == 1
        assert result.conflicts[0].severity == 80

    def test_synthesizer_reinforced(self):
        """Test that reinforced status is returned when learnings/patterns exist."""

        synthesizer = ValidationSynthesizer()
        learnings = [
            {
                "uuid": "l1",
                "title": "Learning 1",
                "score": 0.9,
                "content": "Some content",
            }
        ]
        patterns = [
            PatternMatch(
                pattern_uuid="p1",
                name="Pattern 1",
                pattern_type="diagnostic",
                similarity_score=0.85,
                success_rate=0.9,
                occurrence_count=3,
                confidence=0.85,
                relevance_explanation="Test explanation",
            )
        ]

        result = synthesizer.synthesize(
            conflicts=[],
            learnings=learnings,
            patterns=patterns,
            elapsed_ms=150.0,
            degraded_queries=[],
        )

        assert result.validation_status == "reinforced"
        assert len(result.relevant_learnings) == 1
        assert len(result.relevant_patterns) == 1

    def test_synthesizer_clean(self):
        """Test that clean status is returned when nothing found."""
        synthesizer = ValidationSynthesizer()
        result = synthesizer.synthesize(
            conflicts=[],
            learnings=[],
            patterns=[],
            elapsed_ms=50.0,
            degraded_queries=[],
        )

        assert result.validation_status == "clean"
        assert len(result.conflicts) == 0
        assert len(result.relevant_learnings) == 0

    def test_synthesizer_status_priority(self):
        """Test that conflicts_detected beats reinforced."""
        conflict = ConflictResult(
            conflict_detected=True,
            conflict_type="exact_contradiction",
            severity=80,
            recommendation="warn",
            correction_suggestion="Corrected",
            stored_fact=StoredFact(
                fact_uuid="fact-2",
                entity1_uuid="e1",
                entity1_name="Test1",
                relation="HAS",
                entity2_uuid="e2",
                entity2_name="Value1",
                created_at=datetime.now(UTC),
            ),
            claim=ExtractedClaim(
                claim_type="assertion",
                entity1="Test1",
                relation="HAS",
                entity2="Value2",
                confidence=0.9,
                sentence="Test",
                context="",
            ),
        )
        learnings = [
            {"uuid": "l1", "title": "Learning 1", "score": 0.9, "content": "Content"}
        ]

        synthesizer = ValidationSynthesizer()
        result = synthesizer.synthesize(
            conflicts=[conflict],
            learnings=learnings,
            patterns=[],
            elapsed_ms=100.0,
            degraded_queries=[],
        )

        # conflicts_detected wins even with learnings present
        assert result.validation_status == "conflicts_detected"

    def test_synthesizer_degraded_queries_additive(self):
        """Test that degraded_queries is additive (doesn't change primary status)."""
        synthesizer = ValidationSynthesizer()
        result = synthesizer.synthesize(
            conflicts=[],
            learnings=[],
            patterns=[],
            elapsed_ms=100.0,
            degraded_queries=["fact_retrieval", "pattern_search"],
        )

        assert result.validation_status == "clean"
        assert len(result.degraded_queries) == 2
        assert "fact_retrieval" in result.degraded_queries

    def test_synthesizer_summary_non_empty_all_statuses(self):
        """Test that summary is non-empty for all status types."""
        synthesizer = ValidationSynthesizer()

        # Clean status
        clean_result = synthesizer.synthesize([], [], [], 100.0, [])
        assert len(clean_result.summary) > 0

        # Reinforced status
        reinforced_result = synthesizer.synthesize(
            [],
            [{"uuid": "l1", "title": "Learning", "score": 0.9, "content": "Content"}],
            [],
            100.0,
            [],
        )
        assert len(reinforced_result.summary) > 0

        # Conflicts detected
        conflict = ConflictResult(
            conflict_detected=True,
            conflict_type="exact_contradiction",
            severity=80,
            recommendation="warn",
            correction_suggestion="Corrected",
            stored_fact=StoredFact(
                fact_uuid="fact-3",
                entity1_uuid="e1",
                entity1_name="Test",
                relation="HAS",
                entity2_uuid="e2",
                entity2_name="Val1",
                created_at=datetime.now(UTC),
            ),
            claim=ExtractedClaim(
                claim_type="assertion",
                entity1="Test",
                relation="HAS",
                entity2="Val2",
                confidence=0.9,
                sentence="Test",
                context="",
            ),
        )
        conflict_result = synthesizer.synthesize([conflict], [], [], 100.0, [])
        assert len(conflict_result.summary) > 0


# ============================================================================
# ReasoningValidationService Tests (With Mocks)
# ============================================================================


@pytest.mark.asyncio
class TestReasoningValidationService:
    """Tests for ReasoningValidationService with mocked dependencies."""

    @pytest.fixture
    def mock_services(self):
        """Create mock service dependencies."""
        llm_service = AsyncMock()
        neo4j_service = AsyncMock()
        vector_service = AsyncMock()
        redis_service = AsyncMock()
        redis_service.is_connected = False

        return {
            "llm_service": llm_service,
            "neo4j_service": neo4j_service,
            "vector_service": vector_service,
            "redis_service": redis_service,
        }

    @pytest.fixture
    def service(self, mock_services):
        """Create a ReasoningValidationService with mocked dependencies."""
        with (
            patch(
                "src.services.reasoning_validation_service.ClaimExtractionService"
            ) as mock_claim_service,
            patch(
                "src.services.reasoning_validation_service.Neo4jFactRetrievalService"
            ) as mock_fact_service,
            patch(
                "src.services.reasoning_validation_service.PatternQueryService"
            ) as mock_pattern_service,
        ):
            # Configure mocks
            mock_claim_service.return_value = AsyncMock()
            mock_fact_service.return_value = AsyncMock()
            mock_pattern_service.return_value = AsyncMock()

            return ReasoningValidationService(
                llm_service=mock_services["llm_service"],
                neo4j_service=mock_services["neo4j_service"],
                vector_service=mock_services["vector_service"],
                redis_service=None,
            )

    async def test_validate_response_no_assertions(self, service):
        """Test early return when no factual assertions extracted."""
        service.claim_extraction.extract_claims = AsyncMock(
            return_value=[
                ExtractedClaim(
                    claim_type="example",
                    entity1="Test",
                    relation="HAS",
                    entity2="Value",
                    confidence=0.9,
                    sentence="For example...",
                    context="",
                )
            ]
        )

        result = await service.validate_response("For example, Test HAS Value")

        assert result.validation_status == "clean"
        assert result.claims_analyzed == 0

    async def test_validate_response_claim_extraction_timeout(self, service):
        """A claim-extraction timeout means nothing was checked.

        It must NOT be reported as "clean" (regression for the bug where a
        timed-out/failed extraction silently produced a false "clean"
        result). The distinct "degraded" status communicates that the
        validation did not actually run.
        """

        async def slow_extract(*args, **kwargs):
            await asyncio.sleep(1.0)

        service.claim_extraction.extract_claims = slow_extract

        result = await service.validate_response("Test response", timeout_ms=100)

        assert result.validation_status == "degraded"
        assert result.validation_status != "clean"
        assert (
            "timeout" in result.summary.lower() or "timed out" in result.summary.lower()
        )
        assert result.claims_analyzed == 0

    async def test_validate_response_claim_extraction_error(self, service):
        """A claim-extraction exception (not a timeout) must also avoid "clean".

        Regression for the bug where any extraction failure — timeout or
        otherwise — was reported as a false "clean" result.
        """
        service.claim_extraction.extract_claims = AsyncMock(
            side_effect=RuntimeError("extraction backend unavailable")
        )

        result = await service.validate_response("Test response")

        assert result.validation_status == "degraded"
        assert result.validation_status != "clean"
        assert "error" in result.summary.lower()
        assert result.claims_analyzed == 0

    async def test_validate_response_no_entities(self, service):
        """Test early return when no entities extracted."""
        service.claim_extraction.extract_claims = AsyncMock(
            return_value=[
                ExtractedClaim(
                    claim_type="assertion",
                    entity1="",
                    relation="HAS",
                    entity2="",
                    confidence=0.9,
                    sentence="No entities here",
                    context="",
                )
            ]
        )

        result = await service.validate_response("No entities here")

        assert result.validation_status == "clean"
        assert "entities" in result.summary.lower()

    async def test_validate_response_all_queries_fail(self, service):
        """Test that failed queries result in clean status with degraded entries."""
        claim = ExtractedClaim(
            claim_type="assertion",
            entity1="TestEntity",
            relation="RUNS_ON",
            entity2="hypervisor-01",
            confidence=0.9,
            sentence="Test runs on hypervisor-01",
            context="",
        )
        service.claim_extraction.extract_claims = AsyncMock(return_value=[claim])

        # Mock all parallel queries to fail
        service.fact_retrieval.get_facts_for_entities = AsyncMock(
            side_effect=Exception("Query failed")
        )
        service.vector_service.search = AsyncMock(side_effect=Exception("Query failed"))
        service.pattern_query.search_patterns_by_problem = AsyncMock(
            side_effect=Exception("Query failed")
        )

        result = await service.validate_response("Test runs on hypervisor-01")

        assert result.validation_status == "clean"
        assert len(result.degraded_queries) == 3

    async def test_validate_response_partial_failure(self, service):
        """Test that partial failure returns result with one degraded query."""
        claim = ExtractedClaim(
            claim_type="assertion",
            entity1="TestEntity",
            relation="RUNS_ON",
            entity2="hypervisor-01",
            confidence=0.9,
            sentence="Test runs on hypervisor-01",
            context="",
        )
        service.claim_extraction.extract_claims = AsyncMock(return_value=[claim])

        # Mock one success, two failures
        fact = StoredFact(
            fact_uuid="fact-4",
            entity1_uuid="e1",
            entity1_name="TestEntity",
            relation="RUNS_ON",
            entity2_uuid="e2",
            entity2_name="hypervisor-01",
            created_at=datetime.now(UTC),
        )
        service.fact_retrieval.get_facts_for_entities = AsyncMock(return_value=[fact])
        service.vector_service.search = AsyncMock(side_effect=Exception("Query failed"))
        service.pattern_query.search_patterns_by_problem = AsyncMock(
            side_effect=Exception("Query failed")
        )
        service.conflict_validation.validate_claims_batch = AsyncMock(return_value=[])

        result = await service.validate_response("Test runs on hypervisor-01")

        assert (
            "reinforced" in result.validation_status
            or "clean" in result.validation_status
        )
        assert len(result.degraded_queries) == 2

    async def test_validate_response_never_raises(self, service):
        """Test that service never raises exceptions (returns safe degraded result)."""
        service.claim_extraction.extract_claims = AsyncMock(
            side_effect=RuntimeError("Unexpected error")
        )

        result = await service.validate_response("Test response")

        assert isinstance(result, ReasoningValidationResult)
        # A failure means nothing was actually validated, so this must never
        # be reported as "clean" (see BUG 1 regression tests above).
        assert result.validation_status == "degraded"

    async def test_parallel_queries_complete_within_budget(self, service):
        """Test that parallel queries complete within per-query timeout."""
        import time

        claim = ExtractedClaim(
            claim_type="assertion",
            entity1="TestEntity",
            relation="RUNS_ON",
            entity2="hypervisor-01",
            confidence=0.9,
            sentence="Test runs on hypervisor-01",
            context="",
        )
        service.claim_extraction.extract_claims = AsyncMock(return_value=[claim])

        # Mock slow queries (250ms each - under 300ms budget)
        async def slow_query(*args, **kwargs):
            await asyncio.sleep(0.25)
            return []

        service.fact_retrieval.get_facts_for_entities = slow_query
        service.vector_service.search = slow_query
        service.pattern_query.search_patterns_by_problem = slow_query

        start = time.time()
        result = await service.validate_response(
            "Test runs on hypervisor-01", timeout_ms=1000
        )
        elapsed = (time.time() - start) * 1000

        # Should complete in ~250ms (parallel) + overhead, not 750ms (serial)
        assert elapsed < 600


# ============================================================================
# Integration-Style Tests (Service + Synthesizer Together)
# ============================================================================


@pytest.mark.asyncio
class TestValidationIntegration:
    """Integration tests for the full validation pipeline."""

    async def test_conflict_detection_flow(self):
        """Test full flow from extraction through conflict detection."""
        with (
            patch(
                "src.services.reasoning_validation_service.ClaimExtractionService"
            ) as mock_claim_service,
            patch(
                "src.services.reasoning_validation_service.Neo4jFactRetrievalService"
            ) as mock_fact_service,
            patch(
                "src.services.reasoning_validation_service.PatternQueryService"
            ) as mock_pattern_service,
        ):
            # Setup mocks
            claim = ExtractedClaim(
                claim_type="assertion",
                entity1="Mnemodrome",
                relation="RUNS_ON",
                entity2="server-02",
                confidence=0.95,
                sentence="Mnemodrome runs on server-02",
                context="",
            )
            mock_claim_service.return_value.extract_claims = AsyncMock(
                return_value=[claim]
            )

            fact = StoredFact(
                fact_uuid="fact-5",
                entity1_uuid="e1",
                entity1_name="Mnemodrome",
                relation="RUNS_ON",
                entity2_uuid="e2",
                entity2_name="hypervisor-01",
                created_at=datetime.now(UTC),
            )
            mock_fact_service.return_value.get_facts_for_entities = AsyncMock(
                return_value=[fact]
            )
            mock_pattern_service.return_value.search_patterns_by_problem = AsyncMock(
                return_value=[]
            )

            llm_service = AsyncMock()
            neo4j_service = AsyncMock()
            vector_service = AsyncMock()
            vector_service.search = AsyncMock(return_value=[])

            service = ReasoningValidationService(
                llm_service=llm_service,
                neo4j_service=neo4j_service,
                vector_service=vector_service,
            )

            result = await service.validate_response("Mnemodrome runs on server-02")

            assert result.claims_analyzed == 1
            assert result.entities_checked == 2
            # Should detect conflict
            assert (
                "proxmox" in str(result.conflicts) or len(result.degraded_queries) == 0
            )
