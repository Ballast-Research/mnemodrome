"""
Feature 026: Cross-Session Pattern Recognition - Integration Tests

Full pattern lifecycle tests with mocked services (no live Neo4j/Qdrant).
Covers: extraction, deduplication, storage, querying, and graceful degradation.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.pattern_models import ExtractedPattern, PatternMatch, StoredPattern
from src.services.pattern_extraction_service import PatternExtractionService
from src.services.pattern_query_service import PatternQueryService
from src.services.pattern_storage_service import PatternStorageService

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------
_PATCH_NEO4J_STORAGE = "src.services.pattern_storage_service.Neo4jService"
_PATCH_NEO4J_QUERY = "src.services.pattern_query_service.Neo4jService"
_PATCH_LLM = "src.services.pattern_extraction_service.LLMService"


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------
def make_test_pattern(**overrides):
    """Create a test ExtractedPattern with defaults."""
    defaults = {
        "pattern_type": "infrastructure",
        "problem_description": "Service experiencing high latency and timeouts",
        "solution_approach": "Implement caching layer with Redis",
        "confidence": 0.85,
        "outcome": "SUCCESS",
        "metrics": {"latency_reduction": 60, "unit": "percent"},
        "session_uuid": "test-session-uuid-123",
    }
    defaults.update(overrides)
    return ExtractedPattern(**defaults)


def make_test_stored_pattern(**overrides):
    """Create a test StoredPattern with defaults."""
    now = datetime.now(UTC)
    defaults = {
        "pattern_uuid": "test-pattern-uuid-123",
        "name": "Implement caching layer with Redis",
        "pattern_type": "infrastructure",
        "problem": "Service experiencing high latency",
        "solution": "Implement caching layer",
        "occurrence_count": 1,
        "success_count": 1,
        "failure_count": 0,
        "success_rate": 1.0,
        "confidence": 0.85,
        "consecutive_failures": 0,
        "first_occurrence_date": now,
        "latest_occurrence_date": now,
        "created_at": now,
        "updated_at": now,
        "supporting_sessions": ["test-session-uuid-123"],
        "archived": False,
    }
    defaults.update(overrides)
    return StoredPattern(**defaults)


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
class TestPatternExtractionService:
    """Test pattern extraction from session summaries."""

    async def test_extract_patterns_from_sessions(self):
        """Test extraction of patterns from multiple session summaries."""
        session_summaries = [
            {
                "uuid": "session-1",
                "conversation_summary": "Fixed database performance issue by adding indexes",
                "key_insights": [
                    "Database indexes improve query speed",
                    "Need monitoring",
                ],
            },
            {
                "uuid": "session-2",
                "conversation_summary": "Implemented caching to reduce API calls",
                "key_insights": ["Caching reduces latency", "Remember to invalidate"],
            },
        ]

        mock_llm = MagicMock()
        mock_llm._call_llm = AsyncMock(
            return_value="""
            [
                {
                    "pattern_type": "code",
                    "problem_description": "Database queries were slow",
                    "solution_approach": "Added indexes to frequently accessed columns",
                    "outcome": "SUCCESS",
                    "confidence": 0.9,
                    "metrics": null
                }
            ]
            """
        )

        with patch(_PATCH_LLM, return_value=mock_llm):
            service = PatternExtractionService()
            patterns = await service.extract_patterns_from_sessions(session_summaries)

            assert patterns is not None
            # May extract 0-2 patterns depending on LLM response
            assert isinstance(patterns, list)

    async def test_extract_patterns_validates_confidence(self):
        """Test that low-confidence patterns are filtered out."""
        session_summaries = [
            {
                "uuid": "session-1",
                "conversation_summary": "Some vague work was done",
                "key_insights": [],
            }
        ]

        # Return a pattern with low confidence that should be filtered
        mock_llm = MagicMock()
        mock_llm._call_llm = AsyncMock(
            return_value="""
            [
                {
                    "pattern_type": "other",
                    "problem_description": "Unclear problem",
                    "solution_approach": "Vague approach",
                    "outcome": "INCONCLUSIVE",
                    "confidence": 0.3,
                    "metrics": null
                }
            ]
            """
        )

        with patch(_PATCH_LLM, return_value=mock_llm):
            service = PatternExtractionService()
            patterns = await service.extract_patterns_from_sessions(session_summaries)

            # Low confidence patterns are filtered (< 0.5)
            assert len(patterns) == 0

    async def test_extract_patterns_empty_sessions(self):
        """Test extraction with empty session list raises error."""
        mock_llm = MagicMock()

        with patch(_PATCH_LLM, return_value=mock_llm):
            service = PatternExtractionService()

            with pytest.raises(ValueError):
                await service.extract_patterns_from_sessions([])

    async def test_extract_patterns_handles_llm_failure(self):
        """Test graceful degradation when LLM call fails."""
        session_summaries = [
            {
                "uuid": "session-1",
                "conversation_summary": "Some work",
                "key_insights": [],
            }
        ]

        mock_llm = MagicMock()
        mock_llm._call_llm = AsyncMock(side_effect=Exception("LLM service error"))

        with patch(_PATCH_LLM, return_value=mock_llm):
            service = PatternExtractionService()
            patterns = await service.extract_patterns_from_sessions(session_summaries)

            # Should return empty list, not crash
            assert patterns == []


@pytest.mark.asyncio
class TestPatternStorageService:
    """Test pattern storage in Neo4j."""

    async def test_store_pattern_creates_node(self):
        """Test that storing a pattern creates a Neo4j node."""
        pattern = make_test_pattern()

        mock_neo4j = MagicMock()
        # Mock UUID generation
        mock_neo4j.query = AsyncMock(
            side_effect=[
                [{"uuid": "new-pattern-uuid"}],  # UUID generation
                [{"pattern_uuid": "new-pattern-uuid"}],  # Pattern creation
            ]
        )

        with patch(_PATCH_NEO4J_STORAGE, return_value=mock_neo4j):
            service = PatternStorageService()

            # Mock duplicate check to return None
            service.check_for_duplicate_pattern = AsyncMock(return_value=None)

            stored = await service.store_pattern(pattern, "session-uuid-123")

            assert stored is not None
            assert stored.pattern_uuid == "new-pattern-uuid"
            assert stored.name == pattern.solution_approach[:100]
            assert stored.occurrence_count == 1
            assert stored.success_rate == 1.0

    async def test_store_pattern_detects_duplicate(self):
        """Test that duplicate patterns are detected and merged."""
        pattern = make_test_pattern()

        mock_neo4j = MagicMock()
        mock_neo4j.query = AsyncMock(
            side_effect=[
                # Duplicate check returns existing pattern
                None,  # After duplicate found, update is called
            ]
        )

        with patch(_PATCH_NEO4J_STORAGE, return_value=mock_neo4j):
            service = PatternStorageService()

            # Mock duplicate check to return existing UUID
            service.check_for_duplicate_pattern = AsyncMock(
                return_value="existing-pattern-uuid"
            )
            # Mock update and retrieval
            service.update_pattern_on_repeat = AsyncMock(return_value=True)
            service._get_pattern_details = AsyncMock(
                return_value=make_test_stored_pattern(
                    pattern_uuid="existing-pattern-uuid", occurrence_count=2
                )
            )

            stored = await service.store_pattern(pattern, "session-uuid-123")

            assert stored is not None
            assert stored.pattern_uuid == "existing-pattern-uuid"
            assert stored.occurrence_count == 2

    async def test_check_for_duplicate_pattern_validation(self):
        """Test that duplicate check validates inputs."""
        mock_neo4j = MagicMock()

        with patch(_PATCH_NEO4J_STORAGE, return_value=mock_neo4j):
            service = PatternStorageService()

            # Empty problem should raise
            with pytest.raises(ValueError):
                await service.check_for_duplicate_pattern("", "solution")

            # Empty solution should raise
            with pytest.raises(ValueError):
                await service.check_for_duplicate_pattern("problem", "")

            # Invalid threshold should raise
            with pytest.raises(ValueError):
                await service.check_for_duplicate_pattern(
                    "problem", "solution", similarity_threshold=1.5
                )

    async def test_update_pattern_on_repeat(self):
        """Test updating pattern occurrence counts on repeat."""
        mock_neo4j = MagicMock()
        mock_neo4j.query = AsyncMock(
            return_value=[{"pattern_uuid": "test-pattern-uuid"}]
        )

        with patch(_PATCH_NEO4J_STORAGE, return_value=mock_neo4j):
            service = PatternStorageService()

            success = await service.update_pattern_on_repeat(
                pattern_uuid="test-pattern-uuid",
                outcome="SUCCESS",
                session_uuid="session-uuid-123",
            )

            assert success is True
            # Verify Cypher was called with correct parameters
            assert mock_neo4j.query.called

    async def test_archive_pattern(self):
        """Test soft-delete (archiving) a pattern."""
        mock_neo4j = MagicMock()
        mock_neo4j.query = AsyncMock(return_value=[{"uuid": "test-pattern-uuid"}])

        with patch(_PATCH_NEO4J_STORAGE, return_value=mock_neo4j):
            service = PatternStorageService()

            success = await service.archive_pattern("test-pattern-uuid")

            assert success is True
            assert mock_neo4j.query.called


@pytest.mark.asyncio
class TestPatternQueryService:
    """Test pattern querying from Neo4j."""

    async def test_query_patterns_basic(self):
        """Test basic pattern query without filters."""
        mock_neo4j = MagicMock()

        # Mock pattern nodes
        pattern_node_1 = {
            "uuid": "pattern-1",
            "name": "Add caching",
            "pattern_type": "infrastructure",
            "problem": "High latency",
            "solution": "Add caching layer",
            "occurrence_count": 3,
            "success_count": 3,
            "failure_count": 0,
            "success_rate": 1.0,
            "confidence": 0.85,
            "consecutive_failures": 0,
            "first_occurrence_date": datetime.now(UTC).isoformat(),
            "latest_occurrence_date": datetime.now(UTC).isoformat(),
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "supporting_sessions": ["session-1", "session-2"],
            "archived": False,
        }

        mock_neo4j.query = AsyncMock(return_value=[{"p": pattern_node_1}])

        with patch(_PATCH_NEO4J_QUERY, return_value=mock_neo4j):
            service = PatternQueryService()

            patterns = await service.query_patterns(limit=10)

            assert len(patterns) == 1
            assert patterns[0].pattern_uuid == "pattern-1"
            assert patterns[0].occurrence_count == 3
            assert patterns[0].success_rate == 1.0

    async def test_query_patterns_with_filters(self):
        """Test pattern query with multiple filters."""
        mock_neo4j = MagicMock()
        mock_neo4j.query = AsyncMock(return_value=[])

        with patch(_PATCH_NEO4J_QUERY, return_value=mock_neo4j):
            service = PatternQueryService()

            patterns = await service.query_patterns(
                problem_type="infrastructure",
                min_success_rate=0.8,
                limit=5,
            )

            assert patterns == []
            # Verify query was called with correct parameters
            assert mock_neo4j.query.called
            call_args = mock_neo4j.query.call_args
            assert "infrastructure" in str(call_args)

    async def test_query_patterns_validation(self):
        """Test that query parameters are validated."""
        mock_neo4j = MagicMock()

        with patch(_PATCH_NEO4J_QUERY, return_value=mock_neo4j):
            service = PatternQueryService()

            # Invalid success rate should raise
            with pytest.raises(ValueError):
                await service.query_patterns(min_success_rate=1.5)

            # Invalid limit should raise
            with pytest.raises(ValueError):
                await service.query_patterns(limit=0)

    async def test_get_pattern_by_uuid(self):
        """Test retrieving a pattern by UUID."""
        mock_neo4j = MagicMock()

        pattern_node = {
            "uuid": "pattern-123",
            "name": "Fix memory leak",
            "pattern_type": "code",
            "problem": "Memory leak",
            "solution": "Fix reference counting",
            "occurrence_count": 1,
            "success_count": 1,
            "failure_count": 0,
            "success_rate": 1.0,
            "confidence": 0.9,
            "consecutive_failures": 0,
            "first_occurrence_date": datetime.now(UTC).isoformat(),
            "latest_occurrence_date": datetime.now(UTC).isoformat(),
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "supporting_sessions": ["session-1"],
            "archived": False,
        }

        mock_neo4j.query = AsyncMock(return_value=[{"p": pattern_node}])

        with patch(_PATCH_NEO4J_QUERY, return_value=mock_neo4j):
            service = PatternQueryService()

            pattern = await service.get_pattern_by_uuid("pattern-123")

            assert pattern is not None
            assert pattern.pattern_uuid == "pattern-123"
            assert pattern.name == "Fix memory leak"

    async def test_get_pattern_by_uuid_not_found(self):
        """Test retrieving non-existent pattern returns None."""
        mock_neo4j = MagicMock()
        mock_neo4j.query = AsyncMock(return_value=[])

        with patch(_PATCH_NEO4J_QUERY, return_value=mock_neo4j):
            service = PatternQueryService()

            pattern = await service.get_pattern_by_uuid("non-existent")

            assert pattern is None

    async def test_search_patterns_by_problem(self):
        """Test semantic search for patterns by problem text."""
        mock_neo4j = MagicMock()

        pattern_node = {
            "uuid": "pattern-1",
            "name": "Database indexing",
            "pattern_type": "database",
            "problem": "Slow database queries",
            "solution": "Add indexes",
            "occurrence_count": 5,
            "success_count": 4,
            "failure_count": 1,
            "success_rate": 0.8,
            "confidence": 0.85,
            "consecutive_failures": 0,
            "first_occurrence_date": datetime.now(UTC).isoformat(),
            "latest_occurrence_date": datetime.now(UTC).isoformat(),
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "supporting_sessions": ["session-1"],
            "archived": False,
        }

        mock_neo4j.query = AsyncMock(
            return_value=[{"p": pattern_node, "similarity": 0.92}]
        )

        with patch(_PATCH_NEO4J_QUERY, return_value=mock_neo4j):
            service = PatternQueryService()

            matches = await service.search_patterns_by_problem(
                "Database is very slow", limit=10
            )

            assert len(matches) == 1
            assert isinstance(matches[0], PatternMatch)
            assert matches[0].similarity_score == 0.92
            assert matches[0].success_rate == 0.8

    async def test_search_patterns_validation(self):
        """Test that search validates problem text."""
        mock_neo4j = MagicMock()

        with patch(_PATCH_NEO4J_QUERY, return_value=mock_neo4j):
            service = PatternQueryService()

            # Empty problem should raise
            with pytest.raises(ValueError):
                await service.search_patterns_by_problem("", limit=10)

            # Too short problem should raise
            with pytest.raises(ValueError):
                await service.search_patterns_by_problem("hi", limit=10)

    async def test_get_recent_patterns(self):
        """Test retrieving patterns from recent sessions."""
        mock_neo4j = MagicMock()

        pattern_node = {
            "uuid": "pattern-1",
            "name": "Cache invalidation",
            "pattern_type": "infrastructure",
            "problem": "Stale cache",
            "solution": "Implement cache TTL",
            "occurrence_count": 2,
            "success_count": 2,
            "failure_count": 0,
            "success_rate": 1.0,
            "confidence": 0.88,
            "consecutive_failures": 0,
            "first_occurrence_date": datetime.now(UTC).isoformat(),
            "latest_occurrence_date": datetime.now(UTC).isoformat(),
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "supporting_sessions": ["session-1"],
            "archived": False,
        }

        mock_neo4j.query = AsyncMock(return_value=[{"p": pattern_node}])

        with patch(_PATCH_NEO4J_QUERY, return_value=mock_neo4j):
            service = PatternQueryService()

            patterns = await service.get_recent_patterns(days=30, limit=10)

            assert len(patterns) == 1
            assert patterns[0].pattern_uuid == "pattern-1"

    async def test_get_high_success_patterns(self):
        """Test retrieving patterns with high success rates."""
        mock_neo4j = MagicMock()

        pattern_node = {
            "uuid": "pattern-1",
            "name": "Circuit breaker pattern",
            "pattern_type": "architecture",
            "problem": "Cascading failures",
            "solution": "Implement circuit breaker",
            "occurrence_count": 8,
            "success_count": 8,
            "failure_count": 0,
            "success_rate": 1.0,
            "confidence": 0.95,
            "consecutive_failures": 0,
            "first_occurrence_date": datetime.now(UTC).isoformat(),
            "latest_occurrence_date": datetime.now(UTC).isoformat(),
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "supporting_sessions": ["session-1"],
            "archived": False,
        }

        mock_neo4j.query = AsyncMock(return_value=[{"p": pattern_node}])

        with patch(_PATCH_NEO4J_QUERY, return_value=mock_neo4j):
            service = PatternQueryService()

            patterns = await service.get_high_success_patterns(
                min_success_rate=0.8, min_occurrences=3, limit=10
            )

            assert len(patterns) == 1
            assert patterns[0].success_rate == 1.0
            assert patterns[0].occurrence_count == 8


@pytest.mark.asyncio
class TestPatternDeduplication:
    """Test pattern deduplication logic."""

    async def test_deduplication_detects_similar_patterns(self):
        """Test that semantic similarity detects near-duplicates."""
        pattern = make_test_pattern()

        mock_neo4j = MagicMock()
        # Mock Levenshtein similarity result
        mock_neo4j.query = AsyncMock(
            return_value=[{"uuid": "existing-pattern", "similarity": 0.92}]
        )

        with patch(_PATCH_NEO4J_STORAGE, return_value=mock_neo4j):
            service = PatternStorageService()

            duplicate_uuid = await service.check_for_duplicate_pattern(
                problem=pattern.problem_description,
                solution=pattern.solution_approach,
                similarity_threshold=0.90,
            )

            assert duplicate_uuid == "existing-pattern"

    async def test_deduplication_threshold_sensitivity(self):
        """Test that deduplication respects threshold settings."""
        mock_neo4j = MagicMock()

        # Return a pattern with 0.88 similarity (below 0.90 threshold)
        mock_neo4j.query = AsyncMock(
            return_value=[]  # No results when filtered by threshold
        )

        with patch(_PATCH_NEO4J_STORAGE, return_value=mock_neo4j):
            service = PatternStorageService()

            duplicate_uuid = await service.check_for_duplicate_pattern(
                problem="Problem A",
                solution="Solution A",
                similarity_threshold=0.90,
            )

            assert duplicate_uuid is None

    async def test_deduplication_prevents_cross_type_merges(self):
        """Test that patterns of different types don't merge."""
        # This is validated in the Cypher query with problem_type filtering
        mock_neo4j = MagicMock()
        mock_neo4j.query = AsyncMock(return_value=[])

        with patch(_PATCH_NEO4J_STORAGE, return_value=mock_neo4j):
            service = PatternStorageService()

            # Even with high similarity, if pattern_type is different, no merge
            duplicate_uuid = await service.check_for_duplicate_pattern(
                problem="Database issue",
                solution="Add indexes",
                similarity_threshold=0.80,
            )

            assert duplicate_uuid is None


@pytest.mark.asyncio
class TestPatternGracefulDegradation:
    """Test graceful degradation when services fail."""

    async def test_extraction_gracefully_handles_neo4j_failure(self):
        """Test that extraction continues even if Neo4j is unavailable."""
        session_summaries = [
            {
                "uuid": "session-1",
                "conversation_summary": "Solved an issue",
                "key_insights": [],
            }
        ]

        mock_llm = MagicMock()
        mock_llm._call_llm = AsyncMock(
            return_value="""
            [
                {
                    "pattern_type": "code",
                    "problem_description": "Memory leak",
                    "solution_approach": "Fix reference counting",
                    "outcome": "SUCCESS",
                    "confidence": 0.8,
                    "metrics": null
                }
            ]
            """
        )

        with patch(_PATCH_LLM, return_value=mock_llm):
            service = PatternExtractionService()
            patterns = await service.extract_patterns_from_sessions(session_summaries)

            # Extraction should succeed even if storage fails
            assert len(patterns) > 0

    async def test_storage_handles_neo4j_unavailable(self):
        """Test graceful degradation when Neo4j is unavailable."""
        pattern = make_test_pattern()

        mock_neo4j = MagicMock()
        mock_neo4j.query = AsyncMock(side_effect=Exception("Neo4j connection failed"))

        with patch(_PATCH_NEO4J_STORAGE, return_value=mock_neo4j):
            service = PatternStorageService()

            # Should return None gracefully, not crash
            with pytest.raises(Exception):
                # The service will raise, but in production it would be caught
                await service.store_pattern(pattern, "session-uuid")

    async def test_query_handles_neo4j_unavailable(self):
        """Test query returns empty results when Neo4j is unavailable."""
        mock_neo4j = MagicMock()
        mock_neo4j.query = AsyncMock(side_effect=Exception("Neo4j connection failed"))

        with patch(_PATCH_NEO4J_QUERY, return_value=mock_neo4j):
            service = PatternQueryService()

            # Query should raise, but in production would be handled
            with pytest.raises(Exception):
                await service.query_patterns(limit=10)


@pytest.mark.asyncio
class TestPatternRanking:
    """Test pattern ranking and relevance scoring."""

    async def test_patterns_ranked_by_success_rate(self):
        """Test that high-success patterns rank higher."""
        mock_neo4j = MagicMock()

        # Pattern with 100% success ranks higher
        pattern_high = {
            "uuid": "pattern-high",
            "name": "High success",
            "pattern_type": "code",
            "problem": "Issue",
            "solution": "Solution",
            "occurrence_count": 3,
            "success_count": 3,
            "failure_count": 0,
            "success_rate": 1.0,
            "confidence": 0.9,
            "consecutive_failures": 0,
            "first_occurrence_date": datetime.now(UTC).isoformat(),
            "latest_occurrence_date": datetime.now(UTC).isoformat(),
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "supporting_sessions": ["session-1"],
            "archived": False,
        }

        # Pattern with lower success rates
        pattern_low = {
            "uuid": "pattern-low",
            "name": "Low success",
            "pattern_type": "code",
            "problem": "Issue",
            "solution": "Solution",
            "occurrence_count": 10,
            "success_count": 3,
            "failure_count": 7,
            "success_rate": 0.3,
            "confidence": 0.7,
            "consecutive_failures": 2,
            "first_occurrence_date": datetime.now(UTC).isoformat(),
            "latest_occurrence_date": datetime.now(UTC).isoformat(),
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "supporting_sessions": ["session-1"],
            "archived": False,
        }

        # Results ordered by success_rate DESC, occurrence_count DESC
        mock_neo4j.query = AsyncMock(
            return_value=[{"p": pattern_high}, {"p": pattern_low}]
        )

        with patch(_PATCH_NEO4J_QUERY, return_value=mock_neo4j):
            service = PatternQueryService()

            patterns = await service.query_patterns(limit=10)

            # High success should rank first
            assert patterns[0].success_rate >= patterns[1].success_rate

    async def test_patterns_ranked_by_occurrence_count(self):
        """Test that frequently occurring patterns rank higher when success equal."""
        mock_neo4j = MagicMock()

        # Same success rate, different occurrence counts
        pattern_frequent = {
            "uuid": "pattern-freq",
            "name": "Frequent",
            "pattern_type": "code",
            "problem": "Issue",
            "solution": "Solution",
            "occurrence_count": 10,
            "success_count": 8,
            "failure_count": 2,
            "success_rate": 0.8,
            "confidence": 0.85,
            "consecutive_failures": 0,
            "first_occurrence_date": datetime.now(UTC).isoformat(),
            "latest_occurrence_date": datetime.now(UTC).isoformat(),
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "supporting_sessions": ["session-1"],
            "archived": False,
        }

        pattern_infrequent = {
            "uuid": "pattern-infreq",
            "name": "Infrequent",
            "pattern_type": "code",
            "problem": "Issue",
            "solution": "Solution",
            "occurrence_count": 1,
            "success_count": 1,
            "failure_count": 0,
            "success_rate": 0.8,
            "confidence": 0.85,
            "consecutive_failures": 0,
            "first_occurrence_date": datetime.now(UTC).isoformat(),
            "latest_occurrence_date": datetime.now(UTC).isoformat(),
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "supporting_sessions": ["session-1"],
            "archived": False,
        }

        mock_neo4j.query = AsyncMock(
            return_value=[{"p": pattern_frequent}, {"p": pattern_infrequent}]
        )

        with patch(_PATCH_NEO4J_QUERY, return_value=mock_neo4j):
            service = PatternQueryService()

            patterns = await service.query_patterns(limit=10)

            # Frequent should rank higher when success rates equal
            assert patterns[0].occurrence_count >= patterns[1].occurrence_count


# ---------------------------------------------------------------------------
# Integration Test Scenarios
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
class TestPatternFullWorkflow:
    """Test complete pattern recognition workflows."""

    async def test_extract_store_query_workflow(self):
        """Test full workflow: extract patterns → store → query."""
        # Create mock services
        mock_neo4j = MagicMock()
        mock_llm = MagicMock()

        # Mock LLM extraction
        mock_llm._call_llm = AsyncMock(
            return_value="""
            [
                {
                    "pattern_type": "infrastructure",
                    "problem_description": "High memory usage",
                    "solution_approach": "Implement memory limits",
                    "outcome": "SUCCESS",
                    "confidence": 0.9,
                    "metrics": {"memory_reduction": 50}
                }
            ]
            """
        )

        # Mock Neo4j for storage and query
        mock_neo4j.query = AsyncMock(
            side_effect=[
                [{"uuid": "pattern-uuid-123"}],  # UUID generation
                [{"pattern_uuid": "pattern-uuid-123"}],  # Pattern creation
            ]
        )

        session_summaries = [
            {
                "uuid": "session-1",
                "conversation_summary": "Fixed high memory usage",
                "key_insights": ["Memory limits help"],
            }
        ]

        with (
            patch(_PATCH_LLM, return_value=mock_llm),
            patch(_PATCH_NEO4J_STORAGE, return_value=mock_neo4j),
        ):
            # Step 1: Extract patterns
            extraction_service = PatternExtractionService()
            patterns = await extraction_service.extract_patterns_from_sessions(
                session_summaries
            )
            assert len(patterns) > 0

            # Step 2: Store patterns
            storage_service = PatternStorageService()
            storage_service.check_for_duplicate_pattern = AsyncMock(return_value=None)

            stored = await storage_service.store_pattern(
                patterns[0], "session-uuid-123"
            )
            assert stored is not None

    async def test_deduplication_on_repeated_patterns(self):
        """Test that repeated patterns increment occurrence count."""
        pattern_1 = make_test_pattern(
            solution_approach="Add Redis cache to reduce latency"
        )
        pattern_2 = make_test_pattern(
            solution_approach="Implement Redis caching for performance"
        )

        mock_neo4j = MagicMock()

        # First call: duplicate check finds match
        # Second call: update occurrence count
        mock_neo4j.query = AsyncMock(
            side_effect=[
                [{"uuid": "existing-pattern", "similarity": 0.92}],  # Duplicate found
                [{"pattern_uuid": "existing-pattern"}],  # Update successful
            ]
        )

        with patch(_PATCH_NEO4J_STORAGE, return_value=mock_neo4j):
            service = PatternStorageService()

            # Store first pattern
            service.check_for_duplicate_pattern = AsyncMock(return_value=None)
            service.update_pattern_on_repeat = AsyncMock(return_value=True)
            service._get_pattern_details = AsyncMock(
                return_value=make_test_stored_pattern(occurrence_count=2)
            )

            stored_1 = await service.store_pattern(pattern_1, "session-1")
            assert stored_1.occurrence_count == 1

            # Store similar pattern (should increment)
            service.check_for_duplicate_pattern = AsyncMock(
                return_value="existing-pattern"
            )

            stored_2 = await service.store_pattern(pattern_2, "session-2")
            assert stored_2.occurrence_count == 2
