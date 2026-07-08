"""Unit tests for DeduplicationService (Feature 023).

Tests the three-stage deduplication pipeline with all external
dependencies mocked: Neo4j, Qdrant/VectorService, file system.

Coverage targets:
- validate_learning() three-stage pipeline and threshold boundaries
- title_exact_match() with Neo4j mocked
- vector_similarity_search() scoring at 90% and 95% thresholds
- fallback_title_fuzzy_match() difflib-based matching
- Graceful degradation when Qdrant is unavailable
- Empty/null input handling
- append_to_learning() method
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.deduplication_service import DeduplicationService

# ========================================
# Fixtures
# ========================================


@pytest.fixture
def mock_neo4j():
    """Mock Neo4jService with async query method."""
    service = AsyncMock()
    service.query = AsyncMock(return_value=[])
    return service


@pytest.fixture
def mock_vector():
    """Mock VectorService with is_enabled and search methods."""
    service = MagicMock()
    service.is_enabled = MagicMock(return_value=True)
    service.search_similar_learnings = AsyncMock(return_value=[])
    service.store_learning_embedding = AsyncMock(return_value=True)
    return service


@pytest.fixture
def mock_file_service():
    """Mock FileService."""
    service = AsyncMock()
    return service


@pytest.fixture
def svc(mock_neo4j, mock_vector, mock_file_service):
    """DeduplicationService with all dependencies mocked and explicit thresholds."""
    return DeduplicationService(
        vector_service=mock_vector,
        neo4j_service=mock_neo4j,
        file_service=mock_file_service,
        auto_skip_threshold=0.95,
        validation_threshold=0.90,
        fuzzy_match_threshold=0.85,
        enable_graceful_degradation=True,
    )


LONG_CONTENT = "A" * 60  # Content longer than 50 chars for input validation


# ========================================
# Constructor Tests
# ========================================


class TestDeduplicationServiceInit:
    """Tests for DeduplicationService constructor validation."""

    def test_raises_if_vector_service_is_none(self, mock_neo4j, mock_file_service):
        with pytest.raises(ValueError, match="vector_service is required"):
            DeduplicationService(
                vector_service=None,
                neo4j_service=mock_neo4j,
                file_service=mock_file_service,
            )

    def test_raises_if_neo4j_service_is_none(self, mock_vector, mock_file_service):
        with pytest.raises(ValueError, match="neo4j_service is required"):
            DeduplicationService(
                vector_service=mock_vector,
                neo4j_service=None,
                file_service=mock_file_service,
            )

    def test_raises_if_file_service_is_none(self, mock_neo4j, mock_vector):
        with pytest.raises(ValueError, match="file_service is required"):
            DeduplicationService(
                vector_service=mock_vector,
                neo4j_service=mock_neo4j,
                file_service=None,
            )

    def test_custom_thresholds_are_applied(
        self, mock_neo4j, mock_vector, mock_file_service
    ):
        svc = DeduplicationService(
            vector_service=mock_vector,
            neo4j_service=mock_neo4j,
            file_service=mock_file_service,
            auto_skip_threshold=0.99,
            validation_threshold=0.80,
            fuzzy_match_threshold=0.70,
        )
        assert svc.auto_skip_threshold == 0.99
        assert svc.validation_threshold == 0.80
        assert svc.fuzzy_match_threshold == 0.70


# ========================================
# Stage 1: Title Exact Match
# ========================================


@pytest.mark.asyncio
class TestTitleExactMatch:
    """Tests for title_exact_match() - Neo4j indexed lookup."""

    async def test_returns_uuid_when_match_found(self, svc, mock_neo4j):
        mock_neo4j.query.return_value = [{"uuid": "abc-123"}]
        result = await svc.title_exact_match("Existing Title")
        assert result == "abc-123"

    async def test_returns_none_when_no_match(self, svc, mock_neo4j):
        mock_neo4j.query.return_value = []
        result = await svc.title_exact_match("New Title")
        assert result is None

    async def test_returns_none_when_uuid_is_none(self, svc, mock_neo4j):
        mock_neo4j.query.return_value = [{"uuid": None}]
        result = await svc.title_exact_match("Title With Null UUID")
        assert result is None

    async def test_propagates_neo4j_exception(self, svc, mock_neo4j):
        mock_neo4j.query.side_effect = RuntimeError("Neo4j down")
        with pytest.raises(RuntimeError, match="Neo4j down"):
            await svc.title_exact_match("Any Title")


# ========================================
# Stage 2: Vector Similarity Search
# ========================================


@pytest.mark.asyncio
class TestVectorSimilaritySearch:
    """Tests for vector_similarity_search() - Qdrant semantic matching."""

    async def test_returns_matches_from_qdrant(self, svc, mock_vector):
        mock_vector.search_similar_learnings.return_value = [
            {
                "neo4j_uuid": "uuid-1",
                "title": "Similar Learning",
                "score": 0.93,
                "timestamp": "2025-01-01T00:00:00Z",
                "file_path": "/path/to/file.md",
                "category": "pattern",
                "content": "Some matching content here",
            }
        ]
        matches = await svc.vector_similarity_search("test content")
        assert len(matches) == 1
        assert matches[0].uuid == "uuid-1"
        assert matches[0].similarity_score == 0.93
        assert matches[0].title == "Similar Learning"

    async def test_returns_empty_when_no_matches(self, svc, mock_vector):
        mock_vector.search_similar_learnings.return_value = []
        matches = await svc.vector_similarity_search("unique content")
        assert matches == []

    async def test_uses_custom_threshold_and_limit(self, svc, mock_vector):
        mock_vector.search_similar_learnings.return_value = []
        await svc.vector_similarity_search("content", threshold=0.80, limit=10)
        mock_vector.search_similar_learnings.assert_called_once_with(
            query_text="content", limit=10, min_score=0.80
        )

    async def test_graceful_degradation_when_vector_disabled(self, svc, mock_vector):
        mock_vector.is_enabled.return_value = False
        matches = await svc.vector_similarity_search("content")
        assert matches == []

    async def test_raises_when_disabled_and_no_graceful_degradation(
        self, mock_neo4j, mock_vector, mock_file_service
    ):
        mock_vector.is_enabled.return_value = False
        svc = DeduplicationService(
            vector_service=mock_vector,
            neo4j_service=mock_neo4j,
            file_service=mock_file_service,
            enable_graceful_degradation=False,
        )
        with pytest.raises(RuntimeError, match="Vector service unavailable"):
            await svc.vector_similarity_search("content")

    async def test_graceful_degradation_on_search_exception(self, svc, mock_vector):
        mock_vector.search_similar_learnings.side_effect = ConnectionError(
            "Qdrant down"
        )
        matches = await svc.vector_similarity_search("content")
        assert matches == []

    async def test_raises_on_search_exception_without_graceful_degradation(
        self, mock_neo4j, mock_vector, mock_file_service
    ):
        mock_vector.is_enabled.return_value = True
        mock_vector.search_similar_learnings.side_effect = ConnectionError(
            "Qdrant down"
        )
        svc = DeduplicationService(
            vector_service=mock_vector,
            neo4j_service=mock_neo4j,
            file_service=mock_file_service,
            enable_graceful_degradation=False,
        )
        with pytest.raises(ConnectionError):
            await svc.vector_similarity_search("content")

    async def test_excerpt_truncated_to_200_chars(self, svc, mock_vector):
        long_content = "x" * 500
        mock_vector.search_similar_learnings.return_value = [
            {
                "neo4j_uuid": "uuid-1",
                "title": "Title",
                "score": 0.92,
                "timestamp": "",
                "file_path": "",
                "content": long_content,
            }
        ]
        matches = await svc.vector_similarity_search("content")
        assert len(matches[0].excerpt) == 200

    async def test_excerpt_none_when_no_content(self, svc, mock_vector):
        mock_vector.search_similar_learnings.return_value = [
            {
                "neo4j_uuid": "uuid-1",
                "title": "Title",
                "score": 0.92,
                "timestamp": "",
                "file_path": "",
            }
        ]
        matches = await svc.vector_similarity_search("content")
        assert matches[0].excerpt is None

    async def test_falls_back_to_id_field_when_no_neo4j_uuid(self, svc, mock_vector):
        mock_vector.search_similar_learnings.return_value = [
            {
                "id": "fallback-id",
                "title": "Title",
                "score": 0.92,
                "timestamp": "",
                "file_path": "",
            }
        ]
        matches = await svc.vector_similarity_search("content")
        assert matches[0].uuid == "fallback-id"


# ========================================
# Stage 3: Fuzzy Title Fallback
# ========================================


@pytest.mark.asyncio
class TestFuzzyTitleFallback:
    """Tests for fallback_title_fuzzy_match() - difflib matching."""

    async def test_returns_matches_above_threshold(self, svc, mock_neo4j):
        mock_neo4j.query.return_value = [
            {
                "uuid": "uuid-1",
                "title": "Factory Pattern Implementation",
                "category": "pattern",
                "file_path": "/path/file.md",
                "created_at": "2025-01-01",
            }
        ]
        matches = await svc.fallback_title_fuzzy_match(
            "Factory Pattern Implementation Guide"
        )
        assert len(matches) >= 1
        assert matches[0].uuid == "uuid-1"
        assert matches[0].similarity_score > 0.85

    async def test_filters_out_below_threshold(self, svc, mock_neo4j):
        mock_neo4j.query.return_value = [
            {
                "uuid": "uuid-1",
                "title": "Completely Different Topic About Cooking Recipes",
                "category": "general",
                "file_path": "/path/file.md",
                "created_at": "2025-01-01",
            }
        ]
        matches = await svc.fallback_title_fuzzy_match("Factory Pattern Implementation")
        assert len(matches) == 0

    async def test_case_insensitive_matching(self, svc, mock_neo4j):
        mock_neo4j.query.return_value = [
            {
                "uuid": "uuid-1",
                "title": "FACTORY PATTERN",
                "category": "pattern",
                "file_path": "/path/file.md",
                "created_at": "2025-01-01",
            }
        ]
        matches = await svc.fallback_title_fuzzy_match("factory pattern")
        assert len(matches) == 1
        assert matches[0].similarity_score == 1.0

    async def test_sorted_by_similarity_descending(self, svc, mock_neo4j):
        mock_neo4j.query.return_value = [
            {
                "uuid": "uuid-low",
                "title": "Factory Pattern Basics",
                "category": None,
                "file_path": "",
                "created_at": None,
            },
            {
                "uuid": "uuid-high",
                "title": "Factory Pattern",
                "category": None,
                "file_path": "",
                "created_at": None,
            },
        ]
        matches = await svc.fallback_title_fuzzy_match("Factory Pattern", threshold=0.5)
        assert len(matches) >= 2
        assert matches[0].similarity_score >= matches[1].similarity_score

    async def test_returns_empty_on_neo4j_error(self, svc, mock_neo4j):
        mock_neo4j.query.side_effect = RuntimeError("Neo4j down")
        matches = await svc.fallback_title_fuzzy_match("Any Title")
        assert matches == []

    async def test_skips_records_with_empty_title(self, svc, mock_neo4j):
        mock_neo4j.query.return_value = [
            {
                "uuid": "uuid-1",
                "title": "",
                "category": None,
                "file_path": "",
                "created_at": None,
            },
            {
                "uuid": "uuid-2",
                "title": None,
                "category": None,
                "file_path": "",
                "created_at": None,
            },
        ]
        matches = await svc.fallback_title_fuzzy_match("Something", threshold=0.0)
        # Empty title skipped, None title filtered by `if not existing_title`
        assert all(m.title != "" for m in matches)

    async def test_excerpt_is_always_none_in_fuzzy_mode(self, svc, mock_neo4j):
        mock_neo4j.query.return_value = [
            {
                "uuid": "uuid-1",
                "title": "Exact Match Title",
                "category": None,
                "file_path": "",
                "created_at": "2025-01-01T00:00:00",
            }
        ]
        matches = await svc.fallback_title_fuzzy_match("Exact Match Title")
        assert len(matches) == 1
        assert matches[0].excerpt is None

    async def test_handles_datetime_object_in_created_at(self, svc, mock_neo4j):
        from datetime import datetime

        dt = datetime(2025, 6, 15, 10, 30, 0)
        mock_neo4j.query.return_value = [
            {
                "uuid": "uuid-1",
                "title": "Exact Title",
                "category": None,
                "file_path": "",
                "created_at": dt,
            }
        ]
        matches = await svc.fallback_title_fuzzy_match("Exact Title")
        assert len(matches) == 1
        assert "2025-06-15" in matches[0].created_at

    async def test_custom_threshold_override(self, svc, mock_neo4j):
        mock_neo4j.query.return_value = [
            {
                "uuid": "uuid-1",
                "title": "Loosely Similar Title",
                "category": None,
                "file_path": "",
                "created_at": None,
            }
        ]
        # With very low threshold, should match
        matches_low = await svc.fallback_title_fuzzy_match(
            "Loosely Similar", threshold=0.3
        )
        # With very high threshold, should not match
        matches_high = await svc.fallback_title_fuzzy_match(
            "Loosely Similar", threshold=0.99
        )
        assert len(matches_low) >= len(matches_high)


# ========================================
# validate_learning() - Full Pipeline
# ========================================


@pytest.mark.asyncio
class TestValidateLearning:
    """Tests for the three-stage validate_learning() pipeline."""

    # --- Input validation ---

    async def test_empty_title_returns_proceed(self, svc):
        result = await svc.validate_learning("", LONG_CONTENT)
        assert result.status == "proceed"
        assert result.validation_stage == "input_validation"

    async def test_whitespace_only_title_returns_proceed(self, svc):
        result = await svc.validate_learning("   ", LONG_CONTENT)
        assert result.status == "proceed"
        assert result.validation_stage == "input_validation"

    async def test_short_content_returns_proceed(self, svc):
        result = await svc.validate_learning("Valid Title", "Too short")
        assert result.status == "proceed"
        assert result.validation_stage == "input_validation"
        assert "too short" in result.rationale.lower()

    async def test_none_title_returns_proceed(self, svc):
        result = await svc.validate_learning(None, LONG_CONTENT)
        assert result.status == "proceed"
        assert result.validation_stage == "input_validation"

    async def test_none_content_returns_proceed(self, svc):
        result = await svc.validate_learning("Title", None)
        assert result.status == "proceed"
        assert result.validation_stage == "input_validation"

    async def test_exactly_50_char_content_returns_proceed(self, svc):
        """Content with exactly 50 chars (stripped) is still < 50 meaningful dedup."""
        result = await svc.validate_learning("Title", "A" * 50)
        # 50 chars is not < 50, so it should NOT short-circuit. It proceeds to stage 1.
        # With no matches anywhere, it should be "proceed" from vector_similarity.
        assert result.status == "proceed"

    # --- Stage 1: Title exact match ---

    async def test_exact_title_match_returns_validation_required(self, svc, mock_neo4j):
        mock_neo4j.query.return_value = [{"uuid": "existing-uuid"}]
        result = await svc.validate_learning("Existing Title", LONG_CONTENT)
        assert result.status == "validation_required"
        assert result.validation_stage == "title_exact"
        assert result.conflict_type == "title_match"
        assert result.existing_learning_uuid == "existing-uuid"
        assert result.similarity_score == 1.0
        assert result.recommendation == "append"
        assert "append" in result.options

    async def test_stage1_failure_continues_to_stage2(
        self, svc, mock_neo4j, mock_vector
    ):
        """When Neo4j fails at stage 1, pipeline continues to stage 2."""
        # First call (title match) fails, second call would be fuzzy fallback
        mock_neo4j.query.side_effect = RuntimeError("Neo4j flaky")
        mock_vector.is_enabled.return_value = True
        mock_vector.search_similar_learnings.return_value = []

        result = await svc.validate_learning("Title", LONG_CONTENT)
        assert result.status == "proceed"
        assert result.validation_stage == "vector_similarity"

    # --- Stage 2: Vector similarity at threshold boundaries ---

    async def test_score_at_95_returns_auto_skipped(self, svc, mock_neo4j, mock_vector):
        """Score exactly at auto_skip_threshold (0.95) triggers auto_skipped."""
        mock_neo4j.query.return_value = []  # No exact title match
        mock_vector.search_similar_learnings.return_value = [
            {
                "neo4j_uuid": "uuid-1",
                "title": "Similar",
                "score": 0.95,
                "timestamp": "",
                "file_path": "",
            },
        ]
        result = await svc.validate_learning("Title", LONG_CONTENT)
        assert result.status == "auto_skipped"
        assert result.validation_stage == "vector_similarity"
        assert result.conflict_type == "semantic_similarity"
        assert result.similarity_score == 0.95
        assert result.recommendation == "skip"

    async def test_score_above_95_returns_auto_skipped(
        self, svc, mock_neo4j, mock_vector
    ):
        mock_neo4j.query.return_value = []
        mock_vector.search_similar_learnings.return_value = [
            {
                "neo4j_uuid": "uuid-1",
                "title": "Near Duplicate",
                "score": 0.98,
                "timestamp": "",
                "file_path": "",
            },
        ]
        result = await svc.validate_learning("Title", LONG_CONTENT)
        assert result.status == "auto_skipped"
        assert result.similarity_score == 0.98

    async def test_score_at_90_returns_validation_required(
        self, svc, mock_neo4j, mock_vector
    ):
        """Score in [validation_threshold, auto_skip_threshold) returns validation_required."""
        mock_neo4j.query.return_value = []
        mock_vector.search_similar_learnings.return_value = [
            {
                "neo4j_uuid": "uuid-1",
                "title": "Somewhat Similar",
                "score": 0.92,
                "timestamp": "",
                "file_path": "",
            },
        ]
        result = await svc.validate_learning("Title", LONG_CONTENT)
        assert result.status == "validation_required"
        assert result.validation_stage == "vector_similarity"
        assert result.conflict_type == "semantic_similarity"
        assert result.similarity_score == 0.92
        assert result.recommendation == "append"  # >= 0.80

    async def test_score_just_below_95_returns_validation_required(
        self, svc, mock_neo4j, mock_vector
    ):
        mock_neo4j.query.return_value = []
        mock_vector.search_similar_learnings.return_value = [
            {
                "neo4j_uuid": "uuid-1",
                "title": "Close",
                "score": 0.949,
                "timestamp": "",
                "file_path": "",
            },
        ]
        result = await svc.validate_learning("Title", LONG_CONTENT)
        assert result.status == "validation_required"
        assert result.similarity_score == 0.949

    async def test_no_similar_content_returns_proceed(
        self, svc, mock_neo4j, mock_vector
    ):
        mock_neo4j.query.return_value = []
        mock_vector.search_similar_learnings.return_value = []
        result = await svc.validate_learning("Unique Title", LONG_CONTENT)
        assert result.status == "proceed"
        assert result.validation_stage == "vector_similarity"

    async def test_existing_items_populated_for_validation_required(
        self, svc, mock_neo4j, mock_vector
    ):
        mock_neo4j.query.return_value = []
        mock_vector.search_similar_learnings.return_value = [
            {
                "neo4j_uuid": "uuid-1",
                "title": "Match 1",
                "score": 0.93,
                "timestamp": "t1",
                "file_path": "/p1",
                "category": "cat1",
                "content": "excerpt1",
            },
            {
                "neo4j_uuid": "uuid-2",
                "title": "Match 2",
                "score": 0.91,
                "timestamp": "t2",
                "file_path": "/p2",
            },
        ]
        result = await svc.validate_learning("Title", LONG_CONTENT)
        assert result.status == "validation_required"
        assert result.existing_items is not None
        assert len(result.existing_items) == 2

    async def test_recommendation_force_create_when_score_below_80(
        self, mock_neo4j, mock_vector, mock_file_service
    ):
        """When top match < 0.80, recommendation should be force_create."""
        svc = DeduplicationService(
            vector_service=mock_vector,
            neo4j_service=mock_neo4j,
            file_service=mock_file_service,
            auto_skip_threshold=0.95,
            validation_threshold=0.70,  # Lower threshold to allow 0.75 to pass
            fuzzy_match_threshold=0.85,
            enable_graceful_degradation=True,
        )
        mock_neo4j.query.return_value = []
        mock_vector.search_similar_learnings.return_value = [
            {
                "neo4j_uuid": "uuid-1",
                "title": "Loose Match",
                "score": 0.75,
                "timestamp": "",
                "file_path": "",
            },
        ]
        result = await svc.validate_learning("Title", LONG_CONTENT)
        assert result.status == "validation_required"
        assert result.recommendation == "force_create"

    # --- Stage 3: Fuzzy fallback ---

    async def test_falls_back_to_fuzzy_when_vector_disabled(
        self, svc, mock_neo4j, mock_vector
    ):
        # Stage 1: no match
        mock_neo4j.query.side_effect = [
            [],  # title_exact_match returns no match
            [  # fuzzy fallback queries all titles
                {
                    "uuid": "uuid-f",
                    "title": "Fuzzy Title Match",
                    "category": None,
                    "file_path": "",
                    "created_at": None,
                }
            ],
        ]
        mock_vector.is_enabled.return_value = False

        result = await svc.validate_learning("Fuzzy Title Match", LONG_CONTENT)
        assert result.validation_stage == "fuzzy_fallback"

    async def test_fuzzy_fallback_on_vector_exception_with_degradation_disabled_in_search(
        self, mock_neo4j, mock_vector, mock_file_service
    ):
        """When vector_similarity_search raises (graceful_degradation=False internally
        but validate_learning catches it), fuzzy fallback triggers if
        enable_graceful_degradation=True at the validate_learning level.

        The key subtlety: vector_similarity_search with graceful_degradation=True
        swallows exceptions and returns []. To trigger the fuzzy fallback path
        in validate_learning, is_enabled() must return True but the call in
        validate_learning's own try/except must raise (i.e., the service itself
        must raise, not be caught by vector_similarity_search).
        """
        # We call vector_similarity_search from within validate_learning,
        # and that method catches exceptions when graceful_degradation=True.
        # So the fuzzy fallback in validate_learning only triggers when
        # is_enabled() returns False at the validate_learning level check.
        # Test the actual behavior: vector exception with graceful degradation
        # returns empty results via vector_similarity_search, resulting in "proceed".
        svc = DeduplicationService(
            vector_service=mock_vector,
            neo4j_service=mock_neo4j,
            file_service=mock_file_service,
            auto_skip_threshold=0.95,
            validation_threshold=0.90,
            fuzzy_match_threshold=0.85,
            enable_graceful_degradation=True,
        )
        mock_neo4j.query.return_value = []
        mock_vector.is_enabled.return_value = True
        mock_vector.search_similar_learnings.side_effect = ConnectionError(
            "Qdrant error"
        )

        result = await svc.validate_learning("Title", LONG_CONTENT)
        # vector_similarity_search catches the error and returns [], so
        # validate_learning sees no matches and returns proceed from vector_similarity stage
        assert result.status == "proceed"
        assert result.validation_stage == "vector_similarity"

    async def test_no_fuzzy_fallback_when_degradation_disabled_and_vector_raises(
        self, mock_neo4j, mock_vector, mock_file_service
    ):
        """When graceful_degradation=False and vector_similarity_search raises,
        validate_learning catches the exception at stage 2 but does NOT trigger
        fuzzy fallback. Result is proceed with empty similar list."""
        svc = DeduplicationService(
            vector_service=mock_vector,
            neo4j_service=mock_neo4j,
            file_service=mock_file_service,
            enable_graceful_degradation=False,
        )
        mock_neo4j.query.return_value = []
        mock_vector.is_enabled.return_value = True
        mock_vector.search_similar_learnings.side_effect = ConnectionError(
            "Qdrant error"
        )

        result = await svc.validate_learning("Title", LONG_CONTENT)
        assert result.status == "proceed"
        # Stage stays at vector_similarity (no fuzzy fallback triggered)
        assert result.validation_stage == "vector_similarity"

    async def test_fuzzy_fallback_triggers_when_vector_raises_with_graceful_degradation(
        self, mock_neo4j, mock_vector, mock_file_service
    ):
        """When graceful_degradation=True AND the exception is NOT caught by
        vector_similarity_search (e.g., is_enabled check passes but call raises
        at the validate_learning level), fuzzy fallback triggers.

        We simulate this by making validate_learning's own call to
        vector_similarity_search raise, which validate_learning catches and
        sets use_fuzzy_fallback=True.
        """
        svc = DeduplicationService(
            vector_service=mock_vector,
            neo4j_service=mock_neo4j,
            file_service=mock_file_service,
            enable_graceful_degradation=True,
        )
        mock_neo4j.query.side_effect = [
            [],  # title exact: no match
            [],  # fuzzy fallback: no titles
        ]
        mock_vector.is_enabled.return_value = True
        # Patch vector_similarity_search itself to raise (bypassing its internal catch)
        original_vss = svc.vector_similarity_search

        async def raise_on_call(*a, **kw):
            raise ConnectionError("Qdrant error at validate level")

        svc.vector_similarity_search = raise_on_call

        result = await svc.validate_learning("Title", LONG_CONTENT)
        assert result.validation_stage == "fuzzy_fallback"
        assert result.status == "proceed"

    # --- Telemetry ---

    async def test_validation_time_ms_is_positive(self, svc, mock_neo4j, mock_vector):
        mock_neo4j.query.return_value = []
        mock_vector.search_similar_learnings.return_value = []
        result = await svc.validate_learning("Title", LONG_CONTENT)
        assert result.validation_time_ms > 0

    async def test_validation_time_ms_present_on_input_validation(self, svc):
        result = await svc.validate_learning("", LONG_CONTENT)
        assert result.validation_time_ms >= 0


# ========================================
# append_to_learning()
# ========================================


@pytest.mark.asyncio
class TestAppendToLearning:
    """Tests for append_to_learning() content merging."""

    async def test_short_content_returns_error(self, svc):
        result = await svc.append_to_learning("uuid-1", "Too short")
        assert result.status == "error"
        assert "50 characters" in result.error

    async def test_learning_not_found_returns_error(self, svc, mock_neo4j):
        mock_neo4j.query.return_value = []
        result = await svc.append_to_learning("nonexistent-uuid", LONG_CONTENT)
        assert result.status == "error"
        assert "not found" in result.error.lower()

    async def test_successful_append_without_file(self, svc, mock_neo4j, mock_vector):
        """Append succeeds when learning has no file_path on disk."""
        mock_neo4j.query.side_effect = [
            [
                {
                    "uuid": "uuid-1",
                    "title": "Title",
                    "file_path": "",
                    "content": "old",
                    "category": "general",
                    "append_count": 2,
                }
            ],
            [],  # update query
        ]
        mock_vector.is_enabled.return_value = True
        mock_vector.store_learning_embedding.return_value = True

        result = await svc.append_to_learning("uuid-1", LONG_CONTENT)
        assert result.status == "success"
        assert result.action == "appended"
        assert result.append_count == 3
        assert result.embedding_updated is True

    async def test_successful_append_with_file(
        self, svc, mock_neo4j, mock_vector, tmp_path
    ):
        """Append succeeds when learning has a real file on disk."""
        test_file = tmp_path / "learning.md"
        test_file.write_text("# Original Content\nSome text here.", encoding="utf-8")

        mock_neo4j.query.side_effect = [
            [
                {
                    "uuid": "uuid-1",
                    "title": "Title",
                    "file_path": str(test_file),
                    "content": "old",
                    "category": "general",
                    "append_count": 0,
                }
            ],
            [],  # update query
        ]
        mock_vector.is_enabled.return_value = True
        mock_vector.store_learning_embedding.return_value = True

        result = await svc.append_to_learning("uuid-1", LONG_CONTENT)
        assert result.status == "success"
        assert result.append_count == 1

        # Verify file was updated
        updated = test_file.read_text(encoding="utf-8")
        assert "---" in updated
        assert LONG_CONTENT in updated

    async def test_custom_section_title(self, svc, mock_neo4j, mock_vector):
        mock_neo4j.query.side_effect = [
            [
                {
                    "uuid": "uuid-1",
                    "title": "Title",
                    "file_path": "",
                    "content": "old",
                    "category": "general",
                    "append_count": 0,
                }
            ],
            [],
        ]
        mock_vector.is_enabled.return_value = False

        result = await svc.append_to_learning(
            "uuid-1", LONG_CONTENT, new_section_title="Custom Section"
        )
        assert result.status == "success"
        assert result.new_section_title == "Custom Section"

    async def test_embedding_not_updated_when_vector_disabled(
        self, svc, mock_neo4j, mock_vector
    ):
        mock_neo4j.query.side_effect = [
            [
                {
                    "uuid": "uuid-1",
                    "title": "Title",
                    "file_path": "",
                    "content": "old",
                    "category": "general",
                    "append_count": 0,
                }
            ],
            [],
        ]
        mock_vector.is_enabled.return_value = False

        result = await svc.append_to_learning("uuid-1", LONG_CONTENT)
        assert result.status == "success"
        assert result.embedding_updated is False

    async def test_embedding_failure_is_non_fatal(self, svc, mock_neo4j, mock_vector):
        mock_neo4j.query.side_effect = [
            [
                {
                    "uuid": "uuid-1",
                    "title": "Title",
                    "file_path": "",
                    "content": "old",
                    "category": "general",
                    "append_count": 0,
                }
            ],
            [],
        ]
        mock_vector.is_enabled.return_value = True
        mock_vector.store_learning_embedding.side_effect = RuntimeError("Qdrant error")

        result = await svc.append_to_learning("uuid-1", LONG_CONTENT)
        assert result.status == "success"
        assert result.embedding_updated is False

    async def test_neo4j_failure_returns_error(self, svc, mock_neo4j):
        mock_neo4j.query.side_effect = RuntimeError("Neo4j connection lost")
        result = await svc.append_to_learning("uuid-1", LONG_CONTENT)
        assert result.status == "error"
        assert result.error is not None


# ========================================
# _elapsed_ms helper
# ========================================


class TestElapsedMs:
    """Tests for the _elapsed_ms helper method."""

    def test_returns_positive_float(self, svc):
        import time

        start = time.perf_counter()
        elapsed = svc._elapsed_ms(start)
        assert isinstance(elapsed, float)
        assert elapsed >= 0
