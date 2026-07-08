"""Unit tests for PatternStorageService."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from src.models.pattern_models import ExtractedPattern, StoredPattern
from src.services.pattern_storage_service import PatternStorageService


@pytest.fixture
def pattern_storage_service():
    """Provide PatternStorageService with mocked Neo4j dependency."""
    with patch("src.services.pattern_storage_service.Neo4jService"):
        service = PatternStorageService()
        service.neo4j_service = AsyncMock()
        return service


class TestPatternStorageServiceInit:
    """Test service initialization."""

    def test_init_with_defaults(self):
        """Test service initializes with defaults."""
        with patch("src.services.pattern_storage_service.Neo4jService"):
            with patch(
                "src.services.pattern_storage_service.PATTERN_RECOGNITION_ENABLED", True
            ):
                service = PatternStorageService()
                assert service.enabled is True
                assert service.neo4j_service is not None


class TestCheckForDuplicatePattern:
    """Test duplicate pattern detection."""

    @pytest.mark.asyncio
    async def test_check_duplicate_validates_inputs(self, pattern_storage_service):
        """Test validates problem and solution are not empty."""
        pattern_storage_service.enabled = True

        with pytest.raises(ValueError, match="problem and solution are required"):
            await pattern_storage_service.check_for_duplicate_pattern("", "solution")

        with pytest.raises(ValueError, match="problem and solution are required"):
            await pattern_storage_service.check_for_duplicate_pattern("problem", "")

    @pytest.mark.asyncio
    async def test_check_duplicate_validates_threshold(self, pattern_storage_service):
        """Test validates similarity threshold bounds."""
        pattern_storage_service.enabled = True

        with pytest.raises(
            ValueError, match="similarity_threshold must be between 0.0 and 1.0"
        ):
            await pattern_storage_service.check_for_duplicate_pattern(
                "problem", "solution", similarity_threshold=1.5
            )

    @pytest.mark.asyncio
    async def test_check_duplicate_returns_uuid_if_found(self, pattern_storage_service):
        """Test returns UUID of duplicate if found."""
        pattern_storage_service.enabled = True
        pattern_storage_service.neo4j_service.query = AsyncMock(
            return_value=[{"uuid": "duplicate-uuid", "similarity": 0.92}]
        )

        result = await pattern_storage_service.check_for_duplicate_pattern(
            "database latency issue", "implement caching layer"
        )

        assert result == "duplicate-uuid"

    @pytest.mark.asyncio
    async def test_check_duplicate_returns_none_if_not_found(
        self, pattern_storage_service
    ):
        """Test returns None if no duplicate found."""
        pattern_storage_service.enabled = True
        pattern_storage_service.neo4j_service.query = AsyncMock(return_value=[])

        result = await pattern_storage_service.check_for_duplicate_pattern(
            "unique problem", "unique solution"
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_check_duplicate_handles_query_error(self, pattern_storage_service):
        """Test gracefully handles query errors."""
        pattern_storage_service.enabled = True
        pattern_storage_service.neo4j_service.query = AsyncMock(
            side_effect=Exception("Database error")
        )

        result = await pattern_storage_service.check_for_duplicate_pattern(
            "problem", "solution"
        )

        assert result is None


class TestStorePattern:
    """Test pattern storage."""

    @pytest.mark.asyncio
    async def test_store_returns_none_when_disabled(self, pattern_storage_service):
        """Test returns None when feature disabled."""
        pattern_storage_service.enabled = False

        pattern = ExtractedPattern(
            pattern_type="code",
            problem_description="Test problem",
            solution_approach="Test solution",
            confidence=0.8,
            outcome="SUCCESS",
        )

        result = await pattern_storage_service.store_pattern(pattern, "session1")

        assert result is None

    @pytest.mark.asyncio
    async def test_store_validates_inputs(self, pattern_storage_service):
        """Test validates pattern and session_uuid."""
        pattern_storage_service.enabled = True

        with pytest.raises(ValueError):
            await pattern_storage_service.store_pattern(None, "session1")

    @pytest.mark.asyncio
    async def test_store_creates_new_pattern_on_unique(self, pattern_storage_service):
        """Test creates new pattern when no duplicate exists."""
        pattern_storage_service.enabled = True
        pattern_storage_service.check_for_duplicate_pattern = AsyncMock(
            return_value=None
        )
        pattern_storage_service._generate_pattern_uuid = AsyncMock(
            return_value="new-uuid"
        )
        pattern_storage_service.neo4j_service.query = AsyncMock(
            return_value=[{"pattern_uuid": "new-uuid"}]
        )

        pattern = ExtractedPattern(
            pattern_type="infrastructure",
            problem_description="Database connection pool exhaustion",
            solution_approach="Implement connection queue management",
            confidence=0.85,
            outcome="SUCCESS",
            session_uuid="session1",
        )

        result = await pattern_storage_service.store_pattern(pattern, "session1")

        assert result is not None
        assert result.pattern_uuid == "new-uuid"
        assert result.occurrence_count == 1

    @pytest.mark.asyncio
    async def test_store_updates_existing_on_duplicate(self, pattern_storage_service):
        """Test updates existing pattern on duplicate detection."""
        pattern_storage_service.enabled = True
        pattern_storage_service.check_for_duplicate_pattern = AsyncMock(
            return_value="existing-uuid"
        )
        pattern_storage_service.update_pattern_on_repeat = AsyncMock(return_value=True)
        pattern_storage_service._get_pattern_details = AsyncMock(
            return_value=StoredPattern(
                pattern_uuid="existing-uuid",
                name="Existing Pattern",
                pattern_type="infrastructure",
                problem="Database pool",
                solution="Queue management",
                occurrence_count=2,
                success_count=1,
                failure_count=1,
                success_rate=0.5,
                confidence=0.85,
                consecutive_failures=0,
                first_occurrence_date=datetime.now(UTC),
                latest_occurrence_date=datetime.now(UTC),
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
                supporting_sessions=["session1"],
                archived=False,
            )
        )

        pattern = ExtractedPattern(
            pattern_type="infrastructure",
            problem_description="Database connection pool exhaustion",
            solution_approach="Implement connection queue management",
            confidence=0.85,
            outcome="FAILED",
            session_uuid="session2",
        )

        result = await pattern_storage_service.store_pattern(pattern, "session2")

        assert result is not None
        assert result.pattern_uuid == "existing-uuid"
        assert result.occurrence_count == 2


class TestUpdatePatternOnRepeat:
    """Test pattern update on repeat occurrence."""

    @pytest.mark.asyncio
    async def test_update_increments_occurrence(self, pattern_storage_service):
        """Test increments occurrence count."""
        pattern_storage_service.neo4j_service.query = AsyncMock(
            return_value=[{"pattern_uuid": "pattern-1"}]
        )

        result = await pattern_storage_service.update_pattern_on_repeat(
            "pattern-1", "SUCCESS", "session2"
        )

        assert result is True
        # Verify query called with correct params
        call_args = pattern_storage_service.neo4j_service.query.call_args
        cypher = call_args[0][0]
        assert "p.occurrence_count = p.occurrence_count + 1" in cypher

    @pytest.mark.asyncio
    async def test_update_updates_success_count_on_success(
        self, pattern_storage_service
    ):
        """Test updates success count on SUCCESS outcome."""
        pattern_storage_service.neo4j_service.query = AsyncMock(
            return_value=[{"pattern_uuid": "pattern-1"}]
        )

        result = await pattern_storage_service.update_pattern_on_repeat(
            "pattern-1", "SUCCESS", "session2"
        )

        assert result is True
        call_args = pattern_storage_service.neo4j_service.query.call_args
        cypher = call_args[0][0]
        assert (
            "p.success_count = p.success_count + CASE WHEN $outcome = 'SUCCESS' THEN 1 ELSE 0 END"
            in cypher
        )

    @pytest.mark.asyncio
    async def test_update_returns_false_on_not_found(self, pattern_storage_service):
        """Test returns False if pattern not found."""
        pattern_storage_service.neo4j_service.query = AsyncMock(return_value=[])

        result = await pattern_storage_service.update_pattern_on_repeat(
            "nonexistent", "SUCCESS", "session2"
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_update_handles_errors_gracefully(self, pattern_storage_service):
        """Test returns False on error."""
        pattern_storage_service.neo4j_service.query = AsyncMock(
            side_effect=Exception("Database error")
        )

        result = await pattern_storage_service.update_pattern_on_repeat(
            "pattern-1", "SUCCESS", "session2"
        )

        assert result is False


class TestGetSimilarPatterns:
    """Test finding semantically similar patterns."""

    @pytest.mark.asyncio
    async def test_get_similar_returns_tuples(self, pattern_storage_service):
        """Test returns list of (uuid, similarity_score) tuples."""
        pattern_storage_service.neo4j_service.query = AsyncMock(
            return_value=[
                {"uuid": "similar-1", "similarity": 0.95},
                {"uuid": "similar-2", "similarity": 0.88},
            ]
        )

        result = await pattern_storage_service.get_similar_patterns("pattern-1")

        assert len(result) == 2
        assert result[0] == ("similar-1", 0.95)
        assert result[1] == ("similar-2", 0.88)

    @pytest.mark.asyncio
    async def test_get_similar_returns_empty_on_none_found(
        self, pattern_storage_service
    ):
        """Test returns empty list if no similar patterns."""
        pattern_storage_service.neo4j_service.query = AsyncMock(return_value=[])

        result = await pattern_storage_service.get_similar_patterns("pattern-1")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_similar_handles_errors(self, pattern_storage_service):
        """Test returns empty list on error."""
        pattern_storage_service.neo4j_service.query = AsyncMock(
            side_effect=Exception("Error")
        )

        result = await pattern_storage_service.get_similar_patterns("pattern-1")

        assert result == []


class TestArchivePattern:
    """Test pattern archival (soft delete)."""

    @pytest.mark.asyncio
    async def test_archive_sets_archived_flag(self, pattern_storage_service):
        """Test archives pattern by setting flag."""
        pattern_storage_service.neo4j_service.query = AsyncMock(
            return_value=[{"uuid": "pattern-1"}]
        )

        result = await pattern_storage_service.archive_pattern("pattern-1")

        assert result is True
        call_args = pattern_storage_service.neo4j_service.query.call_args
        cypher = call_args[0][0]
        assert "SET p.archived = true" in cypher

    @pytest.mark.asyncio
    async def test_archive_returns_false_if_not_found(self, pattern_storage_service):
        """Test returns False if pattern not found."""
        pattern_storage_service.neo4j_service.query = AsyncMock(return_value=[])

        result = await pattern_storage_service.archive_pattern("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_archive_handles_errors(self, pattern_storage_service):
        """Test returns False on error."""
        pattern_storage_service.neo4j_service.query = AsyncMock(
            side_effect=Exception("Error")
        )

        result = await pattern_storage_service.archive_pattern("pattern-1")

        assert result is False


class TestGeneratePatternUuid:
    """Test UUID generation."""

    @pytest.mark.asyncio
    async def test_generate_uuid_calls_neo4j(self, pattern_storage_service):
        """Test calls Neo4j randomUUID function."""
        pattern_storage_service.neo4j_service.query = AsyncMock(
            return_value=[{"uuid": "generated-uuid-123"}]
        )

        result = await pattern_storage_service._generate_pattern_uuid()

        assert result == "generated-uuid-123"
        pattern_storage_service.neo4j_service.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_uuid_returns_none_on_error(self, pattern_storage_service):
        """Test returns None if query fails."""
        pattern_storage_service.neo4j_service.query = AsyncMock(return_value=[])

        result = await pattern_storage_service._generate_pattern_uuid()

        assert result is None


class TestGeneratePatternName:
    """Test pattern name generation."""

    def test_generates_name_from_solution(self, pattern_storage_service):
        """Test generates name from solution approach."""
        solution = "Implement Redis caching layer for API responses"

        name = pattern_storage_service._generate_pattern_name(solution)

        assert name.startswith("Implement Redis caching")
        assert len(name) <= 100

    def test_truncates_long_names(self, pattern_storage_service):
        """Test truncates names to max length."""
        long_solution = "A" * 200

        name = pattern_storage_service._generate_pattern_name(long_solution)

        assert len(name) <= 100

    def test_removes_trailing_punctuation(self, pattern_storage_service):
        """Test removes trailing punctuation."""
        solution = "Implement connection pooling!!"

        name = pattern_storage_service._generate_pattern_name(solution)

        assert name == "Implement connection pooling"

    def test_returns_fallback_on_empty(self, pattern_storage_service):
        """Test returns fallback for empty input."""
        name = pattern_storage_service._generate_pattern_name("")

        assert name == "Unknown Pattern"


class TestGetPatternDetails:
    """Test pattern detail retrieval."""

    @pytest.mark.asyncio
    async def test_get_details_returns_none_if_not_found(self, pattern_storage_service):
        """Test returns None if pattern not found."""
        pattern_storage_service.neo4j_service.query = AsyncMock(return_value=[])

        result = await pattern_storage_service._get_pattern_details("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_details_returns_stored_pattern(self, pattern_storage_service):
        """Test returns StoredPattern object."""
        mock_node = {
            "uuid": "pattern-1",
            "name": "Test Pattern",
            "pattern_type": "code",
            "problem": "Memory leak",
            "solution": "GC implementation",
            "occurrence_count": 2,
            "success_count": 2,
            "failure_count": 0,
            "success_rate": 1.0,
            "confidence": 0.9,
            "consecutive_failures": 0,
            "first_occurrence_date": datetime.now(UTC),
            "latest_occurrence_date": datetime.now(UTC),
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "supporting_sessions": ["session1"],
            "archived": False,
        }

        pattern_storage_service.neo4j_service.query = AsyncMock(
            return_value=[{"p": mock_node}]
        )

        result = await pattern_storage_service._get_pattern_details("pattern-1")

        assert result is not None
        assert result.pattern_uuid == "pattern-1"
        assert result.success_rate == 1.0
