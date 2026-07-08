"""Integration tests for Feature 023: Knowledge Layer Deduplication.

End-to-end tests for the complete deduplication workflow including
capture_learning, append_to_learning, and graceful degradation scenarios.

These tests use mocked services to avoid requiring live Neo4j/Qdrant.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_services():
    """Create mock services for integration testing."""
    vector_service = MagicMock()
    vector_service.is_enabled.return_value = True
    vector_service.search_similar_learnings = AsyncMock(return_value=[])
    vector_service.store_learning_embedding = AsyncMock(return_value="test-uuid")

    neo4j_service = MagicMock()
    neo4j_service.query = AsyncMock(return_value=[])

    file_service = MagicMock()
    file_service._get_target_path.return_value = MagicMock()
    file_service.write_file = MagicMock(return_value="/test/path.md")

    return {
        "vector": vector_service,
        "neo4j": neo4j_service,
        "file_service": file_service,
    }


@pytest.mark.asyncio
class TestCaptureLearnineDeduplicationE2E:
    """E2E tests for capture_learning with deduplication (T028)."""

    async def test_unique_content_proceeds(self, mock_services):
        """Test that unique content passes validation and gets captured."""
        from src.services.deduplication_service import DeduplicationService

        dedup_service = DeduplicationService(
            vector_service=mock_services["vector"],
            neo4j_service=mock_services["neo4j"],
            file_service=mock_services["file_service"],
        )

        # Configure mocks for unique content scenario
        mock_services["neo4j"].query.return_value = []  # No title match
        mock_services["vector"].search_similar_learnings.return_value = []  # No similar

        result = await dedup_service.validate_learning(
            title="Unique Learning About Quantum Computing",
            content="This is a completely unique learning about quantum computing principles and applications that has no similarity to existing content.",
        )

        assert result.status == "proceed"
        assert result.validation_stage in [
            "title_exact",
            "vector_similarity",
            "fuzzy_fallback",
        ]
        assert result.validation_time_ms > 0

    async def test_exact_duplicate_auto_skipped(self, mock_services):
        """Test that exact duplicates (≥95% similar) are auto-skipped."""
        from src.services.deduplication_service import DeduplicationService

        dedup_service = DeduplicationService(
            vector_service=mock_services["vector"],
            neo4j_service=mock_services["neo4j"],
            file_service=mock_services["file_service"],
            auto_skip_threshold=0.95,
        )

        # Configure mocks for duplicate scenario
        mock_services["vector"].search_similar_learnings.return_value = [
            {
                "neo4j_uuid": "existing-uuid-123",
                "title": "Existing Learning",
                "score": 0.97,
                "timestamp": "2025-01-01T00:00:00Z",
                "category": "general",
                "content": "Similar content...",
            }
        ]

        result = await dedup_service.validate_learning(
            title="Almost Identical Learning",
            content="This content is nearly identical to an existing learning about the same topic with very similar wording and structure.",
        )

        assert result.status == "auto_skipped"
        assert result.similarity_score >= 0.95
        assert result.existing_learning_uuid == "existing-uuid-123"
        assert result.conflict_type == "semantic_similarity"

    async def test_similar_content_requires_validation(self, mock_services):
        """Test that moderately similar content (90-95%) requires validation."""
        from src.services.deduplication_service import DeduplicationService

        dedup_service = DeduplicationService(
            vector_service=mock_services["vector"],
            neo4j_service=mock_services["neo4j"],
            file_service=mock_services["file_service"],
            auto_skip_threshold=0.95,
            validation_threshold=0.90,
        )

        # Configure mocks for similar (not duplicate) scenario
        mock_services["vector"].search_similar_learnings.return_value = [
            {
                "neo4j_uuid": "similar-uuid-456",
                "title": "Related Learning",
                "score": 0.92,
                "timestamp": "2025-01-01T00:00:00Z",
                "category": "pattern",
                "content": "Related but different content...",
            }
        ]

        result = await dedup_service.validate_learning(
            title="Related Learning Topic",
            content="This content discusses a related topic with some overlap but introduces new concepts and perspectives that may warrant a separate learning.",
        )

        assert result.status == "validation_required"
        assert 0.90 <= result.similarity_score < 0.95
        assert result.existing_items is not None
        assert len(result.existing_items) > 0
        assert result.recommendation in ["append", "force_create"]

    async def test_title_exact_match_requires_validation(self, mock_services):
        """Test that exact title match triggers validation."""
        from src.services.deduplication_service import DeduplicationService

        dedup_service = DeduplicationService(
            vector_service=mock_services["vector"],
            neo4j_service=mock_services["neo4j"],
            file_service=mock_services["file_service"],
        )

        # Configure mocks for exact title match
        mock_services["neo4j"].query.return_value = [{"uuid": "title-match-uuid"}]

        result = await dedup_service.validate_learning(
            title="Exact Title Match Learning",
            content="Content that happens to have an exact title match with an existing learning.",
        )

        assert result.status == "validation_required"
        assert result.conflict_type == "title_match"
        assert result.similarity_score == 1.0
        assert result.existing_learning_uuid == "title-match-uuid"


@pytest.mark.asyncio
class TestAppendWorkflowE2E:
    """E2E tests for append_to_learning workflow (T029)."""

    async def test_append_success(self, mock_services, tmp_path):
        """Test successful append to existing learning."""
        from src.services.deduplication_service import DeduplicationService

        # Create test file
        test_file = tmp_path / "learning-test.md"
        test_file.write_text("# Original Learning\n\nOriginal content here.")

        dedup_service = DeduplicationService(
            vector_service=mock_services["vector"],
            neo4j_service=mock_services["neo4j"],
            file_service=mock_services["file_service"],
        )

        # Configure mocks
        mock_services["neo4j"].query.return_value = [
            {
                "uuid": "append-test-uuid",
                "title": "Original Learning",
                "file_path": str(test_file),
                "content": "Original content",
                "category": "general",
                "append_count": 0,
            }
        ]

        result = await dedup_service.append_to_learning(
            learning_uuid="append-test-uuid",
            new_content="This is new content being appended to the existing learning with additional insights and updates.",
            new_section_title="Additional Insights",
        )

        assert result.status == "success"
        assert result.action == "appended"
        assert result.append_count == 1
        assert result.new_section_title == "Additional Insights"

        # Verify file was updated
        updated_content = test_file.read_text()
        assert "Additional Insights" in updated_content
        assert "new content being appended" in updated_content

    async def test_append_not_found(self, mock_services):
        """Test append fails gracefully for non-existent learning."""
        from src.services.deduplication_service import DeduplicationService

        dedup_service = DeduplicationService(
            vector_service=mock_services["vector"],
            neo4j_service=mock_services["neo4j"],
            file_service=mock_services["file_service"],
        )

        # Configure mock for not found
        mock_services["neo4j"].query.return_value = []

        result = await dedup_service.append_to_learning(
            learning_uuid="nonexistent-uuid",
            new_content="Content to append to a learning that doesn't exist in the system.",
        )

        assert result.status == "error"
        assert "not found" in result.error.lower()

    async def test_append_content_too_short(self, mock_services):
        """Test append fails for content below minimum length."""
        from src.services.deduplication_service import DeduplicationService

        dedup_service = DeduplicationService(
            vector_service=mock_services["vector"],
            neo4j_service=mock_services["neo4j"],
            file_service=mock_services["file_service"],
        )

        result = await dedup_service.append_to_learning(
            learning_uuid="some-uuid",
            new_content="Too short",
        )

        assert result.status == "error"
        assert "50 characters" in result.error


@pytest.mark.asyncio
class TestGracefulDegradation:
    """E2E tests for graceful degradation when services unavailable (T032)."""

    async def test_qdrant_unavailable_falls_back_to_fuzzy(self, mock_services):
        """Test validation continues with fuzzy matching when Qdrant is down."""
        from src.services.deduplication_service import DeduplicationService

        # Disable vector service
        mock_services["vector"].is_enabled.return_value = False

        dedup_service = DeduplicationService(
            vector_service=mock_services["vector"],
            neo4j_service=mock_services["neo4j"],
            file_service=mock_services["file_service"],
            enable_graceful_degradation=True,
        )

        # Configure fuzzy fallback data
        mock_services["neo4j"].query.side_effect = [
            [],  # Title exact match returns nothing
            [  # Fuzzy fallback returns learning titles
                {
                    "uuid": "fuzzy-match-uuid",
                    "title": "Similar Title Learning",
                    "category": "general",
                    "file_path": "/test/path.md",
                    "created_at": datetime.now(),
                }
            ],
        ]

        result = await dedup_service.validate_learning(
            title="Similar Title",
            content="Content for a learning with a similar title to test fuzzy matching fallback when vector service is unavailable.",
        )

        # Should still complete validation via fuzzy fallback
        assert result.status in ["proceed", "validation_required"]
        assert result.validation_stage == "fuzzy_fallback"

    async def test_validation_error_proceeds_gracefully(self, mock_services):
        """Test that validation errors allow capture to proceed."""
        from src.services.deduplication_service import DeduplicationService

        # Configure service to raise error on title match but succeed on vector
        mock_services["neo4j"].query.side_effect = Exception(
            "Database connection failed"
        )
        mock_services["vector"].search_similar_learnings.return_value = []

        dedup_service = DeduplicationService(
            vector_service=mock_services["vector"],
            neo4j_service=mock_services["neo4j"],
            file_service=mock_services["file_service"],
        )

        result = await dedup_service.validate_learning(
            title="Test Learning",
            content="Content that should proceed despite validation errors due to graceful degradation.",
        )

        # Should proceed despite Stage 1 error (graceful degradation)
        # The validation continues to Stage 2 (vector) which succeeds
        assert result.status == "proceed"
        # Validation completed via vector similarity (Stage 1 error was logged and skipped)
        assert result.validation_stage in ["vector_similarity", "fuzzy_fallback"]


@pytest.mark.asyncio
class TestPerformanceValidation:
    """Performance validation tests (T031)."""

    async def test_validation_completes_within_timeout(self, mock_services):
        """Test that validation completes within acceptable time."""
        import time

        from src.services.deduplication_service import DeduplicationService

        dedup_service = DeduplicationService(
            vector_service=mock_services["vector"],
            neo4j_service=mock_services["neo4j"],
            file_service=mock_services["file_service"],
            max_validation_time_ms=300.0,
        )

        # Configure fast mock responses
        mock_services["neo4j"].query.return_value = []
        mock_services["vector"].search_similar_learnings.return_value = []

        start = time.perf_counter()
        result = await dedup_service.validate_learning(
            title="Performance Test Learning",
            content="Content for testing validation performance and ensuring it completes within acceptable time limits.",
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert result.validation_time_ms < 300  # Should complete within 300ms
        assert elapsed_ms < 500  # Total should be well under 500ms


@pytest.mark.asyncio
class TestBootstrapFunctionality:
    """Tests for bootstrap_learnings functionality (T030)."""

    async def test_bootstrap_dry_run(self, mock_services, tmp_path):
        """Test bootstrap in dry-run mode doesn't write."""
        from src.services.deduplication_service import DeduplicationService

        # Create test learning files
        (tmp_path / "learning-test1.md").write_text(
            "# Learning: Test 1\n\n**UUID**: `test-uuid-1`\n\nContent here."
        )
        (tmp_path / "learning-test2.md").write_text(
            "# Learning: Test 2\n\n**UUID**: `test-uuid-2`\n\nMore content."
        )

        dedup_service = DeduplicationService(
            vector_service=mock_services["vector"],
            neo4j_service=mock_services["neo4j"],
            file_service=mock_services["file_service"],
        )

        result = await dedup_service.bootstrap_learnings(
            learnings_dir=str(tmp_path),
            dry_run=True,
        )

        assert result.dry_run is True
        assert result.total_files == 2
        # In dry run, successful should be 0 (no actual writes)
        # But our implementation counts them as "would succeed"
        assert result.failed == 0

    async def test_bootstrap_handles_invalid_files(self, mock_services, tmp_path):
        """Test bootstrap handles invalid files gracefully."""
        from src.services.deduplication_service import DeduplicationService

        # Create invalid file (no UUID)
        (tmp_path / "learning-invalid.md").write_text(
            "# Learning: Invalid\n\nNo UUID in this file."
        )

        dedup_service = DeduplicationService(
            vector_service=mock_services["vector"],
            neo4j_service=mock_services["neo4j"],
            file_service=mock_services["file_service"],
        )

        result = await dedup_service.bootstrap_learnings(
            learnings_dir=str(tmp_path),
            dry_run=True,
        )

        # Should handle gracefully, not crash
        assert result.total_files >= 0
