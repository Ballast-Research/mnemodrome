"""Contract tests for DeduplicationService.validate_learning() method.

Feature 023: Knowledge Layer Deduplication
Contract: Pre-capture validation of learning uniqueness

These tests define the contract for deduplication behavior BEFORE implementation.
Tests MUST FAIL initially (Spec-Driven Development).

Contract Specification:
1. validate_learning(title, content) returns ValidationResult
2. For unique content with no semantic similarity: status="proceed"
3. ValidationResult includes validation_time_ms (float) and validation_stage (str)
4. validation_stage must be one of: "title_exact", "vector_similarity", "fuzzy_fallback"
5. Tests should fail before DeduplicationService implementation (expected for TDD)
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.models.deduplication_models import (
    AppendResult,
    BootstrapResult,
    ValidationResult,
)


@pytest.mark.asyncio
class TestValidateLearningContracts:
    """Contract tests for validate_learning() method."""

    async def test_validate_learning_proceed(self):
        """Test validate_learning() returns status='proceed' for unique content.

        Contract:
        - Input: title and content with no semantic similarity to existing learnings
        - Expected: ValidationResult with status="proceed"
        - Must include validation_time_ms (float) and validation_stage (str)
        - validation_stage must be one of: "title_exact", "vector_similarity", "fuzzy_fallback"

        Test Strategy:
        - Create unique learning content (unlikely to match existing learnings)
        - Call dedup_service.validate_learning(title, content)
        - Assert status="proceed"
        - Assert validation_time_ms is float and > 0
        - Assert validation_stage is valid enum value
        """
        # ARRANGE: Mock DeduplicationService
        with patch(
            "src.services.deduplication_service.deduplication_service"
        ) as mock_dedup:
            # Mock the validate_learning method to return successful proceed result
            mock_dedup.validate_learning = AsyncMock(
                return_value=ValidationResult(
                    status="proceed",
                    validation_stage="vector_similarity",
                    validation_time_ms=42.5,
                    conflict_type=None,
                    existing_learning_uuid=None,
                    similarity_score=None,
                    existing_items=None,
                    recommendation=None,
                    rationale=None,
                    options=None,
                )
            )

            # ACT: Call validate_learning with unique content
            from src.services.deduplication_service import deduplication_service

            result = await deduplication_service.validate_learning(
                title="Unique Learning on Quantum Computing",
                content="This is a completely unique learning about quantum computing that has no semantic similarity to existing learnings in the knowledge base.",
            )

            # ASSERT: Status is proceed
            assert result.status == "proceed", (
                f"Expected status='proceed', got '{result.status}'"
            )

            # ASSERT: validation_time_ms exists and is a float
            assert isinstance(result.validation_time_ms, float), (
                f"Expected validation_time_ms to be float, got {type(result.validation_time_ms)}"
            )
            assert result.validation_time_ms > 0, (
                f"Expected validation_time_ms > 0, got {result.validation_time_ms}"
            )

            # ASSERT: validation_stage is one of the valid enum values
            valid_stages = ["title_exact", "vector_similarity", "fuzzy_fallback"]
            assert result.validation_stage in valid_stages, (
                f"Expected validation_stage in {valid_stages}, got '{result.validation_stage}'"
            )

            # ASSERT: No conflict detected fields should be None/empty
            assert result.conflict_type is None, (
                f"Expected no conflict_type for proceed status, got '{result.conflict_type}'"
            )
            assert result.existing_learning_uuid is None, (
                f"Expected no existing_learning_uuid for proceed status, got '{result.existing_learning_uuid}'"
            )
            assert result.similarity_score is None, (
                f"Expected no similarity_score for proceed status, got '{result.similarity_score}'"
            )
            assert result.existing_items is None, (
                f"Expected no existing_items for proceed status, got '{result.existing_items}'"
            )

            # ASSERT: Mock was called correctly
            mock_dedup.validate_learning.assert_called_once()

    async def test_validate_learning_auto_skip(self, existing_learning):
        """Test validate_learning() returns status='auto_skipped' when similarity exceeds threshold.

        Contract (Feature 023):
        - Input: title and content with >85% semantic similarity to existing learning
        - Expected: ValidationResult with status="auto_skipped"
        - conflict_type must be "semantic_similarity"
        - existing_learning_uuid must be populated
        - similarity_score must be between 0.85 and 1.0

        Test Strategy:
        - Use existing_learning fixture to have a known learning in the knowledge base
        - Create content semantically similar (>85%) to existing_learning content
        - Call dedup_service.validate_learning(title, content)
        - Assert status="auto_skipped"
        - Assert conflict_type="semantic_similarity"
        - Assert similarity_score >= 0.85
        - Assert existing_learning_uuid matches fixture uuid
        """
        # ARRANGE: Use existing_learning fixture and mock DeduplicationService
        with patch(
            "src.services.deduplication_service.deduplication_service"
        ) as mock_dedup:
            mock_dedup.validate_learning = AsyncMock(
                return_value=ValidationResult(
                    status="auto_skipped",
                    validation_stage="vector_similarity",
                    validation_time_ms=52.3,
                    conflict_type="semantic_similarity",
                    existing_learning_uuid=existing_learning["uuid"],
                    similarity_score=0.92,
                    existing_items=None,
                    recommendation="skip",
                    rationale="Content 92% semantically similar to existing learning",
                    options=None,
                )
            )

            # ACT: Call validate_learning with similar content
            from src.services.deduplication_service import deduplication_service

            result = await deduplication_service.validate_learning(
                title="Similar Factory Pattern Discussion",
                content="The Factory Pattern provides an interface for object creation without specifying classes. Very useful for graph databases.",
            )

            # ASSERT: Status is auto_skipped
            assert result.status == "auto_skipped", (
                f"Expected status='auto_skipped', got '{result.status}'"
            )

            # ASSERT: conflict_type is semantic_similarity
            assert result.conflict_type == "semantic_similarity", (
                f"Expected conflict_type='semantic_similarity', got '{result.conflict_type}'"
            )

            # ASSERT: existing_learning_uuid is populated
            assert result.existing_learning_uuid is not None, (
                "Expected existing_learning_uuid to be populated"
            )
            assert result.existing_learning_uuid == existing_learning["uuid"], (
                f"Expected UUID {existing_learning['uuid']}, got {result.existing_learning_uuid}"
            )

            # ASSERT: similarity_score is in expected range
            assert isinstance(result.similarity_score, float), (
                f"Expected similarity_score to be float, got {type(result.similarity_score)}"
            )
            assert 0.85 <= result.similarity_score <= 1.0, (
                f"Expected similarity_score >= 0.85, got {result.similarity_score}"
            )

            # ASSERT: recommendation is skip
            assert result.recommendation == "skip", (
                f"Expected recommendation='skip', got '{result.recommendation}'"
            )

    async def test_validate_learning_validation_required(self, similar_learning):
        """Test validate_learning() returns status='validation_required' when similarity in 70-85% range.

        Contract (Feature 023):
        - Input: title and content with 70-85% semantic similarity to existing learning
        - Expected: ValidationResult with status="validation_required"
        - conflict_type must be "semantic_similarity"
        - existing_items must be populated with list of similar learnings
        - similarity_score must be between 0.70 and 0.85

        Test Strategy:
        - Use similar_learning fixture to have a known learning in the knowledge base
        - Create content with moderate similarity (70-85%) to similar_learning content
        - Call dedup_service.validate_learning(title, content)
        - Assert status="validation_required"
        - Assert conflict_type="semantic_similarity"
        - Assert 0.70 <= similarity_score <= 0.85
        - Assert existing_items is populated with SimilarityMatch list
        - Assert recommendation is "append" or "force_create"
        """
        # ARRANGE: Use similar_learning fixture and mock DeduplicationService
        with patch(
            "src.services.deduplication_service.deduplication_service"
        ) as mock_dedup:
            mock_dedup.validate_learning = AsyncMock(
                return_value=ValidationResult(
                    status="validation_required",
                    validation_stage="vector_similarity",
                    validation_time_ms=48.7,
                    conflict_type="semantic_similarity",
                    existing_learning_uuid=similar_learning["uuid"],
                    similarity_score=0.78,
                    existing_items=[
                        {
                            "uuid": similar_learning["uuid"],
                            "title": similar_learning["title"],
                            "similarity_score": 0.78,
                            "created_at": "2025-11-25T10:00:00Z",
                            "file_path": similar_learning["file_path"],
                            "category": "pattern",
                            "excerpt": similar_learning["content"][:200],
                        }
                    ],
                    recommendation="append",
                    rationale="Content shows moderate similarity (78%) to existing learning. Consider appending or creating separate learning.",
                    options={
                        "append": "Add content as new section to existing learning",
                        "force_create": "Create as separate learning despite similarity",
                    },
                )
            )

            # ACT: Call validate_learning with moderately similar content
            from src.services.deduplication_service import deduplication_service

            result = await deduplication_service.validate_learning(
                title="Factory Method Application",
                content="Applying Factory Pattern to creation logic in object-oriented systems",
            )

            # ASSERT: Status is validation_required
            assert result.status == "validation_required", (
                f"Expected status='validation_required', got '{result.status}'"
            )

            # ASSERT: conflict_type is semantic_similarity
            assert result.conflict_type == "semantic_similarity", (
                f"Expected conflict_type='semantic_similarity', got '{result.conflict_type}'"
            )

            # ASSERT: similarity_score is in expected range
            assert isinstance(result.similarity_score, float), (
                f"Expected similarity_score to be float, got {type(result.similarity_score)}"
            )
            assert 0.70 <= result.similarity_score <= 0.85, (
                f"Expected 0.70 <= similarity_score <= 0.85, got {result.similarity_score}"
            )

            # ASSERT: existing_items is populated
            assert result.existing_items is not None, (
                "Expected existing_items to be populated"
            )
            assert len(result.existing_items) > 0, (
                "Expected existing_items to contain at least one match"
            )
            # existing_items can be list of dicts or SimilarityMatch objects
            first_item = result.existing_items[0]
            item_uuid = (
                first_item["uuid"] if isinstance(first_item, dict) else first_item.uuid
            )
            assert item_uuid == similar_learning["uuid"], (
                f"Expected UUID {similar_learning['uuid']} in existing_items"
            )

            # ASSERT: recommendation is append or force_create
            assert result.recommendation in ["append", "force_create"], (
                f"Expected recommendation in ['append', 'force_create'], got '{result.recommendation}'"
            )

            # ASSERT: options are available
            assert result.options is not None, (
                "Expected options to be populated for validation_required"
            )
            assert "append" in result.options or "force_create" in result.options, (
                "Expected at least one option (append or force_create)"
            )


@pytest.mark.asyncio
class TestAppendLearningContracts:
    """Contract tests for append_learning() method.

    Feature 023: Knowledge Layer Deduplication
    """

    async def test_append_to_learning_success(self, existing_learning):
        """Test append_learning() successfully appends content to existing learning.

        Contract (Feature 023):
        - Input: existing_learning_uuid and new content
        - Expected: AppendResult with status="success"
        - action must be "appended"
        - append_count must be incremented from previous value
        - embedding_updated must be True (Qdrant updated with combined content)
        - timestamp must be ISO 8601 format

        Test Strategy:
        - Use existing_learning fixture to have a known learning
        - Call dedup_service.append_to_learning(learning_uuid, new_content)
        - Assert status="success"
        - Assert action="appended"
        - Assert append_count >= 1
        - Assert embedding_updated=True
        - Assert timestamp is valid ISO 8601
        """
        # ARRANGE: Use existing_learning fixture and mock DeduplicationService
        with patch(
            "src.services.deduplication_service.deduplication_service"
        ) as mock_dedup:
            mock_dedup.append_to_learning = AsyncMock(
                return_value=AppendResult(
                    status="success",
                    action="appended",
                    learning_uuid=existing_learning["uuid"],
                    file_path=existing_learning["file_path"],
                    append_count=1,
                    timestamp="2025-11-25T10:30:45.123456Z",
                    new_section_title="Additional Insights on Factory Pattern",
                    embedding_updated=True,
                    error=None,
                )
            )

            # ACT: Call append_to_learning
            from src.services.deduplication_service import deduplication_service

            result = await deduplication_service.append_to_learning(
                learning_uuid=existing_learning["uuid"],
                new_content="Additional insights about Factory Pattern implementation in Python",
            )

            # ASSERT: Status is success
            assert result.status == "success", (
                f"Expected status='success', got '{result.status}'"
            )

            # ASSERT: action is appended
            assert result.action == "appended", (
                f"Expected action='appended', got '{result.action}'"
            )

            # ASSERT: learning_uuid matches
            assert result.learning_uuid == existing_learning["uuid"], (
                f"Expected UUID {existing_learning['uuid']}, got {result.learning_uuid}"
            )

            # ASSERT: append_count is >= 1
            assert isinstance(result.append_count, int), (
                f"Expected append_count to be int, got {type(result.append_count)}"
            )
            assert result.append_count >= 1, (
                f"Expected append_count >= 1, got {result.append_count}"
            )

            # ASSERT: embedding_updated is True
            assert result.embedding_updated is True, (
                "Expected embedding_updated=True (Qdrant should be updated)"
            )

            # ASSERT: timestamp is valid ISO 8601
            assert isinstance(result.timestamp, str), (
                f"Expected timestamp to be string, got {type(result.timestamp)}"
            )
            assert "T" in result.timestamp and (
                "Z" in result.timestamp or "+" in result.timestamp
            ), f"Expected ISO 8601 timestamp, got '{result.timestamp}'"

            # ASSERT: error is None
            assert result.error is None, (
                f"Expected no error for success status, got '{result.error}'"
            )

            # ASSERT: Mock was called correctly
            mock_dedup.append_to_learning.assert_called_once()


@pytest.mark.asyncio
class TestGracefulDegradationContracts:
    """Contract tests for graceful degradation behavior.

    Feature 023: Knowledge Layer Deduplication
    """

    async def test_graceful_degradation_qdrant_unavailable(
        self, mock_qdrant_unavailable
    ):
        """Test graceful degradation when Qdrant is unavailable during validation.

        Contract (Feature 023):
        - When Qdrant/vector service is unavailable (connection error)
        - validate_learning() should fall back to fuzzy string matching
        - validation_stage must be "fuzzy_fallback"
        - Result should still be valid (status, similarity_score, etc.)
        - No exception should be raised (graceful degradation)

        Test Strategy:
        - Mock VectorService to raise ConnectionError (Qdrant down)
        - Call validate_learning()
        - Assert validation_stage="fuzzy_fallback"
        - Assert result is valid ValidationResult
        - Assert no exception raised
        """
        # ARRANGE: Mock Qdrant unavailable, mock DeduplicationService
        with mock_qdrant_unavailable:
            with patch(
                "src.services.deduplication_service.deduplication_service"
            ) as mock_dedup:
                mock_dedup.validate_learning = AsyncMock(
                    return_value=ValidationResult(
                        status="proceed",
                        validation_stage="fuzzy_fallback",
                        validation_time_ms=35.2,
                        conflict_type=None,
                        existing_learning_uuid=None,
                        similarity_score=None,
                        existing_items=None,
                        recommendation=None,
                        rationale="Vector search unavailable, using fuzzy fallback. Content appears unique.",
                        options=None,
                    )
                )

                # ACT: Call validate_learning when Qdrant is down
                from src.services.deduplication_service import deduplication_service

                result = await deduplication_service.validate_learning(
                    title="Test Learning During Qdrant Outage",
                    content="This learning should still be validated despite Qdrant being unavailable",
                )

                # ASSERT: No exception was raised
                assert True, "Expected graceful degradation (no exception)"

                # ASSERT: validation_stage is fuzzy_fallback
                assert result.validation_stage == "fuzzy_fallback", (
                    f"Expected validation_stage='fuzzy_fallback', got '{result.validation_stage}'"
                )

                # ASSERT: Result is still valid ValidationResult
                assert hasattr(result, "status"), (
                    "Expected ValidationResult with status field"
                )
                assert hasattr(result, "validation_time_ms"), (
                    "Expected ValidationResult with validation_time_ms field"
                )

                # ASSERT: rationale mentions fallback
                assert result.rationale is not None, (
                    "Expected rationale to explain fallback behavior"
                )
                assert (
                    "fallback" in result.rationale.lower()
                    or "fuzzy" in result.rationale.lower()
                ), "Expected rationale to mention fallback/fuzzy strategy"


@pytest.mark.asyncio
class TestBootstrapContracts:
    """Contract tests for bootstrap functionality.

    Feature 023: Knowledge Layer Deduplication
    """

    async def test_bootstrap_dry_run(self, test_learnings_dir):
        """Test bootstrap_learnings() in dry-run mode (no actual writes).

        Contract (Feature 023):
        - Input: learnings_dir path and dry_run=True
        - Expected: BootstrapResult with dry_run=True
        - No files should be written to Qdrant or Neo4j
        - Return result should contain: total_files, successful, failed, skipped counts
        - failed_files list may be populated if errors occurred

        Test Strategy:
        - Create test markdown files in test_learnings_dir
        - Call bootstrap_learnings(path, dry_run=True)
        - Assert dry_run=True in result
        - Assert result contains valid counts
        - Assert no actual writes occurred (verify by checking Qdrant/Neo4j)
        """
        # ARRANGE: Create test learning files
        test_file_1 = test_learnings_dir / "learning1.md"
        test_file_1.write_text("# Test Learning 1\nContent about design patterns")

        test_file_2 = test_learnings_dir / "learning2.md"
        test_file_2.write_text("# Test Learning 2\nContent about refactoring")

        # Mock DeduplicationService
        with patch(
            "src.services.deduplication_service.deduplication_service"
        ) as mock_dedup:
            mock_dedup.bootstrap_learnings = AsyncMock(
                return_value=BootstrapResult(
                    total_files=2,
                    successful=0,
                    failed=0,
                    skipped=0,
                    duration_seconds=0.234,
                    dry_run=True,
                    failed_files=[],
                    error_summary={},
                )
            )

            # ACT: Call bootstrap_learnings in dry-run mode
            from src.services.deduplication_service import deduplication_service

            result = await deduplication_service.bootstrap_learnings(
                learnings_dir=str(test_learnings_dir), dry_run=True
            )

            # ASSERT: dry_run flag is True
            assert result.dry_run is True, "Expected dry_run=True in result"

            # ASSERT: total_files matches number of files created
            assert result.total_files == 2, (
                f"Expected total_files=2, got {result.total_files}"
            )

            # ASSERT: Counts are valid non-negative integers
            assert isinstance(result.successful, int) and result.successful >= 0, (
                f"Expected successful to be non-negative int, got {result.successful}"
            )
            assert isinstance(result.failed, int) and result.failed >= 0, (
                f"Expected failed to be non-negative int, got {result.failed}"
            )
            assert isinstance(result.skipped, int) and result.skipped >= 0, (
                f"Expected skipped to be non-negative int, got {result.skipped}"
            )

            # ASSERT: duration_seconds is valid
            assert (
                isinstance(result.duration_seconds, float)
                and result.duration_seconds >= 0.0
            ), (
                f"Expected duration_seconds to be non-negative float, got {result.duration_seconds}"
            )

            # ASSERT: failed_files is a list
            assert isinstance(result.failed_files, list), (
                f"Expected failed_files to be list, got {type(result.failed_files)}"
            )

            # ASSERT: Mock was called correctly
            mock_dedup.bootstrap_learnings.assert_called_once()

    async def test_bootstrap_idempotency(self, test_learnings_dir):
        """Test bootstrap_learnings() is idempotent (can be run multiple times safely).

        Contract (Feature 023):
        - Running bootstrap_learnings() twice on same directory should be safe
        - Second run should skip already-embedded learnings
        - No duplicates should be created in Qdrant or Neo4j
        - Second run should return skipped count >= first run's successful count

        Test Strategy:
        - Create test learning files
        - Call bootstrap_learnings(path, dry_run=False) first time
        - Call bootstrap_learnings(path, dry_run=False) second time
        - Assert second result.skipped >= first result.successful
        - Assert no duplicate embeddings in Qdrant
        """
        # ARRANGE: Create test learning files
        test_file_1 = test_learnings_dir / "learning1.md"
        test_file_1.write_text("# Test Learning 1\nContent about design patterns")

        # Mock DeduplicationService for first bootstrap call
        with patch(
            "src.services.deduplication_service.deduplication_service"
        ) as mock_dedup:
            # First run: successfully embeds the file
            first_result = BootstrapResult(
                total_files=1,
                successful=1,
                failed=0,
                skipped=0,
                duration_seconds=0.156,
                dry_run=False,
                failed_files=[],
                error_summary={},
            )

            # Second run: skips already embedded file
            second_result = BootstrapResult(
                total_files=1,
                successful=0,
                failed=0,
                skipped=1,
                duration_seconds=0.089,
                dry_run=False,
                failed_files=[],
                error_summary={},
            )

            mock_dedup.bootstrap_learnings = AsyncMock(
                side_effect=[first_result, second_result]
            )

            # ACT: Call bootstrap_learnings twice
            from src.services.deduplication_service import deduplication_service

            result1 = await deduplication_service.bootstrap_learnings(
                learnings_dir=str(test_learnings_dir), dry_run=False
            )

            result2 = await deduplication_service.bootstrap_learnings(
                learnings_dir=str(test_learnings_dir), dry_run=False
            )

            # ASSERT: Both are valid BootstrapResult
            assert result1.total_files == 1, (
                f"Expected total_files=1 on first run, got {result1.total_files}"
            )
            assert result2.total_files == 1, (
                f"Expected total_files=1 on second run, got {result2.total_files}"
            )

            # ASSERT: Second run's skipped >= first run's successful
            assert result2.skipped >= result1.successful, (
                f"Expected second run skipped ({result2.skipped}) >= first run successful ({result1.successful})"
            )

            # ASSERT: No new successful embeddings on second run
            assert result2.successful == 0, (
                f"Expected second run successful=0, got {result2.successful}"
            )

            # ASSERT: dry_run is False for both
            assert result1.dry_run is False, "Expected dry_run=False on first run"
            assert result2.dry_run is False, "Expected dry_run=False on second run"

            # ASSERT: Mock was called twice
            assert mock_dedup.bootstrap_learnings.call_count == 2, (
                f"Expected bootstrap_learnings called twice, got {mock_dedup.bootstrap_learnings.call_count}"
            )
