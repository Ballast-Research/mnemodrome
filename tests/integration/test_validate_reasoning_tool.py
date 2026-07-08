"""
Integration tests for validate_reasoning MCP tool (Feature 028).

Tests the tool endpoint, markdown output, and error handling.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from src.mcp_server import _format_reasoning_validation_result, validate_reasoning
from src.models.reasoning_validation_models import (
    ConflictSummary,
    LearningReference,
    PatternReference,
    ReasoningValidationResult,
)


@pytest.mark.asyncio
class TestValidateReasoningTool:
    """Integration tests for validate_reasoning MCP tool."""

    async def test_tool_registered(self):
        """Test that validate_reasoning tool is callable."""
        assert callable(validate_reasoning)
        # Should have proper function signature
        import inspect

        sig = inspect.signature(validate_reasoning)
        assert "proposed_response" in sig.parameters
        assert "timeout_ms" in sig.parameters

    async def test_tool_llm_unavailable(self):
        """Test graceful handling when LLM service is unavailable."""
        with patch("src.mcp_server.llm_service_singleton", None):
            result = await validate_reasoning("Test response")
            assert isinstance(result, str)
            assert (
                "unavailable" in result.lower() or "not initialized" in result.lower()
            )
            assert not result.startswith("Traceback")  # No exception

    async def test_tool_clean_response(self):
        """Test tool with clean response (no conflicts)."""
        mock_service = AsyncMock()
        mock_result = ReasoningValidationResult(
            validation_status="clean",
            summary="No conflicts or relevant context found.",
            claims_analyzed=2,
            entities_checked=2,
            elapsed_ms=150.0,
        )
        mock_service.validate_response = AsyncMock(return_value=mock_result)

        with (
            patch("src.mcp_server.llm_service_singleton", AsyncMock()),
            patch("src.mcp_server.get_vector_service") as mock_vector,
            patch("src.mcp_server.neo4j_service", AsyncMock()),
            patch("src.mcp_server.redis_service", AsyncMock()),
            patch(
                "src.mcp_server.ReasoningValidationService", return_value=mock_service
            ),
        ):
            result = await validate_reasoning("Clean test response")

            assert isinstance(result, str)
            assert "clean" in result.lower()
            assert "## Summary" in result

    async def test_tool_conflicting_response(self):
        """Test tool with conflicting response."""
        conflict = ConflictSummary(
            conflict_type="exact_contradiction",
            severity=80,
            claim="Mnemodrome RUNS_ON server-02",
            stored_fact="Mnemodrome RUNS_ON hypervisor-01",
            recommendation="warn",
            correction_suggestion="Mnemodrome RUNS_ON hypervisor-01",
        )
        mock_service = AsyncMock()
        mock_result = ReasoningValidationResult(
            validation_status="conflicts_detected",
            conflicts=[conflict],
            summary="1 conflict(s) detected. Review before responding.",
            claims_analyzed=1,
            entities_checked=2,
            elapsed_ms=200.0,
        )
        mock_service.validate_response = AsyncMock(return_value=mock_result)

        with (
            patch("src.mcp_server.llm_service_singleton", AsyncMock()),
            patch("src.mcp_server.get_vector_service") as mock_vector,
            patch("src.mcp_server.neo4j_service", AsyncMock()),
            patch("src.mcp_server.redis_service", AsyncMock()),
            patch(
                "src.mcp_server.ReasoningValidationService", return_value=mock_service
            ),
        ):
            result = await validate_reasoning("Conflicting response")

            assert isinstance(result, str)
            assert (
                "conflicts_detected" in result.lower() or "conflict" in result.lower()
            )
            assert "## Conflicts Detected" in result

    async def test_tool_timeout_graceful(self):
        """Test that very low timeout doesn't raise exception."""
        mock_service = AsyncMock()
        mock_result = ReasoningValidationResult(
            validation_status="clean",
            summary="Claim extraction timed out",
            elapsed_ms=1.0,
        )
        mock_service.validate_response = AsyncMock(return_value=mock_result)

        with (
            patch("src.mcp_server.llm_service_singleton", AsyncMock()),
            patch("src.mcp_server.get_vector_service") as mock_vector,
            patch("src.mcp_server.neo4j_service", AsyncMock()),
            patch("src.mcp_server.redis_service", AsyncMock()),
            patch(
                "src.mcp_server.ReasoningValidationService", return_value=mock_service
            ),
        ):
            # Should not raise exception even with 1ms timeout
            result = await validate_reasoning("Test response", timeout_ms=1)

            assert isinstance(result, str)
            assert not result.startswith("Traceback")

    async def test_tool_output_markdown_structure(self):
        """Test that output contains all expected markdown sections."""
        learning = LearningReference(
            uuid="l-123",
            title="Test Learning",
            similarity_score=0.85,
        )
        pattern = PatternReference(
            pattern_uuid="p-123",
            name="Test Pattern",
            pattern_type="diagnostic",
            similarity_score=0.9,
            success_rate=0.85,
            occurrence_count=3,
            relevance_explanation="Good match",
        )
        mock_service = AsyncMock()
        mock_result = ReasoningValidationResult(
            validation_status="reinforced",
            relevant_learnings=[learning],
            relevant_patterns=[pattern],
            summary="Response consistent with 1 learning(s) and 1 pattern(s).",
            claims_analyzed=2,
            entities_checked=2,
            elapsed_ms=180.0,
        )
        mock_service.validate_response = AsyncMock(return_value=mock_result)

        with (
            patch("src.mcp_server.llm_service_singleton", AsyncMock()),
            patch("src.mcp_server.get_vector_service") as mock_vector,
            patch("src.mcp_server.neo4j_service", AsyncMock()),
            patch("src.mcp_server.redis_service", AsyncMock()),
            patch(
                "src.mcp_server.ReasoningValidationService", return_value=mock_service
            ),
        ):
            result = await validate_reasoning("Test response")

            # Check for all expected sections
            assert "# Reasoning Validation Result" in result
            assert "## Summary" in result
            assert "## Conflicts Detected" in result
            assert "## Relevant Learnings" in result
            assert "## Relevant Patterns" in result
            assert "## Diagnostics" in result


# ============================================================================
# Formatter Tests
# ============================================================================


class TestReasoningValidationFormatter:
    """Tests for _format_reasoning_validation_result formatter."""

    def test_formatter_clean_status(self):
        """Test formatter with clean status."""
        result = ReasoningValidationResult(
            validation_status="clean",
            summary="No conflicts or relevant context found.",
            claims_analyzed=0,
            entities_checked=0,
            elapsed_ms=50.0,
        )

        markdown = _format_reasoning_validation_result(result)

        assert "# Reasoning Validation Result" in markdown
        assert "clean" in markdown
        assert "## Summary" in markdown
        assert "## Conflicts Detected (0)" in markdown

    def test_formatter_with_conflicts(self):
        """Test formatter with conflicts."""
        conflict = ConflictSummary(
            conflict_type="exact_contradiction",
            severity=75,
            claim="Mnemodrome RUNS_ON server-02",
            stored_fact="Mnemodrome RUNS_ON hypervisor-01",
            recommendation="auto_correct",
            correction_suggestion="Mnemodrome RUNS_ON hypervisor-01",
        )
        result = ReasoningValidationResult(
            validation_status="conflicts_detected",
            conflicts=[conflict],
            summary="1 conflict(s) detected. Review before responding.",
            claims_analyzed=1,
            entities_checked=1,
            elapsed_ms=200.0,
        )

        markdown = _format_reasoning_validation_result(result)

        assert "conflicts_detected" in markdown
        assert "## Conflicts Detected (1)" in markdown
        assert "### Conflict 1" in markdown
        assert "exact_contradiction" in markdown
        assert "75/100" in markdown
        assert "Correction:" in markdown

    def test_formatter_without_correction_suggestion(self):
        """Test that Correction line is omitted when None."""
        conflict = ConflictSummary(
            conflict_type="partial_mismatch",
            severity=35,
            claim="Test claim",
            stored_fact="Stored fact",
            recommendation="skip",
            correction_suggestion=None,
        )
        result = ReasoningValidationResult(
            validation_status="conflicts_detected",
            conflicts=[conflict],
        )

        markdown = _format_reasoning_validation_result(result)

        assert "Correction:" not in markdown

    def test_formatter_with_learnings(self):
        """Test formatter with relevant learnings."""
        now = datetime.now(UTC)
        learning = LearningReference(
            uuid="abc-123",
            title="Factory Pattern Guide",
            similarity_score=0.92,
            created_at=now,
        )
        result = ReasoningValidationResult(
            validation_status="reinforced",
            relevant_learnings=[learning],
            summary="Response consistent with 1 learning(s) and 0 pattern(s).",
        )

        markdown = _format_reasoning_validation_result(result)

        assert "## Relevant Learnings (1)" in markdown
        assert "abc-123" in markdown
        assert "Factory Pattern Guide" in markdown
        assert "0.92" in markdown

    def test_formatter_with_patterns(self):
        """Test formatter with relevant patterns."""
        pattern = PatternReference(
            pattern_uuid="p-xyz",
            name="Timeout Handling",
            pattern_type="diagnostic",
            similarity_score=0.88,
            success_rate=0.95,
            occurrence_count=12,
            relevance_explanation="Pattern matches problem well",
        )
        result = ReasoningValidationResult(
            validation_status="reinforced",
            relevant_patterns=[pattern],
            summary="Response consistent with 0 learning(s) and 1 pattern(s).",
        )

        markdown = _format_reasoning_validation_result(result)

        assert "## Relevant Patterns (1)" in markdown
        assert "p-xyz" in markdown
        assert "Timeout Handling" in markdown
        assert "95%" in markdown
        assert "12 occurrences" in markdown

    def test_formatter_diagnostics_section(self):
        """Test that diagnostics section includes all metrics."""
        result = ReasoningValidationResult(
            validation_status="clean",
            claims_analyzed=5,
            entities_checked=3,
            elapsed_ms=187.5,
            degraded_queries=["fact_retrieval"],
        )

        markdown = _format_reasoning_validation_result(result)

        assert "## Diagnostics" in markdown
        assert "Claims analyzed: 5" in markdown
        assert "Entities checked: 3" in markdown
        assert "fact_retrieval" in markdown

    def test_formatter_degraded_queries_empty(self):
        """Test formatter with no degraded queries."""
        result = ReasoningValidationResult(
            validation_status="clean",
            degraded_queries=[],
        )

        markdown = _format_reasoning_validation_result(result)

        assert "Degraded queries: []" in markdown

    def test_formatter_status_and_elapsed(self):
        """Test that status and elapsed time are in header."""
        result = ReasoningValidationResult(
            validation_status="reinforced",
            elapsed_ms=342.75,
        )

        markdown = _format_reasoning_validation_result(result)

        assert "**Status**: reinforced" in markdown
        assert "**Elapsed**: 343ms" in markdown
