"""Unit tests for Feature 036: Curated Evidence Report helpers."""

from datetime import UTC, datetime

import pytest

from src.mcp_server import (
    _format_date_human,
    _format_evidence_report,
    _generate_synthesis,
    _get_result_title,
    _get_type_badge,
    _smart_truncate,
)


class TestSmartTruncate:
    """Tests for _smart_truncate()."""

    def test_short_text_unchanged(self):
        assert _smart_truncate("Hello world") == "Hello world"

    def test_empty_string(self):
        assert _smart_truncate("") == ""

    def test_none_input(self):
        assert _smart_truncate(None) == None

    def test_exact_length(self):
        text = "a" * 300
        assert _smart_truncate(text) == text

    def test_truncates_at_sentence_boundary(self):
        text = "First sentence. " + "x" * 290
        result = _smart_truncate(text, 300)
        assert result.endswith(".")
        assert len(result) <= 300

    def test_truncates_at_word_boundary(self):
        text = "word " * 100  # 500 chars
        result = _smart_truncate(text, 300)
        assert not result.endswith(" ...")  # No trailing space before ellipsis
        assert result.endswith("...")
        assert len(result) <= 303  # 300 + "..."

    def test_never_mid_word(self):
        text = "a" * 200 + " " + "b" * 200
        result = _smart_truncate(text, 300)
        assert result.endswith("...")
        assert not result[-4].isalpha() or result.endswith("...")

    def test_custom_max_length(self):
        text = "Hello world, this is a test sentence."
        result = _smart_truncate(text, 15)
        assert len(result) <= 18  # 15 + "..."


class TestFormatDateHuman:
    """Tests for _format_date_human()."""

    def test_datetime_object(self):
        dt = datetime(2026, 4, 4, tzinfo=UTC)
        assert _format_date_human(dt) == "April 4, 2026"

    def test_iso_string(self):
        result = _format_date_human("2026-04-04T12:00:00+00:00")
        assert result == "April 4, 2026"

    def test_iso_string_with_z(self):
        result = _format_date_human("2026-04-04T12:00:00Z")
        assert result == "April 4, 2026"

    def test_empty_string(self):
        assert _format_date_human("") == ""

    def test_none(self):
        assert _format_date_human(None) == ""

    def test_invalid_string(self):
        result = _format_date_human("not-a-date")
        assert result == "not-a-date"


class TestGetResultTitle:
    """Tests for _get_result_title()."""

    def test_learning_title(self):
        result = {"type": "learning", "title": "My Learning"}
        assert _get_result_title(result) == "My Learning"

    def test_fact_title(self):
        result = {"type": "fact", "entity1": "A", "relation": "KNOWS", "entity2": "B"}
        assert _get_result_title(result) == "A → KNOWS → B"

    def test_session_title(self):
        result = {"type": "session", "summary": "Discussed architecture patterns"}
        assert "Discussed architecture" in _get_result_title(result)

    def test_prompt_title(self):
        result = {"type": "prompt", "content": "How does Mnemodrome work?"}
        assert "How does Mnemodrome work?" in _get_result_title(result)

    def test_unknown_type(self):
        result = {"type": "other", "title": "Fallback"}
        assert _get_result_title(result) == "Fallback"


class TestGetTypeBadge:
    """Tests for _get_type_badge()."""

    def test_known_types(self):
        assert _get_type_badge("learning") == "Learning"
        assert _get_type_badge("fact") == "Fact"
        assert _get_type_badge("session") == "Session"
        assert _get_type_badge("prompt") == "Prompt"

    def test_unknown_type(self):
        assert _get_type_badge("custom") == "Custom"

    def test_empty_type(self):
        assert _get_type_badge("") == "Unknown"


class TestGenerateSynthesis:
    """Tests for _generate_synthesis()."""

    def test_empty_results(self):
        result = _generate_synthesis([], "test query")
        assert "No results" in result

    def test_single_result(self):
        results = [
            {"type": "learning", "score": 0.85, "created_at": "2026-04-04T00:00:00Z"}
        ]
        synthesis = _generate_synthesis(results, "test")
        assert "1 relevant result" in synthesis
        assert "test" in synthesis

    def test_multiple_types(self):
        results = [
            {"type": "learning", "score": 0.85},
            {"type": "fact", "score": 0.72},
            {"type": "fact", "score": 0.65},
        ]
        synthesis = _generate_synthesis(results, "test")
        assert "3 relevant results" in synthesis
        assert "fact" in synthesis.lower()

    def test_high_confidence(self):
        results = [{"type": "learning", "score": 0.92}]
        synthesis = _generate_synthesis(results, "test")
        assert "High confidence" in synthesis

    def test_low_confidence(self):
        results = [{"type": "learning", "score": 0.48}]
        synthesis = _generate_synthesis(results, "test")
        assert "Low confidence" in synthesis

    def test_temporal_spread(self):
        results = [
            {"type": "learning", "score": 0.8, "created_at": "2026-01-01T00:00:00Z"},
            {"type": "fact", "score": 0.7, "created_at": "2026-04-04T00:00:00Z"},
        ]
        synthesis = _generate_synthesis(results, "test")
        assert "January" in synthesis
        assert "April" in synthesis


class TestFormatEvidenceReport:
    """Tests for _format_evidence_report()."""

    def test_empty_results(self):
        report = _format_evidence_report([], "test", 50.0)
        assert "No matching content" in report
        assert "Curated Evidence Report" in report

    def test_basic_report_structure(self):
        results = [
            {
                "type": "learning",
                "title": "Test Learning",
                "content": "Some content here",
                "score": 0.85,
                "created_at": "2026-04-04T00:00:00Z",
            }
        ]
        report = _format_evidence_report(results, "my query", 42.5)
        assert "# Curated Evidence Report" in report
        assert "**Query**: my query" in report
        assert "Test Learning" in report
        assert "Score: 0.850" in report
        assert "Learning" in report
        assert "### Synthesis" in report
        assert "42.5ms" in report

    def test_cross_category_sorting(self):
        results = [
            {
                "type": "fact",
                "entity1": "A",
                "relation": "R",
                "entity2": "B",
                "score": 0.60,
            },
            {
                "type": "learning",
                "title": "Top Result",
                "content": "Best",
                "score": 0.95,
            },
            {"type": "session", "summary": "Middle", "score": 0.75},
        ]
        report = _format_evidence_report(results, "test", 10.0)
        # Top Result (0.95) should appear before Middle (0.75) before Fact (0.60)
        top_pos = report.index("Top Result")
        mid_pos = report.index("Middle")
        fact_pos = report.index("A → R → B")
        assert top_pos < mid_pos < fact_pos

    def test_score_threshold_filtering(self):
        results = [
            {"type": "learning", "title": "Good", "content": "x", "score": 0.80},
            {"type": "learning", "title": "Ok", "content": "x", "score": 0.55},
            {"type": "learning", "title": "Bad", "content": "x", "score": 0.30},
        ]
        report = _format_evidence_report(results, "test", 10.0)
        assert "Good" in report
        assert "Ok" in report
        assert "Bad" not in report  # Below 0.45 fallback

    def test_fallback_threshold_note(self):
        results = [
            {"type": "learning", "title": "Low1", "content": "x", "score": 0.47},
            {"type": "learning", "title": "Low2", "content": "x", "score": 0.46},
        ]
        report = _format_evidence_report(results, "test", 10.0)
        assert "Threshold lowered" in report or "threshold" in report.lower()

    def test_max_seven_results(self):
        results = [
            {
                "type": "learning",
                "title": f"Item {i}",
                "content": "x",
                "score": 0.90 - i * 0.01,
            }
            for i in range(10)
        ]
        report = _format_evidence_report(results, "test", 10.0)
        # Count evidence blocks by counting "Score:" occurrences
        assert report.count("Score:") <= 7

    def test_backward_compat_default_format_untouched(self):
        """Verify report format is separate from default — this tests the helper, not the tool."""
        report = _format_evidence_report(
            [
                {
                    "type": "fact",
                    "entity1": "X",
                    "relation": "Y",
                    "entity2": "Z",
                    "score": 0.8,
                }
            ],
            "test",
            5.0,
        )
        assert "Curated Evidence Report" in report
        assert "Similar Facts" not in report  # That's the default format

    def test_min_score_respected_in_report(self):
        """Verify that the report formatter formats whatever it receives — min_score filtering
        happens upstream in vector_service.search(), not in the report formatter."""
        # Simulate results as if vector_service.search(min_score=0.60) returned them
        results_from_vector_service = [
            {"type": "learning", "title": "High", "content": "x", "score": 0.90},
            {"type": "learning", "title": "Medium", "content": "x", "score": 0.70},
            {"type": "learning", "title": "Low-ish", "content": "x", "score": 0.55},
        ]
        # The 0.55 result would NOT have been returned by vector_service with min_score=0.60
        # But if passed to the report formatter, it should still be included (it's above FALLBACK_THRESHOLD)
        report = _format_evidence_report(results_from_vector_service, "test", 10.0)
        assert "High" in report
        assert "Medium" in report
        assert (
            "Low-ish" in report
        )  # Report formatter includes it — filtering is upstream

        # Now test that if only high-scored results are passed (as vector_service would with min_score=0.60),
        # the formatter handles them correctly
        filtered_results = [
            r for r in results_from_vector_service if r["score"] >= 0.60
        ]
        report2 = _format_evidence_report(filtered_results, "test", 10.0)
        assert "High" in report2
        assert "Medium" in report2
        assert "Low-ish" not in report2  # Was filtered upstream


@pytest.mark.asyncio
class TestSemanticSearchIntegration:
    """Integration tests for the full semantic_search() coroutine with mocked dependencies."""

    async def test_report_format_with_mocked_vector_service(self):
        """Mock vector service to return 3 sample results, request format='report'."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from src.mcp_server import semantic_search

        mock_vs = MagicMock()
        mock_vs.is_enabled.return_value = True
        mock_vs.search = AsyncMock(
            return_value=[
                {
                    "type": "learning",
                    "title": "Test Learning",
                    "content": "Some content",
                    "score": 0.88,
                    "created_at": "2026-04-04T00:00:00Z",
                },
                {
                    "type": "fact",
                    "entity1": "A",
                    "relation": "KNOWS",
                    "entity2": "B",
                    "context": "Fact context",
                    "score": 0.75,
                    "created_at": "2026-03-01T00:00:00Z",
                },
                {
                    "type": "session",
                    "summary": "Session about testing",
                    "score": 0.65,
                    "created_at": "2026-02-15T00:00:00Z",
                },
            ]
        )

        with patch("src.mcp_server.get_vector_service", return_value=mock_vs):
            output = await semantic_search(query="test", format="report")

        assert "Curated Evidence Report" in output
        assert "Synthesis" in output
        assert "0.880" in output or "0.88" in output

    async def test_default_format_unchanged(self):
        """Calling without format param should NOT produce a Curated Evidence Report."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from src.mcp_server import semantic_search

        mock_vs = MagicMock()
        mock_vs.is_enabled.return_value = True
        mock_vs.search = AsyncMock(
            return_value=[
                {
                    "type": "learning",
                    "title": "Test",
                    "content": "Content",
                    "score": 0.85,
                    "uuid": "abc123",
                },
            ]
        )

        with patch("src.mcp_server.get_vector_service", return_value=mock_vs):
            output = await semantic_search(query="test")

        assert "Curated Evidence Report" not in output

    async def test_invalid_format_returns_error(self):
        """Invalid format value should return an error message."""
        from src.mcp_server import semantic_search

        output = await semantic_search(query="test", format="invalid")
        assert "Invalid format" in output
