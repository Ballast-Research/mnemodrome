"""Unit tests for PatternExtractionService."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.models.pattern_models import ExtractedPattern
from src.services.pattern_extraction_service import PatternExtractionService


@pytest.fixture
def pattern_extraction_service():
    """Provide PatternExtractionService with mocked dependencies."""
    with patch("src.services.pattern_extraction_service.LLMService"):
        service = PatternExtractionService()
        service.llm_service = AsyncMock()
        return service


class TestPatternExtractionServiceInit:
    """Test service initialization."""

    def test_init_creates_service_with_defaults(self):
        """Test service initializes with default configuration."""
        with patch("src.services.pattern_extraction_service.LLMService"):
            with patch(
                "src.services.pattern_extraction_service.PATTERN_RECOGNITION_ENABLED",
                True,
            ):
                service = PatternExtractionService()
                assert service.enabled is True
                assert service.llm_service is not None

    def test_init_respects_disabled_config(self):
        """Test service respects PATTERN_RECOGNITION_ENABLED configuration."""
        with patch("src.services.pattern_extraction_service.LLMService"):
            with patch(
                "src.services.pattern_extraction_service.PATTERN_RECOGNITION_ENABLED",
                False,
            ):
                service = PatternExtractionService()
                assert service.enabled is False


class TestExtractPatternsFromSessions:
    """Test pattern extraction from multiple sessions."""

    @pytest.mark.asyncio
    async def test_extract_returns_empty_when_disabled(
        self, pattern_extraction_service
    ):
        """Test extraction returns empty list when disabled."""
        pattern_extraction_service.enabled = False

        result = await pattern_extraction_service.extract_patterns_from_sessions(
            [{"uuid": "session1", "conversation_summary": "test"}]
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_extract_raises_on_empty_sessions(self, pattern_extraction_service):
        """Test extraction raises ValueError on empty sessions list."""
        pattern_extraction_service.enabled = True

        with pytest.raises(ValueError, match="session_summaries cannot be empty"):
            await pattern_extraction_service.extract_patterns_from_sessions([])

    @pytest.mark.asyncio
    async def test_extract_processes_valid_sessions(self, pattern_extraction_service):
        """Test extraction processes valid sessions successfully."""
        pattern_extraction_service.enabled = True

        # Mock the internal extraction method
        mock_patterns = [
            ExtractedPattern(
                pattern_type="infrastructure",
                problem_description="Database connectivity issue",
                solution_approach="Implemented connection pooling",
                confidence=0.85,
                outcome="SUCCESS",
                session_uuid="session1",
            )
        ]
        pattern_extraction_service._extract_patterns_from_session = AsyncMock(
            return_value=mock_patterns
        )

        sessions = [
            {
                "uuid": "session1",
                "conversation_summary": "Fixed database connectivity",
                "key_insights": ["Connection pooling helps"],
            }
        ]

        result = await pattern_extraction_service.extract_patterns_from_sessions(
            sessions
        )

        assert len(result) == 1
        assert result[0].outcome == "SUCCESS"
        assert result[0].pattern_type == "infrastructure"

    @pytest.mark.asyncio
    async def test_extract_handles_session_errors_gracefully(
        self, pattern_extraction_service
    ):
        """Test extraction continues on individual session errors."""
        pattern_extraction_service.enabled = True

        # Mock one success, one failure
        async def side_effect(session):
            if session.get("uuid") == "session2":
                raise Exception("LLM error")
            return [
                ExtractedPattern(
                    pattern_type="code",
                    problem_description="Test problem",
                    solution_approach="Test solution",
                    confidence=0.8,
                    outcome="SUCCESS",
                    session_uuid="session1",
                )
            ]

        pattern_extraction_service._extract_patterns_from_session = AsyncMock(
            side_effect=side_effect
        )

        sessions = [
            {"uuid": "session1", "conversation_summary": "Summary 1"},
            {"uuid": "session2", "conversation_summary": "Summary 2"},
        ]

        result = await pattern_extraction_service.extract_patterns_from_sessions(
            sessions
        )

        # Should have one successful pattern despite error
        assert len(result) == 1


class TestBuildExtractionPrompt:
    """Test extraction prompt building."""

    def test_builds_valid_extraction_prompt(self, pattern_extraction_service):
        """Test prompt is built with correct structure."""
        session_text = "Implemented caching strategy for API"

        prompt = pattern_extraction_service._build_extraction_prompt(session_text)

        assert "SESSION SUMMARY:" in prompt
        assert "pattern_type" in prompt
        assert "problem_description" in prompt
        assert "solution_approach" in prompt
        assert "outcome" in prompt
        assert "confidence" in prompt
        assert "SUCCESS|FAILED|PARTIAL|INCONCLUSIVE" in prompt

    def test_prompt_includes_session_text(self, pattern_extraction_service):
        """Test prompt includes provided session text."""
        session_text = "Unique session content"

        prompt = pattern_extraction_service._build_extraction_prompt(session_text)

        assert session_text in prompt


class TestParseExtractionResponse:
    """Test LLM response parsing."""

    def test_parses_valid_json_response(self, pattern_extraction_service):
        """Test parsing valid JSON array response."""
        response = json.dumps(
            [
                {
                    "pattern_type": "infrastructure",
                    "problem_description": "Database latency",
                    "solution_approach": "Added indexing",
                    "outcome": "SUCCESS",
                    "confidence": 0.9,
                    "metrics": {"latency_improvement": 40},
                }
            ]
        )

        patterns = pattern_extraction_service._parse_extraction_response(
            response, "session1"
        )

        assert len(patterns) == 1
        assert patterns[0].pattern_type == "infrastructure"
        assert patterns[0].confidence == 0.9

    def test_parses_markdown_wrapped_json(self, pattern_extraction_service):
        """Test parsing JSON wrapped in markdown code blocks."""
        response = """```json
[
    {
        "pattern_type": "code",
        "problem_description": "Memory leak in service",
        "solution_approach": "Implemented garbage collection",
        "outcome": "SUCCESS",
        "confidence": 0.75,
        "metrics": null
    }
]
```"""

        patterns = pattern_extraction_service._parse_extraction_response(
            response, "session1"
        )

        assert len(patterns) == 1
        assert patterns[0].pattern_type == "code"

    def test_filters_low_confidence_patterns(self, pattern_extraction_service):
        """Test low confidence patterns are filtered out."""
        response = json.dumps(
            [
                {
                    "pattern_type": "testing",
                    "problem_description": "Test flakiness",
                    "solution_approach": "Added retries",
                    "outcome": "PARTIAL",
                    "confidence": 0.3,  # Below 0.5 threshold
                },
                {
                    "pattern_type": "architecture",
                    "problem_description": "Scalability issue",
                    "solution_approach": "Microservices refactor",
                    "outcome": "SUCCESS",
                    "confidence": 0.95,
                },
            ]
        )

        patterns = pattern_extraction_service._parse_extraction_response(
            response, "session1"
        )

        # Only high confidence pattern should be included
        assert len(patterns) == 1
        assert patterns[0].confidence == 0.95

    def test_handles_invalid_json(self, pattern_extraction_service):
        """Test graceful handling of invalid JSON."""
        response = "{ invalid json ]"

        patterns = pattern_extraction_service._parse_extraction_response(
            response, "session1"
        )

        assert patterns == []

    def test_handles_empty_array_response(self, pattern_extraction_service):
        """Test handling of empty JSON array."""
        response = "[]"

        patterns = pattern_extraction_service._parse_extraction_response(
            response, "session1"
        )

        assert patterns == []

    def test_handles_non_array_response(self, pattern_extraction_service):
        """Test handling of non-array JSON response."""
        response = json.dumps({"error": "unexpected format"})

        patterns = pattern_extraction_service._parse_extraction_response(
            response, "session1"
        )

        assert patterns == []


class TestCalculateConfidenceScore:
    """Test confidence score calculation."""

    def test_calculates_base_confidence(self, pattern_extraction_service):
        """Test basic confidence calculation."""
        score = pattern_extraction_service.calculate_confidence_score(
            base_confidence=0.8,
            clarity_score=1.0,
            specificity_boost=1.0,
            recency_weight=1.0,
        )

        assert score == 0.8

    def test_applies_clarity_factor(self, pattern_extraction_service):
        """Test clarity score reduces confidence."""
        score = pattern_extraction_service.calculate_confidence_score(
            base_confidence=0.8,
            clarity_score=0.75,
            specificity_boost=1.0,
            recency_weight=1.0,
        )

        assert abs(score - 0.6) < 0.01

    def test_applies_specificity_boost(self, pattern_extraction_service):
        """Test specificity boost increases confidence."""
        score = pattern_extraction_service.calculate_confidence_score(
            base_confidence=0.8,
            clarity_score=1.0,
            specificity_boost=1.2,
            recency_weight=1.0,
        )

        assert score == 0.96

    def test_clamps_to_max_one(self, pattern_extraction_service):
        """Test score is clamped to 1.0 maximum."""
        score = pattern_extraction_service.calculate_confidence_score(
            base_confidence=0.9,
            clarity_score=1.2,
            specificity_boost=1.2,
            recency_weight=1.2,
        )

        assert score == 1.0

    def test_clamps_to_min_zero(self, pattern_extraction_service):
        """Test score is clamped to 0.0 minimum."""
        score = pattern_extraction_service.calculate_confidence_score(
            base_confidence=0.1,
            clarity_score=0.0,
            specificity_boost=0.5,
            recency_weight=0.5,
        )

        assert score == 0.0

    def test_all_factors_combined(self, pattern_extraction_service):
        """Test all factors work together correctly."""
        score = pattern_extraction_service.calculate_confidence_score(
            base_confidence=0.85,
            clarity_score=0.9,
            specificity_boost=1.1,
            recency_weight=0.95,
        )

        expected = 0.85 * 0.9 * 1.1 * 0.95
        assert abs(score - expected) < 0.01
