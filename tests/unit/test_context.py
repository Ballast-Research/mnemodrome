"""Tests for Phase 3 context loading functionality."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.services.context_formatter import format_context_markdown
from src.services.keyword_extractor import extract_keywords


class TestKeywordExtractor:
    """Tests for keyword extraction."""

    def test_extract_from_simple_prompt(self):
        """Test keyword extraction from simple prompt."""
        keyword = extract_keywords("Tell me about Mnemodrome")
        assert keyword == "mnemodrome"

    def test_filters_stop_words(self):
        """Test that stop words are filtered out."""
        keyword = extract_keywords("What is the status of the project?")
        assert keyword in ["status", "project"]

    def test_empty_prompt_returns_none(self):
        """Test empty prompt returns None."""
        assert extract_keywords("") is None
        assert extract_keywords(None) is None

    def test_filters_short_words(self):
        """Test that short words (<=3 chars) are filtered."""
        keyword = extract_keywords("I am at work")
        assert keyword == "work"


class TestContextFormatter:
    """Tests for context markdown formatting."""

    def test_formats_empty_results(self):
        """Test formatting with no results."""
        result = format_context_markdown([])
        assert "No recent context found" in result

    def test_formats_single_relationship(self):
        """Test formatting a single relationship."""
        mock_record = {
            "source": {"name": "Entity1", "summary": "First entity"},
            "r": MagicMock(type="RELATED_TO", valid_at=datetime.now()),
            "target": {"name": "Entity2", "summary": "Second entity"},
        }

        result = format_context_markdown([mock_record])

        assert "Entity1 → Entity2" in result
        assert "RELATED_TO" in result
        assert "First entity" in result
        assert "Second entity" in result

    def test_handles_missing_properties(self):
        """Test graceful handling of missing properties."""
        mock_record = {
            "source": {"name": "Entity1"},
            "r": MagicMock(type="RELATED_TO"),
            "target": {"name": "Entity2"},
        }

        result = format_context_markdown([mock_record])

        assert "Entity1 → Entity2" in result
        assert "RELATED_TO" in result

    def test_counts_relationships(self):
        """Test that relationship count is displayed."""
        mock_records = [
            {
                "source": {"name": "E1"},
                "r": MagicMock(type="REL"),
                "target": {"name": "E2"},
            },
            {
                "source": {"name": "E3"},
                "r": MagicMock(type="REL"),
                "target": {"name": "E4"},
            },
        ]

        result = format_context_markdown(mock_records)
        assert "2 relationship(s)" in result


class TestTemporalQuery:
    """Tests for temporal query logic (integration-style, can be skipped if no test DB)."""

    @pytest.mark.skip(reason="Requires Neo4j test instance")
    @pytest.mark.asyncio
    async def test_filters_invalidated_relationships(self):
        """Test that invalidated relationships are filtered out."""
        # This would require a test Neo4j instance with sample data
        pass

    @pytest.mark.skip(reason="Requires Neo4j test instance")
    @pytest.mark.asyncio
    async def test_temporal_ordering(self):
        """Test that results are ordered by valid_at DESC."""
        pass

    @pytest.mark.skip(reason="Requires Neo4j test instance")
    @pytest.mark.asyncio
    async def test_keyword_filtering(self):
        """Test that keyword filtering works when provided."""
        pass
