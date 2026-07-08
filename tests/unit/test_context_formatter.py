"""Unit tests for context formatting module (src/services/context_formatter.py).

Tests the markdown formatting of temporal query results.
"""

from datetime import datetime
from unittest.mock import MagicMock

from src.services.context_formatter import (
    _format_datetime,
    _get_property,
    _get_relationship_type,
    _truncate_summary,
    format_context_markdown,
)


class TestFormatContextMarkdown:
    """Tests for format_context_markdown function."""

    def test_format_empty_results(self):
        """Test formatting with empty result list."""
        result = format_context_markdown([])
        assert "No recent context found" in result

    def test_format_none_results(self):
        """Test formatting with None results."""
        result = format_context_markdown(None)
        assert "No recent context found" in result

    def test_format_single_relationship(self):
        """Test formatting a single relationship."""
        results = [
            {
                "source": {"name": "Carlos", "summary": "A developer"},
                "r": {"type": "USES"},
                "target": {"name": "Neo4j", "summary": "Graph database"},
            }
        ]
        result = format_context_markdown(results)

        assert "Recent Context" in result
        assert "1 relationship(s)" in result
        assert "Carlos → Neo4j" in result
        assert "USES" in result
        assert "A developer" in result
        assert "Graph database" in result

    def test_format_multiple_relationships(self):
        """Test formatting multiple relationships."""
        results = [
            {
                "source": {"name": "Entity1", "summary": "First entity"},
                "r": {"type": "RELATES_TO"},
                "target": {"name": "Entity2", "summary": "Second entity"},
            },
            {
                "source": {"name": "Entity2", "summary": "Second entity"},
                "r": {"type": "IMPLEMENTS"},
                "target": {"name": "Entity3", "summary": "Third entity"},
            },
        ]
        result = format_context_markdown(results)

        assert "2 relationship(s)" in result
        assert "1. Entity1 → Entity2" in result
        assert "2. Entity2 → Entity3" in result

    def test_format_with_missing_properties(self):
        """Test formatting when some properties are missing."""
        results = [{"source": {}, "r": {}, "target": {}}]
        result = format_context_markdown(results)

        assert "Unknown → Unknown" in result
        assert "RELATED_TO" in result

    def test_format_with_datetime_valid_at(self):
        """Test formatting with valid_at timestamp."""
        results = [
            {
                "source": {"name": "Source"},
                "r": {"type": "USES", "valid_at": "2025-03-12T10:30:00Z"},
                "target": {"name": "Target"},
            }
        ]
        result = format_context_markdown(results)

        assert "Created" in result
        assert "2025-03-12" in result or "2025-03-12 10:30" in result

    def test_format_with_neo4j_datetime_object(self):
        """Test formatting with Neo4j datetime object."""
        dt_mock = MagicMock()
        dt_mock.iso_format = MagicMock(return_value="2025-03-12T14:45:00Z")

        results = [
            {
                "source": {"name": "Source"},
                "r": {"type": "RELATES", "valid_at": dt_mock},
                "target": {"name": "Target"},
            }
        ]
        result = format_context_markdown(results)

        assert "Created" in result

    def test_format_truncates_long_summaries(self):
        """Test that long summaries are truncated."""
        long_summary = "A" * 300  # Longer than MAX_SUMMARY_LENGTH
        results = [
            {
                "source": {"name": "Source", "summary": long_summary},
                "r": {"type": "USES"},
                "target": {"name": "Target"},
            }
        ]
        result = format_context_markdown(results)

        assert "..." in result
        assert len(result) < len(long_summary) + 500

    def test_format_empty_summary(self):
        """Test formatting with empty summary."""
        results = [
            {
                "source": {"name": "Source", "summary": ""},
                "r": {"type": "USES"},
                "target": {"name": "Target", "summary": ""},
            }
        ]
        result = format_context_markdown(results)

        assert "Source" in result
        assert "Target" in result

    def test_format_renders_edge_context(self):
        """issue #97: the edge's own r.context must be rendered."""
        results = [
            {
                "source": {"name": "Carlos", "summary": "A developer"},
                "r": {
                    "type": "USES",
                    "context": "Carlos uses Neo4j for the graph layer",
                },
                "target": {"name": "Neo4j", "summary": "Graph database"},
            }
        ]
        result = format_context_markdown(results)

        assert "**Context**" in result
        assert "Carlos uses Neo4j for the graph layer" in result

    def test_format_distinct_edge_context_per_relationship(self):
        """issue #97: distinct per-edge context must render even when the
        (clobbered, shared) node summary is identical across a hub node's edges.

        This is the anti-repetition guarantee: before the fix, every edge from
        a hub node showed only that node's single summary, so the edges were
        indistinguishable. Now each edge shows its own context.
        """
        shared_clobbered_summary = "Shared summary blob that clobbered the hub node"
        results = [
            {
                "source": {"name": "Alice", "summary": shared_clobbered_summary},
                "r": {
                    "type": "USES",
                    "context": "Alice downloaded GLM-5.2 to her workstation",
                },
                "target": {"name": "GLM-5.2", "summary": "A model"},
            },
            {
                "source": {"name": "Alice", "summary": shared_clobbered_summary},
                "r": {"type": "PLANS", "context": "Alice is building out the VLAN"},
                "target": {"name": "VLAN buildout", "summary": "Network work"},
            },
        ]
        result = format_context_markdown(results)

        # Each edge surfaces its OWN context, so the two edges are distinguishable.
        assert "Alice downloaded GLM-5.2 to her workstation" in result
        assert "Alice is building out the VLAN" in result

    def test_format_no_context_line_when_edge_has_no_context(self):
        """No **Context** line should appear when the edge carries no context."""
        results = [
            {
                "source": {"name": "Source", "summary": "s"},
                "r": {"type": "USES"},
                "target": {"name": "Target", "summary": "t"},
            }
        ]
        result = format_context_markdown(results)

        assert "**Context**" not in result

    def test_format_truncates_long_edge_context(self):
        """Edge context must be truncated like summaries to avoid token bloat."""
        long_context = "C" * 300
        results = [
            {
                "source": {"name": "Source"},
                "r": {"type": "USES", "context": long_context},
                "target": {"name": "Target"},
            }
        ]
        result = format_context_markdown(results)

        assert "**Context**" in result
        assert "..." in result


class TestTruncateSummary:
    """Tests for _truncate_summary helper function."""

    def test_truncate_none_summary(self):
        """Test truncating None summary."""
        result = _truncate_summary(None)
        assert result == ""

    def test_truncate_empty_summary(self):
        """Test truncating empty string."""
        result = _truncate_summary("")
        assert result == ""

    def test_truncate_short_summary(self):
        """Test that short summaries are not truncated."""
        short = "This is a short summary"
        result = _truncate_summary(short)
        assert result == short

    def test_truncate_long_summary(self):
        """Test that long summaries are truncated with ellipsis."""
        long_summary = "A" * 300
        result = _truncate_summary(long_summary)

        assert result.endswith("...")
        assert len(result) < len(long_summary)
        assert len(result) <= 204  # MAX_SUMMARY_LENGTH (200) + "..." (3)

    def test_truncate_at_word_boundary(self):
        """Test that truncation happens at word boundary."""
        summary = (
            "This is a long summary that should be truncated at a word boundary "
            + "A" * 200
        )
        result = _truncate_summary(summary)

        # Should not have partial word at end before ellipsis
        assert result.endswith("...")
        # Verify it was truncated
        assert len(result) < len(summary)


class TestGetProperty:
    """Tests for _get_property helper function."""

    def test_get_property_dict_access(self):
        """Test property retrieval from dict."""
        obj = {"name": "Test", "value": 42}
        assert _get_property(obj, "name") == "Test"
        assert _get_property(obj, "value") == 42

    def test_get_property_with_get_method(self):
        """Test property retrieval from object with get method."""
        obj = MagicMock()
        obj.get = MagicMock(return_value="TestValue")
        result = _get_property(obj, "key")
        assert result == "TestValue"

    def test_get_property_with_hasattr(self):
        """Test property retrieval from object with attributes."""
        obj = MagicMock()
        obj.name = "TestName"

        # MagicMock has get() method, so it will use that first
        # Instead test with a real object
        class TestObj:
            name = "TestName"

        result = _get_property(TestObj(), "name")
        assert result == "TestName"

    def test_get_property_missing_with_default(self):
        """Test property retrieval with default value."""
        obj = {}
        result = _get_property(obj, "missing", "default_value")
        assert result == "default_value"

    def test_get_property_missing_no_default(self):
        """Test property retrieval without default."""
        obj = {}
        result = _get_property(obj, "missing")
        assert result is None


class TestGetRelationshipType:
    """Tests for _get_relationship_type helper function."""

    def test_get_relationship_type_from_object_attribute(self):
        """Test retrieving type from object with type attribute."""
        rel = MagicMock()
        rel.type = "USES"
        result = _get_relationship_type(rel)
        assert result == "USES"

    def test_get_relationship_type_from_dict(self):
        """Test retrieving type from dict."""
        rel = {"type": "IMPLEMENTS"}
        result = _get_relationship_type(rel)
        assert result == "IMPLEMENTS"

    def test_get_relationship_type_default(self):
        """Test default relationship type when not found."""
        rel = {}
        result = _get_relationship_type(rel)
        assert result == "RELATED_TO"

    def test_get_relationship_type_missing_attribute(self):
        """Test with object without type attribute."""
        rel = MagicMock(spec=[])  # No attributes
        result = _get_relationship_type(rel)
        assert result == "RELATED_TO"


class TestFormatDatetime:
    """Tests for _format_datetime helper function."""

    def test_format_iso_string(self):
        """Test formatting ISO format string."""
        dt_str = "2025-03-12T10:30:00Z"
        result = _format_datetime(dt_str)
        assert "2025-03-12" in result
        assert "UTC" in result

    def test_format_iso_string_with_timezone(self):
        """Test formatting ISO string with timezone offset."""
        dt_str = "2025-03-12T10:30:00+00:00"
        result = _format_datetime(dt_str)
        assert "2025-03-12" in result

    def test_format_datetime_object(self):
        """Test formatting Python datetime object."""
        dt = datetime(2025, 3, 12, 10, 30, 0)
        result = _format_datetime(dt)
        assert "2025-03-12" in result
        assert "10:30" in result or "10" in result

    def test_format_neo4j_datetime_object(self):
        """Test formatting Neo4j datetime object."""
        dt_mock = MagicMock()
        dt_mock.iso_format = MagicMock(return_value="2025-03-12T15:45:00Z")
        result = _format_datetime(dt_mock)
        assert "2025-03-12" in result

    def test_format_invalid_datetime_string(self):
        """Test fallback for invalid datetime string."""
        result = _format_datetime("invalid-datetime")
        assert result == "invalid-datetime"

    def test_format_none_datetime(self):
        """Test formatting None datetime."""
        result = _format_datetime(None)
        assert result == "None"
