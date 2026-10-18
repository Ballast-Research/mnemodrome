"""
Regression tests for the load_context_cascade markdown formatter.

Bug: load_context_cascade returned hits with similarity scores but empty
bodies (title/content/summary blank). Root cause: the formatter read the
"text" key for vector-search prompt hits, but VectorService.search_similar_prompts
(and its *_by_embedding variant) stores the hit text under "matched_text" —
"text" is never set on the returned dict, so every prompt hit rendered with
an empty body despite a real similarity score.

These tests reproduce the empty-body case with mocked cascade results
(the shapes VectorService / Neo4j actually return) and assert the fix:
every rendered item carries non-empty text, and hits with no recoverable
text are dropped rather than shown with an empty body.
"""

from src.mcp_server import _format_context_cascade_result


class TestContextCascadeFormatter:
    """Tests for _format_context_cascade_result."""

    def test_greenfield_returns_empty_string(self):
        result = _format_context_cascade_result({"is_greenfield": True}, 10.0)
        assert result == ""

    def test_similar_prompt_matched_text_key_is_rendered(self):
        """Regression: prompts use 'matched_text', not 'text'."""
        cascade_result = {
            "is_greenfield": False,
            "similar_prompts": [
                {
                    "id": "abc123",
                    "score": 0.91,
                    "matched_text": "How do I configure the reverse proxy?",
                    "timestamp": "2026-01-01T00:00:00",
                    "neo4j_uuid": "uuid-1",
                }
            ],
            "similar_learnings": [],
            "entity_context": [],
            "facts": [],
        }

        markdown = _format_context_cascade_result(cascade_result, 50.0)

        assert "0.910" in markdown
        assert "How do I configure the reverse proxy?" in markdown
        # The old bug rendered a bare "...\n\n" with no preceding text.
        assert "  ...\n\n" not in markdown

    def test_similar_prompt_with_no_recoverable_text_is_dropped(self):
        """A hit with a real score but no text anywhere must not render empty."""
        cascade_result = {
            "is_greenfield": False,
            "similar_prompts": [
                {"id": "stale-1", "score": 0.88, "neo4j_uuid": "uuid-missing"}
            ],
            "similar_learnings": [],
            "entity_context": [],
            "facts": [],
        }

        markdown = _format_context_cascade_result(cascade_result, 50.0)

        assert "0.880" not in markdown
        assert "Previous Similar Questions" not in markdown
        # Nothing else rendered either -> explicit "no displayable context".
        assert "No displayable context found" in markdown

    def test_similar_learning_title_and_content_rendered(self):
        cascade_result = {
            "is_greenfield": False,
            "similar_prompts": [],
            "similar_learnings": [
                {
                    "id": "l1",
                    "score": 0.8,
                    "title": "Reverse Proxy Setup",
                    "content": "Use nginx with the dmz01 config.",
                    "neo4j_uuid": "uuid-2",
                }
            ],
            "entity_context": [],
            "facts": [],
        }

        markdown = _format_context_cascade_result(cascade_result, 50.0)

        assert "Reverse Proxy Setup" in markdown
        assert "Use nginx with the dmz01 config." in markdown

    def test_similar_learning_with_missing_record_is_dropped(self):
        """Stale Qdrant point whose source record has no title/content."""
        cascade_result = {
            "is_greenfield": False,
            "similar_prompts": [],
            "similar_learnings": [
                {"id": "l-stale", "score": 0.77, "title": None, "content": None}
            ],
            "entity_context": [],
            "facts": [],
        }

        markdown = _format_context_cascade_result(cascade_result, 50.0)

        assert "Related Learnings" not in markdown
        assert "No displayable context found" in markdown

    def test_similar_learning_with_title_only_marks_content_unavailable(self):
        cascade_result = {
            "is_greenfield": False,
            "similar_prompts": [],
            "similar_learnings": [
                {
                    "id": "l-partial",
                    "score": 0.7,
                    "title": "Partial Record",
                    "content": "",
                }
            ],
            "entity_context": [],
            "facts": [],
        }

        markdown = _format_context_cascade_result(cascade_result, 50.0)

        assert "Partial Record" in markdown
        assert "[content unavailable]" in markdown

    def test_facts_and_entities_rendered(self):
        cascade_result = {
            "is_greenfield": False,
            "similar_prompts": [],
            "similar_learnings": [],
            "entity_context": [{"uuid": "e1", "name": "Mnemodrome"}],
            "facts": [
                {
                    "entity1": "Mnemodrome",
                    "relation": "RUNS_ON",
                    "entity2": "server01",
                }
            ],
        }

        markdown = _format_context_cascade_result(cascade_result, 50.0)

        assert "Mnemodrome" in markdown
        assert "RUNS_ON" in markdown
        assert "server01" in markdown

    def test_fact_with_missing_entity_is_dropped(self):
        cascade_result = {
            "is_greenfield": False,
            "similar_prompts": [],
            "similar_learnings": [],
            "entity_context": [],
            "facts": [
                {"entity1": "Mnemodrome", "relation": "RUNS_ON", "entity2": None}
            ],
        }

        markdown = _format_context_cascade_result(cascade_result, 50.0)

        assert "Related Facts" not in markdown
        assert "No displayable context found" in markdown

    def test_total_items_count_excludes_dropped_hits(self):
        cascade_result = {
            "is_greenfield": False,
            "similar_prompts": [
                {"id": "ok", "score": 0.9, "matched_text": "real text"},
                {"id": "stale", "score": 0.8},  # dropped: no text
            ],
            "similar_learnings": [],
            "entity_context": [],
            "facts": [],
        }

        markdown = _format_context_cascade_result(cascade_result, 123.0)

        assert "Retrieved 1 items in 123.0ms" in markdown
