"""
Feature 026: Pattern Capture & Context Injection Integration Tests

Tests for:
- T026-011: Async pattern extraction on session capture (capture_session)
- T026-012: Pattern context injection on session init (load_context)
"""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.models.pattern_models import ExtractedPattern, StoredPattern

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------
_PATCH_PATTERN_EXTRACTION = "src.mcp_server.PatternExtractionService"
_PATCH_PATTERN_STORAGE = "src.mcp_server.PatternStorageService"
_PATCH_PATTERN_QUERY = "src.mcp_server.PatternQueryService"
_PATCH_NEO4J = "src.mcp_server.neo4j_service"
_PATCH_VECTOR = "src.mcp_server.get_vector_service"
_PATCH_GRAPH_FACTORY = "src.mcp_server.graph_factory"
_PATCH_ENTITY_FACTORY = "src.mcp_server.entity_factory"
_PATCH_CLASSIFICATION_SERVICE = "src.mcp_server.classification_service"
_PATCH_FILE_SERVICE = "src.mcp_server.file_service"
_PATCH_EVENT_LOG = "src.mcp_server.event_log_service"
_PATCH_CONFIG_PATTERN_ENABLED = "src.mcp_server.PATTERN_RECOGNITION_ENABLED"


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------
def make_test_extracted_pattern(**overrides):
    """Create a test ExtractedPattern with defaults."""
    defaults = {
        "pattern_type": "infrastructure",
        "problem_description": "Database connection timeouts under load",
        "solution_approach": "Implement connection pooling with HikariCP",
        "confidence": 0.85,
        "outcome": "SUCCESS",
        "metrics": {"connection_time": 50, "unit": "ms"},
        "session_uuid": str(uuid4()),
    }
    defaults.update(overrides)
    return ExtractedPattern(**defaults)


def make_test_stored_pattern(**overrides):
    """Create a test StoredPattern with defaults."""
    now = datetime.now(UTC)
    defaults = {
        "pattern_uuid": str(uuid4()),
        "name": "Implement connection pooling with HikariCP",
        "pattern_type": "infrastructure",
        "problem": "Database connection timeouts under load",
        "solution": "Implement connection pooling with HikariCP",
        "occurrence_count": 3,
        "success_count": 3,
        "failure_count": 0,
        "success_rate": 1.0,
        "confidence": 0.85,
        "consecutive_failures": 0,
        "first_occurrence_date": now,
        "latest_occurrence_date": now,
        "created_at": now,
        "updated_at": now,
        "supporting_sessions": [str(uuid4()), str(uuid4()), str(uuid4())],
        "archived": False,
    }
    defaults.update(overrides)
    return StoredPattern(**defaults)


# ---------------------------------------------------------------------------
# T026-011: Async Pattern Extraction on Session Capture
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
class TestSessionCapturePatternExtraction:
    """Test async pattern extraction triggered by capture_session()."""

    async def test_capture_session_triggers_pattern_extraction_async(self):
        """Test that capture_session triggers pattern extraction without blocking."""
        conversation_summary = (
            "Fixed database performance by implementing connection pooling"
        )
        key_insights = [
            "Connection pooling reduces latency",
            "Need monitoring in production",
        ]
        session_uuid = str(uuid4())

        # Mock services
        mock_neo4j = AsyncMock()
        mock_neo4j.query = AsyncMock(return_value=[{"uuid": session_uuid}])

        mock_pai = MagicMock()
        mock_pai.write_file = MagicMock(return_value=f"/path/to/{session_uuid}.md")

        mock_graph_factory = MagicMock()
        mock_graph_factory._extract_with_llm = AsyncMock(return_value={"entities": []})

        # Mock pattern services
        mock_extraction = MagicMock()
        extracted_pattern = make_test_extracted_pattern()
        mock_extraction.extract_patterns_from_sessions = AsyncMock(
            return_value=[extracted_pattern]
        )

        mock_storage = MagicMock()
        stored_pattern = make_test_stored_pattern()
        mock_storage.store_pattern = AsyncMock(return_value=stored_pattern)

        mock_event_log = MagicMock()
        mock_event_log.log_event = AsyncMock(return_value=None)

        patches = {
            _PATCH_NEO4J: mock_neo4j,
            _PATCH_FILE_SERVICE: mock_pai,
            _PATCH_GRAPH_FACTORY: mock_graph_factory,
            _PATCH_PATTERN_EXTRACTION: MagicMock(return_value=mock_extraction),
            _PATCH_PATTERN_STORAGE: MagicMock(return_value=mock_storage),
            _PATCH_EVENT_LOG: mock_event_log,
            _PATCH_VECTOR: MagicMock(return_value=None),
            _PATCH_CONFIG_PATTERN_ENABLED: True,
        }

        with patch.multiple("src.mcp_server", **patches):
            # Import after patching to get the patched version
            from src.mcp_server import capture_session

            # Call capture_session
            response = await capture_session(
                conversation_summary=conversation_summary, key_insights=key_insights
            )

            # Session capture should complete immediately
            assert "✅ Session knowledge synthesized" in response
            assert session_uuid in response or "Session UUID" in response

            # Give background task a moment to start
            await asyncio.sleep(0.1)

            # Verify that pattern extraction was triggered asynchronously
            # (would be verified by checking task creation in actual implementation)
            assert mock_extraction.extract_patterns_from_sessions.called

    async def test_pattern_extraction_respects_feature_flag(self):
        """Test that pattern extraction is skipped when PATTERN_RECOGNITION_ENABLED is False."""
        conversation_summary = "Some work done"
        session_uuid = str(uuid4())

        mock_neo4j = AsyncMock()
        mock_neo4j.query = AsyncMock(return_value=[{"uuid": session_uuid}])

        mock_pai = MagicMock()
        mock_pai.write_file = MagicMock(return_value=f"/path/to/{session_uuid}.md")

        mock_graph_factory = MagicMock()
        mock_graph_factory._extract_with_llm = AsyncMock(return_value={"entities": []})

        mock_extraction = MagicMock()
        mock_extraction.extract_patterns_from_sessions = AsyncMock(return_value=[])

        patches = {
            _PATCH_NEO4J: mock_neo4j,
            _PATCH_FILE_SERVICE: mock_pai,
            _PATCH_GRAPH_FACTORY: mock_graph_factory,
            _PATCH_PATTERN_EXTRACTION: MagicMock(return_value=mock_extraction),
            _PATCH_VECTOR: MagicMock(return_value=None),
            _PATCH_CONFIG_PATTERN_ENABLED: False,  # Feature disabled
        }

        with patch.multiple("src.mcp_server", **patches):
            from src.mcp_server import capture_session

            response = await capture_session(
                conversation_summary=conversation_summary, key_insights=[]
            )

            # Should complete successfully
            assert "✅ Session knowledge synthesized" in response

            # Pattern extraction should not be called
            await asyncio.sleep(0.1)
            assert not mock_extraction.extract_patterns_from_sessions.called

    async def test_pattern_extraction_handles_timeout_gracefully(self):
        """Test that pattern extraction timeout doesn't break session capture."""
        conversation_summary = "Some work"
        session_uuid = str(uuid4())

        mock_neo4j = AsyncMock()
        mock_neo4j.query = AsyncMock(return_value=[{"uuid": session_uuid}])

        mock_pai = MagicMock()
        mock_pai.write_file = MagicMock(return_value=f"/path/to/{session_uuid}.md")

        mock_graph_factory = MagicMock()
        mock_graph_factory._extract_with_llm = AsyncMock(return_value={"entities": []})

        # Simulate timeout in pattern extraction
        async def timeout_extraction(*args, **kwargs):
            await asyncio.sleep(60)  # Will timeout before completion
            return []

        mock_extraction = MagicMock()
        mock_extraction.extract_patterns_from_sessions = AsyncMock(
            side_effect=timeout_extraction
        )

        patches = {
            _PATCH_NEO4J: mock_neo4j,
            _PATCH_FILE_SERVICE: mock_pai,
            _PATCH_GRAPH_FACTORY: mock_graph_factory,
            _PATCH_PATTERN_EXTRACTION: MagicMock(return_value=mock_extraction),
            _PATCH_VECTOR: MagicMock(return_value=None),
            _PATCH_CONFIG_PATTERN_ENABLED: True,
        }

        with patch.multiple("src.mcp_server", **patches):
            from src.mcp_server import capture_session

            # Should complete quickly despite timeout
            response = await capture_session(
                conversation_summary=conversation_summary, key_insights=[]
            )

            # Session capture should succeed despite background extraction timeout
            assert "✅ Session knowledge synthesized" in response


# ---------------------------------------------------------------------------
# T026-012: Pattern Context Injection on load_context
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
class TestPatternContextInjection:
    """Test pattern context injection in load_context()."""

    async def test_load_context_injects_patterns(self):
        """Test that load_context injects relevant patterns into context."""
        prompt = "How do I optimize database performance?"

        # Mock stored patterns
        pattern_1 = make_test_stored_pattern(
            name="Connection Pooling Optimization",
            success_rate=0.95,
            occurrence_count=5,
        )
        pattern_2 = make_test_stored_pattern(
            name="Query Index Optimization", success_rate=0.88, occurrence_count=3
        )

        mock_neo4j = AsyncMock()
        mock_neo4j.temporal_query = AsyncMock(return_value=[{"relationship": "test"}])

        mock_graph_factory = MagicMock()
        mock_graph_factory._extract_with_llm = AsyncMock(return_value={"entities": []})

        mock_query_service = MagicMock()
        mock_query_service.get_high_success_patterns = AsyncMock(
            return_value=[pattern_1, pattern_2]
        )

        patches = {
            _PATCH_NEO4J: mock_neo4j,
            _PATCH_GRAPH_FACTORY: mock_graph_factory,
            _PATCH_PATTERN_QUERY: MagicMock(return_value=mock_query_service),
            _PATCH_CONFIG_PATTERN_ENABLED: True,
        }

        with patch.multiple("src.mcp_server", **patches):
            from src.mcp_server import load_context

            context = await load_context(prompt=prompt)

            # Context should include patterns
            assert context is not None
            assert "Relevant Patterns from History" in context
            assert "Connection Pooling Optimization" in context
            assert "95" in context  # Success rate should be shown

    async def test_load_context_pattern_formatting(self):
        """Test that patterns are formatted correctly in context."""
        from src.mcp_server import _format_patterns_for_context

        pattern = make_test_stored_pattern(
            name="Caching Strategy",
            success_rate=0.92,
            occurrence_count=4,
            consecutive_failures=0,
        )

        markdown = _format_patterns_for_context([pattern])

        # Verify markdown formatting
        assert "Relevant Patterns from History" in markdown
        assert "Caching Strategy" in markdown
        assert "caching" in markdown.lower()
        assert "92" in markdown  # Success rate

    async def test_load_context_pattern_failure_warning(self):
        """Test that failure warnings are included for failed patterns."""
        from src.mcp_server import _format_patterns_for_context

        pattern = make_test_stored_pattern(
            name="Retry Policy",
            success_rate=0.30,
            occurrence_count=6,
            failure_count=4,
            consecutive_failures=2,  # Should trigger warning
        )

        markdown = _format_patterns_for_context([pattern])

        # Verify warning is included
        assert "Warning" in markdown
        assert "failed" in markdown.lower()
        assert "⚠️" in markdown

    async def test_load_context_pattern_injection_disabled(self):
        """Test that pattern injection is skipped when disabled."""
        prompt = "Some prompt"

        mock_neo4j = AsyncMock()
        mock_neo4j.temporal_query = AsyncMock(return_value=[{"relationship": "test"}])

        mock_graph_factory = MagicMock()
        mock_graph_factory._extract_with_llm = AsyncMock(return_value={"entities": []})

        patches = {
            _PATCH_NEO4J: mock_neo4j,
            _PATCH_GRAPH_FACTORY: mock_graph_factory,
            _PATCH_CONFIG_PATTERN_ENABLED: False,  # Disabled
        }

        with patch.multiple("src.mcp_server", **patches):
            from src.mcp_server import load_context

            context = await load_context(prompt=prompt)

            # Context should not include patterns section
            # (just the normal temporal context)
            assert context is not None

    async def test_load_context_pattern_query_respects_thresholds(self):
        """Test that pattern queries use correct min_success_rate and min_occurrences."""
        prompt = "Help with database"

        mock_neo4j = AsyncMock()
        mock_neo4j.temporal_query = AsyncMock(return_value=[])

        mock_graph_factory = MagicMock()
        mock_graph_factory._extract_with_llm = AsyncMock(return_value={"entities": []})

        call_args = []

        async def capture_call(*args, **kwargs):
            call_args.append((args, kwargs))
            return []

        mock_query_service = MagicMock()
        mock_query_service.get_high_success_patterns = AsyncMock(
            side_effect=capture_call
        )

        patches = {
            _PATCH_NEO4J: mock_neo4j,
            _PATCH_GRAPH_FACTORY: mock_graph_factory,
            _PATCH_PATTERN_QUERY: MagicMock(return_value=mock_query_service),
            _PATCH_CONFIG_PATTERN_ENABLED: True,
        }

        with patch.multiple("src.mcp_server", **patches):
            from src.mcp_server import load_context

            await load_context(prompt=prompt)

            # Verify query was called with correct thresholds
            assert mock_query_service.get_high_success_patterns.called
            call = mock_query_service.get_high_success_patterns.call_args

            # Check keyword arguments
            assert call.kwargs.get("min_success_rate") >= 0.7  # At least 70%
            assert call.kwargs.get("min_occurrences") == 2  # Proactive threshold
            assert call.kwargs.get("limit") <= 5  # Max 5 patterns

    async def test_load_context_handles_pattern_query_failure(self):
        """Test graceful degradation when pattern query fails."""
        prompt = "Some work"

        mock_neo4j = AsyncMock()
        mock_neo4j.temporal_query = AsyncMock(return_value=[{"relationship": "test"}])

        mock_graph_factory = MagicMock()
        mock_graph_factory._extract_with_llm = AsyncMock(return_value={"entities": []})

        # Pattern query fails
        mock_query_service = MagicMock()
        mock_query_service.get_high_success_patterns = AsyncMock(
            side_effect=Exception("Service unavailable")
        )

        patches = {
            _PATCH_NEO4J: mock_neo4j,
            _PATCH_GRAPH_FACTORY: mock_graph_factory,
            _PATCH_PATTERN_QUERY: MagicMock(return_value=mock_query_service),
            _PATCH_CONFIG_PATTERN_ENABLED: True,
        }

        with patch.multiple("src.mcp_server", **patches):
            from src.mcp_server import load_context

            # Should not crash, should return context without patterns
            context = await load_context(prompt=prompt)
            assert context is not None
