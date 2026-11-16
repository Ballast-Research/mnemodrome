"""Failed layer writes must be reported honestly (GitLab #124 / H1).

Nothing retries a failed Markdown or vector write, so tool responses must say
the write FAILED and must never claim it was "queued for retry".
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.contract import test_mcp_tool_handlers as h

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mod(monkeypatch):
    import src.mcp_server as m

    monkeypatch.setattr(m, "_services_initialized", True)
    monkeypatch.setattr(m, "ACTR_ENABLED", False)
    monkeypatch.setattr(m, "neo4j_service", h._mock_neo4j())
    monkeypatch.setattr(m, "event_log_service", h._mock_event_log())
    monkeypatch.setattr(m, "entity_cache", h._mock_entity_cache())
    monkeypatch.setattr(m, "graph_factory", h._mock_graph_factory())
    monkeypatch.setattr(m, "classification_service", h._mock_classification_service())
    monkeypatch.setattr(m, "entity_factory", h._mock_entity_factory())
    monkeypatch.setattr(m, "relationship_factory", h._mock_relationship_factory())
    llm = h._mock_llm_service()
    llm.classification_degraded_count = 0
    llm.extraction_degraded_count = 0
    monkeypatch.setattr(m, "llm_service_singleton", llm)
    monkeypatch.setattr(m, "get_vector_service", lambda: h._mock_vector_service())
    monkeypatch.setattr(
        m, "_fire_and_forget", lambda coro, description="": coro.close()
    )
    monkeypatch.setattr("src.config.EVENT_LOG_AUTO_CAPTURE", False)
    failing = MagicMock()
    failing.write_file = MagicMock(side_effect=OSError("disk full"))
    monkeypatch.setattr(m, "file_service", failing)
    return m


def _assert_honest(text: str):
    assert "queued for retry" not in text
    assert "retry" not in text.lower()
    assert "FAILED" in text
    assert "Markdown file layer" in text
    assert "Neo4j" in text


async def test_capture_fact_markdown_failure_is_honest(mod):
    result = await mod.capture_fact(
        entity1="Python",
        relation="IS_LANGUAGE_OF",
        entity2="Mnemodrome",
        entity1_type="Technology",
        entity2_type="Project",
    )
    _assert_honest(result)


async def test_capture_context_markdown_failure_is_honest(mod):
    result = await mod.capture_context(context="Alice works at Acme Corp.")
    _assert_honest(result)


async def test_update_fact_markdown_failure_is_honest(mod):
    mod.neo4j_service.find_relationship = AsyncMock(
        return_value={
            "uuid": "old-uuid",
            "entity1_name": "A",
            "entity1_uuid": "e1",
            "relation_type": "RUNS_ON",
            "entity2_name": "old",
            "created_at": "2025-09-15T10:00:00Z",
        }
    )
    mod.neo4j_service.query = AsyncMock(return_value=[{"uuid": "e2"}])
    mod.conflict_detection_service = MagicMock(
        assess_complexity=AsyncMock(
            return_value={"complexity_score": 25, "auto_resolvable": True}
        )
    )
    result = await mod.update_fact(
        entity1="A",
        relation="RUNS_ON",
        entity2_old="old",
        entity2_new="new",
        conflict_resolution="auto",
    )
    assert result["status"] == "success"
    assert result["dual_memory_status"]["markdown"] == "failed"
    assert result["queued_retries"] == 0
    _assert_honest(result["warning"])
    assert "queued for retry" not in str(result)


async def test_append_response_vector_not_updated_is_honest(mod):
    from src.models.deduplication_models import AppendResult

    result = AppendResult(
        status="success",
        action="appended",
        learning_uuid="u",
        file_path="/x.md",
        new_section_title="S",
        append_count=1,
        timestamp="2026-01-01T00:00:00Z",
        embedding_updated=False,
    )
    text = mod._format_append_success_response(result)
    assert "queued" not in text.lower()
    assert "vector layer" in text
    assert "NOT regenerated" in text


async def test_capture_context_degraded_markdown_failure_is_honest(mod):
    mod.graph_factory.build_from_extraction = AsyncMock(
        return_value={
            "entities_created": 0,
            "relationships_created": 0,
            "entity_uuids": {},
            "entity_types": [],
            "relationships": [],
        }
    )
    result = await mod.capture_context(context="Alice works at Acme Corp.")
    assert "DEGRADED" in result
    assert "FAILED" in result
    assert "Markdown file layer" in result
    assert "retry" not in result.lower().replace("nothing is retried", "")
