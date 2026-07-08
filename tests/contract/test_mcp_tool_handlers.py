"""Smoke tests for all MCP tool handlers in src/mcp_server.py.

Priority 1 coverage gap: mcp_server.py has 0% coverage (1,668 lines, 29 handlers).
Each handler is called with mocked service dependencies, verifying:
  - No unhandled exceptions
  - Return value is a string (MCP tools return strings)

Services are mocked at the module level to prevent real I/O.
"""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# All tests in this module are async
pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_neo4j():
    """Create a mock neo4j_service with common methods."""
    svc = AsyncMock()
    svc.ping = AsyncMock()
    svc.query = AsyncMock(return_value=[])
    svc.create_entity = AsyncMock(return_value=None)
    svc.delete_entity = AsyncMock(return_value=None)
    svc.temporal_query = AsyncMock(return_value=[])
    svc.temporal_query_by_entities = AsyncMock(return_value=[])
    svc.find_relationship = AsyncMock(return_value=None)
    svc.invalidate_relationship = AsyncMock(
        return_value={
            "fields_set": ["invalid_at"],
            "invalidated_at": datetime.now(UTC).isoformat(),
            "invalid_at": datetime.now(UTC).isoformat(),
        }
    )
    svc.execute_update_transaction = AsyncMock(
        return_value={
            "old_fact_uuid": str(uuid4()),
            "new_fact_uuid": str(uuid4()),
            "invalidated_at": datetime.now(UTC).isoformat(),
            "supersession_chain": ["old", "new"],
        }
    )
    return svc


def _mock_redis():
    svc = MagicMock()
    svc.is_connected = False
    svc.disconnect = AsyncMock()
    return svc


def _mock_pai():
    svc = MagicMock()
    svc.write_file = MagicMock(return_value="/tmp/test-file.md")
    svc.delete_file = MagicMock()
    svc.validate_path = MagicMock()
    svc._get_target_path = MagicMock(
        return_value=MagicMock(__truediv__=lambda s, o: f"/tmp/{o}")
    )
    svc.storage_path = MagicMock(__truediv__=lambda s, o: f"/tmp/{o}")
    return svc


def _mock_event_log():
    svc = MagicMock()
    svc.session_id = "test-session-12345678"
    svc.log_event = AsyncMock(return_value=1)
    svc.query_events = AsyncMock(return_value=[])
    svc.get_session_timeline = AsyncMock(return_value=[])
    svc.get_database_size = AsyncMock(return_value=1.0)
    svc.get_event_count = AsyncMock(return_value=10)
    svc.postgres = None
    svc._auto_disabled = False
    svc._metrics = MagicMock(
        get_p95_write=MagicMock(return_value=1.0),
        get_p99_write=MagicMock(return_value=2.0),
        get_p95_query=MagicMock(return_value=1.0),
        get_p99_query=MagicMock(return_value=2.0),
        sla_violations=0,
        last_violation_at=None,
    )
    return svc


def _mock_entity_cache():
    svc = MagicMock()
    return svc


def _mock_vector_service():
    svc = AsyncMock()
    svc.is_enabled = MagicMock(return_value=False)
    svc.search = AsyncMock(return_value=[])
    svc.health_check = AsyncMock(
        return_value={
            "status": "healthy",
            "enabled": True,
            "initialized": True,
            "stats": {},
            "performance": {},
        }
    )
    svc.store_prompt_embedding = AsyncMock(return_value=str(uuid4()))
    svc.store_document = AsyncMock()
    svc.store_fact_embedding = AsyncMock()
    svc.store_session_embedding = AsyncMock()
    svc.search_similar_learnings = AsyncMock(return_value=[])
    return svc


def _mock_context_retrieval():
    svc = AsyncMock()
    svc.retrieve_context_for_prompt = AsyncMock(
        return_value={
            "is_greenfield": True,
            "similar_prompts": [],
            "similar_learnings": [],
            "entity_context": [],
            "facts": [],
            "procedures": [],
        }
    )
    return svc


def _mock_llm_service():
    svc = AsyncMock()
    svc.generate = AsyncMock(return_value="test response")
    svc.chat = AsyncMock(return_value="test response")
    return svc


def _mock_graph_factory():
    svc = AsyncMock()
    svc.build_from_extraction = AsyncMock(
        return_value={
            "entities_created": 2,
            "relationships_created": 1,
            "entity_uuids": {"TestEntity1": str(uuid4()), "TestEntity2": str(uuid4())},
            "entity_types": ["Person", "Organization"],
            "relationships": [
                {
                    "source_name": "TestEntity1",
                    "relation_type": "WORKS_AT",
                    "target_name": "TestEntity2",
                    "uuid": str(uuid4()),
                }
            ],
        }
    )
    svc._extract_with_llm = AsyncMock(
        return_value={
            "entities": [{"name": "TestEntity", "role": "entity"}],
            "relationships": [],
        }
    )
    return svc


def _mock_classification_service():
    svc = AsyncMock()
    svc.classify_entity = AsyncMock(return_value="Person")
    svc.classify_batch = AsyncMock(return_value=["Person", "Organization"])
    return svc


def _mock_entity_factory():
    svc = AsyncMock()
    svc.create_or_resolve = AsyncMock(return_value=str(uuid4()))
    return svc


def _mock_relationship_factory():
    svc = AsyncMock()
    svc.create = AsyncMock(
        return_value={
            "source_name": "Entity1",
            "relation_type": "RELATES_TO",
            "target_name": "Entity2",
            "uuid": str(uuid4()),
        }
    )
    return svc


def _mock_procedure_service():
    svc = AsyncMock()
    svc.create_procedure = AsyncMock(return_value=(str(uuid4()), True))
    svc.find_procedures = AsyncMock(return_value=[])
    svc.enrich_procedure = AsyncMock()
    return svc


def _mock_activation_service():
    svc = AsyncMock()
    svc.update_access_tracking = AsyncMock()
    svc.strengthen_hebbian_links = AsyncMock()
    svc.update_confidence = AsyncMock()
    return svc


def _mock_conflict_detection():
    svc = MagicMock()
    svc.calculate_complexity_score = MagicMock(return_value=10)
    svc.assess_complexity = AsyncMock(
        return_value={
            "complexity_score": 10,
            "auto_resolvable": True,
        }
    )
    return svc


@pytest.fixture(autouse=True)
def _patch_all_services(monkeypatch):
    """Apply all service mocks before each test.

    Uses monkeypatch to set attributes on the imported module so handler
    functions see the mocked services during execution.
    """
    import src.mcp_server as mod

    # Ensure _services_initialized is True so handlers don't try init
    monkeypatch.setattr(mod, "_services_initialized", True)

    # Patch config flags
    monkeypatch.setattr(mod, "ACTR_ENABLED", False)
    monkeypatch.setattr(mod, "PATTERN_RECOGNITION_ENABLED", False)
    monkeypatch.setattr(mod, "EVOLUTION_ENABLED", False)

    # Patch each service
    monkeypatch.setattr(mod, "neo4j_service", _mock_neo4j())
    monkeypatch.setattr(mod, "redis_service", _mock_redis())
    monkeypatch.setattr(mod, "file_service", _mock_pai())
    monkeypatch.setattr(mod, "event_log_service", _mock_event_log())
    monkeypatch.setattr(mod, "entity_cache", _mock_entity_cache())
    monkeypatch.setattr(mod, "graph_factory", _mock_graph_factory())
    monkeypatch.setattr(mod, "classification_service", _mock_classification_service())
    monkeypatch.setattr(mod, "entity_factory", _mock_entity_factory())
    monkeypatch.setattr(mod, "relationship_factory", _mock_relationship_factory())
    monkeypatch.setattr(mod, "activation_service", _mock_activation_service())
    monkeypatch.setattr(mod, "conflict_detection_service", _mock_conflict_detection())
    monkeypatch.setattr(mod, "procedure_service", _mock_procedure_service())
    monkeypatch.setattr(mod, "llm_service_singleton", _mock_llm_service())
    monkeypatch.setattr(mod, "sync_service", MagicMock())
    monkeypatch.setattr(mod, "temporal_validity_service", MagicMock())
    monkeypatch.setattr(mod, "validation_history_service", MagicMock())

    # Factory functions
    monkeypatch.setattr(mod, "get_vector_service", lambda: _mock_vector_service())
    monkeypatch.setattr(
        mod, "get_context_retrieval_service", lambda: _mock_context_retrieval()
    )

    # Fire-and-forget should be a no-op (consumes the coroutine to prevent warnings)
    def _noop_fire_and_forget(coro, description=""):
        # Close the coroutine to prevent RuntimeWarning
        coro.close()
        return None

    monkeypatch.setattr(mod, "_fire_and_forget", _noop_fire_and_forget)

    # Disable auto_capture_tool event logging
    monkeypatch.setattr("src.config.EVENT_LOG_AUTO_CAPTURE", False)


# ===========================================================================
# Category 1: Health and Diagnostics
# ===========================================================================


class TestHealthDiagnostics:
    """Tests for health, echo, verify_storage, discover_schema handlers."""

    async def test_echo(self):
        from src.mcp_server import echo

        result = await echo(message="hello")
        assert isinstance(result, str)
        assert "hello" in result

    async def test_verify_storage(self):
        from src.mcp_server import verify_storage

        result = await verify_storage()
        assert isinstance(result, str)
        assert "verified" in result.lower() or "Storage" in result

    async def test_health(self):
        from src.mcp_server import health

        result = await health()
        assert isinstance(result, str)
        data = json.loads(result)
        assert "status" in data

    async def test_discover_schema(self):
        from src.mcp_server import discover_schema

        result = await discover_schema()
        assert isinstance(result, str)
        assert "Schema" in result


# ===========================================================================
# Category 2: Context Loading and Retrieval
# ===========================================================================


class TestContextLoading:
    """Tests for load_context, load_context_cascade, semantic_search."""

    async def test_load_context_no_prompt(self):
        from src.mcp_server import load_context

        result = await load_context(prompt="", days=7, limit=10)
        assert isinstance(result, str)

    async def test_load_context_with_prompt(self):
        from src.mcp_server import load_context

        result = await load_context(prompt="test query", days=7, limit=5)
        assert isinstance(result, str)

    async def test_load_context_cascade(self):
        from src.mcp_server import load_context_cascade

        result = await load_context_cascade(prompt="test query", max_results=5)
        assert isinstance(result, str)

    async def test_semantic_search(self):
        from src.mcp_server import semantic_search

        result = await semantic_search(query="test query", search_type="all", limit=5)
        assert isinstance(result, str)

    async def test_semantic_search_min_score(self):
        from src.mcp_server import semantic_search

        result = await semantic_search(
            query="test", search_type="learnings", limit=3, min_score=0.5
        )
        assert isinstance(result, str)


# ===========================================================================
# Category 3: Memory Capture
# ===========================================================================


class TestMemoryCapture:
    """Tests for capture_learning, append_to_learning, capture_session,
    capture_fact, capture_context."""

    async def test_capture_learning(self):
        import src.mcp_server as mod
        from src.mcp_server import capture_learning

        # Neo4j query must return a non-empty result for learning creation
        mod.neo4j_service.query = AsyncMock(
            return_value=[
                {
                    "uuid": "test-uuid",
                    "title": "Test Learning",
                    "category": "test",
                }
            ]
        )

        # Mock dedup service to return "proceed"
        mock_dedup = AsyncMock()
        mock_dedup.validate_learning = AsyncMock(
            return_value=MagicMock(
                status="proceed",
                validation_time_ms=1.0,
            )
        )

        with patch(
            "src.services.deduplication_service.get_deduplication_service",
            return_value=mock_dedup,
        ):
            result = await capture_learning(
                title="Test Learning",
                content="Test content for learning",
                category="test",
                auto_deduplicate=True,
            )

        assert isinstance(result, str)
        assert "Test Learning" in result

    async def test_capture_learning_no_dedup(self):
        import src.mcp_server as mod
        from src.mcp_server import capture_learning

        # Neo4j query must return a non-empty result for learning creation
        mod.neo4j_service.query = AsyncMock(
            return_value=[
                {
                    "uuid": "test-uuid",
                    "title": "Test Learning No Dedup",
                    "category": "test",
                }
            ]
        )

        result = await capture_learning(
            title="Test Learning No Dedup",
            content="Test content",
            category="test",
            auto_deduplicate=False,
        )
        assert isinstance(result, str)
        assert "Test Learning No Dedup" in result

    async def test_append_to_learning(self):
        from src.mcp_server import append_to_learning

        mock_dedup = AsyncMock()
        mock_dedup.append_to_learning = AsyncMock(
            return_value=MagicMock(
                status="success",
                learning_uuid="test-uuid",
                append_count=1,
                file_path="/tmp/test.md",
            )
        )

        with patch(
            "src.services.deduplication_service.get_deduplication_service",
            return_value=mock_dedup,
        ):
            result = await append_to_learning(
                learning_uuid="test-uuid-123",
                new_content="Additional content",
                new_section_title="New Section",
            )
        assert isinstance(result, str)

    async def test_capture_session(self):
        import src.mcp_server as mod
        from src.mcp_server import capture_session

        # Neo4j query should return session creation result
        mod.neo4j_service.query = AsyncMock(
            return_value=[
                {
                    "uuid": str(uuid4()),
                    "summary": "test summary",
                }
            ]
        )

        result = await capture_session(
            conversation_summary="Test session summary",
            key_insights=["Insight 1", "Insight 2"],
        )
        assert isinstance(result, str)
        assert "Test session summary" in result

    async def test_capture_session_no_insights(self):
        import src.mcp_server as mod
        from src.mcp_server import capture_session

        mod.neo4j_service.query = AsyncMock(
            return_value=[
                {
                    "uuid": str(uuid4()),
                    "summary": "test",
                }
            ]
        )

        result = await capture_session(
            conversation_summary="Session without insights",
        )
        assert isinstance(result, str)

    async def test_capture_fact(self):
        from src.mcp_server import capture_fact

        result = await capture_fact(
            entity1="Python",
            relation="IS_LANGUAGE_OF",
            entity2="Mnemodrome",
            context="Testing",
            entity1_type="Technology",
            entity2_type="Project",
        )
        assert isinstance(result, str)
        assert "Python" in result

    async def test_capture_fact_auto_classify(self):
        from src.mcp_server import capture_fact

        result = await capture_fact(
            entity1="Alice",
            relation="WORKS_AT",
            entity2="Acme",
            context="Employment relationship",
        )
        assert isinstance(result, str)

    async def test_capture_context(self):
        from src.mcp_server import capture_context

        result = await capture_context(
            context="Alice works at Acme Corp and uses Python for development."
        )
        assert isinstance(result, str)
        assert "Graph" in result or "captured" in result.lower()


# ===========================================================================
# Category 4: Fact Management (invalidate, update)
# ===========================================================================


class TestFactManagement:
    """Tests for invalidate_fact and update_fact."""

    async def test_invalidate_fact_not_found(self):
        import src.mcp_server as mod
        from src.mcp_server import invalidate_fact

        mod.neo4j_service.find_relationship = AsyncMock(return_value=None)

        result = await invalidate_fact(
            entity1="A",
            relation="REL",
            entity2="B",
            reason="outdated",
        )
        # invalidate_fact returns a dict, not a string
        assert isinstance(result, dict)
        assert result["status"] == "error"

    async def test_invalidate_fact_auto_resolve(self):
        import src.mcp_server as mod
        from src.mcp_server import invalidate_fact

        mod.neo4j_service.find_relationship = AsyncMock(
            return_value={
                "uuid": str(uuid4()),
                "entity1_name": "A",
                "relation_type": "REL",
                "entity2_name": "B",
                "created_at": datetime.now(UTC),
                "invalid_at": None,
                "expired_at": None,
            }
        )
        mod.conflict_detection_service.calculate_complexity_score = MagicMock(
            return_value=10
        )

        result = await invalidate_fact(
            entity1="A",
            relation="REL",
            entity2="B",
            reason="outdated",
        )
        assert isinstance(result, dict)
        assert result["status"] == "success"

    async def test_update_fact_old_not_found(self):
        import src.mcp_server as mod
        from src.mcp_server import update_fact

        mod.neo4j_service.find_relationship = AsyncMock(return_value=None)

        result = await update_fact(
            entity1="A",
            relation="REL",
            entity2_old="B",
            entity2_new="C",
        )
        assert isinstance(result, dict)
        assert result["status"] == "error"

    async def test_update_fact_prompt_mode(self):
        import src.mcp_server as mod
        from src.mcp_server import update_fact

        mod.neo4j_service.find_relationship = AsyncMock(
            return_value={
                "uuid": str(uuid4()),
                "entity1_name": "A",
                "entity1_uuid": str(uuid4()),
                "relation_type": "REL",
                "entity2_name": "B",
                "created_at": datetime.now(UTC),
            }
        )

        result = await update_fact(
            entity1="A",
            relation="REL",
            entity2_old="B",
            entity2_new="C",
            conflict_resolution="prompt",
        )
        assert isinstance(result, dict)
        assert result["status"] == "validation_required"


# ===========================================================================
# Category 5: Event Log
# ===========================================================================


class TestEventLog:
    """Tests for log_event, query_events, get_session_timeline, get_event_log_health."""

    async def test_log_event(self):
        from src.mcp_server import log_event

        result = await log_event(
            event_type="test",
            action="create",
            actor="test_user",
        )
        assert isinstance(result, str)
        assert "Event" in result

    async def test_query_events_empty(self):
        from src.mcp_server import query_events

        result = await query_events()
        assert isinstance(result, str)

    async def test_query_events_with_filters(self):
        from src.mcp_server import query_events

        result = await query_events(
            event_type="test",
            action="create",
            actor="system",
            limit=50,
            offset=0,
        )
        assert isinstance(result, str)

    async def test_get_session_timeline(self):
        from src.mcp_server import get_session_timeline

        result = await get_session_timeline(session_id="", limit=100)
        assert isinstance(result, str)

    async def test_get_event_log_health(self):
        from src.mcp_server import get_event_log_health

        result = await get_event_log_health()
        assert isinstance(result, str)
        assert "Event Log" in result


# ===========================================================================
# Category 6: Vector Layer
# ===========================================================================


class TestVectorLayer:
    """Tests for vector_health, store_prompt_embedding."""

    async def test_vector_health(self):
        from src.mcp_server import vector_health

        result = await vector_health()
        assert isinstance(result, str)
        assert "Vector" in result

    async def test_store_prompt_embedding(self):
        from src.mcp_server import store_prompt_embedding

        result = await store_prompt_embedding(
            prompt_text="What is Mnemodrome?",
            session_id="test-session",
        )
        assert isinstance(result, str)

    async def test_store_prompt_embedding_with_metadata(self):
        from src.mcp_server import store_prompt_embedding

        result = await store_prompt_embedding(
            prompt_text="How does activation work?",
            session_id="test",
            metadata={"source": "test"},
        )
        assert isinstance(result, str)


# ===========================================================================
# Category 7: Validation and Correction
# ===========================================================================


class TestValidationCorrection:
    """Tests for validate_prompt, validate_reasoning, auto_correct_response."""

    async def test_validate_prompt_no_llm(self):
        import src.mcp_server as mod
        from src.mcp_server import validate_prompt

        mod.llm_service_singleton = None

        result = await validate_prompt(prompt_text="test prompt")
        assert isinstance(result, str)
        assert "Unavailable" in result

    async def test_validate_prompt_no_claims(self):
        import src.mcp_server as mod
        from src.mcp_server import validate_prompt

        mod.llm_service_singleton = _mock_llm_service()

        with patch("src.mcp_server.ClaimExtractionService") as MockCES:
            instance = MockCES.return_value
            instance.extract_claims = AsyncMock(return_value=[])
            result = await validate_prompt(prompt_text="test prompt", timeout_ms=5000)

        assert isinstance(result, str)
        assert "No" in result

    async def test_validate_reasoning_no_llm(self):
        import src.mcp_server as mod
        from src.mcp_server import validate_reasoning

        mod.llm_service_singleton = None

        result = await validate_reasoning(proposed_response="test response")
        assert isinstance(result, str)
        assert "Unavailable" in result

    async def test_validate_reasoning_with_service(self):
        import src.mcp_server as mod
        from src.mcp_server import validate_reasoning

        mod.llm_service_singleton = _mock_llm_service()

        mock_result = MagicMock()
        mock_result.validation_status = "clean"
        mock_result.elapsed_ms = 10.0
        mock_result.summary = "No issues found"
        mock_result.conflicts = []
        mock_result.relevant_learnings = []
        mock_result.relevant_patterns = []
        mock_result.claims_analyzed = 0
        mock_result.entities_checked = 0
        mock_result.degraded_queries = []

        with patch("src.mcp_server.ReasoningValidationService") as MockRVS:
            instance = MockRVS.return_value
            instance.validate_response = AsyncMock(return_value=mock_result)
            result = await validate_reasoning(proposed_response="test response")

        assert isinstance(result, str)
        assert "Reasoning Validation" in result

    async def test_auto_correct_response_no_llm(self):
        import src.mcp_server as mod
        from src.mcp_server import auto_correct_response

        mod.llm_service_singleton = None

        result = await auto_correct_response(response_text="test response")
        assert isinstance(result, str)
        assert "Unavailable" in result

    async def test_auto_correct_response_no_claims(self):
        import src.mcp_server as mod
        from src.mcp_server import auto_correct_response

        mod.llm_service_singleton = _mock_llm_service()

        with patch("src.mcp_server.ClaimExtractionService") as MockCES:
            instance = MockCES.return_value
            instance.extract_claims = AsyncMock(return_value=[])
            result = await auto_correct_response(
                response_text="test response",
                timeout_ms=5000,
            )

        assert isinstance(result, str)
        assert "No Corrections" in result or "No factual" in result


# ===========================================================================
# Category 8: Pattern Recognition
# ===========================================================================


class TestPatternRecognition:
    """Tests for extract_patterns_from_recent_sessions, query_patterns."""

    async def test_extract_patterns_invalid_days(self):
        from src.mcp_server import extract_patterns_from_recent_sessions

        result = await extract_patterns_from_recent_sessions(days_back=0)
        assert isinstance(result, str)
        assert "Error" in result

    async def test_extract_patterns_no_sessions(self):
        import src.mcp_server as mod
        from src.mcp_server import extract_patterns_from_recent_sessions

        mod.neo4j_service.query = AsyncMock(return_value=[])

        result = await extract_patterns_from_recent_sessions(
            days_back=30, min_sessions=3
        )
        assert isinstance(result, str)
        assert "No sessions" in result or "Pattern" in result

    async def test_query_patterns_invalid_rate(self):
        from src.mcp_server import query_patterns

        result = await query_patterns(min_success_rate=1.5)
        assert isinstance(result, str)
        assert "Error" in result

    async def test_query_patterns_no_results(self):
        from src.mcp_server import query_patterns

        with patch("src.mcp_server.PatternQueryService") as MockPQS:
            instance = MockPQS.return_value
            instance.query_patterns = AsyncMock(return_value=[])

            result = await query_patterns(
                problem_type="", min_success_rate=0.0, limit=10
            )

        assert isinstance(result, str)
        assert "Pattern" in result


# ===========================================================================
# Category 9: Procedural Memory
# ===========================================================================


class TestProceduralMemory:
    """Tests for capture_procedure, find_procedure."""

    async def test_capture_procedure(self):
        from src.mcp_server import capture_procedure

        result = await capture_procedure(
            symptom="Service won't start",
            diagnostic_steps="1. Check logs\n2. Check ports",
            root_cause="Port conflict",
            fix="Change port in config",
            verification="Service starts successfully",
            problem_type="infrastructure",
            technologies="docker,python",
            outcome="SUCCESS",
        )
        assert isinstance(result, str)
        assert "captured" in result.lower() or "Procedure" in result

    async def test_capture_procedure_empty_steps(self):
        from src.mcp_server import capture_procedure

        result = await capture_procedure(
            symptom="Test",
            diagnostic_steps="",
            root_cause="Test",
            fix="Test",
            verification="Test",
        )
        assert isinstance(result, str)
        assert "Error" in result or "required" in result.lower()

    async def test_find_procedure_no_results(self):
        from src.mcp_server import find_procedure

        result = await find_procedure(
            problem_description="Service won't start",
            problem_type="infrastructure",
        )
        assert isinstance(result, str)
        assert "No matching" in result or "procedures" in result.lower()

    async def test_find_procedure_with_results(self):
        import src.mcp_server as mod
        from src.mcp_server import find_procedure

        mock_match = MagicMock()
        mock_match.procedure_uuid = str(uuid4())
        mock_match.problem_type = "infrastructure"
        mock_match.symptom = "Service won't start"
        mock_match.diagnostic_steps = ["Check logs", "Check ports"]
        mock_match.root_cause = "Port conflict"
        mock_match.fix = "Change port"
        mock_match.verification = "Service starts"
        mock_match.outcome = "SUCCESS"
        mock_match.technologies = ["docker"]
        mock_match.similarity_score = 0.95
        mock_match.linked_pattern_name = None
        mock_match.linked_pattern_uuid = None

        mod.procedure_service.find_procedures = AsyncMock(return_value=[mock_match])

        result = await find_procedure(
            problem_description="Service won't start",
        )
        assert isinstance(result, str)
        assert "Service won't start" in result


# ===========================================================================
# Category 10: Evolution and Activation
# ===========================================================================


class TestEvolutionActivation:
    """Tests for suggest_instruction_updates, get_activation_diagnostics."""

    async def test_suggest_instruction_updates_disabled(self):
        import src.mcp_server as mod
        from src.mcp_server import suggest_instruction_updates

        mod.EVOLUTION_ENABLED = False

        result = await suggest_instruction_updates()
        assert isinstance(result, str)
        assert "disabled" in result.lower()

    async def test_suggest_instruction_updates_enabled_no_candidates(self):
        import src.mcp_server as mod
        from src.mcp_server import suggest_instruction_updates

        mod.EVOLUTION_ENABLED = True

        with patch("src.mcp_server.EvolutionCandidateService") as MockECS:
            instance = MockECS.return_value
            instance.generate_candidates = AsyncMock(
                return_value={
                    "candidates": [],
                    "skills_analyzed": 5,
                    "patterns_evaluated": 10,
                }
            )

            result = await suggest_instruction_updates()

        assert isinstance(result, str)
        assert "No evolution" in result or "candidates" in result.lower()

    async def test_get_activation_diagnostics_no_args(self):
        from src.mcp_server import get_activation_diagnostics

        result = await get_activation_diagnostics()
        assert isinstance(result, str)
        assert "provide" in result.lower() or "entity_name" in result

    async def test_get_activation_diagnostics_not_found(self):
        import src.mcp_server as mod
        from src.mcp_server import get_activation_diagnostics

        mod.neo4j_service.query = AsyncMock(return_value=[])

        result = await get_activation_diagnostics(entity_name="NonExistent")
        assert isinstance(result, str)
        assert "not found" in result.lower()

    async def test_get_activation_diagnostics_found(self):
        import src.mcp_server as mod
        from src.mcp_server import get_activation_diagnostics

        entity_uuid = str(uuid4())

        # First call: entity lookup, second call: relationships
        mod.neo4j_service.query = AsyncMock(
            side_effect=[
                [
                    {
                        "uuid": entity_uuid,
                        "name": "TestEntity",
                        "entity_type": "Person",
                        "access_count": 5,
                        "last_accessed": datetime.now(UTC),
                        "created_at": datetime.now(UTC),
                    }
                ],
                [],  # no relationships
            ]
        )

        result = await get_activation_diagnostics(entity_name="TestEntity")
        assert isinstance(result, str)
        assert "TestEntity" in result
        assert "Activation" in result
