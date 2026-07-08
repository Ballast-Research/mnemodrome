"""
Unit tests for ProcedureService (Feature 031: Procedural Memory Layer).

Tests the create, enrich, find, and deduplication workflows with fully mocked
backing services (Neo4j, Qdrant, EntityFactory, EventLog, FileService).
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.procedure_models import Procedure, ProcedureMatch
from src.services.procedure_service import COLLECTION_PROCEDURES, ProcedureService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_test_procedure(**overrides):
    """Build a valid Procedure with sensible defaults."""
    defaults = {
        "procedure_uuid": "test-uuid-123",
        "problem_type": "infrastructure",
        "symptom": "Service is responding slowly with high latency",
        "diagnostic_steps": ["Check CPU usage", "Check memory usage", "Check network"],
        "root_cause": "Memory leak in background worker process",
        "fix": "Restart the worker and apply memory limit fix",
        "verification": "Response times returned to normal under 200ms",
        "outcome": "SUCCESS",
        "technologies": ["Docker", "Python"],
    }
    defaults.update(overrides)
    return Procedure(**defaults)


def _make_mock_vector_service(enabled=True, initialized=True, embedding=None):
    """Return a mock that quacks like VectorService."""
    if embedding is None:
        embedding = [0.1] * 768
    vs = MagicMock()
    vs.enabled = enabled
    vs._initialized = initialized
    vs.embed_text = AsyncMock(return_value=embedding)
    vs.client = MagicMock()
    return vs


def _make_qdrant_hit(uuid="existing-uuid-999", score=0.95, payload=None):
    """Build a fake Qdrant ScoredPoint."""
    hit = MagicMock()
    hit.score = score
    hit.payload = payload or {
        "uuid": uuid,
        "problem_type": "infrastructure",
        "symptom": "Service is responding slowly with high latency",
        "root_cause": "Memory leak in background worker process",
        "technologies": ["Docker", "Python"],
        "outcome": "SUCCESS",
        "created_at": datetime.utcnow().isoformat(),
    }
    return hit


def _qdrant_result_with_points(points):
    """Wrap a list of hits in the QueryResponse-like object Qdrant returns."""
    result = MagicMock()
    result.points = points
    return result


# ---------------------------------------------------------------------------
# Common patch targets (in the *procedure_service* module namespace)
# ---------------------------------------------------------------------------
_PATCH_NEO4J = "src.services.procedure_service.neo4j_service"
_PATCH_VECTOR = "src.services.procedure_service.get_vector_service"
_PATCH_ENTITY = "src.services.procedure_service.entity_factory"
_PATCH_EVENT = "src.services.procedure_service.event_log_service"
_PATCH_FILE_SERVICE = "src.services.procedure_service.file_service"


# ===================================================================
# 1. create_procedure -- new procedure (no duplicates)
# ===================================================================
class TestCreateProcedureNew:
    """create_procedure() when no duplicate is found."""

    @pytest.mark.asyncio
    async def test_creates_neo4j_node_and_returns_uuid(self):
        """Should write to Neo4j and return (uuid, True)."""
        mock_vs = _make_mock_vector_service()
        # Qdrant search returns no matches
        mock_vs.client.query_points = MagicMock(
            return_value=_qdrant_result_with_points([])
        )

        with (
            patch(_PATCH_NEO4J) as mock_neo4j,
            patch(_PATCH_VECTOR, return_value=mock_vs),
        ):
            mock_neo4j.query = AsyncMock(return_value=[{"uuid": "test-uuid-123"}])

            svc = ProcedureService()
            proc = make_test_procedure()
            result_uuid, is_new = await svc.create_procedure(proc)

            assert result_uuid == "test-uuid-123"
            assert is_new is True
            # Verify Neo4j CREATE was called
            mock_neo4j.query.assert_called_once()
            cypher_arg = mock_neo4j.query.call_args[0][0]
            assert "CREATE (p:Procedure" in cypher_arg

    @pytest.mark.asyncio
    async def test_generates_uuid_when_empty(self):
        """Should auto-generate a UUID when procedure_uuid is empty."""
        mock_vs = _make_mock_vector_service()
        mock_vs.client.query_points = MagicMock(
            return_value=_qdrant_result_with_points([])
        )

        with (
            patch(_PATCH_NEO4J) as mock_neo4j,
            patch(_PATCH_VECTOR, return_value=mock_vs),
        ):
            mock_neo4j.query = AsyncMock(return_value=[{"uuid": "auto"}])

            svc = ProcedureService()
            proc = make_test_procedure(procedure_uuid="")
            result_uuid, is_new = await svc.create_procedure(proc)

            assert result_uuid != ""
            assert is_new is True


# ===================================================================
# 2. create_procedure -- duplicate found (similarity > 0.92)
# ===================================================================
class TestCreateProcedureDuplicate:
    """create_procedure() when Qdrant finds a near-duplicate."""

    @pytest.mark.asyncio
    async def test_returns_existing_uuid_when_duplicate(self):
        """Should return existing UUID and is_new=False."""
        mock_vs = _make_mock_vector_service()
        hit = _make_qdrant_hit(uuid="existing-uuid-999", score=0.95)
        mock_vs.client.query_points = MagicMock(
            return_value=_qdrant_result_with_points([hit])
        )

        with (
            patch(_PATCH_NEO4J) as mock_neo4j,
            patch(_PATCH_VECTOR, return_value=mock_vs),
        ):
            mock_neo4j.query = AsyncMock()

            svc = ProcedureService()
            proc = make_test_procedure()
            result_uuid, is_new = await svc.create_procedure(proc)

            assert result_uuid == "existing-uuid-999"
            assert is_new is False
            # Neo4j CREATE should NOT have been called
            mock_neo4j.query.assert_not_called()


# ===================================================================
# 3. create_procedure -- same symptom, different problem_type (NOT duplicate)
# ===================================================================
class TestCreateProcedureDifferentProblemType:
    """Dedup uses a problem_type filter, so different types should not match."""

    @pytest.mark.asyncio
    async def test_different_problem_type_not_deduplicated(self):
        """Same symptom + different problem_type should create a new procedure."""
        mock_vs = _make_mock_vector_service()
        # Qdrant returns no matches because the filter excludes other types
        mock_vs.client.query_points = MagicMock(
            return_value=_qdrant_result_with_points([])
        )

        with (
            patch(_PATCH_NEO4J) as mock_neo4j,
            patch(_PATCH_VECTOR, return_value=mock_vs),
        ):
            mock_neo4j.query = AsyncMock(return_value=[{"uuid": "new-uuid"}])

            svc = ProcedureService()
            proc = make_test_procedure(problem_type="code")
            result_uuid, is_new = await svc.create_procedure(proc)

            assert is_new is True
            # Verify the Qdrant search was called (dedup was attempted)
            mock_vs.client.query_points.assert_called_once()
            # Verify problem_type filter was passed
            call_kwargs = mock_vs.client.query_points.call_args
            query_filter = call_kwargs.kwargs.get("query_filter") or call_kwargs[1].get(
                "query_filter"
            )
            # The filter object should exist and reference "code"
            assert query_filter is not None


# ===================================================================
# 4. _write_neo4j_node -- Cypher + params
# ===================================================================
class TestWriteNeo4jNode:
    """Verify _write_neo4j_node sends correct Cypher and params."""

    @pytest.mark.asyncio
    async def test_cypher_and_params(self):
        proc = make_test_procedure()

        with patch(_PATCH_NEO4J) as mock_neo4j:
            mock_neo4j.query = AsyncMock(return_value=[{"uuid": proc.procedure_uuid}])

            svc = ProcedureService()
            await svc._write_neo4j_node(proc)

            mock_neo4j.query.assert_called_once()
            cypher_arg, params_arg = mock_neo4j.query.call_args[0]

            assert "CREATE (p:Procedure" in cypher_arg
            assert "RETURN p.uuid as uuid" in cypher_arg

            assert params_arg["uuid"] == "test-uuid-123"
            assert params_arg["problem_type"] == "infrastructure"
            assert params_arg["symptom"] == proc.symptom
            assert params_arg["root_cause"] == proc.root_cause
            assert params_arg["fix"] == proc.fix
            assert params_arg["verification"] == proc.verification
            assert params_arg["outcome"] == "SUCCESS"
            # JSON-encoded list fields
            assert json.loads(params_arg["diagnostic_steps"]) == proc.diagnostic_steps
            assert json.loads(params_arg["technologies"]) == proc.technologies
            assert params_arg["pattern_uuid"] == ""
            assert params_arg["session_uuid"] == ""

    @pytest.mark.asyncio
    async def test_raises_on_empty_result(self):
        """Should raise if Neo4j returns empty result."""
        proc = make_test_procedure()

        with patch(_PATCH_NEO4J) as mock_neo4j:
            mock_neo4j.query = AsyncMock(return_value=[])

            svc = ProcedureService()
            with pytest.raises(Exception, match="Failed to create Procedure node"):
                await svc._write_neo4j_node(proc)


# ===================================================================
# 5. _check_duplicate -- Qdrant search with filter + threshold
# ===================================================================
class TestCheckDuplicate:
    """Verify _check_duplicate queries Qdrant with correct parameters."""

    @pytest.mark.asyncio
    async def test_returns_uuid_on_match(self):
        mock_vs = _make_mock_vector_service()
        hit = _make_qdrant_hit(uuid="dup-uuid-777", score=0.96)
        mock_vs.client.query_points = MagicMock(
            return_value=_qdrant_result_with_points([hit])
        )

        with patch(_PATCH_VECTOR, return_value=mock_vs):
            svc = ProcedureService()
            result = await svc._check_duplicate(
                symptom="Service is responding slowly with high latency",
                root_cause="Memory leak in background worker process",
                problem_type="infrastructure",
            )

            assert result == "dup-uuid-777"
            # Verify embed_text was called with symptom + root_cause
            mock_vs.embed_text.assert_called_once_with(
                "Service is responding slowly with high latency "
                "Memory leak in background worker process"
            )
            # Verify query_points params
            call_kwargs = mock_vs.client.query_points.call_args
            assert call_kwargs.kwargs["collection_name"] == COLLECTION_PROCEDURES
            assert call_kwargs.kwargs["limit"] == 1

    @pytest.mark.asyncio
    async def test_returns_none_when_no_match(self):
        mock_vs = _make_mock_vector_service()
        mock_vs.client.query_points = MagicMock(
            return_value=_qdrant_result_with_points([])
        )

        with patch(_PATCH_VECTOR, return_value=mock_vs):
            svc = ProcedureService()
            result = await svc._check_duplicate(
                symptom="Service is responding slowly with high latency",
                root_cause="Memory leak in background worker process",
                problem_type="infrastructure",
            )
            assert result is None


# ===================================================================
# 6. _check_duplicate -- Qdrant unavailable (returns None gracefully)
# ===================================================================
class TestCheckDuplicateQdrantUnavailable:
    """When Qdrant is down, _check_duplicate should return None."""

    @pytest.mark.asyncio
    async def test_returns_none_when_vector_service_disabled(self):
        mock_vs = _make_mock_vector_service(enabled=False)

        with patch(_PATCH_VECTOR, return_value=mock_vs):
            svc = ProcedureService()
            result = await svc._check_duplicate(
                symptom="Service is responding slowly with high latency",
                root_cause="Memory leak in background worker process",
                problem_type="infrastructure",
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_not_initialized(self):
        mock_vs = _make_mock_vector_service(initialized=False)

        with patch(_PATCH_VECTOR, return_value=mock_vs):
            svc = ProcedureService()
            result = await svc._check_duplicate(
                symptom="Service is responding slowly with high latency",
                root_cause="Memory leak in background worker process",
                problem_type="infrastructure",
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_embed_fails(self):
        mock_vs = _make_mock_vector_service()
        mock_vs.embed_text = AsyncMock(return_value=None)

        with patch(_PATCH_VECTOR, return_value=mock_vs):
            svc = ProcedureService()
            result = await svc._check_duplicate(
                symptom="Service is responding slowly with high latency",
                root_cause="Memory leak in background worker process",
                problem_type="infrastructure",
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_query_raises(self):
        mock_vs = _make_mock_vector_service()
        mock_vs.client.query_points = MagicMock(
            side_effect=Exception("Qdrant connection refused")
        )

        with patch(_PATCH_VECTOR, return_value=mock_vs):
            svc = ProcedureService()
            result = await svc._check_duplicate(
                symptom="Service is responding slowly with high latency",
                root_cause="Memory leak in background worker process",
                problem_type="infrastructure",
            )
            assert result is None


# ===================================================================
# 7. enrich_procedure -- all 6 enrichment steps called
# ===================================================================
class TestEnrichProcedure:
    """Verify the 6-step enrichment pipeline."""

    @pytest.mark.asyncio
    async def test_all_enrichment_steps_execute(self):
        """All 6 steps should run: embedding, markdown, tech links,
        pattern link, session link, event log."""
        mock_vs = _make_mock_vector_service()

        # Must use a valid hex UUID because store_embedding calls uuid.UUID()
        proc = make_test_procedure(
            procedure_uuid="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            pattern_uuid="pat-abc",
            session_uuid="sess-xyz",
        )

        with (
            patch(_PATCH_NEO4J) as mock_neo4j,
            patch(_PATCH_VECTOR, return_value=mock_vs),
            patch(_PATCH_ENTITY) as mock_entity,
            patch(_PATCH_EVENT) as mock_event,
            patch(_PATCH_FILE_SERVICE) as mock_pai,
        ):
            mock_neo4j.query = AsyncMock(return_value=[{"uuid": proc.procedure_uuid}])
            mock_entity.create_or_resolve = AsyncMock(return_value="tech-uuid-1")
            mock_event.log_event = AsyncMock()
            mock_pai.write_file = MagicMock()

            svc = ProcedureService()
            await svc.enrich_procedure(proc, session_uuid="sess-xyz")

            # 1. Embedding stored
            mock_vs.embed_text.assert_called_once()
            mock_vs.client.upsert.assert_called_once()

            # 2. Markdown written
            mock_pai.write_file.assert_called_once()
            md_content = mock_pai.write_file.call_args[0][1]
            assert proc.symptom[:80] in md_content

            # 3. Technology entities created (2 technologies: Docker, Python)
            assert mock_entity.create_or_resolve.call_count == 2
            tech_calls = [
                c.kwargs["name"] for c in mock_entity.create_or_resolve.call_args_list
            ]
            assert "Docker" in tech_calls
            assert "Python" in tech_calls

            # 4+5. Pattern + Session Neo4j relationships
            # Neo4j query called for: 2 USES_TECHNOLOGY + 1 INSTANTIATES + 1 CAPTURED_IN = 4
            assert mock_neo4j.query.call_count == 4

            # 6. Event logged
            mock_event.log_event.assert_called_once()
            event_kwargs = mock_event.log_event.call_args.kwargs
            assert event_kwargs["event_type"] == "procedure_capture"
            assert event_kwargs["action"] == "captured"


# ===================================================================
# 8. enrich_procedure -- individual step failures handled gracefully
# ===================================================================
class TestEnrichProcedureStepFailures:
    """Individual enrichment failures should be logged but not raised."""

    @pytest.mark.asyncio
    async def test_embedding_failure_does_not_raise(self):
        """If embedding fails, other steps should still run."""
        mock_vs = _make_mock_vector_service()
        mock_vs.embed_text = AsyncMock(side_effect=Exception("Embedding model offline"))

        proc = make_test_procedure()

        with (
            patch(_PATCH_NEO4J) as mock_neo4j,
            patch(_PATCH_VECTOR, return_value=mock_vs),
            patch(_PATCH_ENTITY) as mock_entity,
            patch(_PATCH_EVENT) as mock_event,
            patch(_PATCH_FILE_SERVICE) as mock_pai,
        ):
            mock_neo4j.query = AsyncMock(return_value=[{}])
            mock_entity.create_or_resolve = AsyncMock(return_value="tech-uuid")
            mock_event.log_event = AsyncMock()
            mock_pai.write_file = MagicMock()

            svc = ProcedureService()
            # Should NOT raise
            await svc.enrich_procedure(proc)

            # Markdown and event log should still have been called
            mock_pai.write_file.assert_called_once()
            mock_event.log_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_all_steps_fail_does_not_raise(self):
        """Even if every step fails, enrich_procedure should not propagate."""
        mock_vs = _make_mock_vector_service()
        mock_vs.embed_text = AsyncMock(side_effect=Exception("boom"))

        proc = make_test_procedure()

        with (
            patch(_PATCH_NEO4J) as mock_neo4j,
            patch(_PATCH_VECTOR, return_value=mock_vs),
            patch(_PATCH_ENTITY) as mock_entity,
            patch(_PATCH_EVENT) as mock_event,
            patch(_PATCH_FILE_SERVICE) as mock_pai,
        ):
            mock_neo4j.query = AsyncMock(side_effect=Exception("neo4j down"))
            mock_entity.create_or_resolve = AsyncMock(
                side_effect=Exception("entity fail")
            )
            mock_event.log_event = AsyncMock(side_effect=Exception("event fail"))
            mock_pai.write_file = MagicMock(side_effect=Exception("pai fail"))

            svc = ProcedureService()
            await svc.enrich_procedure(proc)  # should not raise


# ===================================================================
# 9. find_procedures -- via Qdrant
# ===================================================================
class TestFindProceduresQdrant:
    """find_procedures() semantic search via Qdrant."""

    @pytest.mark.asyncio
    async def test_returns_procedure_matches(self):
        mock_vs = _make_mock_vector_service()
        hit = _make_qdrant_hit(uuid="found-uuid-1", score=0.88)
        mock_vs.client.query_points = MagicMock(
            return_value=_qdrant_result_with_points([hit])
        )

        with (
            patch(_PATCH_NEO4J) as mock_neo4j,
            patch(_PATCH_VECTOR, return_value=mock_vs),
        ):
            # Neo4j enrichment returns full procedure data
            mock_neo4j.query = AsyncMock(
                return_value=[
                    {
                        "uuid": "found-uuid-1",
                        "problem_type": "infrastructure",
                        "symptom": "Service is responding slowly with high latency",
                        "diagnostic_steps": json.dumps(["Check CPU", "Check memory"]),
                        "root_cause": "Memory leak in background worker process",
                        "fix": "Restart the worker and apply memory limit fix",
                        "verification": "Response times returned to normal",
                        "outcome": "SUCCESS",
                        "technologies": json.dumps(["Docker"]),
                        "pattern_uuid": "",
                        "created_at": datetime.utcnow().isoformat(),
                    }
                ]
            )

            svc = ProcedureService()
            matches = await svc.find_procedures("slow service high latency")

            assert len(matches) == 1
            assert isinstance(matches[0], ProcedureMatch)
            assert matches[0].procedure_uuid == "found-uuid-1"
            assert matches[0].similarity_score == 0.88
            assert matches[0].problem_type == "infrastructure"

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_embedding(self):
        mock_vs = _make_mock_vector_service()
        mock_vs.embed_text = AsyncMock(return_value=None)

        with patch(_PATCH_VECTOR, return_value=mock_vs):
            svc = ProcedureService()
            matches = await svc.find_procedures("anything")
            assert matches == []


# ===================================================================
# 10. find_procedures -- Neo4j fallback when Qdrant unavailable
# ===================================================================
class TestFindProceduresNeo4jFallback:
    """When Qdrant is disabled, fall back to Neo4j text search."""

    @pytest.mark.asyncio
    async def test_neo4j_fallback_returns_matches(self):
        mock_vs = _make_mock_vector_service(enabled=False)

        with (
            patch(_PATCH_NEO4J) as mock_neo4j,
            patch(_PATCH_VECTOR, return_value=mock_vs),
        ):
            mock_neo4j.query = AsyncMock(
                return_value=[
                    {
                        "p": {
                            "uuid": "neo4j-uuid-1",
                            "problem_type": "infrastructure",
                            "symptom": "Service timeout during deployment",
                            "diagnostic_steps": json.dumps(["Check pods"]),
                            "root_cause": "Memory limit exceeded during startup",
                            "fix": "Increase memory limits in deployment manifest",
                            "verification": "Verify pod readiness probe passes",
                            "outcome": "SUCCESS",
                            "technologies": json.dumps(["Kubernetes"]),
                            "created_at": datetime.utcnow().isoformat(),
                        }
                    }
                ]
            )

            svc = ProcedureService()
            matches = await svc.find_procedures("service timeout deployment memory")

            assert len(matches) == 1
            assert matches[0].procedure_uuid == "neo4j-uuid-1"
            assert matches[0].similarity_score == 0.0  # No semantic score for fallback
            assert matches[0].technologies == ["Kubernetes"]

    @pytest.mark.asyncio
    async def test_neo4j_fallback_when_qdrant_search_fails(self):
        """Qdrant is enabled but search raises -> falls back to Neo4j."""
        mock_vs = _make_mock_vector_service()
        mock_vs.embed_text = AsyncMock(side_effect=Exception("Qdrant unreachable"))

        with (
            patch(_PATCH_NEO4J) as mock_neo4j,
            patch(_PATCH_VECTOR, return_value=mock_vs),
        ):
            mock_neo4j.query = AsyncMock(
                return_value=[
                    {
                        "p": {
                            "uuid": "fallback-uuid",
                            "problem_type": "code",
                            "symptom": "Function returns None unexpectedly",
                            "diagnostic_steps": json.dumps(["Check return"]),
                            "root_cause": "Missing return in error handler path",
                            "fix": "Add explicit return statement in except block",
                            "verification": "Unit tests pass after fix applied",
                            "outcome": "SUCCESS",
                            "technologies": json.dumps(["Python"]),
                            "created_at": None,
                        }
                    }
                ]
            )

            svc = ProcedureService()
            matches = await svc.find_procedures("function returns None unexpectedly")

            assert len(matches) == 1
            assert matches[0].procedure_uuid == "fallback-uuid"


# ===================================================================
# 11. find_procedures -- with problem_type filter
# ===================================================================
class TestFindProceduresWithFilter:
    """Verify problem_type filter is applied to both Qdrant and Neo4j paths."""

    @pytest.mark.asyncio
    async def test_qdrant_search_with_problem_type_filter(self):
        mock_vs = _make_mock_vector_service()
        hit = _make_qdrant_hit(uuid="filtered-uuid", score=0.85)
        mock_vs.client.query_points = MagicMock(
            return_value=_qdrant_result_with_points([hit])
        )

        with (
            patch(_PATCH_NEO4J) as mock_neo4j,
            patch(_PATCH_VECTOR, return_value=mock_vs),
        ):
            mock_neo4j.query = AsyncMock(
                return_value=[
                    {
                        "uuid": "filtered-uuid",
                        "problem_type": "deployment",
                        "symptom": "Deployment pipeline stalls during rollout",
                        "diagnostic_steps": json.dumps(["Check CI logs"]),
                        "root_cause": "Docker image pull rate limit exceeded",
                        "fix": "Add registry mirror to pipeline configuration",
                        "verification": "Deploy succeeds within 2 minutes target",
                        "outcome": "SUCCESS",
                        "technologies": json.dumps(["Docker", "GitLab CI"]),
                        "pattern_uuid": "",
                        "created_at": datetime.utcnow().isoformat(),
                    }
                ]
            )

            svc = ProcedureService()
            matches = await svc.find_procedures(
                "deployment stalls",
                problem_type="deployment",
            )

            assert len(matches) == 1
            # Verify filter was passed to Qdrant
            call_kwargs = mock_vs.client.query_points.call_args
            qf = call_kwargs.kwargs.get("query_filter") or call_kwargs[1].get(
                "query_filter"
            )
            assert qf is not None

    @pytest.mark.asyncio
    async def test_neo4j_fallback_with_problem_type_filter(self):
        mock_vs = _make_mock_vector_service(enabled=False)

        with (
            patch(_PATCH_NEO4J) as mock_neo4j,
            patch(_PATCH_VECTOR, return_value=mock_vs),
        ):
            mock_neo4j.query = AsyncMock(return_value=[])

            svc = ProcedureService()
            matches = await svc.find_procedures(
                "deployment stalls during rollout",
                problem_type="deployment",
            )

            # Verify the Cypher query includes problem_type filter
            cypher_arg = mock_neo4j.query.call_args[0][0]
            assert "p.problem_type = $problem_type" in cypher_arg
            params_arg = mock_neo4j.query.call_args[0][1]
            assert params_arg["problem_type"] == "deployment"
