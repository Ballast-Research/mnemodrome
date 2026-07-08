"""
Feature 031: Procedural Memory Layer - Integration Tests

Full procedure lifecycle tests with mocked services (no live Neo4j/Qdrant).
Covers: capture, deduplication, enrichment, search, graceful degradation,
and context cascade integration.
"""

import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.procedure_models import Procedure, ProcedureMatch
from src.services.context_retrieval_service import ContextRetrievalService
from src.services.procedure_service import COLLECTION_PROCEDURES, ProcedureService

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------
_PATCH_NEO4J = "src.services.procedure_service.neo4j_service"
_PATCH_VECTOR = "src.services.procedure_service.get_vector_service"
_PATCH_ENTITY = "src.services.procedure_service.entity_factory"
_PATCH_EVENT = "src.services.procedure_service.event_log_service"
_PATCH_FILE_SERVICE = "src.services.procedure_service.file_service"

_PATCH_CASCADE_PROC = "src.services.context_retrieval_service.procedure_service"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_test_procedure(**overrides):
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


def _make_vector_service(*, enabled=True, initialized=True, embed_result=None):
    """Build a MagicMock that behaves like VectorService."""
    vs = MagicMock()
    vs.enabled = enabled
    vs._initialized = initialized
    vs.embed_text = AsyncMock(return_value=embed_result or [0.1] * 384)
    vs.client = MagicMock()
    # Default: query_points returns empty
    vs.client.query_points = MagicMock(return_value=SimpleNamespace(points=[]))
    vs.client.upsert = MagicMock()
    return vs


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
class TestProcedureLifecycle:
    """Integration tests for the full procedure lifecycle."""

    # ---- 1. Capture -> Find round-trip ---------------------------------
    async def test_capture_then_find_round_trip(self):
        """Capture a procedure, then find it by the same symptom text."""
        proc = make_test_procedure()
        proc_uuid = proc.procedure_uuid

        vector_svc = _make_vector_service()
        # Dedup check returns no duplicate
        vector_svc.client.query_points = MagicMock(
            return_value=SimpleNamespace(points=[])
        )

        mock_neo4j = MagicMock()
        mock_neo4j.query = AsyncMock(return_value=[{"uuid": proc_uuid}])

        with (
            patch(_PATCH_NEO4J, mock_neo4j),
            patch(_PATCH_VECTOR, return_value=vector_svc),
        ):
            svc = ProcedureService()

            # -- Capture --
            returned_uuid, is_new = await svc.create_procedure(proc)
            assert is_new is True
            assert returned_uuid == proc_uuid

            # -- Prepare find: Qdrant returns a hit for the same symptom --
            hit = SimpleNamespace(
                payload={
                    "uuid": proc_uuid,
                    "problem_type": "infrastructure",
                    "symptom": proc.symptom,
                    "root_cause": proc.root_cause,
                    "technologies": proc.technologies,
                    "outcome": proc.outcome,
                    "created_at": proc.created_at.isoformat(),
                },
                score=0.95,
            )
            vector_svc.client.query_points = MagicMock(
                return_value=SimpleNamespace(points=[hit])
            )

            # Neo4j enrichment for found hit
            mock_neo4j.query = AsyncMock(
                return_value=[
                    {
                        "uuid": proc_uuid,
                        "problem_type": "infrastructure",
                        "symptom": proc.symptom,
                        "diagnostic_steps": json.dumps(proc.diagnostic_steps),
                        "root_cause": proc.root_cause,
                        "fix": proc.fix,
                        "verification": proc.verification,
                        "outcome": proc.outcome,
                        "technologies": json.dumps(proc.technologies),
                        "pattern_uuid": "",
                        "created_at": proc.created_at.isoformat(),
                    }
                ]
            )

            # -- Find --
            results = await svc.find_procedures(query_text=proc.symptom)
            assert len(results) == 1
            match = results[0]
            assert match.procedure_uuid == proc_uuid
            assert match.symptom == proc.symptom
            assert match.similarity_score == 0.95

    # ---- 2. Capture -> Find with rephrased symptom --------------------
    async def test_find_with_rephrased_symptom(self):
        """Capture about Docker failing; search with different wording."""
        proc = make_test_procedure(
            procedure_uuid="docker-proc-001",
            symptom="Docker containers failing to start after deployment",
            root_cause="Port conflict with host network bindings",
            fix="Release conflicting port and redeploy containers",
        )

        vector_svc = _make_vector_service()
        # Dedup: no duplicate
        vector_svc.client.query_points = MagicMock(
            return_value=SimpleNamespace(points=[])
        )

        mock_neo4j = MagicMock()
        mock_neo4j.query = AsyncMock(return_value=[{"uuid": proc.procedure_uuid}])

        with (
            patch(_PATCH_NEO4J, mock_neo4j),
            patch(_PATCH_VECTOR, return_value=vector_svc),
        ):
            svc = ProcedureService()
            await svc.create_procedure(proc)

            # Search with rephrased symptom -> mock Qdrant high similarity
            hit = SimpleNamespace(
                payload={
                    "uuid": proc.procedure_uuid,
                    "problem_type": "infrastructure",
                    "symptom": proc.symptom,
                    "root_cause": proc.root_cause,
                    "technologies": proc.technologies,
                    "outcome": "SUCCESS",
                    "created_at": proc.created_at.isoformat(),
                },
                score=0.88,
            )
            vector_svc.client.query_points = MagicMock(
                return_value=SimpleNamespace(points=[hit])
            )

            mock_neo4j.query = AsyncMock(
                return_value=[
                    {
                        "uuid": proc.procedure_uuid,
                        "problem_type": "infrastructure",
                        "symptom": proc.symptom,
                        "diagnostic_steps": json.dumps(proc.diagnostic_steps),
                        "root_cause": proc.root_cause,
                        "fix": proc.fix,
                        "verification": proc.verification,
                        "outcome": proc.outcome,
                        "technologies": json.dumps(proc.technologies),
                        "pattern_uuid": "",
                        "created_at": proc.created_at.isoformat(),
                    }
                ]
            )

            results = await svc.find_procedures(query_text="containers won't launch")
            assert len(results) == 1
            assert results[0].procedure_uuid == proc.procedure_uuid
            assert results[0].similarity_score == 0.88

    # ---- 3. problem_type filter works ----------------------------------
    async def test_problem_type_filter(self):
        """Only procedures matching the requested problem_type are returned."""
        infra_proc = make_test_procedure(
            procedure_uuid="infra-001",
            problem_type="infrastructure",
        )
        code_proc = make_test_procedure(
            procedure_uuid="code-001",
            problem_type="code",
            symptom="Null pointer exception in the request handler",
            root_cause="Missing null check on optional field access",
        )

        vector_svc = _make_vector_service()
        # No duplicates during creation
        vector_svc.client.query_points = MagicMock(
            return_value=SimpleNamespace(points=[])
        )

        mock_neo4j = MagicMock()
        mock_neo4j.query = AsyncMock(return_value=[{"uuid": "infra-001"}])

        with (
            patch(_PATCH_NEO4J, mock_neo4j),
            patch(_PATCH_VECTOR, return_value=vector_svc),
        ):
            svc = ProcedureService()
            await svc.create_procedure(infra_proc)

            mock_neo4j.query = AsyncMock(return_value=[{"uuid": "code-001"}])
            await svc.create_procedure(code_proc)

            # Search filtered to "infrastructure" -> only infra procedure
            hit = SimpleNamespace(
                payload={
                    "uuid": "infra-001",
                    "problem_type": "infrastructure",
                    "symptom": infra_proc.symptom,
                    "root_cause": infra_proc.root_cause,
                    "technologies": infra_proc.technologies,
                    "outcome": "SUCCESS",
                    "created_at": infra_proc.created_at.isoformat(),
                },
                score=0.85,
            )
            vector_svc.client.query_points = MagicMock(
                return_value=SimpleNamespace(points=[hit])
            )

            mock_neo4j.query = AsyncMock(
                return_value=[
                    {
                        "uuid": "infra-001",
                        "problem_type": "infrastructure",
                        "symptom": infra_proc.symptom,
                        "diagnostic_steps": json.dumps(infra_proc.diagnostic_steps),
                        "root_cause": infra_proc.root_cause,
                        "fix": infra_proc.fix,
                        "verification": infra_proc.verification,
                        "outcome": "SUCCESS",
                        "technologies": json.dumps(infra_proc.technologies),
                        "pattern_uuid": "",
                        "created_at": infra_proc.created_at.isoformat(),
                    }
                ]
            )

            results = await svc.find_procedures(
                query_text="slow service",
                problem_type="infrastructure",
            )
            assert len(results) == 1
            assert results[0].procedure_uuid == "infra-001"
            assert results[0].problem_type == "infrastructure"

    # ---- 4. Deduplication prevents double-capture ----------------------
    async def test_deduplication_prevents_double_capture(self):
        """Second capture with >0.92 similarity returns existing UUID."""
        proc = make_test_procedure(procedure_uuid="orig-uuid-111")

        vector_svc = _make_vector_service()
        # First capture: no duplicate
        vector_svc.client.query_points = MagicMock(
            return_value=SimpleNamespace(points=[])
        )

        mock_neo4j = MagicMock()
        mock_neo4j.query = AsyncMock(return_value=[{"uuid": proc.procedure_uuid}])

        with (
            patch(_PATCH_NEO4J, mock_neo4j),
            patch(_PATCH_VECTOR, return_value=vector_svc),
        ):
            svc = ProcedureService()
            uuid1, is_new1 = await svc.create_procedure(proc)
            assert is_new1 is True
            assert uuid1 == "orig-uuid-111"

            # Second capture: Qdrant returns existing as duplicate (score > 0.92)
            dup_hit = SimpleNamespace(
                payload={"uuid": "orig-uuid-111"},
                score=0.96,
            )
            vector_svc.client.query_points = MagicMock(
                return_value=SimpleNamespace(points=[dup_hit])
            )

            proc2 = make_test_procedure(procedure_uuid="new-uuid-222")
            uuid2, is_new2 = await svc.create_procedure(proc2)
            assert is_new2 is False
            assert uuid2 == "orig-uuid-111"

    # ---- 5. Enrichment completes all artifacts -------------------------
    async def test_enrichment_completes_all_artifacts(self):
        """Directly call enrich_procedure; verify all enrichment tasks run."""
        proc = make_test_procedure(
            procedure_uuid="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        )

        vector_svc = _make_vector_service()
        mock_neo4j = MagicMock()
        mock_neo4j.query = AsyncMock(return_value=[{"uuid": proc.procedure_uuid}])

        mock_entity = MagicMock()
        mock_entity.create_or_resolve = AsyncMock(return_value="tech-uuid-001")

        mock_event = MagicMock()
        mock_event.log_event = AsyncMock()

        mock_pai = MagicMock()
        mock_pai.write_file = MagicMock()

        with (
            patch(_PATCH_NEO4J, mock_neo4j),
            patch(_PATCH_VECTOR, return_value=vector_svc),
            patch(_PATCH_ENTITY, mock_entity),
            patch(_PATCH_EVENT, mock_event),
            patch(_PATCH_FILE_SERVICE, mock_pai),
        ):
            svc = ProcedureService()
            await svc.enrich_procedure(proc, session_uuid="session-001")

            # 1. Qdrant embedding stored
            vector_svc.client.upsert.assert_called_once()
            upsert_kwargs = vector_svc.client.upsert.call_args
            assert upsert_kwargs[1]["collection_name"] == COLLECTION_PROCEDURES

            # 2. Markdown file written
            mock_pai.write_file.assert_called_once()

            # 3. Technology entities created (Docker + Python = 2 calls)
            assert mock_entity.create_or_resolve.call_count == 2

            # 4. USES_TECHNOLOGY relationships created (2 techs = query for
            #    neo4j node creation + 2 relationship queries)
            #    Also CAPTURED_IN relationship query for session_uuid
            neo4j_calls = mock_neo4j.query.call_args_list
            # At least 2 USES_TECHNOLOGY + 1 PRODUCED_PROCEDURE
            assert mock_neo4j.query.call_count >= 3

            # 5. Event logged
            mock_event.log_event.assert_called_once()
            event_kwargs = mock_event.log_event.call_args
            assert event_kwargs[1]["event_type"] == "procedure_capture"
            assert event_kwargs[1]["target"] == proc.procedure_uuid

    # ---- 6. Graceful degradation - Qdrant down -------------------------
    async def test_graceful_degradation_qdrant_down(self):
        """Capture succeeds via Neo4j when Qdrant is unavailable; find falls back."""
        proc = make_test_procedure(procedure_uuid="resilient-uuid-001")

        # Vector service disabled (Qdrant down)
        vector_svc = _make_vector_service(enabled=False, initialized=False)

        mock_neo4j = MagicMock()
        mock_neo4j.query = AsyncMock(return_value=[{"uuid": proc.procedure_uuid}])

        with (
            patch(_PATCH_NEO4J, mock_neo4j),
            patch(_PATCH_VECTOR, return_value=vector_svc),
        ):
            svc = ProcedureService()

            # Capture should still succeed (Neo4j write, dedup skipped)
            uuid_out, is_new = await svc.create_procedure(proc)
            assert is_new is True
            assert uuid_out == proc.procedure_uuid

            # Verify Neo4j was called for the write
            assert mock_neo4j.query.call_count >= 1

            # Find should fall back to Neo4j text search
            mock_neo4j.query = AsyncMock(
                return_value=[
                    {
                        "p": {
                            "uuid": proc.procedure_uuid,
                            "problem_type": "infrastructure",
                            "symptom": proc.symptom,
                            "diagnostic_steps": json.dumps(proc.diagnostic_steps),
                            "root_cause": proc.root_cause,
                            "fix": proc.fix,
                            "verification": proc.verification,
                            "outcome": "SUCCESS",
                            "technologies": json.dumps(proc.technologies),
                            "created_at": proc.created_at.isoformat(),
                        }
                    }
                ]
            )

            results = await svc.find_procedures(
                query_text="slowly high latency service responding"
            )
            assert len(results) >= 1
            assert results[0].procedure_uuid == proc.procedure_uuid
            # Neo4j fallback has similarity_score 0.0
            assert results[0].similarity_score == 0.0

    # ---- 7. Context cascade includes procedures -----------------------
    async def test_context_cascade_includes_procedures(self):
        """retrieve_context_for_prompt includes procedures for problem prompts."""
        proc_match = ProcedureMatch(
            procedure_uuid="cascade-proc-001",
            problem_type="infrastructure",
            symptom="Database connection timeout under load",
            diagnostic_steps=["Check connection pool", "Check DB CPU"],
            root_cause="Connection pool exhaustion",
            fix="Increase pool size and add connection timeout",
            verification="Sustained load test with no timeouts",
            outcome="SUCCESS",
            technologies=["PostgreSQL", "Python"],
            similarity_score=0.91,
            created_at=datetime.utcnow(),
        )

        mock_proc_svc = MagicMock()
        mock_proc_svc.find_procedures = AsyncMock(return_value=[proc_match])

        # Build a ContextRetrievalService with mocked internals
        mock_vector = MagicMock()
        mock_vector.enabled = True
        mock_vector._initialized = True
        mock_vector.initialize = AsyncMock()
        mock_vector.search_similar_prompts = AsyncMock(return_value=[])
        mock_vector.search_similar_learnings = AsyncMock(return_value=[])

        with (
            patch(_PATCH_CASCADE_PROC, mock_proc_svc),
            patch(
                "src.services.context_retrieval_service.get_vector_service",
                return_value=mock_vector,
            ),
        ):
            svc = ContextRetrievalService()
            result = await svc.retrieve_context_for_prompt(
                prompt="Database connection timeout error under heavy load",
                include_procedures=True,
            )

            # Procedure search should have been called
            mock_proc_svc.find_procedures.assert_called_once()

            # Procedures present in result
            assert "procedures" in result
            assert len(result["procedures"]) == 1
            assert result["procedures"][0].procedure_uuid == "cascade-proc-001"
