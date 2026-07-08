"""
Performance SLA Validation for Procedural Memory Layer (Feature 031)

Verifies that procedure service code paths meet latency SLAs:
- capture_procedure sync path: <500ms (p95)
- find_procedure search: <800ms (p95)
- enrich_procedure fire-and-forget: completes within 5s
- Dedup check overhead: <100ms
- Neo4j fallback search: <800ms

All tests use mocked services to isolate code-path latency from
infrastructure latency.
"""

import json
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.procedure_models import Procedure
from src.services.procedure_service import ProcedureService

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------
_PATCH_NEO4J = "src.services.procedure_service.neo4j_service"
_PATCH_VECTOR = "src.services.procedure_service.get_vector_service"
_PATCH_ENTITY = "src.services.procedure_service.entity_factory"
_PATCH_EVENT = "src.services.procedure_service.event_log_service"
_PATCH_FILE_SERVICE = "src.services.procedure_service.file_service"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_test_procedure(**overrides):
    defaults = {
        "procedure_uuid": "perf-test-uuid",
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


def _make_mock_vector_service(enabled=True, initialized=True):
    vs = MagicMock()
    vs.enabled = enabled
    vs._initialized = initialized
    vs.embed_text = AsyncMock(return_value=[0.1] * 768)
    vs.client = MagicMock()
    return vs


def _qdrant_result_with_points(points):
    result = MagicMock()
    result.points = points
    return result


def _percentile(values, pct):
    """Return the *pct*-th percentile from a sorted list of values."""
    sorted_v = sorted(values)
    idx = int(len(sorted_v) * pct / 100)
    idx = min(idx, len(sorted_v) - 1)
    return sorted_v[idx]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.performance
class TestProcedurePerformanceSLA:
    """SLA validation tests for procedural memory service."""

    # ---- Test 1: capture_procedure sync path under 500ms ----
    @pytest.mark.asyncio
    async def test_capture_procedure_sync_under_500ms(self):
        """
        Time create_procedure() with mocked Neo4j and Qdrant.
        Run 10 iterations and assert p95 < 500ms.
        """
        svc = ProcedureService()

        mock_neo4j = MagicMock()
        mock_neo4j.query = AsyncMock(return_value=[{"uuid": "perf-test-uuid"}])

        mock_vs = _make_mock_vector_service()
        # Dedup check returns no duplicate
        mock_vs.client.query_points = MagicMock(
            return_value=_qdrant_result_with_points([])
        )

        timings_ms: list[float] = []

        with (
            patch(_PATCH_NEO4J, mock_neo4j),
            patch(_PATCH_VECTOR, return_value=mock_vs),
        ):
            for i in range(10):
                proc = make_test_procedure(procedure_uuid=f"perf-cap-{i}")
                start = time.perf_counter()
                uuid, is_new = await svc.create_procedure(proc)
                elapsed_ms = (time.perf_counter() - start) * 1000
                timings_ms.append(elapsed_ms)
                assert is_new is True

        p95 = _percentile(timings_ms, 95)
        print("\n=== capture_procedure sync timings (ms) ===")
        print(
            f"  min={min(timings_ms):.2f}  median={_percentile(timings_ms, 50):.2f}  p95={p95:.2f}"
        )
        assert p95 < 500, f"p95 capture_procedure latency {p95:.2f}ms exceeds 500ms SLA"

    # ---- Test 2: find_procedure search under 800ms ----
    @pytest.mark.asyncio
    async def test_find_procedure_search_under_800ms(self):
        """
        Time find_procedures() with mocked Qdrant search and Neo4j
        enrichment.  Run 10 iterations and assert p95 < 800ms.
        """
        svc = ProcedureService()

        # Build a fake Qdrant hit
        hit = MagicMock()
        hit.payload = {
            "uuid": "perf-find-uuid",
            "problem_type": "infrastructure",
            "symptom": "High latency",
            "root_cause": "Memory leak",
            "technologies": ["Docker"],
            "outcome": "SUCCESS",
            "created_at": datetime.utcnow().isoformat(),
        }
        hit.score = 0.88

        mock_vs = _make_mock_vector_service()
        mock_vs.client.query_points = MagicMock(
            return_value=_qdrant_result_with_points([hit])
        )

        mock_neo4j = MagicMock()
        mock_neo4j.query = AsyncMock(
            return_value=[
                {
                    "uuid": "perf-find-uuid",
                    "problem_type": "infrastructure",
                    "symptom": "High latency",
                    "diagnostic_steps": json.dumps(["Step 1"]),
                    "root_cause": "Memory leak",
                    "fix": "Restart the worker process",
                    "verification": "Check response times",
                    "outcome": "SUCCESS",
                    "technologies": json.dumps(["Docker"]),
                    "pattern_uuid": "",
                    "created_at": datetime.utcnow().isoformat(),
                }
            ]
        )

        timings_ms: list[float] = []

        with (
            patch(_PATCH_NEO4J, mock_neo4j),
            patch(_PATCH_VECTOR, return_value=mock_vs),
        ):
            for _ in range(10):
                start = time.perf_counter()
                matches = await svc.find_procedures("high latency in service")
                elapsed_ms = (time.perf_counter() - start) * 1000
                timings_ms.append(elapsed_ms)
                assert len(matches) >= 1

        p95 = _percentile(timings_ms, 95)
        print("\n=== find_procedures timings (ms) ===")
        print(
            f"  min={min(timings_ms):.2f}  median={_percentile(timings_ms, 50):.2f}  p95={p95:.2f}"
        )
        assert p95 < 800, f"p95 find_procedures latency {p95:.2f}ms exceeds 800ms SLA"

    # ---- Test 3: enrich_procedure completes within 5s ----
    @pytest.mark.asyncio
    async def test_enrich_procedure_completes_within_5s(self):
        """
        Time enrich_procedure() with all services mocked.
        Must complete within 5 seconds.
        """
        svc = ProcedureService()

        # Valid hex UUID required for embedding storage (uuid.UUID parsing)
        proc = make_test_procedure(
            procedure_uuid="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            technologies=["Docker", "Python"],
            pattern_uuid="bbbbbbbb-0000-0000-0000-000000000001",
            session_uuid="cccccccc-0000-0000-0000-000000000001",
        )

        mock_neo4j = MagicMock()
        mock_neo4j.query = AsyncMock(return_value=[{"uuid": proc.procedure_uuid}])

        mock_vs = _make_mock_vector_service()
        mock_vs.client.upsert = MagicMock(return_value=None)

        mock_entity = MagicMock()
        mock_entity.create_or_resolve = AsyncMock(return_value="tech-uuid-1")

        mock_event = MagicMock()
        mock_event.log_event = AsyncMock(return_value=None)

        mock_pai = MagicMock()
        mock_pai.write_file = MagicMock(return_value=None)

        with (
            patch(_PATCH_NEO4J, mock_neo4j),
            patch(_PATCH_VECTOR, return_value=mock_vs),
            patch(_PATCH_ENTITY, mock_entity),
            patch(_PATCH_EVENT, mock_event),
            patch(_PATCH_FILE_SERVICE, mock_pai),
        ):
            start = time.perf_counter()
            await svc.enrich_procedure(
                proc, session_uuid="cccccccc-0000-0000-0000-000000000001"
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

        print("\n=== enrich_procedure timing ===")
        print(f"  elapsed={elapsed_ms:.2f}ms  (SLA: <5000ms)")
        assert elapsed_ms < 5000, (
            f"enrich_procedure took {elapsed_ms:.2f}ms, exceeds 5s SLA"
        )

    # ---- Test 4: Dedup check adds minimal overhead ----
    @pytest.mark.asyncio
    async def test_dedup_check_minimal_overhead(self):
        """
        Time _check_duplicate() with mocked Qdrant.
        Must add less than 100ms overhead.
        """
        svc = ProcedureService()

        mock_vs = _make_mock_vector_service()
        mock_vs.client.query_points = MagicMock(
            return_value=_qdrant_result_with_points([])
        )

        timings_ms: list[float] = []

        with patch(_PATCH_VECTOR, return_value=mock_vs):
            for _ in range(10):
                start = time.perf_counter()
                result = await svc._check_duplicate(
                    symptom="Service is responding slowly",
                    root_cause="Memory leak in worker",
                    problem_type="infrastructure",
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                timings_ms.append(elapsed_ms)
                assert result is None

        p95 = _percentile(timings_ms, 95)
        print("\n=== _check_duplicate timings (ms) ===")
        print(
            f"  min={min(timings_ms):.2f}  median={_percentile(timings_ms, 50):.2f}  p95={p95:.2f}"
        )
        assert p95 < 100, f"p95 _check_duplicate overhead {p95:.2f}ms exceeds 100ms"

    # ---- Test 5: Neo4j fallback search under 800ms ----
    @pytest.mark.asyncio
    async def test_neo4j_fallback_search_under_800ms(self):
        """
        Time find_procedures() when Qdrant is disabled (falls back to
        Neo4j text search).  Run 10 iterations and assert p95 < 800ms.
        """
        svc = ProcedureService()

        # Qdrant disabled -> triggers Neo4j fallback path
        mock_vs = _make_mock_vector_service(enabled=False, initialized=False)

        mock_neo4j = MagicMock()
        mock_neo4j.query = AsyncMock(
            return_value=[
                {
                    "p": {
                        "uuid": "fallback-uuid",
                        "problem_type": "infrastructure",
                        "symptom": "High latency in service",
                        "diagnostic_steps": json.dumps(["Check CPU"]),
                        "root_cause": "Memory leak",
                        "fix": "Restart worker",
                        "verification": "Check response times",
                        "outcome": "SUCCESS",
                        "technologies": json.dumps(["Docker"]),
                        "created_at": datetime.utcnow().isoformat(),
                    }
                }
            ]
        )

        timings_ms: list[float] = []

        with (
            patch(_PATCH_NEO4J, mock_neo4j),
            patch(_PATCH_VECTOR, return_value=mock_vs),
        ):
            for _ in range(10):
                start = time.perf_counter()
                matches = await svc.find_procedures("high latency service")
                elapsed_ms = (time.perf_counter() - start) * 1000
                timings_ms.append(elapsed_ms)
                assert len(matches) >= 1

        p95 = _percentile(timings_ms, 95)
        print("\n=== Neo4j fallback search timings (ms) ===")
        print(
            f"  min={min(timings_ms):.2f}  median={_percentile(timings_ms, 50):.2f}  p95={p95:.2f}"
        )
        assert p95 < 800, f"p95 Neo4j fallback latency {p95:.2f}ms exceeds 800ms SLA"
