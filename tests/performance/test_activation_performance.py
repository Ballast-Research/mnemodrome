"""
Performance Benchmark for ACT-R Activation Scoring (Feature 034).

Measures activation overhead and validates NFR-001: average overhead < 50ms.

Tests:
1. Activation scoring overhead: _apply_activation_scoring() with/without ACTR_ENABLED
2. Pure function performance: calculate_activation(), compute_final_score(), bayesian_update()

All tests use mocked Neo4j to isolate code-path latency from infrastructure latency.
"""

import time
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from src.config import ACTR_DECAY_EXPONENT
from src.services.activation_service import (
    bayesian_update,
    calculate_activation,
    compute_final_score,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _percentile(values, pct):
    """Return the *pct*-th percentile from a sorted list of values."""
    sorted_v = sorted(values)
    idx = int(len(sorted_v) * pct / 100)
    idx = min(idx, len(sorted_v) - 1)
    return sorted_v[idx]


def _make_mock_neo4j_results(num_entities: int = 5):
    """Create realistic mock Neo4j activation data."""
    now = datetime.now(UTC)
    results = []
    for i in range(num_entities):
        results.append(
            {
                "uuid": f"test-entity-{i}",
                "access_count": 5 + (i * 2),  # Vary access counts
                "last_accessed": now - timedelta(days=i),
                "created_at": now - timedelta(days=30),
            }
        )
    return results


def _make_mock_results(num_results: int = 10):
    """Create realistic mock retrieval results with source/target structure."""
    results = []
    for i in range(num_results):
        results.append(
            {
                "source": {
                    "uuid": f"entity-source-{i}",
                    "type": "Fact",
                    "title": f"Source Entity {i}",
                },
                "target": {
                    "uuid": f"entity-target-{i}",
                    "type": "Fact",
                    "title": f"Target Entity {i}",
                },
                "relation": "RELATES_TO",
                "weight": 0.8 - (i * 0.05),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Test Class
# ---------------------------------------------------------------------------


@pytest.mark.performance
class TestActivationPerformance:
    """Performance benchmarks for ACT-R activation scoring."""

    # ---- Test 1: Activation Scoring Overhead ----
    @pytest.mark.asyncio
    async def test_activation_scoring_overhead_under_50ms_average(self):
        """
        Measure overhead of _apply_activation_scoring() with ACTR_ENABLED=true
        vs disabled (baseline).

        Runs 50 iterations with mocked Neo4j retrieval, asserts average
        overhead < 50ms per call (NFR-001).
        """
        from src.mcp_server import _apply_activation_scoring

        # Create realistic test data
        mock_results = _make_mock_results(num_results=10)
        mock_neo4j_data = _make_mock_neo4j_results(num_entities=20)

        # Content scores: simulate relevance ranking
        content_scores = [1.0 - (i / 10) for i in range(10)]

        timings_enabled_ms: list[float] = []
        timings_disabled_ms: list[float] = []

        # Test with ACTR_ENABLED=true
        with patch("src.mcp_server.ACTR_ENABLED", True):
            with patch(
                "src.mcp_server.neo4j_service.query", new_callable=AsyncMock
            ) as mock_query:
                mock_query.return_value = mock_neo4j_data

                for i in range(50):
                    start = time.perf_counter()
                    result = await _apply_activation_scoring(
                        results=mock_results.copy(),
                        content_scores=content_scores.copy(),
                    )
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    timings_enabled_ms.append(elapsed_ms)
                    assert len(result) == 10, "All results should be returned"

        # Test with ACTR_ENABLED=false (baseline: should skip activation)
        with patch("src.mcp_server.ACTR_ENABLED", False):
            for i in range(50):
                start = time.perf_counter()
                result = await _apply_activation_scoring(
                    results=mock_results.copy(),
                    content_scores=content_scores.copy(),
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                timings_disabled_ms.append(elapsed_ms)
                assert result == mock_results, "Should return unchanged when disabled"

        # Calculate overhead
        avg_enabled = sum(timings_enabled_ms) / len(timings_enabled_ms)
        avg_disabled = sum(timings_disabled_ms) / len(timings_disabled_ms)
        overhead = avg_enabled - avg_disabled

        p95_enabled = _percentile(timings_enabled_ms, 95)
        p95_disabled = _percentile(timings_disabled_ms, 95)

        print("\n=== Activation Scoring Overhead ===")
        print(f"  ACTR_ENABLED=true:  avg={avg_enabled:.2f}ms  p95={p95_enabled:.2f}ms")
        print(
            f"  ACTR_ENABLED=false: avg={avg_disabled:.2f}ms  p95={p95_disabled:.2f}ms"
        )
        print(f"  Overhead: {overhead:.2f}ms (average per call)")

        # NFR-001: Average overhead must be < 50ms
        assert overhead < 50.0, (
            f"Activation overhead {overhead:.2f}ms exceeds 50ms SLA (NFR-001)"
        )

    # ---- Test 2: calculate_activation() Pure Function ----
    @pytest.mark.asyncio
    async def test_calculate_activation_under_1ms(self):
        """
        Time pure calculate_activation() function (no I/O).

        Run 1000 iterations with varying access counts and last_accessed times.
        Assert average < 1ms per call.
        """
        now = datetime.now(UTC)
        timings_ms: list[float] = []

        # Run 1000 iterations with varied parameters
        for i in range(1000):
            access_count = i % 20  # 0-19
            days_ago = i % 180  # 0-179 days old

            start = time.perf_counter()
            activation = calculate_activation(
                access_count=access_count,
                last_accessed=now - timedelta(days=days_ago),
                decay_exponent=ACTR_DECAY_EXPONENT,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            timings_ms.append(elapsed_ms)

            # Activation should be a reasonable float
            assert isinstance(activation, float)
            assert -10 < activation < 10  # Realistic range

        avg = sum(timings_ms) / len(timings_ms)
        p95 = _percentile(timings_ms, 95)
        p99 = _percentile(timings_ms, 99)

        print("\n=== calculate_activation() Pure Function ===")
        print(f"  avg={avg:.4f}ms  p95={p95:.4f}ms  p99={p99:.4f}ms")

        assert avg < 1.0, f"Average latency {avg:.4f}ms exceeds 1ms target"

    # ---- Test 3: compute_final_score() Pure Function ----
    @pytest.mark.asyncio
    async def test_compute_final_score_under_1ms(self):
        """
        Time pure compute_final_score() function (no I/O).

        Run 1000 iterations with varying content/activation/hebbian values.
        Assert average < 1ms per call.
        """
        timings_ms: list[float] = []

        # Run 1000 iterations with varied parameters
        for i in range(1000):
            content_match = (i % 100) / 100.0  # 0.0-1.0
            activation = (i % 60) / 10.0  # 0.0-6.0
            hebbian_boost = (i % 20) / 100.0  # 0.0-0.2

            start = time.perf_counter()
            score = compute_final_score(
                content_match=content_match,
                activation=activation,
                hebbian_boost=hebbian_boost,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            timings_ms.append(elapsed_ms)

            # Score should be a reasonable float
            assert isinstance(score, float)
            assert 0.0 <= score <= 2.0  # Realistic range with activation + softplus

        avg = sum(timings_ms) / len(timings_ms)
        p95 = _percentile(timings_ms, 95)
        p99 = _percentile(timings_ms, 99)

        print("\n=== compute_final_score() Pure Function ===")
        print(f"  avg={avg:.4f}ms  p95={p95:.4f}ms  p99={p99:.4f}ms")

        assert avg < 1.0, f"Average latency {avg:.4f}ms exceeds 1ms target"

    # ---- Test 4: bayesian_update() Pure Function ----
    @pytest.mark.asyncio
    async def test_bayesian_update_under_1ms(self):
        """
        Time pure bayesian_update() function (no I/O).

        Run 1000 iterations with varying confidence and evidence strengths.
        Assert average < 1ms per call.
        """
        timings_ms: list[float] = []

        # Run 1000 iterations with varied parameters
        for i in range(1000):
            current_confidence = 0.1 + (i % 89) / 100.0  # 0.1-0.99
            evidence_strength = 1.0 + (i % 30) / 10.0  # 1.0-4.0
            is_supporting = (i % 2) == 0  # Alternate true/false

            start = time.perf_counter()
            new_confidence = bayesian_update(
                current_confidence=current_confidence,
                evidence_strength=evidence_strength,
                is_supporting=is_supporting,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            timings_ms.append(elapsed_ms)

            # Result should be clamped in valid range
            assert isinstance(new_confidence, float)
            assert 0.01 <= new_confidence <= 0.99

        avg = sum(timings_ms) / len(timings_ms)
        p95 = _percentile(timings_ms, 95)
        p99 = _percentile(timings_ms, 99)

        print("\n=== bayesian_update() Pure Function ===")
        print(f"  avg={avg:.4f}ms  p95={p95:.4f}ms  p99={p99:.4f}ms")

        assert avg < 1.0, f"Average latency {avg:.4f}ms exceeds 1ms target"

    # ---- Test 5: Activation Scoring with Timing Breakdown ----
    @pytest.mark.asyncio
    async def test_activation_scoring_timing_breakdown(self):
        """
        Detailed timing breakdown of _apply_activation_scoring() phases:
        - Neo4j query fetch
        - Activation calculation
        - Score blending
        - Result re-ranking
        - Background fire-and-forget setup

        Helps identify bottlenecks and validate overhead allocation.
        """
        from src.mcp_server import _apply_activation_scoring

        mock_results = _make_mock_results(num_results=20)
        mock_neo4j_data = _make_mock_neo4j_results(num_entities=40)
        content_scores = [1.0 - (i / 20) for i in range(20)]

        # Single run with detailed timing (with ACTR_ENABLED=true)
        with patch("src.mcp_server.ACTR_ENABLED", True):
            with patch(
                "src.mcp_server.neo4j_service.query", new_callable=AsyncMock
            ) as mock_query:
                mock_query.return_value = mock_neo4j_data

                start_total = time.perf_counter()
                result = await _apply_activation_scoring(
                    results=mock_results.copy(),
                    content_scores=content_scores.copy(),
                )
                elapsed_total_ms = (time.perf_counter() - start_total) * 1000

                # Verify query was called (activation fetch)
                assert mock_query.called, "Neo4j query should be called"
                assert len(result) == 20, "All results should be returned"

        print("\n=== Activation Scoring Timing Breakdown ===")
        print(f"  Total latency: {elapsed_total_ms:.2f}ms")
        print("  - Includes: Neo4j query, activation calc, score blend, re-ranking")
        print("  - Fire-and-forget: update_access_tracking, strengthen_hebbian_links")

        # Total should still be reasonable (< 100ms for 20 results)
        assert elapsed_total_ms < 100.0, (
            f"Activation scoring latency {elapsed_total_ms:.2f}ms is higher than expected"
        )
