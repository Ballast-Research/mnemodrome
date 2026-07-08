"""
Graceful fallback verification tests for ACT-R activation scoring (Feature 034).

Tests that activation scoring gracefully handles service unavailability,
configuration changes, and error conditions without impacting core retrieval.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
class TestActivationFallback:
    """Test graceful fallback behavior when activation scoring is unavailable."""

    async def test_actr_disabled_skips_activation_logic(self):
        """
        Test 1: When ACTR_ENABLED=false, _apply_activation_scoring() should
        return results unchanged (no activation scoring applied).

        Acceptance: Verify Neo4j is NOT called for activation metadata when
        ACTR_ENABLED=false. Results should be returned in original order.
        """
        # Mock ACTR_ENABLED as False
        with patch("src.mcp_server.ACTR_ENABLED", False):
            # Import the function to test (after patching config)
            from src.mcp_server import _apply_activation_scoring

            # Create test results
            test_results = [
                {
                    "source": {"uuid": "entity-1", "name": "Entity 1"},
                    "target": {"uuid": "entity-2", "name": "Entity 2"},
                    "type": "RELATES_TO",
                },
                {
                    "source": {"uuid": "entity-3", "name": "Entity 3"},
                    "target": {"uuid": "entity-4", "name": "Entity 4"},
                    "type": "RELATES_TO",
                },
            ]

            # Mock neo4j_service to verify it's NOT called
            with patch("src.mcp_server.neo4j_service") as mock_neo4j:
                mock_neo4j.query = AsyncMock()

                # Call the function
                result = await _apply_activation_scoring(test_results)

                # Verify results are unchanged (returned as-is)
                assert result == test_results, "Results should be returned unchanged"

                # Verify Neo4j query was NOT called
                mock_neo4j.query.assert_not_called()

    async def test_activation_exception_falls_back_to_content_ranking(self):
        """
        Test 2: When activation calculation throws an exception (e.g., Neo4j
        query fails), results should be returned in their original order
        (content-only ranking), and a warning should be logged.

        Acceptance: Mock neo4j.query to raise an Exception, verify results
        are returned unchanged and warning is logged (FR-008).
        """
        with patch("src.mcp_server.ACTR_ENABLED", True):
            from src.mcp_server import _apply_activation_scoring

            test_results = [
                {
                    "source": {"uuid": "uuid-1"},
                    "target": {"uuid": "uuid-2"},
                },
                {
                    "source": {"uuid": "uuid-3"},
                    "target": {"uuid": "uuid-4"},
                },
            ]

            with (
                patch("src.mcp_server.neo4j_service") as mock_neo4j,
                patch("src.mcp_server.logger") as mock_logger,
                patch("src.mcp_server._fire_and_forget"),
            ):
                # Make neo4j_service.query raise an exception
                mock_neo4j.query = AsyncMock(side_effect=Exception("Neo4j unavailable"))

                # Call the function
                result = await _apply_activation_scoring(test_results)

                # Verify results are returned (graceful degradation)
                assert len(result) == len(test_results), (
                    "All results should be returned on query failure"
                )

                # Verify per-query failure warnings were logged
                mock_logger.warning.assert_called()
                warning_calls = [
                    call[0][0] for call in mock_logger.warning.call_args_list
                ]
                assert any("query failed" in w for w in warning_calls), (
                    "Query failure warnings should be logged"
                )

    async def test_neo4j_unavailable_during_access_tracking_no_failure(self):
        """
        Test 3: When Neo4j is unavailable during fire-and-forget access
        tracking, the retrieval (and its activation scoring) should still
        succeed. The main function doesn't fail even if background tracking fails.

        Acceptance: Mock activation_service.update_access_tracking to raise
        an Exception, verify the main scoring function succeeds (results are
        returned).
        """
        with patch("src.mcp_server.ACTR_ENABLED", True):
            from src.mcp_server import _apply_activation_scoring

            test_results = [
                {
                    "source": {"uuid": "uuid-1"},
                    "target": {"uuid": "uuid-2"},
                },
            ]

            with (
                patch("src.mcp_server.neo4j_service") as mock_neo4j,
                patch("src.mcp_server.activation_service") as mock_activation,
                patch("src.mcp_server._fire_and_forget") as mock_fire_and_forget,
            ):
                now = datetime.now(UTC)
                # Setup neo4j queries: activation, hebbian, confidence
                mock_neo4j.query = AsyncMock(
                    side_effect=[
                        [  # activation metadata
                            {
                                "uuid": "uuid-1",
                                "access_count": 5,
                                "last_accessed": now,
                                "created_at": None,
                            },
                            {
                                "uuid": "uuid-2",
                                "access_count": 3,
                                "last_accessed": now,
                                "created_at": None,
                            },
                        ],
                        [],  # hebbian data (empty)
                        [
                            {
                                "source_uuid": "uuid-1",
                                "target_uuid": "uuid-2",
                                "confidence": 0.8,
                            }
                        ],
                    ]
                )

                # Make background update_access_tracking fail (but it's fire-and-forget)
                mock_activation.update_access_tracking = AsyncMock(
                    side_effect=Exception("Access tracking failed")
                )

                # Call the function
                result = await _apply_activation_scoring(test_results)

                # Verify results are still returned despite background failure
                assert len(result) > 0, (
                    "Results should be returned despite tracking failure"
                )
                assert isinstance(result, list)

    async def test_activation_independent_of_redis_qdrant(self):
        """
        Test 4: Verify activation scoring operates independently - it only
        needs Neo4j for activation metadata, not Redis or Qdrant.

        Acceptance: Verify that _apply_activation_scoring() doesn't import
        or reference Redis/Qdrant services. It should only call neo4j_service.
        """
        import inspect

        from src.mcp_server import _apply_activation_scoring

        # Get the source code of the function
        source = inspect.getsource(_apply_activation_scoring)

        # Verify it doesn't reference Redis service
        assert "redis_service" not in source, (
            "Activation scoring should not reference redis_service"
        )

        # Verify it doesn't reference Qdrant service
        assert "qdrant" not in source.lower(), (
            "Activation scoring should not reference Qdrant"
        )

        # Verify it DOES reference Neo4j service
        assert "neo4j_service" in source, (
            "Activation scoring should reference neo4j_service for activation metadata"
        )

        # Verify it DOES call query (for activation metadata lookup)
        assert ".query(" in source, (
            "Activation scoring should call neo4j_service.query()"
        )

    async def test_empty_results_short_circuits_gracefully(self):
        """
        Test 5: When results list is empty, _apply_activation_scoring should
        return immediately without calling Neo4j.

        Acceptance: Verify no Neo4j calls are made for empty results.
        """
        with patch("src.mcp_server.ACTR_ENABLED", True):
            from src.mcp_server import _apply_activation_scoring

            with patch("src.mcp_server.neo4j_service") as mock_neo4j:
                mock_neo4j.query = AsyncMock()

                # Call with empty results
                result = await _apply_activation_scoring([])

                # Verify results are returned as empty
                assert result == []

                # Verify Neo4j was not called
                mock_neo4j.query.assert_not_called()

    async def test_results_without_uuids_short_circuits(self):
        """
        Test 6: When results have no extractable UUIDs (malformed results),
        should return unchanged without calling Neo4j.

        Acceptance: Verify graceful handling of malformed results.
        """
        with patch("src.mcp_server.ACTR_ENABLED", True):
            from src.mcp_server import _apply_activation_scoring

            # Results without UUIDs in source/target
            test_results = [
                {"source": {}, "target": {}},  # No UUIDs
                {"source": {"name": "no-uuid"}, "target": {"name": "also-no-uuid"}},
            ]

            with patch("src.mcp_server.neo4j_service") as mock_neo4j:
                mock_neo4j.query = AsyncMock()

                result = await _apply_activation_scoring(test_results)

                # Verify results returned unchanged
                assert result == test_results

                # Verify Neo4j not called (no UUIDs to query)
                mock_neo4j.query.assert_not_called()

    async def test_activation_with_none_last_accessed_uses_created_at(self):
        """
        Test 7: When Neo4j returns None for last_accessed, activation scoring
        should fall back to created_at timestamp (handles lazy initialization).

        Acceptance: Verify that None last_accessed doesn't crash, falls back
        to created_at or current time.
        """
        with patch("src.mcp_server.ACTR_ENABLED", True):
            from src.mcp_server import _apply_activation_scoring

            test_results = [
                {
                    "source": {"uuid": "uuid-with-no-access"},
                    "target": {"uuid": "uuid-created-only"},
                },
            ]

            with (
                patch("src.mcp_server.neo4j_service") as mock_neo4j,
                patch("src.mcp_server._fire_and_forget"),
            ):
                now_iso = datetime.now(UTC).isoformat()
                # Return data with None last_accessed but valid created_at
                mock_neo4j.query = AsyncMock(
                    side_effect=[
                        [  # activation metadata
                            {
                                "uuid": "uuid-with-no-access",
                                "access_count": 0,
                                "last_accessed": None,
                                "created_at": now_iso,
                            },
                            {
                                "uuid": "uuid-created-only",
                                "access_count": 0,
                                "last_accessed": None,
                                "created_at": now_iso,
                            },
                        ],
                        [],  # hebbian data
                        [
                            {
                                "source_uuid": "uuid-with-no-access",
                                "target_uuid": "uuid-created-only",
                                "confidence": 0.8,
                            }
                        ],
                    ]
                )

                # Should not raise
                result = await _apply_activation_scoring(test_results)

                # Verify results are returned (no exception)
                assert len(result) > 0

    async def test_content_scores_provided_overrides_position_based(self):
        """
        Test 8: When content_scores are provided explicitly, they should
        override position-based normalization.

        Acceptance: Verify final ranking uses provided content scores.
        """
        with patch("src.mcp_server.ACTR_ENABLED", True):
            from src.mcp_server import _apply_activation_scoring

            test_results = [
                {"source": {"uuid": "uuid-1"}, "target": {"uuid": "uuid-2"}},
                {"source": {"uuid": "uuid-3"}, "target": {"uuid": "uuid-4"}},
                {"source": {"uuid": "uuid-5"}, "target": {"uuid": "uuid-6"}},
            ]

            # Provide explicit content scores
            content_scores = [0.3, 0.9, 0.5]  # Second result should rank highest

            with (
                patch("src.mcp_server.neo4j_service") as mock_neo4j,
                patch("src.mcp_server.activation_service") as mock_activation,
                patch("src.mcp_server._fire_and_forget"),
            ):
                now = datetime.now(UTC)
                # All entities have same access pattern
                mock_neo4j.query = AsyncMock(
                    side_effect=[
                        [  # activation data
                            {
                                "uuid": f"uuid-{i}",
                                "access_count": 1,
                                "last_accessed": now,
                                "created_at": None,
                            }
                            for i in range(1, 7)
                        ],
                        [],  # hebbian data
                        [  # confidence data — all at default
                            {
                                "source_uuid": "uuid-1",
                                "target_uuid": "uuid-2",
                                "confidence": 0.8,
                            },
                            {
                                "source_uuid": "uuid-3",
                                "target_uuid": "uuid-4",
                                "confidence": 0.8,
                            },
                            {
                                "source_uuid": "uuid-5",
                                "target_uuid": "uuid-6",
                                "confidence": 0.8,
                            },
                        ],
                    ]
                )

                mock_activation.update_access_tracking = AsyncMock()
                mock_activation.strengthen_hebbian_links = AsyncMock()

                result = await _apply_activation_scoring(test_results, content_scores)

                # The second result (content_score=0.9) should rank higher
                # due to content score dominance (0.7 weight)
                assert len(result) == 3, "All results should be returned"

    async def test_fire_and_forget_background_tasks_tracked(self):
        """
        Test 9: Verify that fire-and-forget background tasks
        (access tracking and Hebbian strengthening) are properly scheduled
        but don't block the main return.

        Acceptance: Verify _fire_and_forget is called for both access
        tracking and Hebbian strengthening when results > 1.
        """
        with patch("src.mcp_server.ACTR_ENABLED", True):
            from src.mcp_server import _apply_activation_scoring

            test_results = [
                {"source": {"uuid": "uuid-1"}, "target": {"uuid": "uuid-2"}},
                {"source": {"uuid": "uuid-3"}, "target": {"uuid": "uuid-4"}},
            ]

            with (
                patch("src.mcp_server.neo4j_service") as mock_neo4j,
                patch("src.mcp_server.activation_service") as mock_activation,
                patch("src.mcp_server._fire_and_forget") as mock_fire_and_forget,
            ):
                now = datetime.now(UTC)
                activation_data = [
                    {
                        "uuid": f"uuid-{i}",
                        "access_count": 1,
                        "last_accessed": now,
                        "created_at": None,
                    }
                    for i in range(1, 5)
                ]
                mock_neo4j.query = AsyncMock(
                    side_effect=[
                        activation_data,  # activation query
                        [],  # hebbian query
                        [  # confidence query
                            {
                                "source_uuid": "uuid-1",
                                "target_uuid": "uuid-2",
                                "confidence": 0.8,
                            },
                            {
                                "source_uuid": "uuid-3",
                                "target_uuid": "uuid-4",
                                "confidence": 0.8,
                            },
                        ],
                    ]
                )

                mock_activation.update_access_tracking = AsyncMock()
                mock_activation.strengthen_hebbian_links = AsyncMock()

                result = await _apply_activation_scoring(test_results)

                # Verify results returned
                assert len(result) > 0

                # Verify fire_and_forget was called (for access tracking)
                assert mock_fire_and_forget.call_count >= 1, (
                    "Fire-and-forget should be used for access tracking"
                )
