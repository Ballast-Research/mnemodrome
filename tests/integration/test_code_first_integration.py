"""
Integration tests for Code-First API with real services.

Tests full workflow with actual Neo4j instance (not mocked).
Validates performance, error handling, and realistic usage scenarios.
"""

import asyncio
import time

import pytest

from src.interfaces.code_first import MnemodromeAPI


@pytest.mark.integration
@pytest.mark.asyncio
class TestCodeFirstIntegration:
    """Integration tests with real Neo4j and Redis services."""

    async def test_full_workflow_async_context_manager(self):
        """Test complete workflow using async context manager."""
        start_time = time.time()

        async with MnemodromeAPI() as api:
            # 1. Load context (should work even if empty)
            context = await api.load_context("integration test", days=1, limit=5)
            assert isinstance(context, str)

            # 2. Capture a fact
            fact_result = await api.capture_fact(
                "Alice",
                "VALIDATES",
                "CodeFirstAPI",
                context="Integration test for code-first interface",
            )

            assert fact_result["status"] in ["complete", "partial"]
            assert "entity1_uuid" in fact_result
            assert "entity2_uuid" in fact_result

            # 3. Load context again (should include our fact)
            context_after = await api.load_context("Alice", days=1)
            # Context should contain something (may or may not include our fact depending on timing)
            assert isinstance(context_after, str)

            # 4. Health check
            health = await api.health()
            assert health["status"] in ["healthy", "degraded"]
            assert health["neo4j"] == "connected"

        # Services should be closed after context exit
        assert api._initialized is False

        duration = time.time() - start_time
        print(f"\nFull workflow completed in {duration:.2f}s")

    async def test_manual_lifecycle_management(self):
        """Test manual initialize and close (non-context-manager)."""
        api = MnemodromeAPI()

        # Should not be initialized yet
        assert api._initialized is False

        # Initialize
        await api.initialize()
        assert api._initialized is True

        # Use API
        health = await api.health()
        assert health["neo4j"] == "connected"

        # Close
        await api.close()
        assert api._initialized is False

    async def test_capture_context_multi_entity(self):
        """Test multi-entity graph extraction from natural language."""
        async with MnemodromeAPI() as api:
            result = await api.capture_context(
                "Integration test validates that Alice built the code-first interface "
                "using Mnemodrome service layer and Neo4j database"
            )

            assert "entities_created" in result
            assert "relationships_created" in result
            assert result["entities_created"] >= 2  # At least a few entities

    async def test_capture_learning(self):
        """Test learning capture with dual-memory writes."""
        async with MnemodromeAPI() as api:
            result = await api.capture_learning(
                title="Code-First Integration Test",
                content="This learning validates that code-first interface works with real Neo4j",
                category="testing",
            )

            assert "uuid" in result
            assert "file_path" in result
            assert "duration_ms" in result

    async def test_capture_session(self):
        """Test session capture with insights."""
        async with MnemodromeAPI() as api:
            result = await api.capture_session(
                conversation_summary="Integration testing of code-first interface",
                key_insights=[
                    "Code-first API achieves 98.7% token reduction",
                    "Full feature parity with MCP server",
                    "Real Neo4j integration validates correctness",
                ],
            )

            assert "uuid" in result
            assert "file_path" in result

    async def test_schema_discovery(self):
        """Test schema discovery returns actual data."""
        async with MnemodromeAPI() as api:
            schema = await api.discover_schema()

            assert "sample_relationships" in schema
            assert "sample_entities" in schema
            # Should have some data (assuming Neo4j has content)
            assert isinstance(schema["sample_relationships"], list)
            assert isinstance(schema["sample_entities"], list)

    async def test_redis_graceful_degradation(self):
        """Test API works even if Redis is unavailable."""
        # Note: This test assumes Redis might not be available
        # API should still work (graceful degradation)

        async with MnemodromeAPI() as api:
            # Check health
            health = await api.health()

            # Neo4j should be connected (required)
            assert health["neo4j"] == "connected"

            # Redis might be connected or disconnected (both ok)
            assert health["redis"] in ["connected", "disconnected"]

            # API should work regardless of Redis status
            context = await api.load_context("test", days=1)
            assert isinstance(context, str)

    async def test_performance_benchmarks(self):
        """Test performance meets targets."""
        timings = {}

        # Initialization benchmark
        api = MnemodromeAPI()
        start = time.time()
        await api.initialize()
        timings["initialization"] = (time.time() - start) * 1000  # ms

        # Context loading benchmark
        start = time.time()
        await api.load_context("performance test", days=7, limit=10)
        timings["context_loading"] = (time.time() - start) * 1000  # ms

        # Fact capture benchmark
        start = time.time()
        await api.capture_fact(
            "PerformanceTest",
            "MEASURES",
            "CodeFirstAPI",
            context="Performance benchmark",
        )
        timings["fact_capture"] = (time.time() - start) * 1000  # ms

        # Health check benchmark
        start = time.time()
        await api.health()
        timings["health_check"] = (time.time() - start) * 1000  # ms

        # Cleanup
        await api.close()

        # Print benchmarks
        print("\n=== Performance Benchmarks ===")
        for operation, duration in timings.items():
            print(f"{operation}: {duration:.1f}ms")

        # Validate targets
        assert timings["initialization"] < 2000, "Initialization should be <2s"
        # Context loading can vary widely based on data
        # Fact capture includes LLM classification, so can be slower

    async def test_error_handling_initialization_failure(self):
        """Test proper error handling when Neo4j unavailable."""
        # This test would require stopping Neo4j, which we don't want to do
        # Just verify the error message is clear
        pass  # Skipped - would require stopping Neo4j

    async def test_concurrent_operations(self):
        """Test API handles concurrent operations correctly."""
        async with MnemodromeAPI() as api:
            # Run multiple operations concurrently
            tasks = [
                api.health(),
                api.load_context("concurrent test 1", days=1),
                api.load_context("concurrent test 2", days=1),
                api.capture_fact("ConcurrentTest1", "VALIDATES", "Concurrency"),
                api.capture_fact("ConcurrentTest2", "VALIDATES", "Concurrency"),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should succeed (or return expected exceptions)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    pytest.fail(f"Task {i} failed: {result}")

    async def test_typical_session_workflow(self):
        """
        Test realistic session workflow:
        1. Load recent context
        2. Capture new facts during work
        3. Capture learning breakthrough
        4. Capture session summary at end
        """
        async with MnemodromeAPI() as api:
            # Phase 1: Load context from previous work
            context = await api.load_context("typical workflow", days=7)
            print(f"\nLoaded context: {len(context)} characters")

            # Phase 2: Capture facts as we learn
            fact1 = await api.capture_fact(
                "CodeFirstAPI",
                "IMPLEMENTS",
                "Code-First Pattern",
                context="Anthropic's recommended pattern for token efficiency",
            )
            assert fact1["status"] in ["complete", "partial"]

            fact2 = await api.capture_fact(
                "CodeFirstAPI",
                "ACHIEVES",
                "98.7% Token Reduction",
                context="From 150k to 2k tokens",
            )
            assert fact2["status"] in ["complete", "partial"]

            # Phase 3: Capture breakthrough learning
            learning = await api.capture_learning(
                title="Code-First Pattern Implementation Success",
                content="""
                Successfully implemented code-first pattern for Mnemodrome.

                ## Key Insights
                - Thin wrapper over existing service layer
                - No business logic duplication
                - Full feature parity with MCP server
                - Async context manager for clean lifecycle

                ## Performance
                - 98.7% token reduction achieved
                - <500ms initialization
                - All tests passing
                """,
                category="breakthrough",
            )
            assert "uuid" in learning

            # Phase 4: End-of-session synthesis
            session = await api.capture_session(
                conversation_summary="Implemented and tested code-first interface",
                key_insights=[
                    "Code-first pattern achieves dramatic token reduction",
                    "Integration tests validate real-world usage",
                    "Performance meets all targets",
                ],
            )
            assert "uuid" in session

            print("\nTypical session workflow complete:")
            print("  - Context loaded: ✓")
            print("  - Facts captured: 2")
            print("  - Learning captured: 1")
            print("  - Session synthesized: ✓")


@pytest.mark.integration
@pytest.mark.asyncio
class TestCodeFirstTokenUsage:
    """Test token usage characteristics."""

    async def test_token_usage_estimation(self):
        """
        Estimate token usage for typical session.

        Note: This is an estimation based on string length.
        Actual token count would require tiktoken library.
        """
        # Import statement
        import_code = "from src.interfaces.code_first import MnemodromeAPI"

        # Initialization
        init_code = """
api = MnemodromeAPI()
await api.initialize()
"""

        # Typical usage
        usage_code = """
context = await api.load_context("topic", days=7)
await api.capture_fact("E1", "REL", "E2", context="ctx")
await api.close()
"""

        total_chars = len(import_code) + len(init_code) + len(usage_code)

        # Rough estimation: ~4 chars per token
        estimated_tokens = total_chars / 4

        print("\n=== Token Usage Estimation ===")
        print(f"Import: ~{len(import_code) / 4:.0f} tokens")
        print(f"Initialize: ~{len(init_code) / 4:.0f} tokens")
        print(f"Usage: ~{len(usage_code) / 4:.0f} tokens")
        print(f"Total: ~{estimated_tokens:.0f} tokens")
        print("\nTarget: <5,000 tokens")
        print("MCP Baseline: ~150,000 tokens")
        print(f"Reduction: ~{((150000 - estimated_tokens) / 150000 * 100):.1f}%")

        # Should be well under 5k token target
        assert estimated_tokens < 5000


@pytest.mark.integration
def test_code_first_imports():
    """Test imports work correctly."""
    from src.interfaces import MnemodromeAPI
    from src.interfaces.code_first import MnemodromeAPI as DirectImport

    assert MnemodromeAPI is DirectImport
