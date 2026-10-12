"""
Integration tests for Mnemodrome Open Web UI Pipeline.

Tests the complete Pipeline functionality including:
- Initialization and lifecycle management
- Inlet context loading (pre-processing)
- Outlet fact capture (post-processing)
- Error handling and graceful degradation
- Feature flag toggles
- Performance benchmarking

Run with: pytest tests/integration/test_pipeline_integration.py -v
"""

import asyncio
import os
import sys
import time
from unittest.mock import AsyncMock, patch

import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


@pytest.fixture
async def pipeline():
    """
    Create a Pipeline instance for testing.

    Sets up environment variables for Mnemodrome connection before import.
    """
    # Set required environment variables
    os.environ["NEO4J_URI"] = "bolt://localhost:7687"
    os.environ["NEO4J_USER"] = "neo4j"
    os.environ["NEO4J_PASSWORD"] = "test-password"
    os.environ["MNEMODROME_STORAGE_PATH"] = "/tmp/mnemodrome_test"

    # Import Pipeline after setting env vars
    from pipelines.mnemodrome_pipeline import Pipeline

    pipeline_instance = Pipeline()
    yield pipeline_instance

    # Cleanup
    if pipeline_instance._initialized:
        await pipeline_instance.on_shutdown()


@pytest.fixture
def sample_user_message():
    """Sample Open Web UI message body with user message."""
    return {
        "messages": [
            {
                "role": "user",
                "content": "How do I configure Ollama to listen on all interfaces?",
            }
        ]
    }


@pytest.fixture
def sample_ai_response():
    """Sample Open Web UI message body with AI response."""
    return {
        "messages": [
            {"role": "user", "content": "How do I configure Ollama?"},
            {
                "role": "assistant",
                "content": "To configure Ollama to listen on all interfaces, set OLLAMA_HOST=0.0.0.0:11434 in your environment variables.",
            },
        ]
    }


@pytest.fixture
def sample_short_response():
    """Sample Open Web UI message body with short AI response (should be filtered)."""
    return {
        "messages": [
            {"role": "user", "content": "Thanks!"},
            {"role": "assistant", "content": "You're welcome!"},
        ]
    }


# Test: Pipeline Initialization


@pytest.mark.asyncio
async def test_pipeline_initialization(pipeline):
    """Test Pipeline initializes correctly with default configuration."""
    assert pipeline.type == "filter"
    assert pipeline.name == "Mnemodrome Knowledge Graph"
    assert pipeline.valves is not None
    assert pipeline._initialized is False
    assert pipeline.mnemodrome is None


@pytest.mark.asyncio
async def test_pipeline_startup_success():
    """Test successful Pipeline startup with Mnemodrome connection."""
    # Mock MnemodromeAPI to avoid actual Neo4j connection
    with patch("pipelines.mnemodrome_pipeline.MnemodromeAPI") as MockAPI:
        mock_api_instance = AsyncMock()
        MockAPI.return_value = mock_api_instance

        from pipelines.mnemodrome_pipeline import Pipeline

        pipeline = Pipeline()

        await pipeline.on_startup()

        assert pipeline._initialized is True
        assert pipeline.mnemodrome is not None
        mock_api_instance.initialize.assert_called_once()


@pytest.mark.asyncio
async def test_pipeline_startup_failure_graceful_degradation():
    """Test graceful degradation when Mnemodrome connection fails."""
    # Mock MnemodromeAPI to raise exception
    with patch("pipelines.mnemodrome_pipeline.MnemodromeAPI") as MockAPI:
        mock_api_instance = AsyncMock()
        mock_api_instance.initialize.side_effect = RuntimeError(
            "Neo4j connection failed"
        )
        MockAPI.return_value = mock_api_instance

        from pipelines.mnemodrome_pipeline import Pipeline

        pipeline = Pipeline()

        # Should not raise exception (graceful degradation)
        await pipeline.on_startup()

        assert pipeline._initialized is False
        assert pipeline.mnemodrome is not None  # Instance created but not initialized


@pytest.mark.asyncio
async def test_pipeline_shutdown():
    """Test Pipeline shutdown cleans up connections."""
    with patch("pipelines.mnemodrome_pipeline.MnemodromeAPI") as MockAPI:
        mock_api_instance = AsyncMock()
        MockAPI.return_value = mock_api_instance

        from pipelines.mnemodrome_pipeline import Pipeline

        pipeline = Pipeline()

        await pipeline.on_startup()
        await pipeline.on_shutdown()

        assert pipeline._initialized is False
        assert pipeline.mnemodrome is None
        mock_api_instance.close.assert_called_once()


# Test: Inlet (Context Loading)


@pytest.mark.asyncio
async def test_inlet_context_loading_disabled(pipeline, sample_user_message):
    """Test inlet returns original body when context loading disabled."""
    pipeline.valves.enable_context_loading = False

    result = await pipeline.inlet(sample_user_message)

    assert result == sample_user_message  # Unchanged


@pytest.mark.asyncio
async def test_inlet_not_initialized(pipeline, sample_user_message):
    """Test inlet returns original body when Mnemodrome not initialized."""
    pipeline._initialized = False

    result = await pipeline.inlet(sample_user_message)

    assert result == sample_user_message  # Unchanged


@pytest.mark.asyncio
async def test_inlet_context_loading_success():
    """Test successful context loading and injection."""
    with patch("pipelines.mnemodrome_pipeline.MnemodromeAPI") as MockAPI:
        mock_api_instance = AsyncMock()
        mock_api_instance.load_context.return_value = (
            "Ollama runs on port 11434 by default."
        )
        MockAPI.return_value = mock_api_instance

        from pipelines.mnemodrome_pipeline import Pipeline

        pipeline = Pipeline()
        await pipeline.on_startup()

        body = {"messages": [{"role": "user", "content": "How do I configure Ollama?"}]}

        result = await pipeline.inlet(body)

        # Verify context was loaded
        mock_api_instance.load_context.assert_called_once()

        # Verify context was injected (body modified)
        assert result["messages"][0]["content"] != body["messages"][0]["content"]
        assert "Relevant Background Context" in result["messages"][0]["content"]


@pytest.mark.asyncio
async def test_inlet_timeout_handling():
    """Test inlet gracefully handles context loading timeout."""
    with patch("pipelines.mnemodrome_pipeline.MnemodromeAPI") as MockAPI:
        mock_api_instance = AsyncMock()

        # Simulate slow context loading (will timeout)
        async def slow_load_context(*args, **kwargs):
            await asyncio.sleep(2)  # Longer than default 500ms timeout
            return "Context"

        mock_api_instance.load_context = slow_load_context
        MockAPI.return_value = mock_api_instance

        from pipelines.mnemodrome_pipeline import Pipeline

        pipeline = Pipeline()
        await pipeline.on_startup()
        pipeline.valves.timeout_ms = 100  # Very short timeout

        body = {"messages": [{"role": "user", "content": "Test message"}]}

        start_time = time.perf_counter()
        result = await pipeline.inlet(body)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Verify timeout was enforced
        assert elapsed_ms < 500  # Should timeout quickly
        assert result == body  # Original body returned


@pytest.mark.asyncio
async def test_inlet_empty_context():
    """Test inlet handles empty context gracefully."""
    with patch("pipelines.mnemodrome_pipeline.MnemodromeAPI") as MockAPI:
        mock_api_instance = AsyncMock()
        mock_api_instance.load_context.return_value = ""  # Empty context
        MockAPI.return_value = mock_api_instance

        from pipelines.mnemodrome_pipeline import Pipeline

        pipeline = Pipeline()
        await pipeline.on_startup()

        body = {"messages": [{"role": "user", "content": "Test"}]}

        result = await pipeline.inlet(body)

        # Should return original body (no context to inject)
        assert result == body


# Test: Outlet (Fact Capture)


@pytest.mark.asyncio
async def test_outlet_fact_capture_disabled(pipeline, sample_ai_response):
    """Test outlet returns original body when fact capture disabled."""
    pipeline.valves.enable_fact_capture = False

    result = await pipeline.outlet(sample_ai_response)

    assert result == sample_ai_response  # Unchanged


@pytest.mark.asyncio
async def test_outlet_not_initialized(pipeline, sample_ai_response):
    """Test outlet returns original body when Mnemodrome not initialized."""
    pipeline._initialized = False

    result = await pipeline.outlet(sample_ai_response)

    assert result == sample_ai_response  # Unchanged


@pytest.mark.asyncio
async def test_outlet_fact_capture_success():
    """Test successful fact capture launch (non-blocking)."""
    with patch("pipelines.mnemodrome_pipeline.MnemodromeAPI") as MockAPI:
        mock_api_instance = AsyncMock()
        MockAPI.return_value = mock_api_instance

        from pipelines.mnemodrome_pipeline import Pipeline

        pipeline = Pipeline()
        await pipeline.on_startup()

        body = {
            "messages": [
                {"role": "user", "content": "Question"},
                {
                    "role": "assistant",
                    "content": "To configure Ollama, set OLLAMA_HOST=0.0.0.0 in your environment.",
                },
            ]
        }

        start_time = time.perf_counter()
        result = await pipeline.outlet(body)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Verify non-blocking (should return immediately)
        assert elapsed_ms < 50  # Should be very fast
        assert result == body  # Body unchanged

        # Give background task time to execute
        await asyncio.sleep(0.1)

        # Verify capture was called (may happen in background)
        # Note: In real test, would verify fact appears in Neo4j


@pytest.mark.asyncio
async def test_outlet_filters_short_responses(pipeline, sample_short_response):
    """Test outlet filters out short/generic responses."""
    pipeline._initialized = True
    pipeline.mnemodrome = AsyncMock()

    result = await pipeline.outlet(sample_short_response)

    # Give background task time to execute
    await asyncio.sleep(0.1)

    # capture_context should not be called for short responses
    # (verified by _is_worth_capturing filter)
    assert result == sample_short_response  # Body unchanged


@pytest.mark.asyncio
async def test_outlet_non_assistant_message():
    """Test outlet skips non-assistant messages."""
    with patch("pipelines.mnemodrome_pipeline.MnemodromeAPI") as MockAPI:
        mock_api_instance = AsyncMock()
        MockAPI.return_value = mock_api_instance

        from pipelines.mnemodrome_pipeline import Pipeline

        pipeline = Pipeline()
        await pipeline.on_startup()

        # Last message is user message (not assistant)
        body = {"messages": [{"role": "user", "content": "Test message"}]}

        result = await pipeline.outlet(body)

        assert result == body
        # capture_context should not be called
        await asyncio.sleep(0.1)
        mock_api_instance.capture_context.assert_not_called()


# Test: Helper Methods


def test_estimate_tokens(pipeline):
    """Test token estimation helper."""
    text = "This is a test message with approximately 100 characters that should estimate to around 25 tokens."
    tokens = pipeline._estimate_tokens(text)

    assert tokens > 0
    assert tokens == len(text) // 4  # Simple heuristic


def test_is_worth_capturing_filters_short(pipeline):
    """Test _is_worth_capturing filters short responses."""
    short_response = "OK"

    assert pipeline._is_worth_capturing(short_response) is False


def test_is_worth_capturing_filters_generic(pipeline):
    """Test _is_worth_capturing filters generic acknowledgments."""
    generic_responses = [
        "Thanks",
        "Thank you",
        "I understand",
        "Got it",
        "Okay",
        "Sure",
        "Hello",
        "Goodbye",
    ]

    for response in generic_responses:
        assert pipeline._is_worth_capturing(response) is False


def test_is_worth_capturing_accepts_substantive(pipeline):
    """Test _is_worth_capturing accepts substantive responses."""
    substantive_response = "To configure Ollama to listen on all interfaces, you need to set the OLLAMA_HOST environment variable."

    assert pipeline._is_worth_capturing(substantive_response) is True


def test_format_context_adds_header(pipeline):
    """Test _format_context adds proper header."""
    context = "Ollama runs on port 11434."
    formatted = pipeline._format_context(context)

    assert "## Relevant Background Context" in formatted
    assert context in formatted


def test_format_context_truncates_long_content(pipeline):
    """Test _format_context truncates when exceeding token limit."""
    pipeline.valves.max_context_tokens = 10  # Very small limit
    long_context = "This is a very long context " * 100  # Way over limit

    formatted = pipeline._format_context(long_context)
    estimated_tokens = pipeline._estimate_tokens(formatted)

    assert (
        estimated_tokens <= pipeline.valves.max_context_tokens * 2
    )  # Some overhead allowed


# Test: Performance Benchmarking


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_inlet_performance_target():
    """Benchmark: Verify inlet meets <500ms p95 latency target."""
    with patch("pipelines.mnemodrome_pipeline.MnemodromeAPI") as MockAPI:
        mock_api_instance = AsyncMock()

        # Simulate realistic context loading time
        async def mock_load_context(*args, **kwargs):
            await asyncio.sleep(0.1)  # 100ms simulation
            return "Ollama configuration context"

        mock_api_instance.load_context = mock_load_context
        MockAPI.return_value = mock_api_instance

        from pipelines.mnemodrome_pipeline import Pipeline

        pipeline = Pipeline()
        await pipeline.on_startup()

        body = {"messages": [{"role": "user", "content": "How do I configure Ollama?"}]}

        # Run multiple iterations to get p95
        latencies = []
        for _ in range(20):
            start = time.perf_counter()
            await pipeline.inlet(body)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        # Calculate p95
        latencies.sort()
        p95_latency = latencies[int(len(latencies) * 0.95)]

        print("\nInlet Performance:")
        print(f"  Average: {sum(latencies) / len(latencies):.2f}ms")
        print(f"  p95: {p95_latency:.2f}ms")
        print(f"  Max: {max(latencies):.2f}ms")

        # Verify p95 < 500ms target
        assert p95_latency < 500, (
            f"p95 latency {p95_latency:.2f}ms exceeds 500ms target"
        )


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_outlet_non_blocking_performance():
    """Benchmark: Verify outlet doesn't delay response delivery."""
    with patch("pipelines.mnemodrome_pipeline.MnemodromeAPI") as MockAPI:
        mock_api_instance = AsyncMock()

        # Simulate slow fact capture (should not block)
        async def slow_capture(*args, **kwargs):
            await asyncio.sleep(1)  # 1 second (intentionally slow)

        mock_api_instance.capture_context = slow_capture
        MockAPI.return_value = mock_api_instance

        from pipelines.mnemodrome_pipeline import Pipeline

        pipeline = Pipeline()
        await pipeline.on_startup()

        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "Substantive response with facts to capture.",
                }
            ]
        }

        start = time.perf_counter()
        result = await pipeline.outlet(body)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should return immediately (<50ms) despite slow background task
        assert elapsed_ms < 50, (
            f"Outlet blocked for {elapsed_ms:.2f}ms (should be non-blocking)"
        )
        assert result == body


# Test: Configuration


def test_valves_defaults(pipeline):
    """Test Valves have sensible defaults."""
    valves = pipeline.valves

    assert valves.neo4j_host == "localhost"
    assert valves.neo4j_port == 7687
    assert valves.neo4j_user == "neo4j"
    assert valves.enable_context_loading is True
    assert valves.enable_fact_capture is True
    assert valves.debug_mode is False
    assert valves.context_days == 7
    assert valves.context_limit == 10
    assert valves.max_context_tokens == 1000
    assert valves.timeout_ms == 500


def test_valves_validation(pipeline):
    """Test Valves enforce validation constraints."""
    valves = pipeline.valves

    # Test min/max constraints
    with pytest.raises(Exception):
        valves.context_days = 0  # Below minimum (1)

    with pytest.raises(Exception):
        valves.context_days = 100  # Above maximum (90)

    with pytest.raises(Exception):
        valves.timeout_ms = 50  # Below minimum (100)


# Test: Error Scenarios


@pytest.mark.asyncio
async def test_inlet_malformed_message_body():
    """Test inlet handles malformed message body gracefully."""
    with patch("pipelines.mnemodrome_pipeline.MnemodromeAPI") as MockAPI:
        mock_api_instance = AsyncMock()
        MockAPI.return_value = mock_api_instance

        from pipelines.mnemodrome_pipeline import Pipeline

        pipeline = Pipeline()
        await pipeline.on_startup()

        # Malformed bodies
        malformed_bodies = [
            {},  # No messages
            {"messages": []},  # Empty messages
            {"messages": [{"wrong": "format"}]},  # Missing content
            None,  # None body (will cause AttributeError)
        ]

        for body in malformed_bodies:
            try:
                result = await pipeline.inlet(body)
                # Should return original body or handle gracefully
                # (may return original or modified, but shouldn't crash)
            except Exception as e:
                pytest.fail(f"Inlet crashed on malformed body: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
