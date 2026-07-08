"""
Integration tests for /validate-reasoning HTTP endpoint (Feature 028).

Tests the HTTP endpoint for reasoning-time validation, including:
- Authentication checks
- Input validation (proposed_response required)
- Service availability checks (503 when LLM unavailable)
- JSON response format verification
- ReasoningValidationResult field completeness
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.models.reasoning_validation_models import (
    ConflictSummary,
    LearningReference,
    PatternReference,
    ReasoningValidationResult,
)


@pytest.mark.asyncio
class TestValidateReasoningHTTPEndpoint:
    """Integration tests for /validate-reasoning HTTP POST endpoint."""

    async def test_endpoint_missing_proposed_response_returns_400(self):
        """Test endpoint with empty body returns 400."""
        from starlette.requests import Request

        from src.mcp_server import handle_validate_reasoning_http

        # Mock request with empty body
        mock_request = AsyncMock(spec=Request)
        mock_request.json = AsyncMock(return_value={})
        mock_request.headers = {}

        response = await handle_validate_reasoning_http(mock_request)

        assert response.status_code == 400
        assert b"proposed_response" in response.body

    async def test_endpoint_empty_proposed_response_returns_400(self):
        """Test endpoint with empty proposed_response returns 400."""
        from starlette.requests import Request

        from src.mcp_server import handle_validate_reasoning_http

        mock_request = AsyncMock(spec=Request)
        mock_request.json = AsyncMock(return_value={"proposed_response": ""})
        mock_request.headers = {}

        response = await handle_validate_reasoning_http(mock_request)

        assert response.status_code == 400

    async def test_endpoint_unauthorized_with_token_configured(self):
        """Test endpoint with token configured but wrong token returns 401."""
        from starlette.requests import Request

        from src.mcp_server import handle_validate_reasoning_http

        with patch("src.mcp_server.MNEMODROME_HOOK_TOKEN", "secret-token"):
            mock_request = AsyncMock(spec=Request)
            mock_request.json = AsyncMock(return_value={"proposed_response": "test"})
            # Wrong token
            mock_request.headers = {"Authorization": "Bearer wrong-token"}

            response = await handle_validate_reasoning_http(mock_request)

            assert response.status_code == 401
            assert b"Unauthorized" in response.body

    async def test_endpoint_service_unavailable_returns_503(self):
        """Test endpoint when LLM service is None returns 503."""
        from starlette.requests import Request

        from src.mcp_server import handle_validate_reasoning_http

        mock_request = AsyncMock(spec=Request)
        mock_request.json = AsyncMock(
            return_value={"proposed_response": "test response"}
        )
        mock_request.headers = {}

        with (
            patch("src.mcp_server.llm_service_singleton", None),
            patch("src.mcp_server.MNEMODROME_HOOK_TOKEN", None),
        ):
            response = await handle_validate_reasoning_http(mock_request)

            assert response.status_code == 503
            assert b"unavailable" in response.body.lower()

    async def test_endpoint_returns_200_with_valid_body(self):
        """Test endpoint with valid body returns 200 and JSON result."""
        import json

        from starlette.requests import Request

        from src.mcp_server import handle_validate_reasoning_http

        # Mock the ReasoningValidationService
        mock_result = ReasoningValidationResult(
            validation_status="clean",
            summary="No conflicts detected.",
            claims_analyzed=3,
            entities_checked=5,
            elapsed_ms=150.0,
        )

        mock_service = AsyncMock()
        mock_service.validate_response = AsyncMock(return_value=mock_result)

        mock_request = AsyncMock(spec=Request)
        mock_request.json = AsyncMock(
            return_value={"proposed_response": "test response"}
        )
        mock_request.headers = {}

        with (
            patch("src.mcp_server.llm_service_singleton", AsyncMock()),
            patch("src.mcp_server.get_vector_service", return_value=AsyncMock()),
            patch("src.mcp_server.neo4j_service", AsyncMock()),
            patch("src.mcp_server.redis_service", AsyncMock(is_connected=False)),
            patch(
                "src.mcp_server.ReasoningValidationService", return_value=mock_service
            ),
            patch("src.mcp_server.MNEMODROME_HOOK_TOKEN", None),
        ):
            response = await handle_validate_reasoning_http(mock_request)

            assert response.status_code == 200
            # Verify it's JSON
            body_dict = json.loads(response.body)
            assert "validation_status" in body_dict
            assert body_dict["validation_status"] == "clean"

    async def test_endpoint_result_is_json_not_markdown(self):
        """Verify response body is JSON (not markdown) even for conflicts."""
        import json

        from starlette.requests import Request

        from src.mcp_server import handle_validate_reasoning_http

        conflict = ConflictSummary(
            conflict_type="exact_contradiction",
            severity=80,
            claim="Test claim",
            stored_fact="Stored fact",
            recommendation="warn",
        )

        mock_result = ReasoningValidationResult(
            validation_status="conflicts_detected",
            conflicts=[conflict],
            summary="1 conflict detected.",
            claims_analyzed=1,
            entities_checked=2,
            elapsed_ms=200.0,
        )

        mock_service = AsyncMock()
        mock_service.validate_response = AsyncMock(return_value=mock_result)

        mock_request = AsyncMock(spec=Request)
        mock_request.json = AsyncMock(
            return_value={"proposed_response": "conflicting response"}
        )
        mock_request.headers = {}

        with (
            patch("src.mcp_server.llm_service_singleton", AsyncMock()),
            patch("src.mcp_server.get_vector_service", return_value=AsyncMock()),
            patch("src.mcp_server.neo4j_service", AsyncMock()),
            patch("src.mcp_server.redis_service", AsyncMock(is_connected=False)),
            patch(
                "src.mcp_server.ReasoningValidationService", return_value=mock_service
            ),
            patch("src.mcp_server.MNEMODROME_HOOK_TOKEN", None),
        ):
            response = await handle_validate_reasoning_http(mock_request)

            assert response.status_code == 200
            # Verify it's valid JSON
            body_dict = json.loads(response.body)
            assert isinstance(body_dict, dict)
            assert body_dict["validation_status"] == "conflicts_detected"
            # Verify NOT markdown (no # headers, ** bold, etc.)
            body_text = response.body.decode()
            assert "# " not in body_text
            assert "**" not in body_text

    async def test_endpoint_result_has_all_validation_fields(self):
        """Verify all ReasoningValidationResult fields present in JSON response."""
        import json

        from starlette.requests import Request

        from src.mcp_server import handle_validate_reasoning_http

        learning = LearningReference(
            uuid="test-uuid-1234",
            title="Test Learning",
            similarity_score=0.95,
        )

        pattern = PatternReference(
            pattern_uuid="pattern-123",
            name="Test Pattern",
            pattern_type="diagnostic",
            similarity_score=0.85,
            success_rate=0.92,
            occurrence_count=5,
            relevance_explanation="Matches diagnostic pattern",
        )

        mock_result = ReasoningValidationResult(
            validation_status="reinforced",
            conflicts=[],
            relevant_learnings=[learning],
            relevant_patterns=[pattern],
            claims_analyzed=2,
            entities_checked=3,
            elapsed_ms=175.5,
            degraded_queries=[],
            summary="Validation complete with reinforcement.",
        )

        mock_service = AsyncMock()
        mock_service.validate_response = AsyncMock(return_value=mock_result)

        mock_request = AsyncMock(spec=Request)
        mock_request.json = AsyncMock(
            return_value={"proposed_response": "test response"}
        )
        mock_request.headers = {}

        with (
            patch("src.mcp_server.llm_service_singleton", AsyncMock()),
            patch("src.mcp_server.get_vector_service", return_value=AsyncMock()),
            patch("src.mcp_server.neo4j_service", AsyncMock()),
            patch("src.mcp_server.redis_service", AsyncMock(is_connected=False)),
            patch(
                "src.mcp_server.ReasoningValidationService", return_value=mock_service
            ),
            patch("src.mcp_server.MNEMODROME_HOOK_TOKEN", None),
        ):
            response = await handle_validate_reasoning_http(mock_request)

            assert response.status_code == 200
            body_dict = json.loads(response.body)

            # Verify all 9 fields present
            assert "validation_status" in body_dict
            assert "conflicts" in body_dict
            assert "relevant_learnings" in body_dict
            assert "relevant_patterns" in body_dict
            assert "claims_analyzed" in body_dict
            assert "entities_checked" in body_dict
            assert "elapsed_ms" in body_dict
            assert "degraded_queries" in body_dict
            assert "summary" in body_dict

            # Verify field values
            assert body_dict["validation_status"] == "reinforced"
            assert body_dict["claims_analyzed"] == 2
            assert body_dict["entities_checked"] == 3
            assert len(body_dict["relevant_learnings"]) == 1
            assert body_dict["relevant_learnings"][0]["title"] == "Test Learning"

    async def test_endpoint_invalid_json_returns_400(self):
        """Test endpoint with invalid JSON returns 400."""
        from starlette.requests import Request

        from src.mcp_server import handle_validate_reasoning_http

        mock_request = AsyncMock(spec=Request)
        mock_request.json = AsyncMock(side_effect=ValueError("Invalid JSON"))
        mock_request.headers = {}

        with patch("src.mcp_server.MNEMODROME_HOOK_TOKEN", None):
            response = await handle_validate_reasoning_http(mock_request)

            assert response.status_code == 400

    async def test_endpoint_with_custom_timeout(self):
        """Test endpoint passes custom timeout_ms to service."""
        from starlette.requests import Request

        from src.mcp_server import handle_validate_reasoning_http

        mock_result = ReasoningValidationResult(
            validation_status="clean",
            summary="Test.",
            claims_analyzed=1,
            entities_checked=1,
            elapsed_ms=50.0,
        )

        mock_service = AsyncMock()
        mock_service.validate_response = AsyncMock(return_value=mock_result)

        mock_request = AsyncMock(spec=Request)
        mock_request.json = AsyncMock(
            return_value={"proposed_response": "test response", "timeout_ms": 500}
        )
        mock_request.headers = {}

        with (
            patch("src.mcp_server.llm_service_singleton", AsyncMock()),
            patch("src.mcp_server.get_vector_service", return_value=AsyncMock()),
            patch("src.mcp_server.neo4j_service", AsyncMock()),
            patch("src.mcp_server.redis_service", AsyncMock(is_connected=False)),
            patch(
                "src.mcp_server.ReasoningValidationService", return_value=mock_service
            ),
            patch("src.mcp_server.MNEMODROME_HOOK_TOKEN", None),
        ):
            response = await handle_validate_reasoning_http(mock_request)

            assert response.status_code == 200
            # Verify the custom timeout was passed to service
            mock_service.validate_response.assert_called_once()
            call_args = mock_service.validate_response.call_args
            assert call_args[1]["timeout_ms"] == 500


# End-to-End Test Procedure (requires running Mnemodrome server)
# 1. Start Mnemodrome:
#    cd . && uv run python -m src.mcp_server
#
# 2. Verify /validate-reasoning reachable:
#    curl -X POST http://127.0.0.1:8765/validate-reasoning \
#      -H 'Content-Type: application/json' \
#      -d '{"proposed_response": "test response"}'
#
# 3. Create test transcript with known-conflicting assertion:
#    cat > /tmp/test-transcript.jsonl << 'TRANSCRIPT'
#    {"type": "assistant", "message": {"content": [{"type": "text", "text": "Mnemodrome runs on <your-host>"}]}}
#    TRANSCRIPT
#
# 4. Pipe to hook:
#    echo '{"session_id":"test","transcript_path":"/tmp/test-transcript.jsonl","hook_event_name":"Stop"}' | \
#      MNEMODROME_CONTEXT_ENDPOINT="http://127.0.0.1:8765" node ~/.claude/hooks/reasoning-validation-hook.ts
#
# 5. Verify output:
#    - If conflicts found: stdout contains JSON with systemMessage key
#    - If clean: stdout empty (no output)
#    - Hook always exits 0 (graceful degradation)
#
# 6. In real Claude Code session:
#    - Make a factual claim that contradicts stored knowledge
#    - Verify systemMessage appears at start of next Claude response
#    - Verify marker <!-- REASONING_VALIDATION_INJECTED --> present in message
