"""Tests for Feature 024: Project-Aware Session Init HTTP endpoint."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# We'll test the individual handler functions without needing a full FastMCP setup
# since we're mocking the services


@pytest.mark.asyncio
async def test_project_context_returns_200_with_markdown_for_known_project():
    """Happy path: returns formatted markdown for a project with data."""
    from src.mcp_server import handle_project_context

    # Mock request
    mock_request = MagicMock()
    mock_request.query_params = {"project": "Mnemodrome"}
    mock_request.headers = {}

    # Mock services
    mock_temporal_results = [
        {"name": "Feature 024", "type": "feature", "timestamp": "2026-03-05"}
    ]

    mock_sessions = [
        {"summary": "Implemented project context endpoint", "created_at": "2026-03-05"}
    ]

    mock_learnings = [{"title": "FastMCP Custom Routes", "created_at": "2026-03-05"}]

    with (
        patch("src.mcp_server.neo4j_service") as mock_neo4j,
        patch("src.mcp_server.get_vector_service") as mock_get_vector,
    ):
        # Setup Neo4j mock
        mock_neo4j.temporal_query = AsyncMock(return_value=mock_temporal_results)

        # Setup Vector service mock
        mock_vector_service = AsyncMock()
        mock_vector_service.is_enabled.return_value = True
        mock_vector_service.search = AsyncMock(
            side_effect=[
                mock_sessions,  # First call for sessions
                mock_learnings,  # Second call for learnings
            ]
        )
        mock_get_vector.return_value = mock_vector_service

        # Call handler
        response = await handle_project_context(mock_request)

        # Verify response
        assert response.status_code == 200
        assert response.media_type == "text/markdown"
        content = response.body.decode()
        assert "PROJECT_AWARE_SESSION_INIT" in content
        assert "Mnemodrome" in content
        assert "END_PROJECT_AWARE_SESSION_INIT" in content


@pytest.mark.asyncio
async def test_project_context_returns_400_when_project_param_missing():
    """Missing project param returns 400."""
    from src.mcp_server import handle_project_context

    # Mock request without project param
    mock_request = MagicMock()
    mock_request.query_params = {}
    mock_request.headers = {}

    response = await handle_project_context(mock_request)

    assert response.status_code == 400
    assert b"project param required" in response.body


@pytest.mark.asyncio
async def test_project_context_returns_401_when_token_configured_and_missing():
    """Auth check fires when MNEMODROME_HOOK_TOKEN is set."""
    from src.mcp_server import handle_project_context

    # Mock request without auth
    mock_request = MagicMock()
    mock_request.query_params = {"project": "Mnemodrome"}
    mock_request.headers = {}

    # Patch config to set token
    with patch("src.mcp_server.MNEMODROME_HOOK_TOKEN", "test-token-secret"):
        response = await handle_project_context(mock_request)

        assert response.status_code == 401
        assert b"Unauthorized" in response.body


@pytest.mark.asyncio
async def test_project_context_returns_200_when_no_token_configured():
    """No auth check when token is empty."""
    from src.mcp_server import handle_project_context

    # Mock request
    mock_request = MagicMock()
    mock_request.query_params = {"project": "Mnemodrome"}
    mock_request.headers = {}

    # Patch config with empty token
    with (
        patch("src.mcp_server.MNEMODROME_HOOK_TOKEN", ""),
        patch("src.mcp_server.neo4j_service") as mock_neo4j,
        patch("src.mcp_server.get_vector_service") as mock_get_vector,
    ):
        # Setup mocks
        mock_neo4j.temporal_query = AsyncMock(return_value=[])
        mock_vector_service = AsyncMock()
        mock_vector_service.is_enabled.return_value = True
        mock_vector_service.search = AsyncMock(return_value=[])
        mock_get_vector.return_value = mock_vector_service

        response = await handle_project_context(mock_request)

        # Should succeed without auth
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_project_context_handles_neo4j_failure_gracefully():
    """Returns partial context (empty temporal) when Neo4j query fails."""
    from src.mcp_server import handle_project_context

    # Mock request
    mock_request = MagicMock()
    mock_request.query_params = {"project": "Mnemodrome"}
    mock_request.headers = {}

    mock_sessions = [{"summary": "Recent work session", "created_at": "2026-03-05"}]

    with (
        patch("src.mcp_server.neo4j_service") as mock_neo4j,
        patch("src.mcp_server.get_vector_service") as mock_get_vector,
    ):
        # Setup Neo4j to fail
        mock_neo4j.temporal_query = AsyncMock(
            side_effect=ConnectionError("Neo4j unavailable")
        )

        # Setup Vector service to succeed
        mock_vector_service = AsyncMock()
        mock_vector_service.is_enabled.return_value = True
        mock_vector_service.search = AsyncMock(return_value=mock_sessions)
        mock_get_vector.return_value = mock_vector_service

        response = await handle_project_context(mock_request)

        # Should still return 200 with partial data
        assert response.status_code == 200
        content = response.body.decode()
        assert "PROJECT_AWARE_SESSION_INIT" in content


@pytest.mark.asyncio
async def test_project_context_handles_vector_service_unavailable():
    """Returns temporal context only when vector service is down."""
    from src.mcp_server import handle_project_context

    # Mock request
    mock_request = MagicMock()
    mock_request.query_params = {"project": "Mnemodrome"}
    mock_request.headers = {}

    mock_temporal = [
        {"name": "Feature 024", "type": "feature", "timestamp": "2026-03-05"}
    ]

    with (
        patch("src.mcp_server.neo4j_service") as mock_neo4j,
        patch("src.mcp_server.get_vector_service") as mock_get_vector,
    ):
        # Setup Neo4j to succeed
        mock_neo4j.temporal_query = AsyncMock(return_value=mock_temporal)

        # Setup Vector service to be unavailable
        mock_get_vector.return_value = None

        response = await handle_project_context(mock_request)

        # Should still return 200 with temporal data only
        assert response.status_code == 200
        content = response.body.decode()
        assert "Recent Activity" in content or "PROJECT_AWARE_SESSION_INIT" in content


@pytest.mark.asyncio
async def test_project_context_handles_all_services_failing():
    """Returns minimal context (empty sections) rather than 500."""
    from src.mcp_server import handle_project_context

    # Mock request
    mock_request = MagicMock()
    mock_request.query_params = {"project": "Mnemodrome"}
    mock_request.headers = {}

    with (
        patch("src.mcp_server.neo4j_service") as mock_neo4j,
        patch("src.mcp_server.get_vector_service") as mock_get_vector,
    ):
        # Setup Neo4j to fail
        mock_neo4j.temporal_query = AsyncMock(side_effect=ConnectionError("Neo4j down"))

        # Setup Vector service to fail
        mock_get_vector.return_value = None

        response = await handle_project_context(mock_request)

        # Should return 200 with empty context, not 500
        assert response.status_code == 200
        content = response.body.decode()
        assert "PROJECT_AWARE_SESSION_INIT" in content
        assert "END_PROJECT_AWARE_SESSION_INIT" in content


@pytest.mark.asyncio
async def test_project_context_custom_days_param():
    """Passes days param to temporal query."""
    from src.mcp_server import handle_project_context

    # Mock request with custom days
    mock_request = MagicMock()
    mock_request.query_params = {"project": "Mnemodrome", "days": "30"}
    mock_request.headers = {}

    with (
        patch("src.mcp_server.neo4j_service") as mock_neo4j,
        patch("src.mcp_server.get_vector_service") as mock_get_vector,
    ):
        # Setup mocks
        mock_neo4j.temporal_query = AsyncMock(return_value=[])
        mock_vector_service = AsyncMock()
        mock_vector_service.is_enabled.return_value = True
        mock_vector_service.search = AsyncMock(return_value=[])
        mock_get_vector.return_value = mock_vector_service

        response = await handle_project_context(mock_request)

        # Verify temporal_query was called with custom days
        mock_neo4j.temporal_query.assert_called_once()
        call_kwargs = mock_neo4j.temporal_query.call_args[1]
        assert call_kwargs.get("days") == 30


@pytest.mark.asyncio
async def test_project_context_response_format_contains_required_markers():
    """Response wraps content in PROJECT_AWARE_SESSION_INIT markers."""
    from src.mcp_server import handle_project_context

    # Mock request
    mock_request = MagicMock()
    mock_request.query_params = {"project": "Mnemodrome"}
    mock_request.headers = {}

    with (
        patch("src.mcp_server.neo4j_service") as mock_neo4j,
        patch("src.mcp_server.get_vector_service") as mock_get_vector,
    ):
        # Setup mocks to return empty data
        mock_neo4j.temporal_query = AsyncMock(return_value=[])
        mock_vector_service = AsyncMock()
        mock_vector_service.is_enabled.return_value = True
        mock_vector_service.search = AsyncMock(return_value=[])
        mock_get_vector.return_value = mock_vector_service

        response = await handle_project_context(mock_request)

        # Verify response format
        assert response.status_code == 200
        content = response.body.decode()
        assert "<!-- PROJECT_AWARE_SESSION_INIT -->" in content
        assert "<!-- END_PROJECT_AWARE_SESSION_INIT -->" in content
        assert "# Session Context: Mnemodrome" in content


@pytest.mark.asyncio
async def test_project_vector_search_returns_empty_on_unavailable_service():
    """Vector search helper returns empty list when service is unavailable."""
    from src.mcp_server import _project_vector_search

    with patch("src.mcp_server.get_vector_service") as mock_get_vector:
        # Mock service as unavailable
        mock_get_vector.return_value = None

        result = await _project_vector_search("Mnemodrome", "sessions", 5)

        assert result == []


@pytest.mark.asyncio
async def test_project_vector_search_returns_empty_on_error():
    """Vector search helper returns empty list on exception."""
    from src.mcp_server import _project_vector_search

    with patch("src.mcp_server.get_vector_service") as mock_get_vector:
        # Mock service to raise exception
        mock_service = AsyncMock()
        mock_service.is_enabled.return_value = True
        mock_service.search = AsyncMock(side_effect=RuntimeError("Search failed"))
        mock_get_vector.return_value = mock_service

        result = await _project_vector_search("Mnemodrome", "sessions", 5)

        # Should return empty list instead of raising
        assert result == []
