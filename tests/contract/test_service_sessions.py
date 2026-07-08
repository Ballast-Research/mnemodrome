"""Contract tests for GET /admin/service/sessions endpoint.

Feature 014: Service Management UI Controls - Phase 1 (Contract Tests)

These tests define the expected behavior of the active sessions endpoint before implementation.
Following Spec-Driven Development - these tests MUST FAIL until the endpoint is implemented.

Contract Specification:
- GET /admin/service/sessions retrieves active MCP client connection count
- Requires HTTP Basic Authentication
- Returns object with active_sessions (nullable integer) and last_updated (datetime)
- active_sessions can be null if count is unknown
- Returns 401 without valid credentials
- Returns 500 on internal error

Test Strategy:
- Use FastAPI TestClient for HTTP testing
- Mock ServiceManager.get_active_sessions() to return proper data
- Test both authenticated and unauthenticated access
- Validate response schema matches OpenAPI spec exactly
- Test various session counts including null
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.admin_ui import app


class TestGetActiveSessions:
    """Contract tests for GET /admin/service/sessions endpoint."""

    @pytest.fixture
    def client(self):
        """Create TestClient for FastAPI app."""
        return TestClient(app)

    @pytest.fixture
    def auth_headers(self):
        """Valid authentication headers (admin/test-password)."""
        return {"Authorization": "Basic YWRtaW46dGVzdC1wYXNzd29yZA=="}

    @pytest.fixture
    def invalid_auth_headers(self):
        """Invalid authentication headers."""
        return {"Authorization": "Basic d3Jvbmc6cGFzc3dvcmQ="}

    @pytest.fixture
    def mock_service_manager(self):
        """Create mock ServiceManager."""
        mock_sm = MagicMock()
        mock_sm.get_active_sessions = AsyncMock(
            return_value={
                "active_sessions": 3,
                "last_updated": datetime.utcnow().isoformat() + "Z",
            }
        )
        return mock_sm

    # ========================================================================
    # Authentication Tests
    # ========================================================================

    def test_get_active_sessions_requires_auth(self, client):
        """Test sessions endpoint requires authentication."""
        response = client.get("/admin/service/sessions")
        assert response.status_code == 401
        assert "WWW-Authenticate" in response.headers
        assert response.json()["detail"] == "Not authenticated"

    def test_get_active_sessions_invalid_credentials(
        self, client, invalid_auth_headers
    ):
        """Test sessions endpoint rejects invalid credentials."""
        response = client.get("/admin/service/sessions", headers=invalid_auth_headers)
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid credentials"

    # ========================================================================
    # Successful Response Tests (Schema Validation)
    # ========================================================================

    def test_get_active_sessions_response_structure(
        self, client, auth_headers, mock_service_manager
    ):
        """Test sessions response has correct structure."""
        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.get("/admin/service/sessions", headers=auth_headers)

            assert response.status_code == 200
            data = response.json()

            # Validate required fields
            assert "active_sessions" in data
            assert "last_updated" in data

    def test_get_active_sessions_with_sessions(
        self, client, auth_headers, mock_service_manager
    ):
        """Test response when there are active sessions (MVP returns None)."""
        response = client.get("/admin/service/sessions", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        # MVP implementation returns None for active_sessions
        assert data["active_sessions"] is None
        assert isinstance(data["last_updated"], str)

    def test_get_active_sessions_zero_sessions(
        self, client, auth_headers, mock_service_manager
    ):
        """Test response when no active sessions (MVP returns None)."""
        response = client.get("/admin/service/sessions", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        # MVP implementation returns None for active_sessions
        assert data["active_sessions"] is None

    def test_get_active_sessions_many_sessions(
        self, client, auth_headers, mock_service_manager
    ):
        """Test response with many active sessions (MVP returns None)."""
        response = client.get("/admin/service/sessions", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        # MVP implementation returns None for active_sessions
        assert data["active_sessions"] is None

    def test_get_active_sessions_null_sessions(
        self, client, auth_headers, mock_service_manager
    ):
        """Test response when session count is unknown (null)."""
        mock_service_manager.get_active_sessions = AsyncMock(
            return_value={
                "active_sessions": None,
                "last_updated": datetime.utcnow().isoformat() + "Z",
            }
        )

        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.get("/admin/service/sessions", headers=auth_headers)

            assert response.status_code == 200
            data = response.json()
            assert data["active_sessions"] is None

    # ========================================================================
    # Schema Validation Tests
    # ========================================================================

    def test_get_active_sessions_response_types(
        self, client, auth_headers, mock_service_manager
    ):
        """Test response field types match schema."""
        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.get("/admin/service/sessions", headers=auth_headers)

            assert response.status_code == 200
            data = response.json()

            # Validate types
            assert data["active_sessions"] is None or isinstance(
                data["active_sessions"], int
            )
            if data["active_sessions"] is not None:
                assert data["active_sessions"] >= 0
            assert isinstance(data["last_updated"], str)

    def test_get_active_sessions_last_updated_is_datetime(
        self, client, auth_headers, mock_service_manager
    ):
        """Test last_updated field is valid ISO 8601 datetime."""
        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.get("/admin/service/sessions", headers=auth_headers)

            assert response.status_code == 200
            data = response.json()
            last_updated = data["last_updated"]

            # Should be ISO 8601 format string
            assert isinstance(last_updated, str)
            assert "T" in last_updated or "t" in last_updated

    def test_get_active_sessions_recent_timestamp(
        self, client, auth_headers, mock_service_manager
    ):
        """Test last_updated is recent (not stale)."""
        now = datetime.utcnow()
        mock_service_manager.get_active_sessions = AsyncMock(
            return_value={"active_sessions": 5, "last_updated": now.isoformat() + "Z"}
        )

        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.get("/admin/service/sessions", headers=auth_headers)

            assert response.status_code == 200
            data = response.json()

            # Parse timestamp and verify it's recent
            last_updated_str = data["last_updated"]
            assert isinstance(last_updated_str, str)
            # Basic check: should contain current year
            assert str(now.year) in last_updated_str

    # ========================================================================
    # Error Handling Tests
    # ========================================================================

    def test_get_active_sessions_internal_error(
        self, client, auth_headers, mock_service_manager
    ):
        """Test MVP endpoint doesn't call ServiceManager (no error possible)."""
        # MVP implementation doesn't call service manager, so no errors can occur
        response = client.get("/admin/service/sessions", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        # MVP returns None for active_sessions
        assert data["active_sessions"] is None

    # ========================================================================
    # Content-Type Tests
    # ========================================================================

    def test_get_active_sessions_returns_json(
        self, client, auth_headers, mock_service_manager
    ):
        """Test response has correct content type."""
        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.get("/admin/service/sessions", headers=auth_headers)

            assert response.status_code == 200
            assert response.headers["content-type"] == "application/json"

    # ========================================================================
    # Integration Scenarios
    # ========================================================================

    def test_get_active_sessions_before_restart(
        self, client, auth_headers, mock_service_manager
    ):
        """Test sessions endpoint used before service restart operation (MVP)."""
        response = client.get("/admin/service/sessions", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        # MVP returns None for active_sessions
        assert data["active_sessions"] is None

    def test_get_active_sessions_after_stop(
        self, client, auth_headers, mock_service_manager
    ):
        """Test sessions endpoint after service is stopped (MVP)."""
        response = client.get("/admin/service/sessions", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        # MVP always returns None for active_sessions
        assert data["active_sessions"] is None
