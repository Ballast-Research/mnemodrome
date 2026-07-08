"""Contract tests for POST /admin/service/restart endpoint.

Feature 014: Service Management UI Controls - Phase 1 (Contract Tests)

These tests define the expected behavior of the service restart endpoint before implementation.
Following Spec-Driven Development - these tests MUST FAIL until the endpoint is implemented.

Contract Specification:
- POST /admin/service/restart restarts the Mnemodrome MCP service
- Requires HTTP Basic Authentication
- Request body must contain 'confirm': true
- Returns ControlOperationResult schema on success
- Returns 400 if confirmation missing
- Returns 401 without valid credentials
- Returns 500 on internal error
- Returns 504 if operation times out

Test Strategy:
- Use FastAPI TestClient for HTTP testing
- Mock ServiceManager.restart() to return proper ControlOperationResult
- Test both authenticated and unauthenticated access
- Test request validation (missing confirmation)
- Validate response schema matches OpenAPI spec exactly
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.admin_ui import app


class TestRestartService:
    """Contract tests for POST /admin/service/restart endpoint."""

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
        mock_sm.restart_service = AsyncMock(
            return_value={
                "status": "success",
                "message": "Service restarting",
                "operation_id": "evt_123456",
                "active_sessions": 2,
            }
        )
        return mock_sm

    # ========================================================================
    # Authentication Tests
    # ========================================================================

    def test_restart_service_requires_auth(self, client):
        """Test restart endpoint requires authentication."""
        response = client.post("/admin/service/restart", json={"confirm": True})
        assert response.status_code == 401
        assert "WWW-Authenticate" in response.headers
        assert response.json()["detail"] == "Not authenticated"

    def test_restart_service_invalid_credentials(self, client, invalid_auth_headers):
        """Test restart endpoint rejects invalid credentials."""
        response = client.post(
            "/admin/service/restart",
            json={"confirm": True},
            headers=invalid_auth_headers,
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid credentials"

    # ========================================================================
    # Request Validation Tests
    # ========================================================================

    def test_restart_service_missing_confirmation(self, client, auth_headers):
        """Test 400 response when confirmation is missing."""
        response = client.post("/admin/service/restart", json={}, headers=auth_headers)
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

    def test_restart_service_confirmation_false(self, client, auth_headers):
        """Test 400 response when confirmation is false."""
        response = client.post(
            "/admin/service/restart", json={"confirm": False}, headers=auth_headers
        )
        assert response.status_code == 400

    def test_restart_service_invalid_json(self, client, auth_headers):
        """Test 400 response with malformed JSON."""
        response = client.post(
            "/admin/service/restart",
            data="invalid json",
            headers={**auth_headers, "content-type": "application/json"},
        )
        assert response.status_code in [400, 422]

    # ========================================================================
    # Successful Response Tests (Schema Validation)
    # ========================================================================

    def test_restart_service_successful(
        self, client, auth_headers, mock_service_manager
    ):
        """Test successful restart response."""
        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.post(
                "/admin/service/restart", json={"confirm": True}, headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()

            # Validate required fields
            assert data["status"] == "success"
            assert data["message"] == "Service restarting"

            # Validate optional fields
            assert "operation_id" in data
            assert data["operation_id"] == "evt_123456"
            assert data["active_sessions"] == 2

    def test_restart_service_with_active_sessions(
        self, client, auth_headers, mock_service_manager
    ):
        """Test restart response includes active session count."""
        mock_service_manager.restart_service = AsyncMock(
            return_value={
                "status": "success",
                "message": "Service restarting, disconnecting 5 active sessions",
                "operation_id": "evt_789012",
                "active_sessions": 5,
            }
        )

        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.post(
                "/admin/service/restart", json={"confirm": True}, headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert data["active_sessions"] == 5

    def test_restart_service_no_active_sessions(
        self, client, auth_headers, mock_service_manager
    ):
        """Test restart response when no active sessions."""
        mock_service_manager.restart_service = AsyncMock(
            return_value={
                "status": "success",
                "message": "Service restarting",
                "operation_id": "evt_345678",
                "active_sessions": 0,
            }
        )

        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.post(
                "/admin/service/restart", json={"confirm": True}, headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert data["active_sessions"] == 0

    def test_restart_service_null_active_sessions(
        self, client, auth_headers, mock_service_manager
    ):
        """Test restart response when active_sessions is unknown (null)."""
        mock_service_manager.restart_service = AsyncMock(
            return_value={
                "status": "success",
                "message": "Service restarting",
                "operation_id": "evt_999999",
                "active_sessions": None,
            }
        )

        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.post(
                "/admin/service/restart", json={"confirm": True}, headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert data["active_sessions"] is None

    # ========================================================================
    # Schema Validation Tests
    # ========================================================================

    def test_restart_service_response_schema_types(
        self, client, auth_headers, mock_service_manager
    ):
        """Test response field types match schema."""
        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.post(
                "/admin/service/restart", json={"confirm": True}, headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()

            # Validate types
            assert isinstance(data["status"], str)
            assert data["status"] in ["success", "failure"]
            assert isinstance(data["message"], str)
            assert len(data["message"]) > 0

            # Optional fields
            if "operation_id" in data:
                assert isinstance(data["operation_id"], str)
            if "active_sessions" in data and data["active_sessions"] is not None:
                assert isinstance(data["active_sessions"], int)
                assert data["active_sessions"] >= 0

    # ========================================================================
    # Failure Scenarios
    # ========================================================================

    def test_restart_service_failure_status(
        self, client, auth_headers, mock_service_manager
    ):
        """Test restart response when operation fails."""
        mock_service_manager.restart_service = AsyncMock(
            return_value={
                "status": "failure",
                "message": "Failed to restart service: permission denied",
                "operation_id": "evt_failure_001",
                "active_sessions": None,
            }
        )

        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.post(
                "/admin/service/restart", json={"confirm": True}, headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "failure"

    def test_restart_service_timeout(self, client, auth_headers, mock_service_manager):
        """Test 504 response when restart operation times out."""
        import subprocess

        mock_service_manager.restart_service = AsyncMock(
            side_effect=subprocess.TimeoutExpired("systemctl", 30)
        )

        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.post(
                "/admin/service/restart", json={"confirm": True}, headers=auth_headers
            )

            # Should return 504 Gateway Timeout
            assert response.status_code == 504
            data = response.json()
            assert "detail" in data

    def test_restart_service_internal_error(
        self, client, auth_headers, mock_service_manager
    ):
        """Test 500 response on ServiceManager error."""
        mock_service_manager.restart_service = AsyncMock(
            side_effect=Exception("Database connection failed")
        )

        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.post(
                "/admin/service/restart", json={"confirm": True}, headers=auth_headers
            )

            assert response.status_code == 500
            data = response.json()
            assert "detail" in data

    # ========================================================================
    # Content-Type Tests
    # ========================================================================

    def test_restart_service_returns_json(
        self, client, auth_headers, mock_service_manager
    ):
        """Test response has correct content type."""
        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.post(
                "/admin/service/restart", json={"confirm": True}, headers=auth_headers
            )

            assert response.status_code == 200
            assert response.headers["content-type"] == "application/json"

    def test_restart_service_confirmation_case_sensitivity(
        self, client, auth_headers, mock_service_manager
    ):
        """Test confirmation field is properly validated."""
        # Test with confirm=true (lowercase, standard JSON bool)
        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.post(
                "/admin/service/restart", json={"confirm": True}, headers=auth_headers
            )

            assert response.status_code == 200
