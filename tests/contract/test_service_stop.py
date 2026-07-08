"""Contract tests for POST /admin/service/stop endpoint.

Feature 014: Service Management UI Controls - Phase 1 (Contract Tests)

These tests define the expected behavior of the service stop endpoint before implementation.
Following Spec-Driven Development - these tests MUST FAIL until the endpoint is implemented.

Contract Specification:
- POST /admin/service/stop gracefully stops the Mnemodrome MCP service
- Requires HTTP Basic Authentication
- Request body must contain 'confirm': true
- Returns ControlOperationResult schema on success
- Returns 400 if confirmation missing
- Returns 401 without valid credentials
- Returns 500 on internal error

Test Strategy:
- Use FastAPI TestClient for HTTP testing
- Mock ServiceManager.stop() to return proper ControlOperationResult
- Test both authenticated and unauthenticated access
- Test request validation (missing confirmation)
- Validate response schema matches OpenAPI spec exactly
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.admin_ui import app


class TestStopService:
    """Contract tests for POST /admin/service/stop endpoint."""

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
        mock_sm.stop_service = AsyncMock(
            return_value={
                "status": "success",
                "message": "Service stopping gracefully",
                "operation_id": "evt_123456",
                "active_sessions": 2,
            }
        )
        return mock_sm

    # ========================================================================
    # Authentication Tests
    # ========================================================================

    def test_stop_service_requires_auth(self, client):
        """Test stop endpoint requires authentication."""
        response = client.post("/admin/service/stop", json={"confirm": True})
        assert response.status_code == 401
        assert "WWW-Authenticate" in response.headers
        assert response.json()["detail"] == "Not authenticated"

    def test_stop_service_invalid_credentials(self, client, invalid_auth_headers):
        """Test stop endpoint rejects invalid credentials."""
        response = client.post(
            "/admin/service/stop", json={"confirm": True}, headers=invalid_auth_headers
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid credentials"

    # ========================================================================
    # Request Validation Tests
    # ========================================================================

    def test_stop_service_missing_confirmation(self, client, auth_headers):
        """Test 400 response when confirmation is missing."""
        response = client.post("/admin/service/stop", json={}, headers=auth_headers)
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

    def test_stop_service_confirmation_false(self, client, auth_headers):
        """Test 400 response when confirmation is false."""
        response = client.post(
            "/admin/service/stop", json={"confirm": False}, headers=auth_headers
        )
        assert response.status_code == 400

    def test_stop_service_invalid_json(self, client, auth_headers):
        """Test 400 response with malformed JSON."""
        response = client.post(
            "/admin/service/stop",
            data="invalid json",
            headers={**auth_headers, "content-type": "application/json"},
        )
        assert response.status_code in [400, 422]

    # ========================================================================
    # Successful Response Tests (Schema Validation)
    # ========================================================================

    def test_stop_service_successful(self, client, auth_headers, mock_service_manager):
        """Test successful stop response."""
        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.post(
                "/admin/service/stop", json={"confirm": True}, headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()

            # Validate required fields
            assert data["status"] == "success"
            assert data["message"] == "Service stopping gracefully"

            # Validate optional fields
            assert "operation_id" in data
            assert data["operation_id"] == "evt_123456"
            assert data["active_sessions"] == 2

    def test_stop_service_with_active_sessions(
        self, client, auth_headers, mock_service_manager
    ):
        """Test stop response includes active session count."""
        mock_service_manager.stop_service = AsyncMock(
            return_value={
                "status": "success",
                "message": "Service stopping, disconnecting 8 active sessions",
                "operation_id": "evt_789012",
                "active_sessions": 8,
            }
        )

        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.post(
                "/admin/service/stop", json={"confirm": True}, headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert data["active_sessions"] == 8

    def test_stop_service_no_active_sessions(
        self, client, auth_headers, mock_service_manager
    ):
        """Test stop response when no active sessions."""
        mock_service_manager.stop_service = AsyncMock(
            return_value={
                "status": "success",
                "message": "Service stopped",
                "operation_id": "evt_345678",
                "active_sessions": 0,
            }
        )

        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.post(
                "/admin/service/stop", json={"confirm": True}, headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert data["active_sessions"] == 0

    def test_stop_service_null_active_sessions(
        self, client, auth_headers, mock_service_manager
    ):
        """Test stop response when active_sessions is unknown (null)."""
        mock_service_manager.stop_service = AsyncMock(
            return_value={
                "status": "success",
                "message": "Service stopping",
                "operation_id": "evt_999999",
                "active_sessions": None,
            }
        )

        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.post(
                "/admin/service/stop", json={"confirm": True}, headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert data["active_sessions"] is None

    # ========================================================================
    # Schema Validation Tests
    # ========================================================================

    def test_stop_service_response_schema_types(
        self, client, auth_headers, mock_service_manager
    ):
        """Test response field types match schema."""
        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.post(
                "/admin/service/stop", json={"confirm": True}, headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()

            # Validate types
            assert isinstance(data["status"], str)
            assert data["status"] in ["success", "failure"]
            assert isinstance(data["message"], str)
            assert len(data["message"]) > 0

            # Optional fields
            if "operation_id" in data and data["operation_id"] is not None:
                assert isinstance(data["operation_id"], str)
            if "active_sessions" in data and data["active_sessions"] is not None:
                assert isinstance(data["active_sessions"], int)
                assert data["active_sessions"] >= 0

    # ========================================================================
    # Failure Scenarios
    # ========================================================================

    def test_stop_service_failure_status(
        self, client, auth_headers, mock_service_manager
    ):
        """Test stop response when operation fails."""
        mock_service_manager.stop_service = AsyncMock(
            return_value={
                "status": "failure",
                "message": "Failed to stop service: permission denied",
                "operation_id": "evt_failure_001",
                "active_sessions": None,
            }
        )

        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.post(
                "/admin/service/stop", json={"confirm": True}, headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "failure"

    def test_stop_service_internal_error(
        self, client, auth_headers, mock_service_manager
    ):
        """Test 500 response on ServiceManager error."""
        mock_service_manager.stop_service = AsyncMock(
            side_effect=Exception("Connection lost")
        )

        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.post(
                "/admin/service/stop", json={"confirm": True}, headers=auth_headers
            )

            assert response.status_code == 500
            data = response.json()
            assert "detail" in data

    # ========================================================================
    # Graceful Shutdown Tests
    # ========================================================================

    def test_stop_service_graceful_shutdown_message(
        self, client, auth_headers, mock_service_manager
    ):
        """Test stop initiates graceful shutdown (not force kill)."""
        mock_service_manager.stop_service = AsyncMock(
            return_value={
                "status": "success",
                "message": "Graceful shutdown initiated, waiting for connections to close",
                "operation_id": "evt_graceful_001",
                "active_sessions": 3,
            }
        )

        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.post(
                "/admin/service/stop", json={"confirm": True}, headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert "graceful" in data["message"].lower()

    # ========================================================================
    # Content-Type Tests
    # ========================================================================

    def test_stop_service_returns_json(
        self, client, auth_headers, mock_service_manager
    ):
        """Test response has correct content type."""
        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.post(
                "/admin/service/stop", json={"confirm": True}, headers=auth_headers
            )

            assert response.status_code == 200
            assert response.headers["content-type"] == "application/json"

    def test_stop_service_confirmation_required_strict(
        self, client, auth_headers, mock_service_manager
    ):
        """Test confirmation must be explicitly true, not just truthy."""
        # Test with confirm=1 (truthy but not boolean)
        response = client.post(
            "/admin/service/stop", json={"confirm": 1}, headers=auth_headers
        )
        # Should reject non-boolean confirmation
        assert response.status_code == 400
