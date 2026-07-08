"""Contract tests for GET /admin/service/status endpoint.

Feature 014: Service Management UI Controls - Phase 1 (Contract Tests)

These tests define the expected behavior of the service status endpoint before implementation.
Following Spec-Driven Development - these tests MUST FAIL until the endpoint is implemented.

Contract Specification:
- GET /admin/service/status returns ServiceStatus schema
- Requires HTTP Basic Authentication
- Returns 401 without valid credentials
- Response includes state, version, deployment_type, service_name, last_updated (required)
- Optional fields: uptime_seconds, pid, memory_mb, cpu_percent (nullable)
- Returns 500 on internal error

Test Strategy:
- Use FastAPI TestClient for HTTP testing
- Mock ServiceManager to return proper ServiceStatus data
- Test both authenticated and unauthenticated access
- Validate response schema matches OpenAPI spec exactly
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.admin_ui import app


class TestGetServiceStatus:
    """Contract tests for GET /admin/service/status endpoint."""

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
        mock_sm.get_status = AsyncMock(
            return_value={
                "state": "running",
                "uptime_seconds": 3600,
                "version": "1.0.0",
                "pid": 1234,
                "memory_mb": 145.6,
                "cpu_percent": 2.5,
                "deployment_type": "systemd",
                "service_name": "mnemodrome-mcp.service",
                "last_updated": datetime.utcnow().isoformat() + "Z",
            }
        )
        return mock_sm

    # ========================================================================
    # Authentication Tests
    # ========================================================================

    def test_get_service_status_requires_auth(self, client):
        """Test service status endpoint requires authentication."""
        response = client.get("/admin/service/status")
        assert response.status_code == 401
        assert "WWW-Authenticate" in response.headers
        assert response.json()["detail"] == "Not authenticated"

    def test_get_service_status_invalid_credentials(self, client, invalid_auth_headers):
        """Test service status endpoint rejects invalid credentials."""
        response = client.get("/admin/service/status", headers=invalid_auth_headers)
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid credentials"

    # ========================================================================
    # Successful Response Tests (Schema Validation)
    # ========================================================================

    def test_get_service_status_running_service(
        self, client, auth_headers, mock_service_manager
    ):
        """Test successful status response when service is running."""
        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.get("/admin/service/status", headers=auth_headers)

            assert response.status_code == 200
            data = response.json()

            # Validate required fields
            assert data["state"] == "running"
            assert data["version"] == "1.0.0"
            assert data["deployment_type"] == "systemd"
            assert data["service_name"] == "mnemodrome-mcp.service"
            assert "last_updated" in data

            # Validate optional fields (present in running state)
            assert data["uptime_seconds"] == 3600
            assert data["pid"] == 1234
            assert data["memory_mb"] == 145.6
            assert data["cpu_percent"] == 2.5

    def test_get_service_status_stopped_service(
        self, client, auth_headers, mock_service_manager
    ):
        """Test status response when service is stopped."""
        mock_service_manager.get_status = AsyncMock(
            return_value={
                "state": "stopped",
                "uptime_seconds": None,
                "version": "1.0.0",
                "pid": None,
                "memory_mb": None,
                "cpu_percent": None,
                "deployment_type": "systemd",
                "service_name": "mnemodrome-mcp.service",
                "last_updated": datetime.utcnow().isoformat() + "Z",
            }
        )

        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.get("/admin/service/status", headers=auth_headers)

            assert response.status_code == 200
            data = response.json()

            # Validate required fields
            assert data["state"] == "stopped"
            assert data["version"] == "1.0.0"
            assert data["deployment_type"] == "systemd"
            assert data["service_name"] == "mnemodrome-mcp.service"

            # Optional fields should be null when stopped
            assert data["uptime_seconds"] is None
            assert data["pid"] is None
            assert data["memory_mb"] is None
            assert data["cpu_percent"] is None

    def test_get_service_status_failed_service(
        self, client, auth_headers, mock_service_manager
    ):
        """Test status response when service is in failed state."""
        mock_service_manager.get_status = AsyncMock(
            return_value={
                "state": "failed",
                "uptime_seconds": None,
                "version": "1.0.0",
                "pid": None,
                "memory_mb": None,
                "cpu_percent": None,
                "deployment_type": "systemd",
                "service_name": "mnemodrome-mcp.service",
                "last_updated": datetime.utcnow().isoformat() + "Z",
            }
        )

        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.get("/admin/service/status", headers=auth_headers)

            assert response.status_code == 200
            data = response.json()
            assert data["state"] == "failed"

    def test_get_service_status_starting_service(
        self, client, auth_headers, mock_service_manager
    ):
        """Test status response when service is starting."""
        mock_service_manager.get_status = AsyncMock(
            return_value={
                "state": "starting",
                "uptime_seconds": None,
                "version": "1.0.0",
                "pid": 5678,
                "memory_mb": 50.2,
                "cpu_percent": 25.0,
                "deployment_type": "systemd",
                "service_name": "mnemodrome-mcp.service",
                "last_updated": datetime.utcnow().isoformat() + "Z",
            }
        )

        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.get("/admin/service/status", headers=auth_headers)

            assert response.status_code == 200
            data = response.json()
            assert data["state"] == "starting"
            assert data["pid"] == 5678

    def test_get_service_status_docker_deployment(
        self, client, auth_headers, mock_service_manager
    ):
        """Test status response for Docker-deployed service."""
        mock_service_manager.get_status = AsyncMock(
            return_value={
                "state": "running",
                "uptime_seconds": 7200,
                "version": "1.0.0",
                "pid": 9999,
                "memory_mb": 256.8,
                "cpu_percent": 5.1,
                "deployment_type": "docker",
                "service_name": "mnemodrome-mcp-container",
                "last_updated": datetime.utcnow().isoformat() + "Z",
            }
        )

        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.get("/admin/service/status", headers=auth_headers)

            assert response.status_code == 200
            data = response.json()
            assert data["deployment_type"] == "docker"
            assert data["service_name"] == "mnemodrome-mcp-container"

    # ========================================================================
    # Schema Validation Tests
    # ========================================================================

    def test_get_service_status_response_schema_types(
        self, client, auth_headers, mock_service_manager
    ):
        """Test response field types match schema."""
        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.get("/admin/service/status", headers=auth_headers)

            assert response.status_code == 200
            data = response.json()

            # Validate types
            assert isinstance(data["state"], str)
            assert data["state"] in [
                "running",
                "stopped",
                "failed",
                "starting",
                "stopping",
            ]
            assert isinstance(data["version"], str)
            assert isinstance(data["deployment_type"], str)
            assert data["deployment_type"] in ["systemd", "docker"]
            assert isinstance(data["service_name"], str)
            assert isinstance(data["last_updated"], str)

            # Optional nullable fields
            if data["uptime_seconds"] is not None:
                assert isinstance(data["uptime_seconds"], int)
                assert data["uptime_seconds"] >= 0
            if data["pid"] is not None:
                assert isinstance(data["pid"], int)
            if data["memory_mb"] is not None:
                assert isinstance(data["memory_mb"], (int, float))
                assert data["memory_mb"] >= 0
            if data["cpu_percent"] is not None:
                assert isinstance(data["cpu_percent"], (int, float))
                assert 0 <= data["cpu_percent"] <= 100

    def test_get_service_status_last_updated_is_datetime(
        self, client, auth_headers, mock_service_manager
    ):
        """Test last_updated field is valid ISO 8601 datetime."""
        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.get("/admin/service/status", headers=auth_headers)

            assert response.status_code == 200
            data = response.json()
            last_updated = data["last_updated"]

            # Should be ISO 8601 format string (can be parsed as datetime)
            assert isinstance(last_updated, str)
            assert "T" in last_updated or "t" in last_updated

    # ========================================================================
    # Error Handling Tests
    # ========================================================================

    def test_get_service_status_internal_error(
        self, client, auth_headers, mock_service_manager
    ):
        """Test 500 response on ServiceManager error."""
        mock_service_manager.get_status = AsyncMock(
            side_effect=Exception("Database connection failed")
        )

        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.get("/admin/service/status", headers=auth_headers)

            # Should return 500 error
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data

    # ========================================================================
    # Content-Type Tests
    # ========================================================================

    def test_get_service_status_returns_json(
        self, client, auth_headers, mock_service_manager
    ):
        """Test response has correct content type."""
        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.get("/admin/service/status", headers=auth_headers)

            assert response.status_code == 200
            assert response.headers["content-type"] == "application/json"
