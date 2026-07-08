"""Contract tests for GET /admin/service/logs endpoint.

Feature 014: Service Management UI Controls - Phase 1 (Contract Tests)

These tests define the expected behavior of the service logs endpoint before implementation.
Following Spec-Driven Development - these tests MUST FAIL until the endpoint is implemented.

Contract Specification:
- GET /admin/service/logs retrieves recent log entries
- Requires HTTP Basic Authentication
- Query parameter 'lines' (optional, default 100, range 1-1000)
- Returns LogsResponse schema with array of LogEntry objects
- LogEntry includes: timestamp, severity (enum), message, source (nullable)
- Returns 400 if lines parameter out of range
- Returns 401 without valid credentials
- Returns 500 on internal error
- Returns 504 if log retrieval times out

Test Strategy:
- Use FastAPI TestClient for HTTP testing
- Mock ServiceManager.get_logs() to return proper LogsResponse
- Test query parameter validation
- Test authenticated and unauthenticated access
- Validate response schema matches OpenAPI spec exactly
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.admin_ui import app


class TestGetServiceLogs:
    """Contract tests for GET /admin/service/logs endpoint."""

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
        now = datetime.utcnow()
        mock_sm.get_logs = AsyncMock(
            return_value={
                "logs": [
                    {
                        "timestamp": (now - timedelta(seconds=100)).isoformat() + "Z",
                        "severity": "INFO",
                        "message": "MCP server started on port 8765",
                        "source": "mcp_server",
                    },
                    {
                        "timestamp": (now - timedelta(seconds=50)).isoformat() + "Z",
                        "severity": "DEBUG",
                        "message": "Incoming connection from client",
                        "source": "connection_handler",
                    },
                    {
                        "timestamp": now.isoformat() + "Z",
                        "severity": "INFO",
                        "message": "Health check passed",
                        "source": "health",
                    },
                ],
                "total": 3,
                "lines_requested": 100,
            }
        )
        return mock_sm

    # ========================================================================
    # Authentication Tests
    # ========================================================================

    def test_get_service_logs_requires_auth(self, client):
        """Test logs endpoint requires authentication."""
        response = client.get("/admin/service/logs")
        assert response.status_code == 401
        assert "WWW-Authenticate" in response.headers
        assert response.json()["detail"] == "Not authenticated"

    def test_get_service_logs_invalid_credentials(self, client, invalid_auth_headers):
        """Test logs endpoint rejects invalid credentials."""
        response = client.get("/admin/service/logs", headers=invalid_auth_headers)
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid credentials"

    # ========================================================================
    # Query Parameter Tests
    # ========================================================================

    def test_get_service_logs_default_lines(
        self, client, auth_headers, mock_service_manager
    ):
        """Test default lines parameter is 100."""
        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.get("/admin/service/logs", headers=auth_headers)

            assert response.status_code == 200
            data = response.json()
            assert data["lines_requested"] == 100

    def test_get_service_logs_custom_lines(
        self, client, auth_headers, mock_service_manager
    ):
        """Test custom lines parameter."""
        mock_service_manager.get_logs = AsyncMock(
            return_value={
                "logs": [
                    {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "severity": "INFO",
                        "message": f"Log entry {i}",
                        "source": "test",
                    }
                    for i in range(50)
                ],
                "total": 50,
                "lines_requested": 50,
            }
        )

        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.get("/admin/service/logs?lines=50", headers=auth_headers)

            assert response.status_code == 200
            data = response.json()
            assert data["lines_requested"] == 50
            assert len(data["logs"]) == 50

    def test_get_service_logs_minimum_lines(
        self, client, auth_headers, mock_service_manager
    ):
        """Test minimum lines parameter (1)."""
        mock_service_manager.get_logs = AsyncMock(
            return_value={
                "logs": [
                    {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "severity": "INFO",
                        "message": "Single log entry",
                        "source": "test",
                    }
                ],
                "total": 1,
                "lines_requested": 1,
            }
        )

        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.get("/admin/service/logs?lines=1", headers=auth_headers)

            assert response.status_code == 200
            data = response.json()
            assert data["lines_requested"] == 1

    def test_get_service_logs_maximum_lines(
        self, client, auth_headers, mock_service_manager
    ):
        """Test maximum lines parameter (1000)."""
        mock_service_manager.get_logs = AsyncMock(
            return_value={
                "logs": [
                    {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "severity": "INFO",
                        "message": f"Log entry {i}",
                        "source": "test",
                    }
                    for i in range(1000)
                ],
                "total": 1000,
                "lines_requested": 1000,
            }
        )

        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.get(
                "/admin/service/logs?lines=1000", headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert data["lines_requested"] == 1000

    def test_get_service_logs_lines_out_of_range_too_low(self, client, auth_headers):
        """Test 422 response when lines < 1."""
        response = client.get("/admin/service/logs?lines=0", headers=auth_headers)
        assert response.status_code == 422

    def test_get_service_logs_lines_out_of_range_too_high(self, client, auth_headers):
        """Test 422 response when lines > 1000."""
        response = client.get("/admin/service/logs?lines=1001", headers=auth_headers)
        assert response.status_code == 422

    def test_get_service_logs_lines_non_integer(self, client, auth_headers):
        """Test 400 response when lines is not an integer."""
        response = client.get("/admin/service/logs?lines=abc", headers=auth_headers)
        assert response.status_code in [400, 422]

    # ========================================================================
    # Successful Response Tests (Schema Validation)
    # ========================================================================

    def test_get_service_logs_response_structure(
        self, client, auth_headers, mock_service_manager
    ):
        """Test logs response has correct structure."""
        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.get("/admin/service/logs", headers=auth_headers)

            assert response.status_code == 200
            data = response.json()

            # Validate required fields
            assert "logs" in data
            assert "total" in data
            assert "lines_requested" in data

            # Validate types
            assert isinstance(data["logs"], list)
            assert isinstance(data["total"], int)
            assert isinstance(data["lines_requested"], int)

    def test_get_service_logs_log_entry_schema(
        self, client, auth_headers, mock_service_manager
    ):
        """Test log entry objects match LogEntry schema."""
        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.get("/admin/service/logs", headers=auth_headers)

            assert response.status_code == 200
            data = response.json()
            logs = data["logs"]

            for log_entry in logs:
                # Validate required fields
                assert "timestamp" in log_entry
                assert "severity" in log_entry
                assert "message" in log_entry

                # Validate types
                assert isinstance(log_entry["timestamp"], str)
                assert isinstance(log_entry["severity"], str)
                assert isinstance(log_entry["message"], str)

                # Validate severity enum
                assert log_entry["severity"] in [
                    "DEBUG",
                    "INFO",
                    "WARNING",
                    "ERROR",
                    "CRITICAL",
                ]

                # Optional field
                if "source" in log_entry:
                    assert log_entry["source"] is None or isinstance(
                        log_entry["source"], str
                    )

    def test_get_service_logs_severity_levels(
        self, client, auth_headers, mock_service_manager
    ):
        """Test logs can contain all severity levels."""
        severities = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        now = datetime.utcnow()

        mock_service_manager.get_logs = AsyncMock(
            return_value={
                "logs": [
                    {
                        "timestamp": (now - timedelta(seconds=i)).isoformat() + "Z",
                        "severity": severity,
                        "message": f"Message with {severity} severity",
                        "source": "test",
                    }
                    for i, severity in enumerate(severities)
                ],
                "total": 5,
                "lines_requested": 5,
            }
        )

        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.get("/admin/service/logs", headers=auth_headers)

            assert response.status_code == 200
            data = response.json()
            returned_severities = [log["severity"] for log in data["logs"]]
            assert returned_severities == severities

    def test_get_service_logs_with_source(
        self, client, auth_headers, mock_service_manager
    ):
        """Test log entries with source component."""
        mock_service_manager.get_logs = AsyncMock(
            return_value={
                "logs": [
                    {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "severity": "INFO",
                        "message": "Message from service",
                        "source": "service_manager",
                    }
                ],
                "total": 1,
                "lines_requested": 100,
            }
        )

        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.get("/admin/service/logs", headers=auth_headers)

            assert response.status_code == 200
            data = response.json()
            assert data["logs"][0]["source"] == "service_manager"

    def test_get_service_logs_without_source(
        self, client, auth_headers, mock_service_manager
    ):
        """Test log entries with null source."""
        mock_service_manager.get_logs = AsyncMock(
            return_value={
                "logs": [
                    {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "severity": "WARNING",
                        "message": "Message without source",
                        "source": None,
                    }
                ],
                "total": 1,
                "lines_requested": 100,
            }
        )

        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.get("/admin/service/logs", headers=auth_headers)

            assert response.status_code == 200
            data = response.json()
            assert data["logs"][0]["source"] is None

    def test_get_service_logs_empty_logs(
        self, client, auth_headers, mock_service_manager
    ):
        """Test response when no logs available."""
        mock_service_manager.get_logs = AsyncMock(
            return_value={"logs": [], "total": 0, "lines_requested": 100}
        )

        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.get("/admin/service/logs", headers=auth_headers)

            assert response.status_code == 200
            data = response.json()
            assert data["logs"] == []
            assert data["total"] == 0

    # ========================================================================
    # Timestamp Tests
    # ========================================================================

    def test_get_service_logs_timestamp_format(
        self, client, auth_headers, mock_service_manager
    ):
        """Test log timestamps are ISO 8601 format."""
        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.get("/admin/service/logs", headers=auth_headers)

            assert response.status_code == 200
            data = response.json()

            for log_entry in data["logs"]:
                timestamp = log_entry["timestamp"]
                assert isinstance(timestamp, str)
                assert "T" in timestamp or "t" in timestamp  # ISO 8601 has T

    # ========================================================================
    # Error Handling Tests
    # ========================================================================

    def test_get_service_logs_timeout(self, client, auth_headers, mock_service_manager):
        """Test 504 response when log retrieval times out."""
        import subprocess

        mock_service_manager.get_logs = AsyncMock(
            side_effect=subprocess.TimeoutExpired("journalctl", 30)
        )

        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.get("/admin/service/logs", headers=auth_headers)

            assert response.status_code == 504
            data = response.json()
            assert "detail" in data

    def test_get_service_logs_internal_error(
        self, client, auth_headers, mock_service_manager
    ):
        """Test 500 response on ServiceManager error."""
        mock_service_manager.get_logs = AsyncMock(
            side_effect=Exception("Log file not accessible")
        )

        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.get("/admin/service/logs", headers=auth_headers)

            assert response.status_code == 500
            data = response.json()
            assert "detail" in data

    # ========================================================================
    # Content-Type Tests
    # ========================================================================

    def test_get_service_logs_returns_json(
        self, client, auth_headers, mock_service_manager
    ):
        """Test response has correct content type."""
        with patch(
            "src.admin_ui.get_service_manager", return_value=mock_service_manager
        ):
            response = client.get("/admin/service/logs", headers=auth_headers)

            assert response.status_code == 200
            assert response.headers["content-type"] == "application/json"
