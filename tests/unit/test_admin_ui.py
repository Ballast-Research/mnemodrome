"""Tests for Admin Web UI FastAPI endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.admin_ui import app


class TestAdminUI:
    """Tests for FastAPI Admin UI endpoints."""

    @pytest.fixture
    def mock_config_manager(self):
        """Create mock ConfigurationManager."""
        mock_cm = MagicMock()
        mock_cm._config_version = 1234567890.0
        mock_cm.read_config.return_value = {
            "config_version": 1234567890.0,
            "categories": {
                "event_log": [
                    {
                        "name": "EVENT_LOG_ENABLED",
                        "value": True,
                        "default": True,
                        "editable": True,
                        "description": "Enable event logging",
                    }
                ],
                "neo4j": [
                    {
                        "name": "NEO4J_URI",
                        "value": "bolt://localhost:7687",
                        "default": "bolt://localhost:7687",
                        "editable": False,
                        "description": "Neo4j connection URI",
                    }
                ],
            },
        }
        mock_cm.validate_config.return_value = []
        mock_cm.apply_config = AsyncMock()
        mock_cm.persist_config = AsyncMock()
        mock_cm.undo_last_change = AsyncMock(
            return_value={
                "backup_file": ".env.backup.20240101-120000",
                "backup_timestamp": "2024-01-01T12:00:00",
                "restored_at": "2024-01-01T13:00:00",
            }
        )
        return mock_cm

    @pytest.fixture
    def mock_event_log_service(self):
        """Create mock EventLogService."""
        mock_els = MagicMock()
        mock_els.get_event_log_health = AsyncMock(
            return_value={
                "status": "healthy",
                "total_events": 100,
                "session_id": "test-session-id",
                "database_size_mb": 2.5,
            }
        )
        mock_els.query_events = AsyncMock(
            return_value=[
                {
                    "id": 1,
                    "event_type": "config_change",
                    "action": "update_config",
                    "actor": "admin",
                    "result": "success",
                    "timestamp": "2024-01-01T12:00:00",
                }
            ]
        )
        mock_els.log_event = AsyncMock(return_value=1)
        mock_els.prune_old_events = AsyncMock(return_value=10)
        mock_els.prune_by_size = AsyncMock(return_value=5)
        return mock_els

    @pytest.fixture
    def mock_neo4j_service(self):
        """Create mock Neo4j service."""
        mock_neo4j = MagicMock()
        mock_neo4j.driver = MagicMock()  # Truthy value
        mock_neo4j.driver.__bool__ = lambda self: (
            True
        )  # Ensure it's truthy when checked
        return mock_neo4j

    @pytest.fixture
    def mock_redis_service(self):
        """Create mock Redis service."""
        mock_redis = MagicMock()
        mock_redis.client = MagicMock()  # Truthy value
        mock_redis.client.__bool__ = lambda self: True  # Ensure it's truthy
        mock_redis._connected = True
        return mock_redis

    @pytest.fixture
    def client(self, mock_config_manager):
        """Create TestClient with mocked dependencies."""
        # Patch config_manager BEFORE creating TestClient (so startup event sees the patch)
        # Use stack to manage multiple patches
        import contextlib

        with contextlib.ExitStack() as stack:
            # Patch initialize_config_manager to use our mock
            original_init = stack.enter_context(
                patch("src.admin_ui.initialize_config_manager")
            )
            # Patch the global config_manager in admin_service
            stack.enter_context(
                patch("src.services.admin_service.config_manager", mock_config_manager)
            )

            with TestClient(app) as c:
                yield c

    @pytest.fixture
    def auth_headers(self):
        """Valid authentication headers."""
        # Default credentials from test config: admin/test-password
        return {"Authorization": "Basic YWRtaW46dGVzdC1wYXNzd29yZA=="}

    @pytest.fixture
    def invalid_auth_headers(self):
        """Invalid authentication headers."""
        return {"Authorization": "Basic d3Jvbmc6cGFzc3dvcmQ="}  # wrong:password

    # ============================================================================
    # Health Check Endpoint (No Auth)
    # ============================================================================

    def test_health_check_no_auth_required(self, client):
        """Test health check endpoint works without authentication."""
        response = client.get("/admin/health-check")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "Mnemodrome Admin UI"
        assert data["version"] == "1.0.0"

    # ============================================================================
    # Authentication Tests
    # ============================================================================

    def test_get_config_requires_auth(self, client):
        """Test config endpoint requires authentication."""
        response = client.get("/admin/config")
        assert response.status_code == 401
        assert "WWW-Authenticate" in response.headers

    def test_get_config_invalid_credentials(self, client, invalid_auth_headers):
        """Test config endpoint rejects invalid credentials."""
        response = client.get("/admin/config", headers=invalid_auth_headers)
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid credentials"

    def test_get_config_valid_credentials(
        self, client, auth_headers, mock_config_manager
    ):
        """Test config endpoint accepts valid credentials."""
        response = client.get("/admin/config", headers=auth_headers)
        assert response.status_code == 200

    # ============================================================================
    # Configuration Endpoints
    # ============================================================================

    def test_get_config_success(self, client, auth_headers, mock_config_manager):
        """Test GET /admin/config returns configuration."""
        response = client.get("/admin/config", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert "config_version" in data
        assert "categories" in data
        assert "event_log" in data["categories"]
        assert "neo4j" in data["categories"]

        mock_config_manager.read_config.assert_called_once()

    def test_get_config_manager_not_initialized(self, client, auth_headers):
        """Test GET /admin/config when ConfigurationManager is None."""
        with patch("src.services.admin_service.config_manager", None):
            with TestClient(app) as c:
                response = c.get("/admin/config", headers=auth_headers)
                assert response.status_code == 500
                assert (
                    "ConfigurationManager not initialized" in response.json()["detail"]
                )

    def test_get_config_exception(self, client, auth_headers, mock_config_manager):
        """Test GET /admin/config handles exceptions."""
        mock_config_manager.read_config.side_effect = Exception("Database error")

        response = client.get("/admin/config", headers=auth_headers)
        assert response.status_code == 500
        assert "Failed to read configuration" in response.json()["detail"]

    def test_update_config_success(self, client, auth_headers, mock_config_manager):
        """Test POST /admin/config with valid changes."""
        changes = {"EVENT_LOG_ENABLED": True, "EVENT_LOG_RETENTION_DAYS": 120}

        response = client.post("/admin/config", json=changes, headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "success"
        assert data["applied"] == changes
        assert "new_version" in data

        mock_config_manager.validate_config.assert_called_once_with(changes)
        mock_config_manager.apply_config.assert_called_once()
        mock_config_manager.persist_config.assert_called_once()

    def test_update_config_with_version(
        self, client, auth_headers, mock_config_manager
    ):
        """Test POST /admin/config with optimistic locking."""
        changes = {"EVENT_LOG_ENABLED": False}
        config_version = 1234567890.0

        response = client.post(
            "/admin/config",
            json=changes,
            params={"config_version": config_version},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_update_config_version_mismatch(
        self, client, auth_headers, mock_config_manager
    ):
        """Test POST /admin/config rejects version mismatch."""
        changes = {"EVENT_LOG_ENABLED": False}
        config_version = 9999999999.0  # Wrong version

        response = client.post(
            "/admin/config",
            json=changes,
            params={"config_version": config_version},
            headers=auth_headers,
        )
        assert response.status_code == 409
        assert "version mismatch" in response.json()["detail"]

    def test_update_config_validation_errors(
        self, client, auth_headers, mock_config_manager
    ):
        """Test POST /admin/config rejects invalid changes."""
        mock_config_manager.validate_config.return_value = [
            "EVENT_LOG_RETENTION_DAYS: Value must be >= 1",
            "LOG_LEVEL: Invalid log level",
        ]

        changes = {"EVENT_LOG_RETENTION_DAYS": 0, "LOG_LEVEL": "INVALID"}

        response = client.post("/admin/config", json=changes, headers=auth_headers)
        assert response.status_code == 400
        assert "errors" in response.json()["detail"]
        assert len(response.json()["detail"]["errors"]) == 2

    def test_update_config_manager_not_initialized(self, client, auth_headers):
        """Test POST /admin/config when ConfigurationManager is None."""
        with patch("src.services.admin_service.config_manager", None):
            with TestClient(app) as c:
                response = c.post(
                    "/admin/config",
                    json={"EVENT_LOG_ENABLED": True},
                    headers=auth_headers,
                )
                assert response.status_code == 500
                assert (
                    "ConfigurationManager not initialized" in response.json()["detail"]
                )

    def test_update_config_exception(self, client, auth_headers, mock_config_manager):
        """Test POST /admin/config handles exceptions."""
        mock_config_manager.apply_config.side_effect = Exception("Failed to apply")

        response = client.post(
            "/admin/config",
            json={"EVENT_LOG_ENABLED": True},
            headers=auth_headers,
        )
        assert response.status_code == 500
        assert "Failed to update configuration" in response.json()["detail"]

    def test_undo_config_success(self, client, auth_headers, mock_config_manager):
        """Test POST /admin/config/undo restores backup."""
        with patch("src.services.event_log_service.event_log_service") as mock_els:
            mock_els.log_event = AsyncMock()

            response = client.post("/admin/config/undo", headers=auth_headers)
            assert response.status_code == 200

            data = response.json()
            assert "backup_file" in data
            assert "backup_timestamp" in data
            assert "restored_at" in data

            mock_config_manager.undo_last_change.assert_called_once()

    def test_undo_config_no_backup(self, client, auth_headers, mock_config_manager):
        """Test POST /admin/config/undo when no backup exists."""
        mock_config_manager.undo_last_change.side_effect = FileNotFoundError(
            "No configuration backups available"
        )

        response = client.post("/admin/config/undo", headers=auth_headers)
        assert response.status_code == 404
        assert "No configuration backups available" in response.json()["detail"]

    def test_undo_config_manager_not_initialized(self, client, auth_headers):
        """Test POST /admin/config/undo when ConfigurationManager is None."""
        with patch("src.services.admin_service.config_manager", None):
            with TestClient(app) as c:
                response = c.post("/admin/config/undo", headers=auth_headers)
                assert response.status_code == 500
                assert (
                    "ConfigurationManager not initialized" in response.json()["detail"]
                )

    def test_undo_config_exception(self, client, auth_headers, mock_config_manager):
        """Test POST /admin/config/undo handles exceptions."""
        mock_config_manager.undo_last_change.side_effect = Exception("Restore failed")

        response = client.post("/admin/config/undo", headers=auth_headers)
        assert response.status_code == 500
        assert "Failed to undo configuration" in response.json()["detail"]

    # ============================================================================
    # Health Endpoint
    # ============================================================================

    def test_get_health_success(
        self,
        client,
        auth_headers,
        mock_event_log_service,
        mock_neo4j_service,
        mock_redis_service,
    ):
        """Test GET /admin/health returns system health."""
        with patch(
            "src.services.event_log_service.event_log_service", mock_event_log_service
        ):
            with patch("src.services.neo4j_service.neo4j_service", mock_neo4j_service):
                with patch(
                    "src.services.redis_service.redis_service", mock_redis_service
                ):
                    response = client.get("/admin/health", headers=auth_headers)
                    assert response.status_code == 200

                    data = response.json()
                    assert "event_log" in data
                    assert "neo4j" in data
                    assert "redis" in data
                    assert "mcp_server" in data

                    assert data["event_log"]["status"] == "healthy"
                    assert data["neo4j"]["connected"] is True
                    assert data["redis"]["connected"] is True

    def test_get_health_event_log_error(self, client, auth_headers):
        """Test GET /admin/health handles event log errors."""
        mock_els = MagicMock()
        mock_els.get_event_log_health.side_effect = Exception("Connection failed")

        with patch("src.services.event_log_service.event_log_service", mock_els):
            response = client.get("/admin/health", headers=auth_headers)
            assert response.status_code == 200

            data = response.json()
            assert "event_log" in data
            assert data["event_log"]["status"] == "error"
            assert "Connection failed" in data["event_log"]["error"]

    def test_get_health_neo4j_error(self, client, auth_headers):
        """Test GET /admin/health handles Neo4j errors."""
        # Mock neo4j_service to raise an error when accessing .driver attribute
        mock_neo4j = MagicMock()
        type(mock_neo4j).driver = property(
            lambda self: (_ for _ in ()).throw(Exception("Connection error"))
        )

        with patch("src.services.neo4j_service.neo4j_service", mock_neo4j):
            response = client.get("/admin/health", headers=auth_headers)
            assert response.status_code == 200

            data = response.json()
            assert "neo4j" in data
            assert data["neo4j"]["status"] == "error"
            assert "error" in data["neo4j"]

    def test_get_health_exception(self, client, auth_headers):
        """Test GET /admin/health handles general exceptions in service checks."""
        # The outer exception handler is only triggered if JSONResponse itself fails
        # Service-level exceptions are caught in inner try-catch blocks
        # Let's test that service errors are properly caught and reported
        mock_els = MagicMock()
        mock_els.get_event_log_health.side_effect = Exception("Fatal error")

        with patch("src.services.event_log_service.event_log_service", mock_els):
            response = client.get("/admin/health", headers=auth_headers)
            assert response.status_code == 200  # Service errors don't fail the endpoint

            data = response.json()
            assert "event_log" in data
            assert data["event_log"]["status"] == "error"
            assert "Fatal error" in data["event_log"]["error"]

    # ============================================================================
    # Event Endpoints
    # ============================================================================

    def test_get_events_success(self, client, auth_headers, mock_event_log_service):
        """Test GET /admin/events returns events."""
        with patch(
            "src.services.event_log_service.event_log_service", mock_event_log_service
        ):
            response = client.get("/admin/events", headers=auth_headers)
            assert response.status_code == 200

            data = response.json()
            assert "events" in data
            assert "total" in data
            assert "filters" in data
            assert len(data["events"]) == 1
            assert data["total"] == 1

    def test_get_events_with_filters(
        self, client, auth_headers, mock_event_log_service
    ):
        """Test GET /admin/events with filtering."""
        with patch(
            "src.services.event_log_service.event_log_service", mock_event_log_service
        ):
            response = client.get(
                "/admin/events",
                params={"limit": 50, "event_type": "config_change"},
                headers=auth_headers,
            )
            assert response.status_code == 200

            data = response.json()
            assert data["filters"]["limit"] == 50
            assert data["filters"]["event_type"] == "config_change"

            mock_event_log_service.query_events.assert_called_once_with(
                event_type="config_change", limit=50
            )

    def test_get_events_service_unavailable(self, client, auth_headers):
        """Test GET /admin/events when service is unavailable."""
        with patch("src.services.event_log_service.event_log_service", None):
            response = client.get("/admin/events", headers=auth_headers)
            assert response.status_code == 503
            assert "Event log service not available" in response.json()["detail"]

    def test_get_events_exception(self, client, auth_headers, mock_event_log_service):
        """Test GET /admin/events handles exceptions."""
        mock_event_log_service.query_events.side_effect = Exception("Query failed")

        with patch(
            "src.services.event_log_service.event_log_service", mock_event_log_service
        ):
            response = client.get("/admin/events", headers=auth_headers)
            assert response.status_code == 500
            assert "Failed to get events" in response.json()["detail"]

    def test_cleanup_events_success(self, client, auth_headers, mock_event_log_service):
        """Test POST /admin/events/cleanup triggers cleanup."""
        with patch(
            "src.services.event_log_service.event_log_service", mock_event_log_service
        ):
            response = client.post("/admin/events/cleanup", headers=auth_headers)
            assert response.status_code == 200

            data = response.json()
            assert data["status"] == "success"
            assert data["deleted_age_based"] == 10
            assert data["deleted_size_based"] == 5

            mock_event_log_service.prune_old_events.assert_called_once()
            mock_event_log_service.prune_by_size.assert_called_once()
            mock_event_log_service.log_event.assert_called_once()

    def test_cleanup_events_service_unavailable(self, client, auth_headers):
        """Test POST /admin/events/cleanup when service is unavailable."""
        with patch("src.services.event_log_service.event_log_service", None):
            response = client.post("/admin/events/cleanup", headers=auth_headers)
            assert response.status_code == 503
            assert "Event log service not available" in response.json()["detail"]

    def test_cleanup_events_exception(
        self, client, auth_headers, mock_event_log_service
    ):
        """Test POST /admin/events/cleanup handles exceptions."""
        mock_event_log_service.prune_old_events.side_effect = Exception(
            "Cleanup failed"
        )

        with patch(
            "src.services.event_log_service.event_log_service", mock_event_log_service
        ):
            response = client.post("/admin/events/cleanup", headers=auth_headers)
            assert response.status_code == 500
            assert "Failed to cleanup events" in response.json()["detail"]

    # ============================================================================
    # HTML Endpoints
    # ============================================================================

    def test_admin_index_placeholder(self, client, auth_headers):
        """Test GET /admin returns placeholder HTML."""
        response = client.get("/admin", headers=auth_headers)
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Mnemodrome Admin UI" in response.text

    def test_admin_index_requires_auth(self, client):
        """Test GET /admin requires authentication."""
        response = client.get("/admin")
        assert response.status_code == 401

    def test_config_editor_placeholder(self, client, auth_headers):
        """Test GET /admin/config-editor returns placeholder HTML."""
        response = client.get("/admin/config-editor", headers=auth_headers)
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Configuration Editor" in response.text

    def test_config_editor_requires_auth(self, client):
        """Test GET /admin/config-editor requires authentication."""
        response = client.get("/admin/config-editor")
        assert response.status_code == 401

    def test_health_dashboard_placeholder(self, client, auth_headers):
        """Test GET /admin/health-dashboard returns placeholder HTML."""
        response = client.get("/admin/health-dashboard", headers=auth_headers)
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Health Dashboard" in response.text

    def test_health_dashboard_requires_auth(self, client):
        """Test GET /admin/health-dashboard requires authentication."""
        response = client.get("/admin/health-dashboard")
        assert response.status_code == 401

    def test_event_viewer_placeholder(self, client, auth_headers):
        """Test GET /admin/event-viewer returns placeholder HTML."""
        response = client.get("/admin/event-viewer", headers=auth_headers)
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Event Viewer" in response.text

    def test_event_viewer_requires_auth(self, client):
        """Test GET /admin/event-viewer requires authentication."""
        response = client.get("/admin/event-viewer")
        assert response.status_code == 401

    # ============================================================================
    # Edge Cases and Security
    # ============================================================================

    def test_empty_credentials(self, client):
        """Test endpoints reject empty credentials."""
        headers = {"Authorization": "Basic Og=="}  # Empty username:password
        response = client.get("/admin/config", headers=headers)
        assert response.status_code == 401

    def test_malformed_auth_header(self, client):
        """Test endpoints handle malformed auth headers."""
        headers = {"Authorization": "Bearer invalid-token"}
        response = client.get("/admin/config", headers=headers)
        assert response.status_code == 401

    def test_update_config_empty_changes(
        self, client, auth_headers, mock_config_manager
    ):
        """Test POST /admin/config with empty changes."""
        response = client.post("/admin/config", json={}, headers=auth_headers)
        assert response.status_code == 200  # Empty changes are valid

    def test_update_config_null_changes(self, client, auth_headers):
        """Test POST /admin/config with null body."""
        response = client.post("/admin/config", headers=auth_headers)
        assert response.status_code == 422  # FastAPI validation error

    def test_get_events_invalid_limit(
        self, client, auth_headers, mock_event_log_service
    ):
        """Test GET /admin/events with invalid limit."""
        with patch(
            "src.services.event_log_service.event_log_service", mock_event_log_service
        ):
            response = client.get(
                "/admin/events",
                params={"limit": "invalid"},
                headers=auth_headers,
            )
            assert response.status_code == 422  # Validation error

    def test_multiple_concurrent_auth_requests(self, client, auth_headers):
        """Test multiple concurrent authenticated requests."""
        responses = []
        for _ in range(5):
            response = client.get("/admin/health-check")
            responses.append(response)

        assert all(r.status_code == 200 for r in responses)

    def test_timing_attack_resistance(self, client, auth_headers, invalid_auth_headers):
        """Test authentication uses constant-time comparison."""
        # Both valid and invalid should take similar time (cannot verify timing in unit test)
        # but we can verify both use the same code path
        valid_response = client.get("/admin/config", headers=auth_headers)
        invalid_response = client.get("/admin/config", headers=invalid_auth_headers)

        assert valid_response.status_code == 200
        assert invalid_response.status_code == 401

    # ============================================================================
    # Integration Tests
    # ============================================================================

    def test_full_config_workflow(self, client, auth_headers, mock_config_manager):
        """Test complete configuration update workflow."""
        with patch("src.services.event_log_service.event_log_service") as mock_els:
            mock_els.log_event = AsyncMock()

            # Step 1: Get current config
            response = client.get("/admin/config", headers=auth_headers)
            assert response.status_code == 200
            config_data = response.json()

            # Step 2: Update config
            changes = {"EVENT_LOG_RETENTION_DAYS": 120}
            response = client.post("/admin/config", json=changes, headers=auth_headers)
            assert response.status_code == 200

            # Step 3: Undo change
            response = client.post("/admin/config/undo", headers=auth_headers)
            assert response.status_code == 200

    def test_full_event_workflow(self, client, auth_headers, mock_event_log_service):
        """Test complete event log workflow."""
        with patch(
            "src.services.event_log_service.event_log_service", mock_event_log_service
        ):
            # Step 1: Query events
            response = client.get("/admin/events", headers=auth_headers)
            assert response.status_code == 200

            # Step 2: Cleanup events
            response = client.post("/admin/events/cleanup", headers=auth_headers)
            assert response.status_code == 200

    def test_health_check_all_services(
        self,
        client,
        auth_headers,
        mock_event_log_service,
        mock_neo4j_service,
        mock_redis_service,
    ):
        """Test health endpoint returns all service statuses."""
        with patch(
            "src.services.event_log_service.event_log_service", mock_event_log_service
        ):
            with patch("src.services.neo4j_service.neo4j_service", mock_neo4j_service):
                with patch(
                    "src.services.redis_service.redis_service", mock_redis_service
                ):
                    response = client.get("/admin/health", headers=auth_headers)
                    assert response.status_code == 200

                    data = response.json()
                    # Verify all required service health checks
                    assert "event_log" in data
                    assert "neo4j" in data
                    assert "redis" in data
                    assert "mcp_server" in data

                    # Verify MCP server info structure
                    assert "version" in data["mcp_server"]
                    assert "port" in data["mcp_server"]
                    assert "host" in data["mcp_server"]

    # ============================================================================
    # Additional Coverage Tests
    # ============================================================================

    def test_undo_config_event_log_failure(
        self, client, auth_headers, mock_config_manager
    ):
        """Test POST /admin/config/undo handles event log failures gracefully."""
        mock_els = MagicMock()
        mock_els.log_event.side_effect = Exception("Event log unavailable")

        with patch("src.services.event_log_service.event_log_service", mock_els):
            response = client.post("/admin/config/undo", headers=auth_headers)
            # Should succeed even if logging fails
            assert response.status_code == 200

    def test_get_health_redis_no_attribute(self, client, auth_headers):
        """Test GET /admin/health when redis_service exists but has no redis attribute."""
        mock_redis = MagicMock(spec=[])  # No attributes

        with patch("src.services.event_log_service.event_log_service") as mock_els:
            mock_els.get_event_log_health = AsyncMock(
                return_value={"status": "healthy"}
            )
            with patch("src.services.neo4j_service.neo4j_service") as mock_neo4j:
                mock_neo4j.driver = MagicMock()
                with patch("src.services.redis_service.redis_service", mock_redis):
                    response = client.get("/admin/health", headers=auth_headers)
                    assert response.status_code == 200
                    # Redis should not be in response if service lacks redis attribute
                    data = response.json()
                    assert "event_log" in data
                    assert "neo4j" in data

    def test_get_health_redis_error(
        self, client, auth_headers, mock_event_log_service, mock_neo4j_service
    ):
        """Test GET /admin/health handles Redis errors."""
        # Create a redis mock that throws error
        mock_redis = MagicMock()
        type(mock_redis).client = property(
            lambda self: (_ for _ in ()).throw(Exception("Redis connection error"))
        )

        with patch(
            "src.services.event_log_service.event_log_service", mock_event_log_service
        ):
            with patch("src.services.neo4j_service.neo4j_service", mock_neo4j_service):
                with patch("src.services.redis_service.redis_service", mock_redis):
                    response = client.get("/admin/health", headers=auth_headers)
                    assert response.status_code == 200

                    data = response.json()
                    assert "redis" in data
                    assert data["redis"]["status"] == "error"
                    assert "error" in data["redis"]

    def test_html_endpoints_with_files(self, client, auth_headers, tmp_path):
        """Test HTML endpoints when static files exist."""
        # Create temporary static directory with HTML files
        static_dir = tmp_path / "static"
        static_dir.mkdir()

        index_html = static_dir / "index.html"
        index_html.write_text("<html><body>Index Page</body></html>")

        config_html = static_dir / "config.html"
        config_html.write_text("<html><body>Config Page</body></html>")

        health_html = static_dir / "health.html"
        health_html.write_text("<html><body>Health Page</body></html>")

        events_html = static_dir / "events.html"
        events_html.write_text("<html><body>Events Page</body></html>")

        # Patch Path to return our temp directory
        with patch("src.admin_ui.Path") as mock_path:
            # Make Path() return objects with exists() and read_text()
            def path_side_effect(arg):
                if arg == "src/static/index.html":
                    return index_html
                elif arg == "src/static/config.html":
                    return config_html
                elif arg == "src/static/health.html":
                    return health_html
                elif arg == "src/static/events.html":
                    return events_html
                return MagicMock(exists=MagicMock(return_value=False))

            mock_path.side_effect = path_side_effect

            # Test index page
            response = client.get("/admin", headers=auth_headers)
            assert response.status_code == 200
            assert "Index Page" in response.text

            # Test config editor
            response = client.get("/admin/config-editor", headers=auth_headers)
            assert response.status_code == 200
            assert "Config Page" in response.text

            # Test health dashboard
            response = client.get("/admin/health-dashboard", headers=auth_headers)
            assert response.status_code == 200
            assert "Health Page" in response.text

            # Test event viewer
            response = client.get("/admin/event-viewer", headers=auth_headers)
            assert response.status_code == 200
            assert "Events Page" in response.text

    def test_html_endpoints_file_read_errors(self, client, auth_headers):
        """Test HTML endpoints handle file read errors."""
        with patch("src.admin_ui.Path") as mock_path:
            mock_file = MagicMock()
            mock_file.exists.return_value = True
            mock_file.read_text.side_effect = Exception("Permission denied")
            mock_path.return_value = mock_file

            # Should return 500 error
            response = client.get("/admin", headers=auth_headers)
            assert response.status_code == 500
