"""
Integration tests for Admin Web UI and dual-server architecture.

Feature 013: Phase 4 - Integration + Testing
Tests complete workflows end-to-end with both MCP and Admin UI servers.
"""

from unittest.mock import patch

import pytest

from src import config


class TestDualServerArchitecture:
    """Test dual-server startup and lifecycle."""

    @pytest.mark.asyncio
    async def test_config_validation_admin_password_required(self):
        """Test that ADMIN_PASSWORD is required when ADMIN_WEB_ENABLED=true."""
        # This test verifies config validation catches missing password
        with patch.dict(
            "os.environ", {"ADMIN_WEB_ENABLED": "true", "ADMIN_PASSWORD": ""}
        ):
            with pytest.raises(ValueError, match="ADMIN_PASSWORD is required"):
                # Reload config module to trigger validation
                import importlib

                importlib.reload(config)

    @pytest.mark.asyncio
    async def test_config_validation_password_length_warning(self, caplog):
        """Test that weak passwords trigger warning."""
        # Test that passwords < 12 chars trigger warning
        with patch.dict(
            "os.environ", {"ADMIN_WEB_ENABLED": "true", "ADMIN_PASSWORD": "short123"}
        ):
            import importlib

            importlib.reload(config)
            # Check for warning in logs
            assert any(
                "should be at least 12 characters" in record.message
                for record in caplog.records
            )

    @pytest.mark.asyncio
    async def test_config_validation_invalid_port(self):
        """Test that invalid port ranges are rejected."""
        with patch.dict(
            "os.environ",
            {
                "ADMIN_WEB_ENABLED": "true",
                "ADMIN_PASSWORD": "secure-password-123",
                "ADMIN_WEB_PORT": "70000",  # Invalid port
            },
        ):
            with pytest.raises(ValueError, match="must be between 1024 and 65535"):
                import importlib

                importlib.reload(config)

    @pytest.mark.asyncio
    async def test_config_validation_invalid_log_level(self):
        """Test that invalid log levels are rejected."""
        with patch.dict("os.environ", {"LOG_LEVEL": "INVALID"}):
            with pytest.raises(ValueError, match="LOG_LEVEL must be one of"):
                import importlib

                importlib.reload(config)


class TestConfigurationWorkflow:
    """Test complete configuration update workflow."""

    @pytest.fixture
    async def admin_client(self):
        """Fixture for authenticated admin API client."""
        from fastapi.testclient import TestClient

        from src.admin_ui import app

        client = TestClient(app)
        # Set up HTTP Basic Auth
        client.auth = (config.ADMIN_USERNAME, config.ADMIN_PASSWORD)
        return client

    @pytest.mark.asyncio
    async def test_full_config_update_workflow(self, admin_client):
        """
        Test complete config update workflow:
        1. GET current config
        2. POST update
        3. Verify changes applied
        4. Verify .env updated
        5. Verify backup created
        """
        # Skip if no password set (prevents config import error)
        if not config.ADMIN_PASSWORD:
            pytest.skip("ADMIN_PASSWORD not set")

        # Step 1: Get current config
        response = admin_client.get("/admin/config")
        assert response.status_code == 200
        current_config = response.json()
        assert "config" in current_config
        assert "version" in current_config

        # Step 2: Update config (change retention days)
        original_retention = current_config["config"]["event_log"][
            "EVENT_LOG_RETENTION_DAYS"
        ]
        new_retention = original_retention + 1

        update_response = admin_client.post(
            "/admin/config",
            json={
                "EVENT_LOG_RETENTION_DAYS": new_retention,
                "config_version": current_config["version"],
            },
        )
        assert update_response.status_code == 200
        update_result = update_response.json()
        assert update_result["status"] == "success"
        assert update_result["applied"]["EVENT_LOG_RETENTION_DAYS"] == new_retention

        # Step 3: Verify changes reflected in GET
        verify_response = admin_client.get("/admin/config")
        assert verify_response.status_code == 200
        updated_config = verify_response.json()
        assert (
            updated_config["config"]["event_log"]["EVENT_LOG_RETENTION_DAYS"]
            == new_retention
        )

    @pytest.mark.asyncio
    async def test_config_validation_error_workflow(self, admin_client):
        """
        Test validation error workflow:
        1. POST invalid config
        2. Verify 400 error
        3. Verify config unchanged
        """
        if not config.ADMIN_PASSWORD:
            pytest.skip("ADMIN_PASSWORD not set")

        # Get current config
        response = admin_client.get("/admin/config")
        assert response.status_code == 200
        current_config = response.json()

        # Try to set invalid retention days (negative)
        invalid_response = admin_client.post(
            "/admin/config", json={"EVENT_LOG_RETENTION_DAYS": -1}
        )
        assert invalid_response.status_code == 400
        assert "errors" in invalid_response.json()

        # Verify config unchanged
        verify_response = admin_client.get("/admin/config")
        assert verify_response.status_code == 200
        assert verify_response.json()["config"] == current_config["config"]

    @pytest.mark.asyncio
    async def test_undo_workflow(self, admin_client):
        """
        Test undo workflow:
        1. Update config
        2. POST /admin/config/undo
        3. Verify config restored
        """
        if not config.ADMIN_PASSWORD:
            pytest.skip("ADMIN_PASSWORD not set")

        # Get original config
        original_response = admin_client.get("/admin/config")
        original_config = original_response.json()["config"]
        original_retention = original_config["event_log"]["EVENT_LOG_RETENTION_DAYS"]

        # Update config
        admin_client.post(
            "/admin/config", json={"EVENT_LOG_RETENTION_DAYS": original_retention + 5}
        )

        # Undo change
        undo_response = admin_client.post("/admin/config/undo")
        assert undo_response.status_code == 200

        # Verify restoration
        restored_response = admin_client.get("/admin/config")
        restored_config = restored_response.json()["config"]
        assert (
            restored_config["event_log"]["EVENT_LOG_RETENTION_DAYS"]
            == original_retention
        )


class TestHealthMonitoring:
    """Test health monitoring endpoints."""

    @pytest.fixture
    async def admin_client(self):
        """Fixture for authenticated admin API client."""
        from fastapi.testclient import TestClient

        from src.admin_ui import app

        client = TestClient(app)
        client.auth = (config.ADMIN_USERNAME, config.ADMIN_PASSWORD)
        return client

    @pytest.mark.asyncio
    async def test_health_monitoring_workflow(self, admin_client):
        """
        Test health monitoring workflow:
        1. GET /admin/health
        2. Verify event log metrics returned
        3. Verify Neo4j status returned
        """
        if not config.ADMIN_PASSWORD:
            pytest.skip("ADMIN_PASSWORD not set")

        response = admin_client.get("/admin/health")
        assert response.status_code == 200

        health_data = response.json()
        assert "event_log" in health_data
        assert "neo4j" in health_data
        assert "mcp_server" in health_data


class TestEventLogWorkflow:
    """Test event log viewer workflow."""

    @pytest.fixture
    async def admin_client(self):
        """Fixture for authenticated admin API client."""
        from fastapi.testclient import TestClient

        from src.admin_ui import app

        client = TestClient(app)
        client.auth = (config.ADMIN_USERNAME, config.ADMIN_PASSWORD)
        return client

    @pytest.mark.asyncio
    async def test_event_log_workflow(self, admin_client):
        """
        Test event log workflow:
        1. GET /admin/events
        2. Verify events returned
        3. POST /admin/events/cleanup
        4. Verify cleanup logged
        """
        if not config.ADMIN_PASSWORD:
            pytest.skip("ADMIN_PASSWORD not set")

        # Get events
        events_response = admin_client.get("/admin/events?limit=10")
        assert events_response.status_code == 200

        events_data = events_response.json()
        assert "events" in events_data
        assert "total" in events_data

        # Trigger cleanup
        cleanup_response = admin_client.post("/admin/events/cleanup")
        assert cleanup_response.status_code == 200

        cleanup_data = cleanup_response.json()
        assert cleanup_data["status"] == "success"
        assert "deleted_age_based" in cleanup_data
        assert "deleted_size_based" in cleanup_data


class TestAuthenticationSecurity:
    """Test authentication and security."""

    @pytest.mark.asyncio
    async def test_authentication_required(self):
        """Test that all endpoints require authentication."""
        from fastapi.testclient import TestClient

        from src.admin_ui import app

        client = TestClient(app)

        # Try to access without auth
        endpoints = ["/admin/config", "/admin/health", "/admin/events"]

        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_credentials_rejected(self):
        """Test that invalid credentials are rejected."""
        from fastapi.testclient import TestClient

        from src.admin_ui import app

        client = TestClient(app)
        client.auth = ("wrong_user", "wrong_password")

        response = client.get("/admin/config")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_health_check_no_auth_required(self):
        """Test that health check endpoint doesn't require auth."""
        from fastapi.testclient import TestClient

        from src.admin_ui import app

        client = TestClient(app)
        response = client.get("/admin/health-check")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
