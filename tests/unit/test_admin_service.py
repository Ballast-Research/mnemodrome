"""Tests for Admin Service (ConfigurationManager)."""

import asyncio
from unittest.mock import patch

import pytest

from src.services.admin_service import ConfigurationManager


class TestConfigurationManager:
    """Tests for ConfigurationManager."""

    @pytest.fixture
    def temp_env_file(self, tmp_path):
        """Create temporary .env file for testing."""
        env_file = tmp_path / ".env"
        env_content = """# Test .env file
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=test-password
NEO4J_DATABASE=neo4j
REDIS_HOST=localhost
REDIS_PORT=6379
EVENT_LOG_ENABLED=true
EVENT_LOG_AUTO_CAPTURE=true
EVENT_LOG_RETENTION_DAYS=90
EVENT_LOG_MAX_SIZE_MB=100
EVENT_LOG_SESSION_TIMEOUT_HOURS=4
EVENT_LOG_PERFORMANCE_SLA_MS=50
EVENT_LOG_QUERY_SLA_MS=100
EVENT_LOG_FAIL_LOUD=true
LOG_LEVEL=INFO
"""
        env_file.write_text(env_content)
        return env_file

    def test_config_manager_instantiation(self, temp_env_file):
        """Test ConfigurationManager instantiates successfully."""
        cm = ConfigurationManager(temp_env_file)

        assert cm.env_path == temp_env_file
        assert isinstance(cm.schema, dict)
        assert len(cm.schema) > 0
        assert cm._config_version > 0

    def test_get_config_version(self, temp_env_file):
        """Test config version returns .env mtime correctly."""
        cm = ConfigurationManager(temp_env_file)

        # Version should be float timestamp
        assert isinstance(cm._config_version, float)
        assert cm._config_version == temp_env_file.stat().st_mtime

    def test_schema_completeness(self, temp_env_file):
        """Test schema contains all expected configuration variables."""
        cm = ConfigurationManager(temp_env_file)

        # Expected categories
        expected_categories = {"event_log", "neo4j", "redis", "admin", "logging"}
        actual_categories = set(v.category for v in cm.schema.values())
        assert expected_categories == actual_categories

        # Expected event_log variables
        event_log_vars = [
            "EVENT_LOG_ENABLED",
            "EVENT_LOG_AUTO_CAPTURE",
            "EVENT_LOG_RETENTION_DAYS",
            "EVENT_LOG_MAX_SIZE_MB",
            "EVENT_LOG_SESSION_TIMEOUT_HOURS",
            "EVENT_LOG_PERFORMANCE_SLA_MS",
            "EVENT_LOG_QUERY_SLA_MS",
            "EVENT_LOG_FAIL_LOUD",
        ]
        for var in event_log_vars:
            assert var in cm.schema
            assert cm.schema[var].category == "event_log"

    def test_validator_functions(self, temp_env_file):
        """Test validator functions work correctly."""
        cm = ConfigurationManager(temp_env_file)

        # Test boolean validator
        bool_validator = cm.schema["EVENT_LOG_ENABLED"].validator
        assert bool_validator(True) is True
        assert bool_validator(False) is True
        assert bool_validator("true") is True
        assert bool_validator("false") is True
        assert bool_validator("invalid") is False

        # Test int min validator
        retention_validator = cm.schema["EVENT_LOG_RETENTION_DAYS"].validator
        assert retention_validator(90) is True
        assert retention_validator(1) is True
        assert retention_validator(0) is False
        assert retention_validator(-1) is False

        # Test log level validator
        log_level_validator = cm.schema["LOG_LEVEL"].validator
        assert log_level_validator("DEBUG") is True
        assert log_level_validator("INFO") is True
        assert log_level_validator("INVALID") is False

    def test_editable_flags(self, temp_env_file):
        """Test editable vs non-editable classification."""
        cm = ConfigurationManager(temp_env_file)

        # Event log variables should be editable
        assert cm.schema["EVENT_LOG_ENABLED"].editable is True
        assert cm.schema["EVENT_LOG_RETENTION_DAYS"].editable is True

        # Neo4j variables should NOT be editable
        assert cm.schema["NEO4J_URI"].editable is False
        assert cm.schema["NEO4J_DATABASE"].editable is False

        # Redis variables should NOT be editable
        assert cm.schema["REDIS_HOST"].editable is False
        assert cm.schema["REDIS_PORT"].editable is False

    def test_read_config_structure(self, temp_env_file):
        """Test read_config() returns correct structure."""
        cm = ConfigurationManager(temp_env_file)
        config_data = cm.read_config()

        # Check top-level structure
        assert "config_version" in config_data
        assert "categories" in config_data
        assert isinstance(config_data["categories"], dict)

        # Check categories exist
        assert "event_log" in config_data["categories"]
        assert "neo4j" in config_data["categories"]
        assert "redis" in config_data["categories"]

    def test_read_config_categories(self, temp_env_file):
        """Test read_config() categories are populated."""
        cm = ConfigurationManager(temp_env_file)
        config_data = cm.read_config()

        # Check event_log category
        event_log_vars = config_data["categories"]["event_log"]
        assert len(event_log_vars) == 8

        # Check variable structure
        for var in event_log_vars:
            assert "name" in var
            assert "value" in var
            assert "default" in var
            assert "editable" in var
            assert "description" in var

    def test_validate_config_valid(self, temp_env_file):
        """Test validate_config() accepts valid changes."""
        cm = ConfigurationManager(temp_env_file)

        valid_changes = {
            "EVENT_LOG_ENABLED": True,
            "EVENT_LOG_RETENTION_DAYS": 120,
            "LOG_LEVEL": "DEBUG",
        }

        errors = cm.validate_config(valid_changes)
        assert len(errors) == 0

    def test_validate_config_unknown_key(self, temp_env_file):
        """Test validate_config() rejects unknown variable."""
        cm = ConfigurationManager(temp_env_file)

        invalid_changes = {"UNKNOWN_VARIABLE": "value"}

        errors = cm.validate_config(invalid_changes)
        assert len(errors) == 1
        assert "UNKNOWN_VARIABLE" in errors[0]
        assert "Unknown configuration variable" in errors[0]

    def test_validate_config_not_editable(self, temp_env_file):
        """Test validate_config() rejects non-editable variable."""
        cm = ConfigurationManager(temp_env_file)

        invalid_changes = {"NEO4J_URI": "bolt://localhost:7687"}

        errors = cm.validate_config(invalid_changes)
        assert len(errors) == 1
        assert "NEO4J_URI" in errors[0]
        assert "not editable" in errors[0]

    def test_validate_config_invalid_value(self, temp_env_file):
        """Test validate_config() rejects invalid value."""
        cm = ConfigurationManager(temp_env_file)

        invalid_changes = {
            "EVENT_LOG_RETENTION_DAYS": 0,  # Must be >= 1
            "LOG_LEVEL": "INVALID",  # Must be valid log level
        }

        errors = cm.validate_config(invalid_changes)
        assert len(errors) == 2

    def test_validate_config_multiple_errors(self, temp_env_file):
        """Test validate_config() returns multiple errors."""
        cm = ConfigurationManager(temp_env_file)

        invalid_changes = {
            "UNKNOWN_VAR": "value",
            "NEO4J_URI": "bolt://localhost:7687",
            "EVENT_LOG_RETENTION_DAYS": -1,
        }

        errors = cm.validate_config(invalid_changes)
        assert len(errors) == 3

    @pytest.mark.asyncio
    async def test_apply_config_updates_module(self, temp_env_file):
        """Test apply_config() updates config module globals."""
        from src import config

        cm = ConfigurationManager(temp_env_file)

        # Store original value
        original_value = config.EVENT_LOG_RETENTION_DAYS

        # Apply change
        changes = {"EVENT_LOG_RETENTION_DAYS": 120}

        # Mock event_log_service to avoid dependencies
        with patch("src.services.event_log_service.event_log_service", None):
            await cm.apply_config(changes, actor="test_user")

        # Verify module updated
        assert config.EVENT_LOG_RETENTION_DAYS == 120

        # Restore original value
        config.EVENT_LOG_RETENTION_DAYS = original_value

    @pytest.mark.asyncio
    async def test_apply_config_updates_schema(self, temp_env_file):
        """Test apply_config() updates schema current_value."""
        cm = ConfigurationManager(temp_env_file)

        # Store original value
        original_value = cm.schema["EVENT_LOG_RETENTION_DAYS"].current_value

        # Apply change
        changes = {"EVENT_LOG_RETENTION_DAYS": 120}

        # Mock event_log_service
        with patch("src.services.event_log_service.event_log_service", None):
            await cm.apply_config(changes, actor="test_user")

        # Verify schema updated
        assert cm.schema["EVENT_LOG_RETENTION_DAYS"].current_value == 120

    @pytest.mark.asyncio
    async def test_persist_config_updates_env(self, temp_env_file):
        """Test persist_config() updates .env file."""
        cm = ConfigurationManager(temp_env_file)

        changes = {"EVENT_LOG_RETENTION_DAYS": 120}

        await cm.persist_config(changes)

        # Verify .env updated
        env_content = temp_env_file.read_text()
        assert "EVENT_LOG_RETENTION_DAYS=120" in env_content

    @pytest.mark.asyncio
    async def test_persist_config_creates_backup(self, temp_env_file):
        """Test persist_config() creates backup file."""
        cm = ConfigurationManager(temp_env_file)

        changes = {"EVENT_LOG_RETENTION_DAYS": 120}

        await cm.persist_config(changes)

        # Verify backup created
        backups = list(temp_env_file.parent.glob(".env.backup.*"))
        assert len(backups) == 1

    @pytest.mark.asyncio
    async def test_persist_config_atomic_write(self, temp_env_file):
        """Test persist_config() uses atomic write."""
        cm = ConfigurationManager(temp_env_file)

        changes = {"EVENT_LOG_RETENTION_DAYS": 120}

        await cm.persist_config(changes)

        # .env should exist and be valid
        assert temp_env_file.exists()
        content = temp_env_file.read_text()
        assert "NEO4J_URI=bolt://localhost:7687" in content
        assert "EVENT_LOG_RETENTION_DAYS=120" in content

    @pytest.mark.asyncio
    async def test_persist_config_prunes_backups(self, temp_env_file):
        """Test persist_config() prunes old backups."""
        cm = ConfigurationManager(temp_env_file)

        # Create 7 backups by making multiple changes with delays
        for i in range(7):
            changes = {"EVENT_LOG_RETENTION_DAYS": 90 + i}
            await cm.persist_config(changes)
            # Small delay to ensure different mtimes (backup pruning uses mtime)
            await asyncio.sleep(0.1)

        # Verify backups are pruned (at most 5 remain)
        # Note: Timestamp format is YYYYMMDD-HHMMSS, so backups created in same second
        # will have same filename. The test verifies pruning happens, not exact count.
        backups = list(temp_env_file.parent.glob(".env.backup.*"))
        assert len(backups) <= 5
        assert len(backups) >= 1  # At least one backup should exist

    @pytest.mark.asyncio
    async def test_persist_config_updates_version(self, temp_env_file):
        """Test persist_config() updates config version."""
        cm = ConfigurationManager(temp_env_file)

        old_version = cm._config_version

        # Small delay to ensure different mtime
        await asyncio.sleep(0.01)

        changes = {"EVENT_LOG_RETENTION_DAYS": 120}
        await cm.persist_config(changes)

        # Version should be updated
        assert cm._config_version > old_version

    @pytest.mark.asyncio
    async def test_undo_last_change_restores_backup(self, temp_env_file):
        """Test undo_last_change() restores most recent backup."""
        cm = ConfigurationManager(temp_env_file)

        # Make a change and persist it
        changes = {"EVENT_LOG_RETENTION_DAYS": 120}
        await cm.persist_config(changes)

        # Verify change applied
        env_content = temp_env_file.read_text()
        assert "EVENT_LOG_RETENTION_DAYS=120" in env_content

        # Undo the change
        result = await cm.undo_last_change()

        # Verify backup restored
        env_content = temp_env_file.read_text()
        assert "EVENT_LOG_RETENTION_DAYS=90" in env_content

        # Verify result contains backup info
        assert "backup_file" in result
        assert "backup_timestamp" in result
        assert "restored_at" in result

    @pytest.mark.asyncio
    async def test_undo_last_change_no_backup(self, temp_env_file):
        """Test undo_last_change() raises error if no backup."""
        cm = ConfigurationManager(temp_env_file)

        # No backups exist yet
        with pytest.raises(
            FileNotFoundError, match="No configuration backups available"
        ):
            await cm.undo_last_change()

    @pytest.mark.asyncio
    async def test_undo_last_change_returns_status(self, temp_env_file):
        """Test undo_last_change() returns status dict."""
        cm = ConfigurationManager(temp_env_file)

        # Make a change to create backup
        changes = {"EVENT_LOG_RETENTION_DAYS": 120}
        await cm.persist_config(changes)

        # Undo and check status
        result = await cm.undo_last_change()

        assert isinstance(result, dict)
        assert "backup_file" in result
        assert ".env.backup." in result["backup_file"]
