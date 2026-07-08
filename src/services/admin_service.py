"""
Admin service for Mnemodrome configuration management.

Feature 010: MCP Configuration Web Interface
Provides ConfigurationManager for live runtime config updates.
"""

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class ConfigVariable:
    """Represents a single configuration variable with metadata."""

    name: str
    current_value: Any
    default_value: Any
    editable: bool
    validator: Callable[[Any], bool]
    description: str
    category: str


class ConfigurationManager:
    """
    Manages Mnemodrome configuration with live runtime updates.

    Responsibilities:
    - Read current configuration from config module
    - Validate configuration changes
    - Apply changes to runtime config (module globals)
    - Persist changes to .env file atomically
    - Manage configuration backups and rollback
    """

    def __init__(self, env_path: Path):
        """
        Initialize ConfigurationManager.

        Args:
            env_path: Path to .env file (typically project root / ".env")
        """
        self.env_path = env_path
        self.schema: dict[str, ConfigVariable] = {}
        self._config_version: float = 0.0

        # Build configuration schema
        self.schema = self._build_schema()
        self._config_version = self._get_config_version()

        logger.info(
            f"ConfigurationManager initialized with {len(self.schema)} variables "
            f"(version: {self._config_version})"
        )

    def _build_schema(self) -> dict[str, ConfigVariable]:
        """
        Build configuration schema with all managed variables.

        Returns:
            Dictionary mapping variable name to ConfigVariable metadata
        """
        from src import config

        # Validator functions
        def validate_bool(value: Any) -> bool:
            """Validate boolean values."""
            return isinstance(value, bool) or value in [
                "true",
                "false",
                "True",
                "False",
            ]

        def validate_int_min(min_val: int) -> Callable[[Any], bool]:
            """Create validator for int >= min_val."""

            def validator(value: Any) -> bool:
                try:
                    return int(value) >= min_val
                except (ValueError, TypeError):
                    return False

            return validator

        def validate_int_range(min_val: int, max_val: int) -> Callable[[Any], bool]:
            """Create validator for int in range."""

            def validator(value: Any) -> bool:
                try:
                    int_value = int(value)
                    return min_val <= int_value <= max_val
                except (ValueError, TypeError):
                    return False

            return validator

        def validate_log_level(value: Any) -> bool:
            """Validate log level."""
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            return str(value).upper() in valid_levels

        def validate_uri_bolt(value: Any) -> bool:
            """Validate Neo4j bolt:// URI."""
            return isinstance(value, str) and value.startswith("bolt://")

        def validate_non_empty_string(value: Any) -> bool:
            """Validate non-empty string."""
            return isinstance(value, str) and len(value) > 0

        # Build schema dictionary
        schema: dict[str, ConfigVariable] = {}

        # Event Log Configuration (8 variables, all editable)
        schema["EVENT_LOG_ENABLED"] = ConfigVariable(
            name="EVENT_LOG_ENABLED",
            current_value=config.EVENT_LOG_ENABLED,
            default_value=True,
            editable=True,
            validator=validate_bool,
            description="Enable event logging system",
            category="event_log",
        )

        schema["EVENT_LOG_AUTO_CAPTURE"] = ConfigVariable(
            name="EVENT_LOG_AUTO_CAPTURE",
            current_value=config.EVENT_LOG_AUTO_CAPTURE,
            default_value=True,
            editable=True,
            validator=validate_bool,
            description="Automatically capture MCP tool calls",
            category="event_log",
        )

        schema["EVENT_LOG_RETENTION_DAYS"] = ConfigVariable(
            name="EVENT_LOG_RETENTION_DAYS",
            current_value=config.EVENT_LOG_RETENTION_DAYS,
            default_value=90,
            editable=True,
            validator=validate_int_min(1),
            description="Days to retain event logs (minimum 1)",
            category="event_log",
        )

        schema["EVENT_LOG_MAX_SIZE_MB"] = ConfigVariable(
            name="EVENT_LOG_MAX_SIZE_MB",
            current_value=config.EVENT_LOG_MAX_SIZE_MB,
            default_value=100,
            editable=True,
            validator=validate_int_min(10),
            description="Maximum event log size in MB (minimum 10)",
            category="event_log",
        )

        schema["EVENT_LOG_SESSION_TIMEOUT_HOURS"] = ConfigVariable(
            name="EVENT_LOG_SESSION_TIMEOUT_HOURS",
            current_value=config.EVENT_LOG_SESSION_TIMEOUT_HOURS,
            default_value=4,
            editable=True,
            validator=validate_int_min(0),
            description="Hours of inactivity before session timeout (0 = no timeout)",
            category="event_log",
        )

        schema["EVENT_LOG_PERFORMANCE_SLA_MS"] = ConfigVariable(
            name="EVENT_LOG_PERFORMANCE_SLA_MS",
            current_value=config.EVENT_LOG_PERFORMANCE_SLA_MS,
            default_value=50,
            editable=True,
            validator=validate_int_min(1),
            description="Performance SLA for tool calls in milliseconds (minimum 1)",
            category="event_log",
        )

        schema["EVENT_LOG_QUERY_SLA_MS"] = ConfigVariable(
            name="EVENT_LOG_QUERY_SLA_MS",
            current_value=config.EVENT_LOG_QUERY_SLA_MS,
            default_value=100,
            editable=True,
            validator=validate_int_min(1),
            description="Performance SLA for event queries in milliseconds (minimum 1)",
            category="event_log",
        )

        schema["EVENT_LOG_FAIL_LOUD"] = ConfigVariable(
            name="EVENT_LOG_FAIL_LOUD",
            current_value=config.EVENT_LOG_FAIL_LOUD,
            default_value=True,
            editable=True,
            validator=validate_bool,
            description="Raise exceptions on event log errors (vs silent logging)",
            category="event_log",
        )

        # Admin Web UI Configuration (2 variables)
        # Note: Both require restart to change (prevent self-lockout)
        admin_enabled = getattr(config, "ADMIN_WEB_ENABLED", True)
        admin_port = getattr(config, "ADMIN_WEB_PORT", 8083)

        schema["ADMIN_WEB_ENABLED"] = ConfigVariable(
            name="ADMIN_WEB_ENABLED",
            current_value=admin_enabled,
            default_value=True,
            editable=False,  # Requires restart - prevents self-lockout
            validator=validate_bool,
            description="Enable admin web interface (restart required to change - prevents self-lockout)",
            category="admin",
        )

        schema["ADMIN_WEB_PORT"] = ConfigVariable(
            name="ADMIN_WEB_PORT",
            current_value=admin_port,
            default_value=8083,
            editable=False,  # Requires restart
            validator=validate_int_range(1024, 65535),
            description="Admin web interface port (restart required to change)",
            category="admin",
        )

        # Logging Configuration (1 variable, editable)
        log_level = getattr(config, "LOG_LEVEL", "INFO")
        schema["LOG_LEVEL"] = ConfigVariable(
            name="LOG_LEVEL",
            current_value=log_level,
            default_value="INFO",
            editable=True,
            validator=validate_log_level,
            description="Application log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
            category="logging",
        )

        # Neo4j Configuration (4 variables, NOT editable - restart required)
        schema["NEO4J_URI"] = ConfigVariable(
            name="NEO4J_URI",
            current_value=config.NEO4J_URI,
            default_value="bolt://localhost:7687",
            editable=False,
            validator=validate_uri_bolt,
            description="Neo4j Bolt URI (restart required to change)",
            category="neo4j",
        )

        schema["NEO4J_DATABASE"] = ConfigVariable(
            name="NEO4J_DATABASE",
            current_value=config.NEO4J_DATABASE,
            default_value="neo4j",
            editable=False,
            validator=validate_non_empty_string,
            description="Neo4j database name (restart required to change)",
            category="neo4j",
        )

        schema["NEO4J_USER"] = ConfigVariable(
            name="NEO4J_USER",
            current_value=config.NEO4J_USER,
            default_value="neo4j",
            editable=False,
            validator=validate_non_empty_string,
            description="Neo4j username (restart required to change)",
            category="neo4j",
        )

        # Note: NEO4J_PASSWORD excluded from schema for security
        # It should never be visible or editable via web UI

        # Redis Configuration (2 variables, NOT editable - restart required)
        schema["REDIS_HOST"] = ConfigVariable(
            name="REDIS_HOST",
            current_value=config.REDIS_HOST,
            default_value="localhost",
            editable=False,
            validator=validate_non_empty_string,
            description="Redis host address (restart required to change)",
            category="redis",
        )

        schema["REDIS_PORT"] = ConfigVariable(
            name="REDIS_PORT",
            current_value=config.REDIS_PORT,
            default_value=6379,
            editable=False,
            validator=validate_int_range(1024, 65535),
            description="Redis port (restart required to change)",
            category="redis",
        )

        return schema

    def _get_config_version(self) -> float:
        """
        Get configuration version based on .env file modification time.

        Returns:
            Unix timestamp of .env file mtime (float)
        """
        if not self.env_path.exists():
            logger.warning(f".env file not found at {self.env_path}")
            return 0.0

        return self.env_path.stat().st_mtime

    def read_config(self) -> dict[str, Any]:
        """
        Read current configuration grouped by category.

        Returns:
            Dictionary with config_version and categories
            Structure:
            {
                "config_version": <timestamp>,
                "categories": {
                    "category_name": [
                        {
                            "name": "VAR_NAME",
                            "value": <current_value>,
                            "default": <default_value>,
                            "editable": bool,
                            "description": "..."
                        },
                        ...
                    ],
                    ...
                }
            }
        """
        from src import config

        # Group variables by category
        categories: dict[str, list[dict[str, Any]]] = {}

        for var_name, var_config in self.schema.items():
            category = var_config.category

            # Initialize category list if not exists
            if category not in categories:
                categories[category] = []

            # Read current value from config module
            current_value = getattr(config, var_name, var_config.default_value)

            # Add variable to category
            categories[category].append(
                {
                    "name": var_name,
                    "value": current_value,
                    "default": var_config.default_value,
                    "editable": var_config.editable,
                    "description": var_config.description,
                }
            )

        # Sort variables within each category alphabetically
        for category in categories:
            categories[category].sort(key=lambda x: x["name"])

        return {
            "config_version": self._config_version,
            "categories": categories,
        }

    def validate_config(self, changes: dict[str, Any]) -> list[str]:
        """
        Validate configuration changes.

        Args:
            changes: Dictionary of variable name -> new value

        Returns:
            List of error messages (empty if all valid)
            Error format: "{key}: {problem}. Expected: {description}"
        """
        errors: list[str] = []

        for key, value in changes.items():
            # Check if variable exists in schema
            if key not in self.schema:
                errors.append(
                    f"{key}: Unknown configuration variable. "
                    f"Valid variables: {', '.join(sorted(self.schema.keys()))}"
                )
                continue

            var_config = self.schema[key]

            # Check if variable is editable
            if not var_config.editable:
                errors.append(
                    f"{key}: Configuration variable is not editable (restart required). "
                    f"Expected: {var_config.description}"
                )
                continue

            # Run validator function
            try:
                is_valid = var_config.validator(value)
            except Exception as e:
                errors.append(
                    f"{key}: Validation error - {str(e)}. "
                    f"Expected: {var_config.description}"
                )
                continue

            if not is_valid:
                errors.append(
                    f"{key}: Invalid value '{value}'. "
                    f"Expected: {var_config.description}"
                )

        return errors

    async def apply_config(self, changes: dict[str, Any], actor: str) -> None:
        """
        Apply configuration changes to runtime config module.

        This updates the config module globals immediately without restart.
        Changes must be validated before calling this method.

        Args:
            changes: Dictionary of variable name -> new value
            actor: Identifier of who made the change (for audit trail)

        Note:
            - This method does NOT persist changes to .env file
            - Call persist_config() separately to save to disk
            - Events are logged to event_log_service if available
        """
        from src import config

        # Capture old values for audit trail
        old_values = {key: getattr(config, key, None) for key in changes}

        # Update config module globals
        for key, value in changes.items():
            # Convert string booleans to actual booleans
            if isinstance(value, str) and value.lower() in ["true", "false"]:
                value = value.lower() == "true"

            # Convert string integers to actual integers
            if key in self.schema and isinstance(self.schema[key].current_value, int):
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    pass  # Let validator catch this

            setattr(config, key, value)

            # Update schema current_value
            if key in self.schema:
                self.schema[key].current_value = value

        # Log to event log if available
        try:
            from src.services.event_log_service import event_log_service

            if event_log_service:
                await event_log_service.log_event(
                    event_type="config_change",
                    action="update_config",
                    actor=actor,
                    context={"changes": changes, "old_values": old_values},
                    result="success",
                    metadata={"timestamp": datetime.now(UTC).isoformat()},
                )
        except Exception as log_error:
            # Don't fail the config update if logging fails
            logger.warning(f"Failed to log config change event: {log_error}")

        logger.info(
            f"Applied {len(changes)} configuration changes by {actor}: "
            f"{', '.join(changes.keys())}"
        )

    async def persist_config(self, changes: dict[str, Any]) -> None:
        """
        Persist configuration changes to .env file atomically.

        This method:
        1. Creates timestamped backup of current .env
        2. Updates .env file with new values
        3. Uses atomic write (temp file + rename) to prevent corruption
        4. Prunes old backups (keeps last 5)
        5. Updates config version

        Args:
            changes: Dictionary of variable name -> new value

        Raises:
            OSError: If .env file operations fail
        """
        if not self.env_path.exists():
            raise FileNotFoundError(f".env file not found at {self.env_path}")

        # Read current .env file
        env_lines = self.env_path.read_text().splitlines()

        # Create timestamped backup
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        backup_path = self.env_path.parent / f".env.backup.{timestamp}"

        # Rename current .env to backup
        self.env_path.rename(backup_path)
        logger.info(f"Created configuration backup: {backup_path.name}")

        try:
            # Update values in memory
            updated_keys = set()
            updated_lines = []

            for line in env_lines:
                stripped = line.strip()

                # Preserve empty lines and comments
                if not stripped or stripped.startswith("#"):
                    updated_lines.append(line)
                    continue

                # Parse key=value
                if "=" in line:
                    key = line.split("=", 1)[0].strip()

                    # Update value if in changes
                    if key in changes:
                        # Convert boolean to string
                        value = changes[key]
                        if isinstance(value, bool):
                            value = "true" if value else "false"
                        else:
                            value = str(value)

                        updated_lines.append(f"{key}={value}")
                        updated_keys.add(key)
                    else:
                        # Keep original line
                        updated_lines.append(line)
                else:
                    # Keep malformed lines as-is
                    updated_lines.append(line)

            # Add new variables not in .env
            for key, value in changes.items():
                if key not in updated_keys:
                    # Convert boolean to string
                    if isinstance(value, bool):
                        value = "true" if value else "false"
                    else:
                        value = str(value)

                    updated_lines.append(f"{key}={value}")
                    logger.info(f"Added new variable to .env: {key}")

            # Write atomically using temp file + rename
            temp_path = self.env_path.parent / f".env.tmp.{timestamp}"
            temp_path.write_text("\n".join(updated_lines) + "\n")

            # Atomic rename
            temp_path.rename(self.env_path)
            logger.info(f"Persisted {len(changes)} configuration changes to .env")

        except Exception as e:
            # Restore backup on failure
            if backup_path.exists():
                backup_path.rename(self.env_path)
                logger.error(f"Failed to persist config, restored backup: {e}")
            raise

        # Prune old backups (keep last 5)
        backup_pattern = self.env_path.parent / ".env.backup.*"
        import glob

        backups = sorted(
            [Path(p) for p in glob.glob(str(backup_pattern))],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        # Delete backups beyond the 5 most recent
        for old_backup in backups[5:]:
            old_backup.unlink()
            logger.debug(f"Pruned old backup: {old_backup.name}")

        # Update config version
        self._config_version = self._get_config_version()
        logger.info(f"Updated config version to {self._config_version}")

    async def undo_last_change(self) -> dict[str, Any]:
        """
        Undo the last configuration change by restoring most recent backup.

        This method:
        1. Finds most recent .env.backup.* file
        2. Restores it atomically
        3. Reloads config module
        4. Rebuilds schema
        5. Updates config version

        Returns:
            Status dict with backup filename and timestamp

        Raises:
            FileNotFoundError: If no backups available
        """
        import glob

        # Find most recent backup
        backup_pattern = self.env_path.parent / ".env.backup.*"
        backups = sorted(
            [Path(p) for p in glob.glob(str(backup_pattern))],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not backups:
            raise FileNotFoundError("No configuration backups available")

        most_recent_backup = backups[0]
        backup_mtime = most_recent_backup.stat().st_mtime
        backup_timestamp = datetime.fromtimestamp(backup_mtime, tz=UTC)

        logger.info(f"Restoring configuration from backup: {most_recent_backup.name}")

        # Read backup content
        backup_content = most_recent_backup.read_text()

        # Create temp file for atomic write
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        temp_path = self.env_path.parent / f".env.tmp.{timestamp}"

        try:
            # Write backup content to temp file
            temp_path.write_text(backup_content)

            # Atomic rename to .env
            temp_path.rename(self.env_path)

            logger.info("Configuration restored from backup successfully")

        except Exception as e:
            # Clean up temp file on failure
            if temp_path.exists():
                temp_path.unlink()
            logger.error(f"Failed to restore configuration from backup: {e}")
            raise

        # Reload config module to pick up restored values
        from src import config

        importlib.reload(config)
        logger.info("Reloaded config module with restored values")

        # Rebuild schema with restored values
        self.schema = self._build_schema()
        logger.info("Rebuilt configuration schema")

        # Update config version
        self._config_version = self._get_config_version()

        return {
            "backup_file": most_recent_backup.name,
            "backup_timestamp": backup_timestamp.isoformat(),
            "restored_at": datetime.now(UTC).isoformat(),
        }


# Global ConfigurationManager instance (initialized in mcp_server.py or admin_ui.py)
config_manager: ConfigurationManager | None = None


def get_config_manager() -> ConfigurationManager:
    """
    Get the global ConfigurationManager instance.

    Returns:
        Global ConfigurationManager instance

    Raises:
        RuntimeError: If ConfigurationManager not initialized
    """
    if config_manager is None:
        raise RuntimeError(
            "ConfigurationManager not initialized. "
            "Call initialize_config_manager() first."
        )
    return config_manager


def initialize_config_manager(env_path: Path) -> ConfigurationManager:
    """
    Initialize the global ConfigurationManager instance.

    Args:
        env_path: Path to .env file

    Returns:
        Initialized ConfigurationManager instance
    """
    global config_manager
    config_manager = ConfigurationManager(env_path)
    return config_manager
