"""Deployment detector for Mnemodrome service management.

Detects whether Mnemodrome is running as a systemd service or Docker container
and returns the appropriate deployment type for service control operations.
"""

import os
from enum import Enum
from pathlib import Path

from loguru import logger

from src.config import DEPLOYMENT_TYPE, DOCKER_CONTAINER_NAME, SERVICE_NAME


class DeploymentType(Enum):
    """Supported deployment types for Mnemodrome."""

    SYSTEMD = "systemd"
    DOCKER = "docker"


class DeploymentDetector:
    """Detects and manages deployment type information.

    Provides methods to detect whether Mnemodrome is running as a systemd service
    or Docker container, with support for manual override via configuration.

    Constitutional Principles:
    - Verification-first: Check environment before making assumptions
    - Exception-based: Handle detection failures gracefully
    - Immutability: Deployment type is detected once and cached
    """

    def __init__(self) -> None:
        """Initialize the deployment detector.

        Caches the detected deployment type on first initialization.
        """
        self._detected_type: DeploymentType | None = None

    def detect_deployment_type(self) -> DeploymentType:
        """Detect the current deployment type of Mnemodrome.

        Detection logic (in order of priority):
        1. If DEPLOYMENT_TYPE config is explicitly set, use that (manual override)
        2. Check if running in Docker (DOCKER_CONTAINER environment variable)
        3. Check if systemd unit file exists
        4. Default to systemd (most common case)

        Returns:
            DeploymentType: Either SYSTEMD or DOCKER

        Raises:
            ValueError: If DEPLOYMENT_TYPE config is set to an invalid value

        Note:
            This method uses verification-first principle:
            - Checks environment before making assumptions
            - Logs detection results for debugging
            - Handles edge cases gracefully
        """
        # If already detected and cached, return cached value
        if self._detected_type is not None:
            return self._detected_type

        # Priority 1: Check for explicit configuration override
        if DEPLOYMENT_TYPE:
            logger.info(
                f"Using explicit DEPLOYMENT_TYPE from config: {DEPLOYMENT_TYPE}"
            )
            try:
                self._detected_type = DeploymentType(DEPLOYMENT_TYPE.lower())
                return self._detected_type
            except ValueError as e:
                logger.error(
                    f"Invalid DEPLOYMENT_TYPE config value: {DEPLOYMENT_TYPE}. "
                    f"Must be 'systemd' or 'docker'."
                )
                raise ValueError(
                    f"Invalid DEPLOYMENT_TYPE: {DEPLOYMENT_TYPE}. "
                    f"Must be 'systemd' or 'docker'."
                ) from e

        # Priority 2: Check for Docker environment indicator
        if os.environ.get("DOCKER_CONTAINER"):
            logger.info(
                "Detected Docker deployment via DOCKER_CONTAINER environment variable"
            )
            self._detected_type = DeploymentType.DOCKER
            return self._detected_type

        # Priority 3: Check for systemd unit file existence
        systemd_unit_path = Path(f"/etc/systemd/system/{SERVICE_NAME}.service")
        if systemd_unit_path.exists():
            logger.info(
                f"Detected systemd deployment via unit file: {systemd_unit_path}"
            )
            self._detected_type = DeploymentType.SYSTEMD
            return self._detected_type

        # Priority 4: Default to systemd (most common case)
        logger.info(
            "No explicit deployment type detected, defaulting to systemd. "
            "Set DEPLOYMENT_TYPE=docker if running in Docker."
        )
        self._detected_type = DeploymentType.SYSTEMD
        return self._detected_type

    def get_deployment_type(self) -> DeploymentType:
        """Get the current deployment type (convenience method).

        Returns:
            DeploymentType: Either SYSTEMD or DOCKER
        """
        return self.detect_deployment_type()

    def is_systemd(self) -> bool:
        """Check if deployment is systemd.

        Returns:
            bool: True if running as systemd service
        """
        return self.detect_deployment_type() == DeploymentType.SYSTEMD

    def is_docker(self) -> bool:
        """Check if deployment is Docker.

        Returns:
            bool: True if running as Docker container
        """
        return self.detect_deployment_type() == DeploymentType.DOCKER

    def reset_detection(self) -> None:
        """Reset cached detection to allow re-detection.

        Used for testing purposes when deployment type may change
        between test cases.
        """
        self._detected_type = None
        logger.debug("Deployment type detection cache reset")

    def get_service_identifier(self) -> str:
        """Get the service identifier for the detected deployment type.

        Returns the appropriate identifier for control commands:
        - systemd: service name (e.g., "mnemodrome-mcp.service")
        - docker: container name (e.g., "mnemodrome-mcp")

        Returns:
            str: Service identifier
        """
        if self.is_docker():
            return DOCKER_CONTAINER_NAME

        # For systemd, ensure the .service suffix is present
        if SERVICE_NAME.endswith(".service"):
            return SERVICE_NAME
        return f"{SERVICE_NAME}.service"
