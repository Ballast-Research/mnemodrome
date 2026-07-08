"""Unit tests for deployment detector module.

Tests for the DeploymentDetector class covering:
- systemd deployment detection
- Docker deployment detection
- Manual override via configuration
- Service identifier generation
- Error handling for invalid configurations
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.services.deployment_detector import DeploymentDetector, DeploymentType


class TestDeploymentDetector:
    """Tests for the DeploymentDetector class."""

    @pytest.fixture
    def detector(self) -> DeploymentDetector:
        """Provide a fresh detector instance for each test."""
        detector = DeploymentDetector()
        yield detector
        # Reset cache after each test
        detector.reset_detection()

    # Test: Explicit Configuration Override

    def test_detect_with_explicit_systemd_config(
        self, detector: DeploymentDetector, monkeypatch
    ) -> None:
        """Test that explicit DEPLOYMENT_TYPE=systemd config is used."""
        monkeypatch.setattr(
            "src.services.deployment_detector.DEPLOYMENT_TYPE", "systemd"
        )

        result = detector.detect_deployment_type()

        assert result == DeploymentType.SYSTEMD

    def test_detect_with_explicit_docker_config(
        self, detector: DeploymentDetector, monkeypatch
    ) -> None:
        """Test that explicit DEPLOYMENT_TYPE=docker config is used."""
        monkeypatch.setattr(
            "src.services.deployment_detector.DEPLOYMENT_TYPE", "docker"
        )

        result = detector.detect_deployment_type()

        assert result == DeploymentType.DOCKER

    def test_detect_with_invalid_deployment_type_config(
        self, detector: DeploymentDetector, monkeypatch
    ) -> None:
        """Test that invalid DEPLOYMENT_TYPE config raises ValueError."""
        monkeypatch.setattr(
            "src.services.deployment_detector.DEPLOYMENT_TYPE", "kubernetes"
        )

        with pytest.raises(ValueError, match="Invalid DEPLOYMENT_TYPE"):
            detector.detect_deployment_type()

    # Test: Docker Detection via Environment

    def test_detect_docker_via_environment_variable(
        self, detector: DeploymentDetector, monkeypatch
    ) -> None:
        """Test that DOCKER_CONTAINER environment variable triggers Docker detection."""
        # Ensure no explicit config override
        monkeypatch.setattr("src.services.deployment_detector.DEPLOYMENT_TYPE", "")
        # Set Docker environment variable
        monkeypatch.setenv("DOCKER_CONTAINER", "1")

        result = detector.detect_deployment_type()

        assert result == DeploymentType.DOCKER

    def test_detect_docker_via_environment_empty_value(
        self, detector: DeploymentDetector, monkeypatch
    ) -> None:
        """Test Docker detection with empty DOCKER_CONTAINER value."""
        monkeypatch.setattr("src.services.deployment_detector.DEPLOYMENT_TYPE", "")
        # Set Docker environment variable with empty value (should still be truthy)
        monkeypatch.setenv("DOCKER_CONTAINER", "")

        result = detector.detect_deployment_type()

        # Empty string is falsy, so should fall through to systemd detection
        # This test verifies correct boolean handling
        assert result == DeploymentType.SYSTEMD

    def test_detect_docker_via_environment_any_value(
        self, detector: DeploymentDetector, monkeypatch
    ) -> None:
        """Test Docker detection with any truthy DOCKER_CONTAINER value."""
        monkeypatch.setattr("src.services.deployment_detector.DEPLOYMENT_TYPE", "")
        monkeypatch.setenv("DOCKER_CONTAINER", "true")

        result = detector.detect_deployment_type()

        assert result == DeploymentType.DOCKER

    # Test: Systemd Detection via Unit File

    def test_detect_systemd_via_unit_file(
        self, detector: DeploymentDetector, monkeypatch
    ) -> None:
        """Test that systemd unit file existence triggers systemd detection."""
        monkeypatch.setattr("src.services.deployment_detector.DEPLOYMENT_TYPE", "")
        # Mock that DOCKER_CONTAINER is not set
        monkeypatch.delenv("DOCKER_CONTAINER", raising=False)

        # Mock the Path.exists() check for systemd unit file
        with patch.object(Path, "exists", return_value=True):
            result = detector.detect_deployment_type()

        assert result == DeploymentType.SYSTEMD

    def test_detect_systemd_via_unit_file_with_custom_service_name(
        self, detector: DeploymentDetector, monkeypatch
    ) -> None:
        """Test systemd detection uses SERVICE_NAME from config."""
        monkeypatch.setattr("src.services.deployment_detector.DEPLOYMENT_TYPE", "")
        monkeypatch.delenv("DOCKER_CONTAINER", raising=False)
        monkeypatch.setattr(
            "src.services.deployment_detector.SERVICE_NAME", "custom-mcp"
        )

        # Mock Path construction and exists check
        with patch.object(Path, "exists", return_value=True):
            result = detector.detect_deployment_type()

        assert result == DeploymentType.SYSTEMD

    # Test: Default Detection

    def test_detect_defaults_to_systemd_when_no_indicators(
        self, detector: DeploymentDetector, monkeypatch
    ) -> None:
        """Test that systemd is default when no indicators are present."""
        monkeypatch.setattr("src.services.deployment_detector.DEPLOYMENT_TYPE", "")
        monkeypatch.delenv("DOCKER_CONTAINER", raising=False)

        # Mock that systemd unit file does not exist
        with patch.object(Path, "exists", return_value=False):
            result = detector.detect_deployment_type()

        assert result == DeploymentType.SYSTEMD

    # Test: Detection Priority Order

    def test_explicit_config_takes_priority_over_docker_env(
        self, detector: DeploymentDetector, monkeypatch
    ) -> None:
        """Test that explicit config overrides Docker environment detection."""
        monkeypatch.setattr(
            "src.services.deployment_detector.DEPLOYMENT_TYPE", "systemd"
        )
        monkeypatch.setenv("DOCKER_CONTAINER", "1")

        result = detector.detect_deployment_type()

        # Config should take priority
        assert result == DeploymentType.SYSTEMD

    def test_explicit_config_takes_priority_over_unit_file(
        self, detector: DeploymentDetector, monkeypatch
    ) -> None:
        """Test that explicit config overrides unit file detection."""
        monkeypatch.setattr(
            "src.services.deployment_detector.DEPLOYMENT_TYPE", "docker"
        )
        monkeypatch.delenv("DOCKER_CONTAINER", raising=False)

        with patch.object(Path, "exists", return_value=True):
            result = detector.detect_deployment_type()

        # Config should take priority
        assert result == DeploymentType.DOCKER

    def test_docker_env_takes_priority_over_unit_file(
        self, detector: DeploymentDetector, monkeypatch
    ) -> None:
        """Test that Docker environment takes priority over unit file."""
        monkeypatch.setattr("src.services.deployment_detector.DEPLOYMENT_TYPE", "")
        monkeypatch.setenv("DOCKER_CONTAINER", "1")

        with patch.object(Path, "exists", return_value=True):
            result = detector.detect_deployment_type()

        # Docker env should take priority over unit file
        assert result == DeploymentType.DOCKER

    # Test: Caching

    def test_detection_result_is_cached(
        self, detector: DeploymentDetector, monkeypatch
    ) -> None:
        """Test that detection result is cached after first call."""
        monkeypatch.setattr("src.services.deployment_detector.DEPLOYMENT_TYPE", "")
        monkeypatch.delenv("DOCKER_CONTAINER", raising=False)

        with patch.object(Path, "exists", return_value=True) as mock_exists:
            # First call
            result1 = detector.detect_deployment_type()
            # Reset mock call count
            mock_exists.reset_mock()
            # Second call should use cache
            result2 = detector.detect_deployment_type()

        # Results should be identical
        assert result1 == result2
        # Path.exists() should only be called once (first detection only)
        assert mock_exists.call_count == 0

    def test_reset_detection_clears_cache(
        self, detector: DeploymentDetector, monkeypatch
    ) -> None:
        """Test that reset_detection() clears the cache."""
        monkeypatch.setattr("src.services.deployment_detector.DEPLOYMENT_TYPE", "")
        monkeypatch.delenv("DOCKER_CONTAINER", raising=False)

        with patch.object(Path, "exists", return_value=True) as mock_exists:
            # First detection
            detector.detect_deployment_type()
            assert mock_exists.call_count == 1

            # Reset cache
            detector.reset_detection()

            # Second detection should check again
            detector.detect_deployment_type()
            assert mock_exists.call_count == 2

    # Test: Convenience Methods

    def test_get_deployment_type_calls_detect(
        self, detector: DeploymentDetector, monkeypatch
    ) -> None:
        """Test that get_deployment_type() is alias for detect_deployment_type()."""
        monkeypatch.setattr(
            "src.services.deployment_detector.DEPLOYMENT_TYPE", "docker"
        )

        result = detector.get_deployment_type()

        assert result == DeploymentType.DOCKER

    def test_is_systemd_returns_true_for_systemd(
        self, detector: DeploymentDetector, monkeypatch
    ) -> None:
        """Test that is_systemd() returns True for systemd deployment."""
        monkeypatch.setattr(
            "src.services.deployment_detector.DEPLOYMENT_TYPE", "systemd"
        )

        assert detector.is_systemd() is True
        assert detector.is_docker() is False

    def test_is_docker_returns_true_for_docker(
        self, detector: DeploymentDetector, monkeypatch
    ) -> None:
        """Test that is_docker() returns True for Docker deployment."""
        monkeypatch.setattr(
            "src.services.deployment_detector.DEPLOYMENT_TYPE", "docker"
        )

        assert detector.is_docker() is True
        assert detector.is_systemd() is False

    # Test: Service Identifier Generation

    def test_get_service_identifier_for_systemd(
        self, detector: DeploymentDetector, monkeypatch
    ) -> None:
        """Test that get_service_identifier() returns systemd format."""
        monkeypatch.setattr(
            "src.services.deployment_detector.DEPLOYMENT_TYPE", "systemd"
        )
        monkeypatch.setattr(
            "src.services.deployment_detector.SERVICE_NAME", "mnemodrome-mcp"
        )

        identifier = detector.get_service_identifier()

        assert identifier == "mnemodrome-mcp.service"

    def test_get_service_identifier_for_docker(
        self, detector: DeploymentDetector, monkeypatch
    ) -> None:
        """Test that get_service_identifier() returns Docker format."""
        monkeypatch.setattr(
            "src.services.deployment_detector.DEPLOYMENT_TYPE", "docker"
        )
        monkeypatch.setattr(
            "src.services.deployment_detector.DOCKER_CONTAINER_NAME", "mnemodrome-mcp"
        )

        identifier = detector.get_service_identifier()

        assert identifier == "mnemodrome-mcp"

    def test_get_service_identifier_uses_config_names(
        self, detector: DeploymentDetector, monkeypatch
    ) -> None:
        """Test that get_service_identifier() uses configured service names."""
        monkeypatch.setattr(
            "src.services.deployment_detector.DEPLOYMENT_TYPE", "systemd"
        )
        monkeypatch.setattr(
            "src.services.deployment_detector.SERVICE_NAME", "custom-service"
        )

        identifier = detector.get_service_identifier()

        assert identifier == "custom-service.service"

    # Test: Error Handling

    def test_detect_handles_case_insensitive_config(
        self, detector: DeploymentDetector, monkeypatch
    ) -> None:
        """Test that config values are case-insensitive."""
        monkeypatch.setattr(
            "src.services.deployment_detector.DEPLOYMENT_TYPE", "SYSTEMD"
        )

        result = detector.detect_deployment_type()

        assert result == DeploymentType.SYSTEMD

    def test_detect_handles_mixed_case_config(
        self, detector: DeploymentDetector, monkeypatch
    ) -> None:
        """Test that config handles mixed case values."""
        monkeypatch.setattr(
            "src.services.deployment_detector.DEPLOYMENT_TYPE", "DocKer"
        )

        result = detector.detect_deployment_type()

        assert result == DeploymentType.DOCKER


class TestDeploymentTypeEnum:
    """Tests for the DeploymentType enum."""

    def test_deployment_type_systemd_value(self) -> None:
        """Test SYSTEMD enum member has correct value."""
        assert DeploymentType.SYSTEMD.value == "systemd"

    def test_deployment_type_docker_value(self) -> None:
        """Test DOCKER enum member has correct value."""
        assert DeploymentType.DOCKER.value == "docker"

    def test_deployment_type_enum_creation_from_string(self) -> None:
        """Test creating enum from string value."""
        dt = DeploymentType("systemd")
        assert dt == DeploymentType.SYSTEMD

    def test_deployment_type_enum_invalid_value_raises_error(self) -> None:
        """Test that invalid enum value raises ValueError."""
        with pytest.raises(ValueError):
            DeploymentType("invalid")
