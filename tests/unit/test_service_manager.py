"""
Unit tests for ServiceManager service.

Feature 014: Service Management UI Controls
Tests status retrieval, log parsing, and error handling with mocked subprocess.
"""

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from src.services.deployment_detector import DeploymentDetector, DeploymentType
from src.services.service_manager import (
    LogSeverity,
    ServiceManager,
    ServiceState,
)


class TestDeploymentDetector:
    """Tests for deployment type detection."""

    def test_detect_type_from_env_systemd(self):
        """Detect systemd when DEPLOYMENT_TYPE=systemd."""
        with patch.dict("os.environ", {"DEPLOYMENT_TYPE": "systemd"}):
            detector = DeploymentDetector()
            result = detector.detect_deployment_type()
            assert result == DeploymentType.SYSTEMD

    def test_detect_type_from_env_docker(self):
        """Detect Docker when DEPLOYMENT_TYPE=docker."""
        with patch("src.services.deployment_detector.DEPLOYMENT_TYPE", "docker"):
            detector = DeploymentDetector()
            result = detector.detect_deployment_type()
            assert result == DeploymentType.DOCKER

    def test_detect_type_from_docker_container_env(self):
        """Detect Docker when DOCKER_CONTAINER environment variable is set."""
        env = {"DOCKER_CONTAINER": "true", "DEPLOYMENT_TYPE": ""}
        with patch.dict("os.environ", env, clear=False):
            detector = DeploymentDetector()
            result = detector.detect_deployment_type()
            assert result == DeploymentType.DOCKER

    def test_detect_type_from_systemd_unit_file(self):
        """Detect systemd when unit file exists."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch.dict(
                "os.environ",
                {"DEPLOYMENT_TYPE": "", "DOCKER_CONTAINER": ""},
                clear=False,
            ):
                detector = DeploymentDetector()
                result = detector.detect_deployment_type()
                assert result == DeploymentType.SYSTEMD

    def test_detect_type_defaults_to_systemd(self):
        """Default to systemd when detection is inconclusive."""
        with patch("pathlib.Path.exists", return_value=False):
            with patch.dict(
                "os.environ",
                {"DEPLOYMENT_TYPE": "", "DOCKER_CONTAINER": ""},
                clear=False,
            ):
                detector = DeploymentDetector()
                result = detector.detect_deployment_type()
                assert result == DeploymentType.SYSTEMD


class TestServiceManagerStatusSystemd:
    """Tests for ServiceManager.get_status() in systemd deployment."""

    @pytest.mark.asyncio
    async def test_get_status_running_systemd(self):
        """Test get_status for running systemd service."""
        manager = ServiceManager()
        manager.deployment_type = DeploymentType.SYSTEMD

        systemctl_output = "active"
        systemctl_show_output = "MainPID=12345\n"

        with patch("subprocess.run") as mock_run:
            # First call: is-active
            # Second call: systemctl show
            # Both return successful results
            mock_run.side_effect = [
                MagicMock(
                    stdout=systemctl_output,
                    returncode=0,
                ),
                MagicMock(
                    stdout=systemctl_show_output,
                    returncode=0,
                ),
            ]

            with patch("psutil.Process") as mock_process:
                mock_proc_instance = MagicMock()
                mock_proc_instance.memory_info.return_value = MagicMock(
                    rss=100 * 1024 * 1024
                )
                mock_proc_instance.cpu_percent.return_value = 5.5
                mock_proc_instance.create_time.return_value = (
                    datetime.now(UTC).timestamp() - 3600
                )
                mock_process.return_value = mock_proc_instance

                status = await manager.get_status()

                assert status["state"] == ServiceState.RUNNING.value
                assert status["pid"] == 12345
                assert status["memory_mb"] == pytest.approx(100.0, rel=0.01)
                assert status["cpu_percent"] == 5.5
                assert status["uptime_seconds"] is not None
                assert status["uptime_seconds"] > 3500  # ~3600 - 60 second drift
                assert status["version"] == "1.0.0"
                assert status["deployment_type"] == "systemd"

    @pytest.mark.asyncio
    async def test_get_status_stopped_systemd(self):
        """Test get_status for stopped systemd service."""
        manager = ServiceManager()
        manager.deployment_type = DeploymentType.SYSTEMD

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="inactive",
                returncode=0,
            )

            status = await manager.get_status()

            assert status["state"] == ServiceState.STOPPED.value
            assert status["pid"] is None
            assert status["memory_mb"] is None
            assert status["uptime_seconds"] is None

    @pytest.mark.asyncio
    async def test_get_status_failed_systemd(self):
        """Test get_status when service is in failed state."""
        manager = ServiceManager()
        manager.deployment_type = DeploymentType.SYSTEMD

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="failed",
                returncode=0,
            )

            status = await manager.get_status()

            assert status["state"] == ServiceState.FAILED.value


class TestServiceManagerStatusDocker:
    """Tests for ServiceManager.get_status() in Docker deployment."""

    @pytest.mark.asyncio
    async def test_get_status_running_docker(self):
        """Test get_status for running Docker container."""
        manager = ServiceManager()
        manager.deployment_type = DeploymentType.DOCKER

        docker_output = json.dumps(
            [
                {
                    "State": {
                        "Running": True,
                        "Pid": 54321,
                        "Status": "running",
                    }
                }
            ]
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=docker_output,
                returncode=0,
            )

            with patch("psutil.Process") as mock_process:
                mock_proc_instance = MagicMock()
                mock_proc_instance.memory_info.return_value = MagicMock(
                    rss=200 * 1024 * 1024
                )
                mock_proc_instance.cpu_percent.return_value = 10.0
                mock_proc_instance.create_time.return_value = (
                    datetime.now(UTC).timestamp() - 7200
                )
                mock_process.return_value = mock_proc_instance

                status = await manager.get_status()

                assert status["state"] == ServiceState.RUNNING.value
                assert status["pid"] == 54321
                assert status["memory_mb"] == pytest.approx(200.0, rel=0.01)
                assert status["cpu_percent"] == 10.0
                assert status["deployment_type"] == "docker"

    @pytest.mark.asyncio
    async def test_get_status_stopped_docker(self):
        """Test get_status for stopped Docker container."""
        manager = ServiceManager()
        manager.deployment_type = DeploymentType.DOCKER

        docker_output = json.dumps(
            [
                {
                    "State": {
                        "Running": False,
                        "Pid": 0,
                        "Status": "exited (0)",
                    }
                }
            ]
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=docker_output,
                returncode=0,
            )

            status = await manager.get_status()

            assert status["state"] == ServiceState.STOPPED.value
            assert status["pid"] is None


class TestServiceManagerLogs:
    """Tests for ServiceManager.get_logs()."""

    @pytest.mark.asyncio
    async def test_get_logs_systemd(self):
        """Test get_logs for systemd journalctl."""
        manager = ServiceManager()
        manager.deployment_type = DeploymentType.SYSTEMD

        log_output = """Nov 21 12:34:56 localhost mnemodrome-mcp[1234]: INFO Starting MCP server on port 8765
Nov 21 12:35:00 localhost mnemodrome-mcp[1234]: WARNING Cache not initialized
Nov 21 12:35:05 localhost mnemodrome-mcp[1234]: ERROR Connection failed: timeout"""

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=log_output,
                returncode=0,
            )

            response = await manager.get_logs(lines=100)

            assert response["total"] == 3
            assert response["lines_requested"] == 100
            assert len(response["logs"]) == 3

            # Check parsed entries
            assert response["logs"][0]["severity"] == LogSeverity.INFO.value
            assert "Starting MCP server" in response["logs"][0]["message"]

            assert response["logs"][1]["severity"] == LogSeverity.WARNING.value
            assert "Cache not initialized" in response["logs"][1]["message"]

            assert response["logs"][2]["severity"] == LogSeverity.ERROR.value
            assert "Connection failed" in response["logs"][2]["message"]

    @pytest.mark.asyncio
    async def test_get_logs_docker(self):
        """Test get_logs for Docker logs."""
        manager = ServiceManager()
        manager.deployment_type = DeploymentType.DOCKER

        log_output = """2025-11-21T12:34:56.123456Z INFO MCP server started
2025-11-21T12:35:00.456789Z WARNING Memory usage high
2025-11-21T12:35:05.789012Z CRITICAL Service crash detected"""

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=log_output,
                returncode=0,
            )

            response = await manager.get_logs(lines=100)

            assert response["total"] == 3
            assert len(response["logs"]) == 3

            assert response["logs"][0]["severity"] == LogSeverity.INFO.value
            assert response["logs"][1]["severity"] == LogSeverity.WARNING.value
            assert response["logs"][2]["severity"] == LogSeverity.CRITICAL.value

    @pytest.mark.asyncio
    async def test_get_logs_invalid_lines_parameter(self):
        """Test get_logs with invalid lines parameter."""
        manager = ServiceManager()

        # Test lines = 0
        with pytest.raises(
            ValueError, match="lines must be an integer between 1 and 1000"
        ):
            await manager.get_logs(lines=0)

        # Test lines > 1000
        with pytest.raises(
            ValueError, match="lines must be an integer between 1 and 1000"
        ):
            await manager.get_logs(lines=1001)

    @pytest.mark.asyncio
    async def test_get_logs_subprocess_error(self):
        """Test get_logs when subprocess fails."""
        manager = ServiceManager()
        manager.deployment_type = DeploymentType.SYSTEMD

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("journalctl not found")

            response = await manager.get_logs(lines=100)

            assert response["total"] == 1
            assert response["logs"][0]["severity"] == LogSeverity.ERROR.value
            assert "Unexpected error" in response["logs"][0]["message"]

    @pytest.mark.asyncio
    async def test_get_logs_timeout(self):
        """Test get_logs when subprocess times out."""
        manager = ServiceManager()
        manager.deployment_type = DeploymentType.SYSTEMD

        with patch("subprocess.run") as mock_run:
            import subprocess

            mock_run.side_effect = subprocess.TimeoutExpired("journalctl", 10)

            response = await manager.get_logs(lines=100)

            assert response["total"] == 1
            assert response["logs"][0]["severity"] == LogSeverity.ERROR.value
            assert "timed out" in response["logs"][0]["message"]


class TestServiceManagerLogParsing:
    """Tests for log parsing logic."""

    def test_parse_logs_empty(self):
        """Test parsing empty log output."""
        manager = ServiceManager()
        response_logs = manager._parse_logs("")
        assert len(response_logs) == 0

    def test_parse_logs_systemd_format(self):
        """Test parsing systemd journalctl format."""
        manager = ServiceManager()
        log_line = "Nov 21 12:34:56 hostname service[1234]: ERROR Something went wrong"

        entries = manager._parse_logs(log_line)

        assert len(entries) == 1
        assert entries[0]["severity"] == LogSeverity.ERROR.value
        assert "Something went wrong" in entries[0]["message"]

    def test_parse_logs_docker_format(self):
        """Test parsing Docker log format."""
        manager = ServiceManager()
        log_line = "2025-11-21T12:34:56.123456Z DEBUG Starting component"

        entries = manager._parse_logs(log_line)

        assert len(entries) == 1
        assert entries[0]["severity"] == LogSeverity.DEBUG.value
        assert "Starting component" in entries[0]["message"]

    def test_extract_severity_error(self):
        """Test severity extraction for ERROR level."""
        manager = ServiceManager()

        assert (
            manager._extract_severity("ERROR: Something failed")
            == LogSeverity.ERROR.value
        )
        assert (
            manager._extract_severity("FATAL: Critical issue")
            == LogSeverity.ERROR.value
        )
        assert (
            manager._extract_severity("error in lowercase") == LogSeverity.ERROR.value
        )

    def test_extract_severity_warning(self):
        """Test severity extraction for WARNING level."""
        manager = ServiceManager()

        assert (
            manager._extract_severity("WARNING: Check this")
            == LogSeverity.WARNING.value
        )
        assert manager._extract_severity("WARN: Caution") == LogSeverity.WARNING.value
        assert manager._extract_severity("warning message") == LogSeverity.WARNING.value

    def test_extract_severity_critical(self):
        """Test severity extraction for CRITICAL level."""
        manager = ServiceManager()

        assert (
            manager._extract_severity("CRITICAL: System down")
            == LogSeverity.CRITICAL.value
        )
        assert (
            manager._extract_severity("PANIC: Immediate action")
            == LogSeverity.CRITICAL.value
        )

    def test_extract_severity_debug(self):
        """Test severity extraction for DEBUG level."""
        manager = ServiceManager()

        assert (
            manager._extract_severity("DEBUG: Variable x=5") == LogSeverity.DEBUG.value
        )
        assert manager._extract_severity("debug output") == LogSeverity.DEBUG.value

    def test_extract_severity_defaults_to_info(self):
        """Test severity extraction defaults to INFO."""
        manager = ServiceManager()

        assert manager._extract_severity("Service started") == LogSeverity.INFO.value
        assert manager._extract_severity("Hello world") == LogSeverity.INFO.value

    def test_parse_log_line_with_truncation(self):
        """Test that very long log messages are truncated."""
        manager = ServiceManager()
        long_message = "x" * 15000

        entry = manager._parse_log_line(long_message)

        assert entry is not None
        assert len(entry["message"]) == 10000


class TestServiceManagerErrorHandling:
    """Tests for error handling in ServiceManager."""

    @pytest.mark.asyncio
    async def test_get_status_subprocess_error(self):
        """Test get_status when subprocess fails."""
        manager = ServiceManager()
        manager.deployment_type = DeploymentType.SYSTEMD

        with patch("subprocess.run") as mock_run:
            import subprocess

            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=1, cmd="systemctl", stderr="Permission denied"
            )

            status = await manager.get_status()

            assert status["state"] == ServiceState.FAILED.value
            assert status["pid"] is None

    @pytest.mark.asyncio
    async def test_get_status_timeout(self):
        """Test get_status when subprocess times out."""
        manager = ServiceManager()
        manager.deployment_type = DeploymentType.SYSTEMD

        with patch("subprocess.run") as mock_run:
            import subprocess

            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd="systemctl", timeout=30
            )

            status = await manager.get_status()

            assert status["state"] == ServiceState.FAILED.value

    @pytest.mark.asyncio
    async def test_get_status_process_not_found(self):
        """Test get_status when process cannot be accessed."""
        manager = ServiceManager()
        manager.deployment_type = DeploymentType.SYSTEMD

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(stdout="active", returncode=0),
                MagicMock(stdout="MainPID=12345\n", returncode=0),
            ]

            with patch("psutil.Process") as mock_process:
                import psutil

                mock_process.side_effect = psutil.NoSuchProcess(12345)

                status = await manager.get_status()

                # Should still return running status despite process metrics failure
                assert status["state"] == ServiceState.RUNNING.value
                assert status["pid"] == 12345
                assert status["memory_mb"] is None

    @pytest.mark.asyncio
    async def test_get_status_access_denied_metrics(self):
        """Test get_status when process metrics are denied."""
        manager = ServiceManager()
        manager.deployment_type = DeploymentType.SYSTEMD

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(stdout="active", returncode=0),
                MagicMock(stdout="MainPID=12345\n", returncode=0),
            ]

            with patch("psutil.Process") as mock_process:
                import psutil

                mock_proc_instance = MagicMock()
                mock_proc_instance.memory_info.side_effect = psutil.AccessDenied()
                mock_process.return_value = mock_proc_instance

                status = await manager.get_status()

                # Should still return running status
                assert status["state"] == ServiceState.RUNNING.value
                assert status["memory_mb"] is None


class TestServiceManagerIntegration:
    """Integration tests with realistic subprocess output."""

    @pytest.mark.asyncio
    async def test_full_status_check_cycle(self):
        """Test complete status check flow with realistic data."""
        manager = ServiceManager()
        manager.deployment_type = DeploymentType.SYSTEMD

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(stdout="active", returncode=0),
                MagicMock(stdout="MainPID=5678\nType=simple\n", returncode=0),
            ]

            with patch("psutil.Process") as mock_process:
                mock_proc_instance = MagicMock()
                mock_proc_instance.memory_info.return_value = MagicMock(
                    rss=512 * 1024 * 1024
                )
                mock_proc_instance.cpu_percent.return_value = 2.3
                mock_proc_instance.create_time.return_value = (
                    datetime.now(UTC).timestamp() - 86400
                )
                mock_process.return_value = mock_proc_instance

                status = await manager.get_status()

                # Verify full response structure
                assert status["state"] == ServiceState.RUNNING.value
                assert status["pid"] == 5678
                assert status["memory_mb"] == 512.0
                assert status["cpu_percent"] == 2.3
                assert status["uptime_seconds"] > 86300
                assert status["version"] == "1.0.0"
                assert status["deployment_type"] == "systemd"
                assert "last_updated" in status
                assert status["service_name"] is not None

    @pytest.mark.asyncio
    async def test_full_logs_cycle_systemd(self):
        """Test complete log retrieval flow for systemd."""
        manager = ServiceManager()
        manager.deployment_type = DeploymentType.SYSTEMD

        complex_logs = """Nov 21 10:00:00 host mnemodrome[1000]: INFO Initializing memory service
Nov 21 10:00:01 host mnemodrome[1000]: DEBUG Connecting to Neo4j
Nov 21 10:00:02 host mnemodrome[1000]: INFO Neo4j connection established
Nov 21 10:00:05 host mnemodrome[1000]: WARNING Cache initialization slow
Nov 21 10:00:10 host mnemodrome[1000]: ERROR Failed to load entity cache
Nov 21 10:00:15 host mnemodrome[1000]: CRITICAL Service startup failed"""

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=complex_logs,
                returncode=0,
            )

            response = await manager.get_logs(lines=200)

            assert response["total"] == 6
            assert response["lines_requested"] == 200

            # Verify severity ordering
            severities = [entry["severity"] for entry in response["logs"]]
            assert severities == [
                "INFO",
                "DEBUG",
                "INFO",
                "WARNING",
                "ERROR",
                "CRITICAL",
            ]


class TestServiceManagerConfiguration:
    """Tests for ServiceManager configuration handling."""

    def test_service_manager_initialization(self):
        """Test ServiceManager initialization with configuration."""
        manager = ServiceManager()

        assert manager.service_name is not None
        assert manager.version == "1.0.0"
        assert manager.control_timeout > 0
        assert manager.log_timeout > 0
        assert manager.deployment_type in [
            DeploymentType.SYSTEMD,
            DeploymentType.DOCKER,
        ]

    def test_service_manager_timeouts(self):
        """Test that timeouts are configured properly."""
        manager = ServiceManager()

        # Timeouts should be reasonable values
        assert 10 <= manager.control_timeout <= 120
        assert 5 <= manager.log_timeout <= 60


class TestServiceManagerRestart:
    """Tests for ServiceManager.restart_service()."""

    @pytest.mark.asyncio
    async def test_restart_service_systemd(self):
        """Test restart_service for systemd deployment."""
        manager = ServiceManager()
        manager.deployment_type = DeploymentType.SYSTEMD

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="",
                returncode=0,
            )

            with patch("src.services.service_manager.asyncio.run") as mock_asyncio_run:
                # Mock the event log call to return an operation ID
                mock_asyncio_run.return_value = 123

                result = await manager.restart_service()

                assert result["status"] == "success"
                assert "restarted successfully" in result["message"]
                assert result["active_sessions"] is None

                # Verify subprocess was called correctly
                mock_run.assert_called_once()
                args = mock_run.call_args[0][0]
                assert "systemctl" in args
                assert "restart" in args

    @pytest.mark.asyncio
    async def test_restart_service_docker(self):
        """Test restart_service for Docker deployment."""
        manager = ServiceManager()
        manager.deployment_type = DeploymentType.DOCKER

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="",
                returncode=0,
            )

            with patch("src.services.service_manager.asyncio.run") as mock_asyncio_run:
                mock_asyncio_run.return_value = 456

                result = await manager.restart_service()

                assert result["status"] == "success"
                assert "restarted successfully" in result["message"]

                # Verify subprocess was called with docker
                mock_run.assert_called_once()
                args = mock_run.call_args[0][0]
                assert "docker" in args
                assert "restart" in args

    @pytest.mark.asyncio
    async def test_restart_service_error(self):
        """Test restart_service when subprocess fails."""
        manager = ServiceManager()
        manager.deployment_type = DeploymentType.SYSTEMD

        with patch("subprocess.run") as mock_run:
            import subprocess

            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=1, cmd="systemctl", stderr="Permission denied"
            )

            with patch("src.services.service_manager.asyncio.run") as mock_asyncio_run:
                # Mock the event log call for failure
                mock_asyncio_run.return_value = 789

                with pytest.raises(subprocess.CalledProcessError):
                    await manager.restart_service()

                # Verify event log was called with failure status
                # The mock should have been called to log the failure

    @pytest.mark.asyncio
    async def test_restart_service_timeout(self):
        """Test restart_service when subprocess times out."""
        manager = ServiceManager()
        manager.deployment_type = DeploymentType.SYSTEMD

        with patch("subprocess.run") as mock_run:
            import subprocess

            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd="systemctl", timeout=30
            )

            with patch("src.services.service_manager.asyncio.run") as mock_asyncio_run:
                # Mock the event log call for timeout
                mock_asyncio_run.return_value = 999

                with pytest.raises(subprocess.TimeoutExpired):
                    await manager.restart_service()


class TestServiceManagerStop:
    """Tests for ServiceManager.stop_service()."""

    @pytest.mark.asyncio
    async def test_stop_service_systemd(self):
        """Test stop_service for systemd deployment."""
        manager = ServiceManager()
        manager.deployment_type = DeploymentType.SYSTEMD

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="",
                returncode=0,
            )

            with patch("src.services.service_manager.asyncio.run") as mock_asyncio_run:
                # Mock the event log call to return an operation ID
                mock_asyncio_run.return_value = 111

                result = await manager.stop_service()

                assert result["status"] == "success"
                assert "stopped successfully" in result["message"]
                assert result["active_sessions"] is None

                # Verify subprocess was called correctly
                mock_run.assert_called_once()
                args = mock_run.call_args[0][0]
                assert "systemctl" in args
                assert "stop" in args

    @pytest.mark.asyncio
    async def test_stop_service_docker(self):
        """Test stop_service for Docker deployment."""
        manager = ServiceManager()
        manager.deployment_type = DeploymentType.DOCKER

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="",
                returncode=0,
            )

            with patch("src.services.service_manager.asyncio.run") as mock_asyncio_run:
                mock_asyncio_run.return_value = 222

                result = await manager.stop_service()

                assert result["status"] == "success"
                assert "stopped successfully" in result["message"]

                # Verify subprocess was called with docker
                mock_run.assert_called_once()
                args = mock_run.call_args[0][0]
                assert "docker" in args
                assert "stop" in args

    @pytest.mark.asyncio
    async def test_stop_service_error(self):
        """Test stop_service when subprocess fails."""
        manager = ServiceManager()
        manager.deployment_type = DeploymentType.SYSTEMD

        with patch("subprocess.run") as mock_run:
            import subprocess

            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=1, cmd="systemctl", stderr="Unit not found"
            )

            with patch("src.services.service_manager.asyncio.run") as mock_asyncio_run:
                # Mock the event log call for failure
                mock_asyncio_run.return_value = 333

                with pytest.raises(subprocess.CalledProcessError):
                    await manager.stop_service()

    @pytest.mark.asyncio
    async def test_stop_service_timeout(self):
        """Test stop_service when subprocess times out."""
        manager = ServiceManager()
        manager.deployment_type = DeploymentType.SYSTEMD

        with patch("subprocess.run") as mock_run:
            import subprocess

            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd="systemctl", timeout=30
            )

            with patch("src.services.service_manager.asyncio.run") as mock_asyncio_run:
                # Mock the event log call for timeout
                mock_asyncio_run.return_value = 444

                with pytest.raises(subprocess.TimeoutExpired):
                    await manager.stop_service()


class TestControlOperationResult:
    """Tests for ControlOperationResult structure."""

    def test_control_operation_result_success(self):
        """Test ControlOperationResult structure for success."""
        from src.services.service_manager import ControlOperationResult

        result = ControlOperationResult(
            status="success",
            message="Service restarted",
            operation_id=123,
            active_sessions=2,
        )

        assert result["status"] == "success"
        assert result["message"] == "Service restarted"
        assert result["operation_id"] == 123
        assert result["active_sessions"] == 2

    def test_control_operation_result_failure(self):
        """Test ControlOperationResult structure for failure."""
        from src.services.service_manager import ControlOperationResult

        result = ControlOperationResult(
            status="failure",
            message="Service restart failed: Permission denied",
            operation_id=456,
            active_sessions=None,
        )

        assert result["status"] == "failure"
        assert "Permission denied" in result["message"]
        assert result["operation_id"] == 456
        assert result["active_sessions"] is None
