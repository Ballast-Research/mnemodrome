"""
Service Management for Mnemodrome MCP service.

Feature 014: Service Management UI Controls
Provides ServiceManager for status monitoring, log retrieval, and service control.
"""

import asyncio
import re
import subprocess
import time
from datetime import UTC, datetime
from enum import Enum
from typing import TypedDict

import psutil
from loguru import logger

from src import config
from src.services.deployment_detector import DeploymentDetector, DeploymentType


class ServiceState(str, Enum):
    """Service operational states."""

    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    STARTING = "starting"
    STOPPING = "stopping"


class LogSeverity(str, Enum):
    """Log severity levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ServiceStatus(TypedDict):
    """Service status response structure."""

    state: str
    uptime_seconds: int | None
    version: str
    pid: int | None
    memory_mb: float | None
    cpu_percent: float | None
    deployment_type: str
    service_name: str
    last_updated: str


class LogEntry(TypedDict):
    """Log entry structure."""

    timestamp: str
    severity: str
    message: str
    source: str | None


class LogsResponse(TypedDict):
    """Logs response structure."""

    logs: list[LogEntry]
    total: int
    lines_requested: int


class ControlOperationResult(TypedDict):
    """Control operation result structure."""

    status: str  # "success" or "failure"
    message: str
    operation_id: int | None
    active_sessions: int | None


class ServiceManager:
    """
    Manages Mnemodrome service status, control, and log retrieval.

    Responsibilities:
    - Query service status via systemctl or docker commands
    - Retrieve process metrics using psutil
    - Fetch and parse service logs
    - Control service operations (restart, stop, start)
    """

    def __init__(self) -> None:
        """Initialize ServiceManager with configuration."""
        self.detector = DeploymentDetector()
        self.deployment_type = self.detector.detect_deployment_type()
        self.service_name = config.SERVICE_NAME
        self.version = config.MCP_VERSION
        # Get timeouts from config, with fallbacks if not present
        self.control_timeout = getattr(config, "SERVICE_CONTROL_TIMEOUT", 30)
        self.log_timeout = getattr(config, "LOG_RETRIEVAL_TIMEOUT", 10)

        logger.info(
            f"ServiceManager initialized: {self.deployment_type.value} "
            f"({self.service_name})"
        )

    async def get_status(self) -> ServiceStatus:
        """
        Get current service status.

        Returns:
            ServiceStatus: Current operational status with metrics

        Raises:
            subprocess.CalledProcessError: If status command fails
            subprocess.TimeoutExpired: If command times out
        """
        # Run status check in thread pool to avoid blocking event loop
        return await asyncio.to_thread(self._get_status_sync)

    def _get_status_sync(self) -> ServiceStatus:
        """
        Synchronous service status retrieval.

        Returns:
            ServiceStatus: Current operational status with metrics
        """
        state = ServiceState.STOPPED
        pid: int | None = None
        uptime_seconds: int | None = None
        memory_mb: float | None = None
        cpu_percent: float | None = None

        try:
            if self.deployment_type == DeploymentType.SYSTEMD:
                pid, state = self._get_systemd_status()
            else:  # Docker
                pid, state = self._get_docker_status()

            # Get process metrics if running
            if pid and state == ServiceState.RUNNING:
                try:
                    process = psutil.Process(pid)
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / (1024 * 1024)
                    cpu_percent = process.cpu_percent(interval=0.1)
                    uptime_seconds = int(time.time() - process.create_time())
                except psutil.NoSuchProcess:
                    logger.warning(f"Process {pid} not found for metrics")
                except (psutil.AccessDenied, psutil.TimeoutExpired) as e:
                    logger.warning(f"Cannot access process metrics: {e}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Status check failed: {e.stderr}")
            state = ServiceState.FAILED
        except subprocess.TimeoutExpired:
            logger.error("Status check timed out")
            state = ServiceState.FAILED
        except Exception as e:
            logger.error(f"Unexpected error in status check: {e}")
            state = ServiceState.FAILED

        return ServiceStatus(
            state=state.value,
            uptime_seconds=uptime_seconds,
            version=self.version,
            pid=pid,
            memory_mb=memory_mb,
            cpu_percent=cpu_percent,
            deployment_type=self.deployment_type.value,
            service_name=self.service_name,
            last_updated=datetime.now(UTC).isoformat(),
        )

    def _get_systemd_status(self) -> tuple[int | None, ServiceState]:
        """
        Query systemd service status.

        Returns:
            Tuple of (PID, ServiceState)

        Raises:
            subprocess.CalledProcessError: If status command fails
            subprocess.TimeoutExpired: If command times out
        """
        # Get service state
        result = subprocess.run(
            ["systemctl", "is-active", self.service_name],
            capture_output=True,
            text=True,
            timeout=self.control_timeout,
        )

        state_output = result.stdout.strip().lower()
        state_map = {
            "active": ServiceState.RUNNING,
            "inactive": ServiceState.STOPPED,
            "failed": ServiceState.FAILED,
            "activating": ServiceState.STARTING,
            "deactivating": ServiceState.STOPPING,
        }
        state = state_map.get(state_output, ServiceState.FAILED)

        # Get PID for running service
        pid: int | None = None
        if state == ServiceState.RUNNING:
            try:
                result = subprocess.run(
                    ["systemctl", "show", self.service_name, "-p", "MainPID"],
                    capture_output=True,
                    text=True,
                    timeout=self.control_timeout,
                    check=True,
                )
                # Parse "MainPID=1234"
                match = re.search(r"MainPID=(\d+)", result.stdout)
                if match:
                    pid = int(match.group(1))
            except (subprocess.CalledProcessError, ValueError) as e:
                logger.warning(f"Cannot retrieve systemd PID: {e}")

        return pid, state

    def _get_docker_status(self) -> tuple[int | None, ServiceState]:
        """
        Query Docker container status.

        Returns:
            Tuple of (PID, ServiceState)

        Raises:
            subprocess.CalledProcessError: If status command fails
            subprocess.TimeoutExpired: If command times out
        """
        try:
            result = subprocess.run(
                ["docker", "inspect", self.service_name],
                capture_output=True,
                text=True,
                timeout=self.control_timeout,
                check=True,
            )

            import json

            container_info = json.loads(result.stdout)[0]
            state_obj = container_info.get("State", {})

            # Determine state
            if state_obj.get("Running"):
                state = ServiceState.RUNNING
            elif state_obj.get("Status", "").lower().startswith("exited"):
                state = ServiceState.STOPPED
            else:
                state = ServiceState.FAILED

            # Get PID
            pid = state_obj.get("Pid")
            if pid == 0:
                pid = None

            return pid, state

        except (subprocess.CalledProcessError, ValueError, KeyError, IndexError) as e:
            logger.error(f"Docker status check failed: {e}")
            return None, ServiceState.FAILED

    async def get_logs(self, lines: int = 100) -> LogsResponse:
        """
        Retrieve service logs.

        Args:
            lines: Number of log lines to retrieve (1-1000, default 100)

        Returns:
            LogsResponse: Structured log entries

        Raises:
            ValueError: If lines parameter is invalid
            subprocess.CalledProcessError: If log command fails
            subprocess.TimeoutExpired: If command times out
        """
        # Validate lines parameter
        if not isinstance(lines, int) or lines < 1 or lines > 1000:
            raise ValueError("lines must be an integer between 1 and 1000")

        # Run log retrieval in thread pool
        return await asyncio.to_thread(self._get_logs_sync, lines)

    def _get_logs_sync(self, lines: int) -> LogsResponse:
        """
        Synchronous log retrieval.

        Args:
            lines: Number of log lines to retrieve

        Returns:
            LogsResponse: Structured log entries
        """
        log_entries: list[LogEntry] = []

        try:
            if self.deployment_type == DeploymentType.SYSTEMD:
                raw_logs = self._get_systemd_logs(lines)
            else:  # Docker
                raw_logs = self._get_docker_logs(lines)

            # Parse logs into structured format
            log_entries = self._parse_logs(raw_logs)

        except subprocess.CalledProcessError as e:
            logger.error(f"Log retrieval failed: {e.stderr}")
            log_entries = [
                LogEntry(
                    timestamp=datetime.now(UTC).isoformat(),
                    severity=LogSeverity.ERROR.value,
                    message=f"Failed to retrieve logs: {e.stderr.strip()[:200]}",
                    source=None,
                )
            ]
        except subprocess.TimeoutExpired:
            logger.error("Log retrieval timed out")
            log_entries = [
                LogEntry(
                    timestamp=datetime.now(UTC).isoformat(),
                    severity=LogSeverity.ERROR.value,
                    message="Log retrieval timed out (logs may be too large)",
                    source=None,
                )
            ]
        except Exception as e:
            logger.error(f"Unexpected error in log retrieval: {e}")
            log_entries = [
                LogEntry(
                    timestamp=datetime.now(UTC).isoformat(),
                    severity=LogSeverity.ERROR.value,
                    message=f"Unexpected error: {str(e)[:200]}",
                    source=None,
                )
            ]

        return LogsResponse(
            logs=log_entries, total=len(log_entries), lines_requested=lines
        )

    def _get_systemd_logs(self, lines: int) -> str:
        """
        Retrieve systemd service logs.

        Args:
            lines: Number of lines to retrieve

        Returns:
            Raw log output from journalctl

        Raises:
            subprocess.CalledProcessError: If journalctl fails
            subprocess.TimeoutExpired: If command times out
        """
        result = subprocess.run(
            ["journalctl", "-u", self.service_name, "-n", str(lines), "--no-pager"],
            capture_output=True,
            text=True,
            timeout=self.log_timeout,
            check=True,
        )
        return result.stdout

    def _get_docker_logs(self, lines: int) -> str:
        """
        Retrieve Docker container logs.

        Args:
            lines: Number of lines to retrieve

        Returns:
            Raw log output from docker logs

        Raises:
            subprocess.CalledProcessError: If docker logs fails
            subprocess.TimeoutExpired: If command times out
        """
        result = subprocess.run(
            ["docker", "logs", self.service_name, "--tail", str(lines)],
            capture_output=True,
            text=True,
            timeout=self.log_timeout,
            check=True,
        )
        return result.stdout

    def _parse_logs(self, raw_logs: str) -> list[LogEntry]:
        """
        Parse raw log output into structured LogEntry objects.

        Supports both systemd (journalctl) and Docker log formats.

        Args:
            raw_logs: Raw log output from systemd or Docker

        Returns:
            List of LogEntry objects with parsed timestamp, severity, message
        """
        entries: list[LogEntry] = []

        for line in raw_logs.strip().split("\n"):
            if not line.strip():
                continue

            try:
                entry = self._parse_log_line(line)
                if entry:
                    entries.append(entry)
            except Exception as e:
                # On parse error, create entry with unparsed content
                logger.debug(f"Cannot parse log line: {e}")
                entries.append(
                    LogEntry(
                        timestamp=datetime.now(UTC).isoformat(),
                        severity=LogSeverity.INFO.value,
                        message=line[:1000],  # Truncate very long lines
                        source=None,
                    )
                )

        return entries

    def _parse_log_line(self, line: str) -> LogEntry | None:
        """
        Parse a single log line into structured format.

        Detects and handles multiple log formats:
        - Systemd journalctl (human-readable format with timestamps)
        - Docker logs (various formats, often JSON or simple text)
        - Generic syslog format

        Args:
            line: Single log line to parse

        Returns:
            LogEntry with parsed components, or None if unparseable
        """
        if not line.strip():
            return None

        # Try systemd format first: "Nov 21 12:34:56 hostname service: message"
        systemd_pattern = r"(\w+ +\d+ +\d+:\d+:\d+)\s+\S+\s+\S+\[?\d*\]?:\s*(.*)"
        match = re.match(systemd_pattern, line)
        if match:
            timestamp_str, message = match.groups()
            severity = self._extract_severity(message)
            try:
                # Parse month day time format
                timestamp = self._parse_timestamp(timestamp_str)
            except Exception:
                timestamp = datetime.now(UTC).isoformat()

            return LogEntry(
                timestamp=timestamp,
                severity=severity,
                message=message[:10000],
                source="systemd",
            )

        # Try Docker format: "2025-11-21T12:34:56.123456Z message"
        docker_pattern = r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)\s+(.*)"
        match = re.match(docker_pattern, line)
        if match:
            timestamp_str, message = match.groups()
            severity = self._extract_severity(message)
            try:
                # ISO format timestamp
                timestamp = (
                    datetime.fromisoformat(timestamp_str.rstrip("Z"))
                    .replace(tzinfo=UTC)
                    .isoformat()
                )
            except Exception:
                timestamp = datetime.now(UTC).isoformat()

            return LogEntry(
                timestamp=timestamp,
                severity=severity,
                message=message[:10000],
                source="docker",
            )

        # Fallback: treat whole line as message
        severity = self._extract_severity(line)
        return LogEntry(
            timestamp=datetime.now(UTC).isoformat(),
            severity=severity,
            message=line[:10000],
            source=None,
        )

    def _extract_severity(self, message: str) -> str:
        """
        Extract severity level from log message text.

        Args:
            message: Log message text

        Returns:
            Severity level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        message_upper = message.upper()

        if "ERROR" in message_upper or "FATAL" in message_upper:
            return LogSeverity.ERROR.value
        if "CRITICAL" in message_upper or "PANIC" in message_upper:
            return LogSeverity.CRITICAL.value
        if "WARNING" in message_upper or "WARN" in message_upper:
            return LogSeverity.WARNING.value
        if "DEBUG" in message_upper:
            return LogSeverity.DEBUG.value

        return LogSeverity.INFO.value

    def _parse_timestamp(self, timestamp_str: str) -> str:
        """
        Parse systemd timestamp format into ISO format.

        Systemd format: "Nov 21 12:34:56"
        Assumes current year, converts to UTC ISO format.

        Args:
            timestamp_str: Timestamp string in systemd format

        Returns:
            ISO format timestamp string with UTC timezone
        """
        import datetime as dt

        # Parse "Nov 21 12:34:56"
        parsed = datetime.strptime(
            f"{timestamp_str} {dt.datetime.now().year}", "%b %d %H:%M:%S %Y"
        ).replace(tzinfo=dt.UTC)
        return parsed.isoformat()

    async def restart_service(self) -> ControlOperationResult:
        """
        Restart the service gracefully.

        Logs restart operation to Event Log with actor, timestamp, and result.

        Returns:
            ControlOperationResult dict with status, message, and operation_id

        Raises:
            subprocess.CalledProcessError: If restart command fails
            subprocess.TimeoutExpired: If command times out
        """
        return await asyncio.to_thread(self._restart_service_sync)

    def _restart_service_sync(self) -> ControlOperationResult:
        """
        Synchronous service restart with Event Log integration.

        Returns:
            ControlOperationResult dict with status, message, and operation_id

        Raises:
            subprocess.CalledProcessError: If restart command fails
            subprocess.TimeoutExpired: If command times out
        """
        operation_id: int | None = None

        try:
            # Execute restart command based on deployment type
            if self.deployment_type == DeploymentType.SYSTEMD:
                result = subprocess.run(
                    ["sudo", "systemctl", "restart", self.service_name],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=self.control_timeout,
                )
            else:  # Docker
                result = subprocess.run(
                    ["sudo", "docker", "restart", self.service_name],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=self.control_timeout,
                )

            logger.info(f"Service {self.service_name} restarted successfully")

            # Log success to Event Log (synchronously for now)
            try:
                import asyncio as aio

                from src.services.event_log_service import event_log_service

                # Create a temporary event loop if needed for sync context
                loop = None
                try:
                    loop = aio.get_running_loop()
                except RuntimeError:
                    loop = None

                if event_log_service:
                    if loop:
                        # We're in async context, but this is sync function
                        # Log asynchronously after restart completes
                        # (Fire-and-forget approach for non-blocking behavior)
                        operation_id = aio.run(
                            event_log_service.log_event(
                                event_type="service_control",
                                action="restart",
                                actor="admin",
                                result="success",
                                metadata={
                                    "deployment_type": self.deployment_type.value,
                                    "service_name": self.service_name,
                                },
                            )
                        )
                    else:
                        # Fallback: create new event loop in thread
                        operation_id = aio.run(
                            event_log_service.log_event(
                                event_type="service_control",
                                action="restart",
                                actor="admin",
                                result="success",
                                metadata={
                                    "deployment_type": self.deployment_type.value,
                                    "service_name": self.service_name,
                                },
                            )
                        )
            except Exception as log_error:
                logger.warning(
                    f"Failed to log restart operation to Event Log: {log_error}"
                )

            return ControlOperationResult(
                status="success",
                message=f"Service {self.service_name} restarted successfully",
                operation_id=operation_id,
                active_sessions=None,
            )

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip() if e.stderr else f"Return code: {e.returncode}"
            logger.error(f"Service restart failed: {error_msg}")

            # Log failure to Event Log
            try:
                import asyncio as aio

                from src.services.event_log_service import event_log_service

                if event_log_service:
                    operation_id = aio.run(
                        event_log_service.log_event(
                            event_type="service_control",
                            action="restart",
                            actor="admin",
                            result="failure",
                            metadata={
                                "deployment_type": self.deployment_type.value,
                                "service_name": self.service_name,
                                "error": error_msg,
                                "returncode": e.returncode,
                            },
                        )
                    )
            except Exception as log_error:
                logger.warning(f"Failed to log restart failure: {log_error}")

            raise

        except subprocess.TimeoutExpired:
            timeout_msg = f"Service restart timed out after {self.control_timeout}s"
            logger.error(timeout_msg)

            # Log timeout to Event Log
            try:
                import asyncio as aio

                from src.services.event_log_service import event_log_service

                if event_log_service:
                    operation_id = aio.run(
                        event_log_service.log_event(
                            event_type="service_control",
                            action="restart",
                            actor="admin",
                            result="failure",
                            metadata={
                                "deployment_type": self.deployment_type.value,
                                "service_name": self.service_name,
                                "error": "Operation timed out",
                                "timeout_seconds": self.control_timeout,
                            },
                        )
                    )
            except Exception as log_error:
                logger.warning(f"Failed to log restart timeout: {log_error}")

            raise

    async def stop_service(self) -> ControlOperationResult:
        """
        Stop the service gracefully.

        Logs stop operation to Event Log with actor, timestamp, and result.

        Returns:
            ControlOperationResult dict with status, message, and operation_id

        Raises:
            subprocess.CalledProcessError: If stop command fails
            subprocess.TimeoutExpired: If command times out
        """
        return await asyncio.to_thread(self._stop_service_sync)

    def _stop_service_sync(self) -> ControlOperationResult:
        """
        Synchronous service stop with Event Log integration.

        Returns:
            ControlOperationResult dict with status, message, and operation_id

        Raises:
            subprocess.CalledProcessError: If stop command fails
            subprocess.TimeoutExpired: If command times out
        """
        operation_id: int | None = None

        try:
            # Execute stop command based on deployment type
            if self.deployment_type == DeploymentType.SYSTEMD:
                result = subprocess.run(
                    ["sudo", "systemctl", "stop", self.service_name],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=self.control_timeout,
                )
            else:  # Docker
                result = subprocess.run(
                    ["sudo", "docker", "stop", self.service_name],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=self.control_timeout,
                )

            logger.info(f"Service {self.service_name} stopped successfully")

            # Log success to Event Log
            try:
                import asyncio as aio

                from src.services.event_log_service import event_log_service

                if event_log_service:
                    operation_id = aio.run(
                        event_log_service.log_event(
                            event_type="service_control",
                            action="stop",
                            actor="admin",
                            result="success",
                            metadata={
                                "deployment_type": self.deployment_type.value,
                                "service_name": self.service_name,
                            },
                        )
                    )
            except Exception as log_error:
                logger.warning(
                    f"Failed to log stop operation to Event Log: {log_error}"
                )

            return ControlOperationResult(
                status="success",
                message=f"Service {self.service_name} stopped successfully",
                operation_id=operation_id,
                active_sessions=None,
            )

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip() if e.stderr else f"Return code: {e.returncode}"
            logger.error(f"Service stop failed: {error_msg}")

            # Log failure to Event Log
            try:
                import asyncio as aio

                from src.services.event_log_service import event_log_service

                if event_log_service:
                    operation_id = aio.run(
                        event_log_service.log_event(
                            event_type="service_control",
                            action="stop",
                            actor="admin",
                            result="failure",
                            metadata={
                                "deployment_type": self.deployment_type.value,
                                "service_name": self.service_name,
                                "error": error_msg,
                                "returncode": e.returncode,
                            },
                        )
                    )
            except Exception as log_error:
                logger.warning(f"Failed to log stop failure: {log_error}")

            raise

        except subprocess.TimeoutExpired:
            timeout_msg = f"Service stop timed out after {self.control_timeout}s"
            logger.error(timeout_msg)

            # Log timeout to Event Log
            try:
                import asyncio as aio

                from src.services.event_log_service import event_log_service

                if event_log_service:
                    operation_id = aio.run(
                        event_log_service.log_event(
                            event_type="service_control",
                            action="stop",
                            actor="admin",
                            result="failure",
                            metadata={
                                "deployment_type": self.deployment_type.value,
                                "service_name": self.service_name,
                                "error": "Operation timed out",
                                "timeout_seconds": self.control_timeout,
                            },
                        )
                    )
            except Exception as log_error:
                logger.warning(f"Failed to log stop timeout: {log_error}")

            raise
