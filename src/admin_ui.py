"""
Admin Web UI for Mnemodrome MCP Server.

Feature 010: MCP Configuration Web Interface
Provides FastAPI-based web interface for configuration management and monitoring.
"""

import secrets
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import Body, Depends, FastAPI, HTTPException, Query, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from loguru import logger

from src import config
from src.services import admin_service
from src.services.admin_service import initialize_config_manager

# Create FastAPI app instance
app = FastAPI(
    title="Mnemodrome Admin UI",
    description="Configuration and monitoring interface for Mnemodrome MCP server",
    version="1.0.0",
)

# HTTP Basic Auth security
security = HTTPBasic()


def verify_admin(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """
    Verify admin credentials using HTTP Basic Auth.

    Args:
        credentials: HTTP Basic Auth credentials from request

    Returns:
        Username if authenticated

    Raises:
        HTTPException: 401 if credentials invalid
    """
    # Get correct credentials from config
    correct_username = config.ADMIN_USERNAME
    correct_password = config.ADMIN_PASSWORD

    # Use constant-time comparison to prevent timing attacks
    username_correct = secrets.compare_digest(
        credentials.username.encode("utf8"), correct_username.encode("utf8")
    )
    password_correct = secrets.compare_digest(
        credentials.password.encode("utf8"), correct_password.encode("utf8")
    )

    if not (username_correct and password_correct):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )

    return credentials.username


# Initialize ConfigurationManager on startup
@app.on_event("startup")
async def startup_event():
    """Initialize ConfigurationManager on app startup."""
    env_path = Path(".env")
    initialize_config_manager(env_path)
    logger.info(f"Admin UI started on port {config.ADMIN_WEB_PORT}")


@app.get("/admin/health-check")
async def health_check():
    """Simple health check endpoint (no auth required)."""
    return {"status": "healthy", "service": "Mnemodrome Admin UI", "version": "1.0.0"}


@app.get("/")
async def root():
    """Redirect root URL to admin interface."""
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/admin", status_code=307)


# ============================================================================
# Configuration Endpoints
# ============================================================================


@app.get("/admin/config")
async def get_config(admin: str = Depends(verify_admin)):
    """
    Get current configuration grouped by category.

    Requires authentication.

    Returns:
        Configuration data with version and categories
    """
    try:
        if admin_service.config_manager is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="ConfigurationManager not initialized",
            )

        config_data = admin_service.config_manager.read_config()
        return JSONResponse(content=config_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to read configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read configuration: {str(e)}",
        )


@app.post("/admin/config")
async def update_config(
    changes: dict[str, Any],
    config_version: float | None = None,
    admin: str = Depends(verify_admin),
):
    """
    Update configuration with validation and optimistic locking.

    Requires authentication.

    Args:
        changes: Dictionary of configuration changes
        config_version: Optional version for optimistic locking
        admin: Authenticated username

    Returns:
        Success response with applied changes
    """
    try:
        if admin_service.config_manager is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="ConfigurationManager not initialized",
            )

        # Optimistic locking check
        if config_version is not None:
            current_version = admin_service.config_manager._config_version
            if abs(current_version - config_version) > 0.001:  # Float comparison
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Configuration version mismatch. Current: {current_version}, Provided: {config_version}. Please reload and try again.",
                )

        # Validate changes
        errors = admin_service.config_manager.validate_config(changes)
        if errors:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"errors": errors, "message": "Configuration validation failed"},
            )

        # Apply runtime changes
        await admin_service.config_manager.apply_config(changes, actor=admin)

        # Persist to .env
        await admin_service.config_manager.persist_config(changes)

        return JSONResponse(
            content={
                "status": "success",
                "applied": changes,
                "warnings": [],
                "new_version": admin_service.config_manager._config_version,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update configuration: {str(e)}",
        )


@app.post("/admin/config/undo")
async def undo_config(admin: str = Depends(verify_admin)):
    """
    Undo last configuration change by restoring most recent backup.

    Requires authentication.

    Args:
        admin: Authenticated username

    Returns:
        Backup details and restore status
    """
    try:
        if admin_service.config_manager is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="ConfigurationManager not initialized",
            )

        # Restore backup
        result = await admin_service.config_manager.undo_last_change()

        # Log undo action
        try:
            from src.services.event_log_service import event_log_service

            if event_log_service:
                await event_log_service.log_event(
                    event_type="config_change",
                    action="undo_config",
                    actor=admin,
                    result="success",
                    metadata=result,
                )
        except Exception as log_error:
            logger.warning(f"Failed to log undo action: {log_error}")

        return JSONResponse(content=result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to undo configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to undo configuration: {str(e)}",
        )


@app.get("/admin/config/export")
async def export_config(admin: str = Depends(verify_admin)):
    """
    Export current configuration as .env file.

    Requires authentication.

    Args:
        admin: Authenticated username

    Returns:
        .env file download
    """
    try:
        env_path = Path(".env")
        if not env_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=".env file not found",
            )

        # Read .env file content
        env_content = env_path.read_text()

        # Return as downloadable file
        from fastapi.responses import Response

        return Response(
            content=env_content,
            media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename=.env"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export configuration: {str(e)}",
        )


# ============================================================================
# Health and Monitoring Endpoints
# ============================================================================


@app.get("/admin/health")
async def get_health(admin: str = Depends(verify_admin)):
    """
    Get system health status including event log, Neo4j, Redis, and MCP server.

    Requires authentication.

    Returns:
        System health metrics
    """
    try:
        health_data: dict[str, Any] = {}

        # Event Log health
        try:
            from src.services.event_log_service import event_log_service

            if event_log_service:
                event_log_health = await event_log_service.get_event_log_health()
                health_data["event_log"] = event_log_health
        except Exception as e:
            health_data["event_log"] = {"status": "error", "error": str(e)}

        # Neo4j status
        try:
            from src.services.neo4j_service import neo4j_service

            health_data["neo4j"] = {
                "connected": neo4j_service.driver is not None
                if neo4j_service
                else False,
                "uri": config.NEO4J_URI,
                "database": config.NEO4J_DATABASE,
            }
        except Exception as e:
            health_data["neo4j"] = {"status": "error", "error": str(e)}

        # Redis status (if available)
        try:
            from src.services.redis_service import redis_service

            if redis_service and hasattr(redis_service, "client"):
                health_data["redis"] = {
                    "connected": redis_service.client is not None
                    and redis_service._connected,
                    "host": config.REDIS_HOST,
                    "port": config.REDIS_PORT,
                }
        except Exception as e:
            health_data["redis"] = {"status": "error", "error": str(e)}

        # MCP Server info
        # If we're able to serve this health request, the MCP server is running
        health_data["mcp_server"] = {
            "connected": True,
            "version": config.MCP_VERSION,
            "port": config.MCP_PORT,
            "host": config.MCP_HOST,
        }

        return JSONResponse(content=health_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get health status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get health status: {str(e)}",
        )


# ============================================================================
# Service Management Endpoints
# ============================================================================

# Global ServiceManager instance (singleton)
_service_manager = None


def get_service_manager():
    """
    Get or create a ServiceManager singleton instance.

    Returns:
        ServiceManager instance
    """
    global _service_manager
    if _service_manager is None:
        from src.services.service_manager import ServiceManager

        _service_manager = ServiceManager()
    return _service_manager


@app.get("/admin/service/status")
async def get_service_status(admin: str = Depends(verify_admin)):
    """
    Get current service status including operational state and metrics.

    Requires authentication.

    Args:
        admin: Authenticated username

    Returns:
        ServiceStatus with state, uptime, metrics
    """
    try:
        manager = get_service_manager()
        service_status = await manager.get_status()

        logger.info(f"Service status retrieved by {admin}: {service_status['state']}")

        return JSONResponse(content=service_status)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get service status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get service status: {str(e)}",
        )


@app.post("/admin/service/restart")
async def restart_service(
    request_body: dict[str, Any] = Body(...), admin: str = Depends(verify_admin)
):
    """
    Restart the Mnemodrome service gracefully.

    Requires authentication and confirmation.

    Args:
        request_body: Request body with "confirm" field (must be true)
        admin: Authenticated username

    Returns:
        ControlOperationResult with operation status

    Raises:
        400 if confirm=false or missing
        504 on timeout
        500 on other errors
    """
    try:
        # Validate request body
        if not isinstance(request_body, dict) or "confirm" not in request_body:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing required field: confirm",
            )

        if request_body.get("confirm") is not True:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Confirmation required (confirm must be true)",
            )

        manager = get_service_manager()

        logger.info(f"Service restart requested by {admin}")

        try:
            # Call ServiceManager restart which logs to Event Log internally
            result = await manager.restart_service()
            return JSONResponse(content=result)

        except subprocess.TimeoutExpired:
            logger.error("Service restart timed out")
            raise HTTPException(
                status_code=504,  # Gateway Timeout used for operation timeout
                detail="Service restart operation timed out",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to restart service: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to restart service: {str(e)}",
        )


@app.post("/admin/service/stop")
async def stop_service(
    request_body: dict[str, Any] = Body(...), admin: str = Depends(verify_admin)
):
    """
    Stop the Mnemodrome service gracefully.

    Requires authentication and confirmation.

    Args:
        request_body: Request body with "confirm" field (must be true)
        admin: Authenticated username

    Returns:
        ControlOperationResult with operation status

    Raises:
        400 if confirm=false or missing
        500 on other errors
    """
    try:
        # Validate request body
        if not isinstance(request_body, dict) or "confirm" not in request_body:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing required field: confirm",
            )

        if request_body.get("confirm") is not True:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Confirmation required (confirm must be true)",
            )

        manager = get_service_manager()

        logger.info(f"Service stop requested by {admin}")

        try:
            # Call ServiceManager stop which logs to Event Log internally
            result = await manager.stop_service()
            return JSONResponse(content=result)

        except subprocess.TimeoutExpired:
            logger.error("Service stop timed out")
            raise HTTPException(
                status_code=504,  # Gateway Timeout used for operation timeout
                detail="Service stop operation timed out",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop service: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop service: {str(e)}",
        )


@app.get("/admin/service/logs")
async def get_service_logs(
    lines: int = Query(100, ge=1, le=1000),
    admin: str = Depends(verify_admin),
):
    """
    Retrieve service logs with specified number of lines.

    Requires authentication.

    Args:
        lines: Number of log lines to retrieve (1-1000, default 100)
        admin: Authenticated username

    Returns:
        LogsResponse with structured log entries

    Raises:
        400 if lines parameter is invalid
        504 on subprocess.TimeoutExpired
        500 on other exceptions
    """
    try:
        manager = get_service_manager()

        logger.info(f"Service logs requested by {admin}: {lines} lines")

        try:
            logs = await manager.get_logs(lines)
            return JSONResponse(content=logs)

        except subprocess.TimeoutExpired:
            logger.error(f"Service log retrieval timed out for {admin}")
            raise HTTPException(
                status_code=504,
                detail="Service log retrieval timed out",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve service logs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve service logs: {str(e)}",
        )


@app.get("/admin/service/sessions")
async def get_service_sessions(admin: str = Depends(verify_admin)):
    """
    Retrieve active session count information.

    Requires authentication.

    Note: For MVP, active session tracking returns None. This is a placeholder
    for future enhancement when session tracking is implemented.

    Args:
        admin: Authenticated username

    Returns:
        Dictionary with active_sessions and last_updated timestamp

    Raises:
        500 on exceptions
    """
    try:
        logger.info(f"Service sessions requested by {admin}")

        # MVP: Return None for active_sessions (future enhancement)
        return JSONResponse(
            content={
                "active_sessions": None,
                "last_updated": datetime.now().isoformat(),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve service sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve service sessions: {str(e)}",
        )


# ============================================================================
# Event Log Endpoints
# ============================================================================


@app.get("/admin/events")
async def get_events(
    limit: int = 100,
    event_type: str | None = None,
    admin: str = Depends(verify_admin),
):
    """
    Get recent events from event log with optional filtering.

    Requires authentication.

    Args:
        limit: Maximum number of events to return (default 100)
        event_type: Optional event type filter
        admin: Authenticated username

    Returns:
        List of events with metadata
    """
    try:
        from src.services.event_log_service import event_log_service

        if not event_log_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Event log service not available",
            )

        # Query events
        events = await event_log_service.query_events(
            event_type=event_type, limit=limit
        )

        return JSONResponse(
            content={
                "events": events,
                "total": len(events),
                "filters": {"event_type": event_type, "limit": limit},
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get events: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get events: {str(e)}",
        )


@app.post("/admin/events/cleanup")
async def cleanup_events(admin: str = Depends(verify_admin)):
    """
    Trigger event log cleanup (age-based and size-based).

    Requires authentication.

    Args:
        admin: Authenticated username

    Returns:
        Cleanup results with deleted counts
    """
    try:
        from src.services.event_log_service import event_log_service

        if not event_log_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Event log service not available",
            )

        # Trigger age-based cleanup
        deleted_age = await event_log_service.prune_old_events(
            config.EVENT_LOG_RETENTION_DAYS
        )

        # Trigger size-based cleanup
        deleted_size = await event_log_service.prune_by_size(
            config.EVENT_LOG_MAX_SIZE_MB
        )

        # Log cleanup action
        await event_log_service.log_event(
            event_type="admin_action",
            action="trigger_cleanup",
            actor=admin,
            result="success",
            metadata={
                "deleted_age_based": deleted_age,
                "deleted_size_based": deleted_size,
            },
        )

        return JSONResponse(
            content={
                "status": "success",
                "deleted_age_based": deleted_age,
                "deleted_size_based": deleted_size,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cleanup events: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup events: {str(e)}",
        )


# ============================================================================
# Static Files and HTML Pages
# ============================================================================


@app.get("/admin", response_class=HTMLResponse)
async def admin_index(admin: str = Depends(verify_admin)):
    """Admin UI home page."""
    try:
        html_path = Path("src/static/index.html")
        if not html_path.exists():
            return HTMLResponse(
                content="<h1>Mnemodrome Admin UI</h1><p>Coming soon...</p>",
                status_code=200,
            )
        return HTMLResponse(content=html_path.read_text())
    except Exception as e:
        logger.error(f"Failed to load index.html: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/config-editor", response_class=HTMLResponse)
async def config_editor(admin: str = Depends(verify_admin)):
    """Configuration editor page."""
    try:
        html_path = Path("src/static/config.html")
        if not html_path.exists():
            return HTMLResponse(
                content="<h1>Configuration Editor</h1><p>Coming soon...</p>",
                status_code=200,
            )
        return HTMLResponse(content=html_path.read_text())
    except Exception as e:
        logger.error(f"Failed to load config.html: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/health-dashboard", response_class=HTMLResponse)
async def health_dashboard(admin: str = Depends(verify_admin)):
    """Health monitoring dashboard page."""
    try:
        html_path = Path("src/static/health.html")
        if not html_path.exists():
            return HTMLResponse(
                content="<h1>Health Dashboard</h1><p>Coming soon...</p>",
                status_code=200,
            )
        return HTMLResponse(content=html_path.read_text())
    except Exception as e:
        logger.error(f"Failed to load health.html: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/event-viewer", response_class=HTMLResponse)
async def event_viewer(admin: str = Depends(verify_admin)):
    """Event log viewer page."""
    try:
        html_path = Path("src/static/events.html")
        if not html_path.exists():
            return HTMLResponse(
                content="<h1>Event Viewer</h1><p>Coming soon...</p>",
                status_code=200,
            )
        return HTMLResponse(content=html_path.read_text())
    except Exception as e:
        logger.error(f"Failed to load events.html: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/service", response_class=HTMLResponse)
async def service_management(admin: str = Depends(verify_admin)):
    """Service management page."""
    try:
        html_path = Path("src/static/service.html")
        if not html_path.exists():
            return HTMLResponse(
                content="<h1>Service Management</h1><p>Coming soon...</p>",
                status_code=200,
            )
        return HTMLResponse(content=html_path.read_text())
    except Exception as e:
        logger.error(f"Failed to load service.html: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Mount static files (CSS, JS, etc.)
# NOTE: This must come AFTER all other routes to avoid path conflicts
try:
    static_dir = Path("src/static")
    if static_dir.exists():
        app.mount(
            "/admin/static", StaticFiles(directory=str(static_dir)), name="static"
        )
        logger.info("Static files mounted at /admin/static")
except Exception as e:
    logger.warning(f"Failed to mount static files: {e}")
