"""The product version has one source: pyproject.toml, read via package metadata."""

import tomllib
from importlib.metadata import version
from pathlib import Path

from fastapi.testclient import TestClient

from src import config
from src.admin_ui import app

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _pyproject_version() -> str:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["project"]["version"]


def test_installed_metadata_matches_pyproject():
    assert version("mnemodrome") == _pyproject_version()


def test_mcp_version_matches_pyproject():
    assert config.MCP_VERSION == _pyproject_version()


def test_admin_health_reports_package_version():
    response = TestClient(app).get("/admin/health-check")
    assert response.status_code == 200
    assert response.json()["version"] == _pyproject_version()
    assert app.version == _pyproject_version()
