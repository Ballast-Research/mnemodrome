"""Contract test fixtures — MCP tool contract validation."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_mcp_context():
    """Mock MCP server context for tool handler testing."""
    ctx = MagicMock()
    ctx.session = MagicMock()
    return ctx
