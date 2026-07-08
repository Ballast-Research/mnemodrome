"""Integration test fixtures — auto-skip when services unavailable."""

import os
import socket

import pytest


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a TCP port is accepting connections."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((host, port))
        s.close()
        return True
    except (TimeoutError, ConnectionRefusedError, OSError):
        return False


@pytest.fixture(scope="session")
def neo4j_available():
    """Check if Neo4j is reachable. Skip test if not."""
    host = os.getenv("NEO4J_HOST", "localhost")
    port = int(os.getenv("NEO4J_BOLT_PORT", "7687"))
    if not _port_open(host, port):
        pytest.skip(f"Neo4j not available at {host}:{port}")
    return True


@pytest.fixture(scope="session")
def redis_available():
    """Check if Redis is reachable. Skip test if not."""
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    if not _port_open(host, port):
        pytest.skip(f"Redis not available at {host}:{port}")
    return True


@pytest.fixture(scope="session")
def qdrant_available():
    """Check if Qdrant is reachable. Skip test if not."""
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    if not _port_open(host, port):
        pytest.skip(f"Qdrant not available at {host}:{port}")
    return True
