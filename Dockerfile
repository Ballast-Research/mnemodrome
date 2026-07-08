# Mnemodrome MCP Server Dockerfile
# Python 3.11 base for compatibility with existing codebase

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install uv package manager
RUN pip install --no-cache-dir uv

# Copy dependency manifests first (for layer caching)
COPY pyproject.toml uv.lock ./

# Install the EXACT, hash-verified dependency set from the committed lockfile.
# Supply-chain hardening: the build never re-resolves version ranges, so it
# cannot silently pull a newer (potentially compromised) release than the one
# we locked and reviewed. --require-hashes verifies each artifact's integrity.
# Includes the [postgres] extra (asyncpg) so EVENT_LOG_BACKEND=postgres works
# out of the box.
RUN uv export --frozen --extra postgres --no-emit-project --no-dev \
        --format requirements-txt -o /tmp/requirements.lock \
 && uv pip install --system --require-hashes -r /tmp/requirements.lock \
 && rm -f /tmp/requirements.lock

# Copy application source code
COPY src/ ./src/
COPY .env.example .env

# Install the project itself for package metadata (drives MCP version reporting)
# WITHOUT re-resolving dependencies — they are already pinned + verified above.
RUN uv pip install --system --no-deps .

# Create directory for SQLite database (will be volume-mounted)
RUN mkdir -p /data/mnemodrome/events

# Create log directory
RUN mkdir -p /var/log/mnemodrome && \
    chmod 755 /var/log/mnemodrome

# Set environment variables for containerized deployment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=INFO \
    MCP_TRANSPORT=streamable-http

# Expose MCP (Streamable HTTP) transport and Admin UI ports
EXPOSE 8765 8083

# Health check — verify server is responding via dedicated /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -sf http://localhost:8765/health | grep -q '"status"' || exit 1

# Run MCP server via Python module execution
CMD ["python", "-m", "src.mcp_server"]
