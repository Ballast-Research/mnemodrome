Status: active · Owner: the maintainers · Verified: 2026-11-22 · Verify-on: change to src/mcp_server.py, src/config.py, docker-compose.yml or docs/release/versioning-policy.md

# Architecture

The components, the data stores, their boundaries, and the public API surface.
Detail lives in the pages linked below; this page does not repeat it.

## Rules

1. Six memory layers, each in its own store: graph (Neo4j), vector (Qdrant),
   file (Markdown), event log (SQLite, or PostgreSQL opt-in), procedural, and
   cache (Redis). The layer table in the README is the reference.
   (2026-11-22, [README "Architecture"](../../README.md#architecture))
2. MCP is the product interface. MCP tools describe themselves: the docstrings
   in `src/mcp_server.py` are the tool documentation.
   (2026-11-22, [CONTRIBUTING.md "User Documentation"](../../CONTRIBUTING.md#user-documentation))
3. The public API surface is exactly five categories: MCP tools, HTTP routes,
   configuration, CLI, stored data. A change to any of them is an API change.
   (2026-11-18, [ADR-002](../decisions/ADR-002-semver-and-api-surface.md),
   [versioning policy](../release/versioning-policy.md#public-api-surface))
4. New tools and environment variables start `experimental` for at least one
   minor release. (2026-11-18, [ADR-002](../decisions/ADR-002-semver-and-api-surface.md))
5. Never delete, only supersede: a changed fact invalidates the old one with a
   reason and keeps its history. (2026-11-22, [PHILOSOPHY.md](../product/PHILOSOPHY.md))
6. A newer minor or patch starts on older stored data with no manual step.
   A data change that needs a manual migration is a major release.
   (2026-11-18, [versioning policy](../release/versioning-policy.md#bump-rules))
7. Every vector collection records the embedder that built it. Switching the
   embedding provider on existing data goes through a reindex, never in place.
   (2026-11-21, [INSTALLATION.md](../../INSTALLATION.md#switching-provider-on-existing-data))

## Procedures

- Change an MCP tool, variable, route, CLI flag or stored data:
  [versioning policy](../release/versioning-policy.md), plus a CHANGELOG entry.
- A generated listing of the API surface is planned, checked by a script.
  Until it exists, the README sections below are the listing.

## Truth

- Overview and layer table: [README "Architecture"](../../README.md#architecture).
- Tool catalog: [README "MCP Tools"](../../README.md#mcp-tools); code in
  [`src/mcp_server.py`](../../src/mcp_server.py).
- Configuration: [README "Configuration"](../../README.md#configuration),
  [`.env.example`](../../.env.example), [`src/config.py`](../../src/config.py).
- Services and stores: [`src/services/`](../../src/services/),
  [`docker-compose.yml`](../../docker-compose.yml),
  [`scripts/init-event-logs.sql`](../../scripts/init-event-logs.sql).
- CLI: [`src/cli/`](../../src/cli/).
