# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.0.x   | Yes                |
| < 1.0   | No                 |

## Reporting a Vulnerability

If you discover a security vulnerability in Mnemodrome, please report it privately. Do not open a public issue for security problems.

**Preferred**: Use GitHub's private vulnerability reporting. Open the repository's **Security** tab and click **Report a vulnerability**. This keeps the report confidential and tracked until a fix is released.

**Alternative**: Email security@icarlosmendez.info.

**Please include**:
- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Suggested fix (if any)

**Timeline**:
- **Acknowledgment**: Within 48 hours of report
- **Assessment**: Within 7 days
- **Fix**: Within 90 days for confirmed vulnerabilities
- **Disclosure**: Coordinated with reporter after fix is released

## Scope

**In scope**:
- MCP server code (`src/`)
- Docker configuration (`Dockerfile`, `docker-compose.yml`)
- Dependency vulnerabilities in pinned packages
- Authentication/authorization issues
- Data exposure through MCP tools

**Out of scope**:
- Vulnerabilities in third-party services (Neo4j, Redis, Qdrant, Ollama)
- Issues requiring physical access to the host
- Social engineering attacks
- Denial of service attacks against self-hosted instances

## Supply Chain Integrity

Mnemodrome is built to minimize supply-chain risk, and every point below is verifiable in this repository:

- **No Node/npm dependencies.** Mnemodrome is a pure Python project — there is no `package.json`, no `node_modules`, and nothing is fetched from the npm registry at build or run time. The entire class of npm-ecosystem worm attacks (self-propagating install scripts, typosquats, compromised transitive packages) structurally does not apply.
- **Reproducible, hash-verified builds.** The Docker image installs the exact dependency set from the committed [`uv.lock`](uv.lock) using `uv export --frozen … | uv pip install --require-hashes`. The build never re-resolves version ranges, so it cannot silently pull a newer — potentially compromised — release than the one that was locked and reviewed; each artifact is checked against its recorded hash.
- **Pinned images.** All service images are pinned to explicit versions: `python:3.11-slim`, `neo4j:5.26-community`, `qdrant/qdrant:v1.18.2`, `redis:7-alpine`, `postgres:16-alpine`.
- **Hash-pinned CI.** Every GitHub Actions step is pinned to a full commit SHA rather than a movable tag, closing the moved-tag attack vector.
- **Auditable.** `uv.lock` is committed and public; you can review the complete dependency graph and reproduce the build yourself.

Supply-chain issues are explicitly in scope — if you find a gap in any of the above, please report it via the process above.

## Security Best Practices for Operators

- Run Mnemodrome behind a firewall; do not expose ports 8765/8083 to the public internet
- Use strong passwords for Neo4j and Redis
- Keep Docker images and dependencies updated
- Review `.env` file permissions (should be readable only by the service user)
- Enable Neo4j authentication in production deployments
