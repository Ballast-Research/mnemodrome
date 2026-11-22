Status: active · Owner: the maintainers · Verified: 2026-11-22 · Verify-on: change to INSTALLATION.md, docker-compose.yml, install.sh, .github/workflows/release.yml or SECURITY.md

# Operations

Running Mnemodrome yourself: install, upgrade, verify, troubleshoot.
This page covers self-hosters. The maintainers' own instance, its deploys and
its incidents are not described here: maintainers use the overlay.

## Rules

1. Run released images, built only from `v*` tags on GitHub. Pre-releases
   (`rc`) never move the `latest` image tag.
   (2026-11-18, [ADR-001](../decisions/ADR-001-github-single-upstream.md),
   [release](../release/README.md))
2. Within a major version, an upgrade starts on older stored data with no
   manual step. Read the CHANGELOG `Upgrade note:` lines before you upgrade.
   (2026-11-18, [versioning policy](../release/versioning-policy.md#bump-rules))
3. Do not expose the MCP or Admin UI ports to the public internet. Put a TLS
   reverse proxy in front when clients cross a network you do not trust.
   (2026-11-22, [SECURITY.md](../../SECURITY.md#security-best-practices-for-operators))
4. Supported versions are listed in SECURITY.md.
   (2026-11-22, [SECURITY.md](../../SECURITY.md#supported-versions))
5. Switch the embedding provider on existing data only by the reindex
   procedure; it is resumable and can be undone.
   (2026-11-21, [INSTALLATION.md](../../INSTALLATION.md#switching-provider-on-existing-data))

## Procedures

- Install: [INSTALLATION.md "Quick Start"](../../INSTALLATION.md#quick-start-one-command-installer)
  or ["Manual Installation"](../../INSTALLATION.md#manual-installation).
- Verify a running instance: [INSTALLATION.md "Verification"](../../INSTALLATION.md#verification).
- Run as a service: [INSTALLATION.md "Running as Service"](../../INSTALLATION.md#running-as-service).
- Turn on agent keys: [INSTALLATION.md "Agent keys"](../../INSTALLATION.md#agent-keys-authentication).
- Fix a problem: [INSTALLATION.md "Troubleshooting"](../../INSTALLATION.md#troubleshooting).
- Maintainers: deploy, rollback and incident runbooks are in the overlay
  (see [AGENTS.md](../../AGENTS.md) section 5).

## Truth

- Services, ports and volumes: [`docker-compose.yml`](../../docker-compose.yml),
  [`Dockerfile`](../../Dockerfile).
- Installer and its flags: [`install.sh`](../../install.sh),
  [INSTALLATION.md "Installer flags reference"](../../INSTALLATION.md#installer-flags-reference).
- Configuration: [`.env.example`](../../.env.example).
- What a release tag builds: [`.github/workflows/release.yml`](../../.github/workflows/release.yml).
- Changes per version: [`CHANGELOG.md`](../../CHANGELOG.md).
