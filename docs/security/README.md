Status: active · Owner: the maintainers · Verified: 2026-11-22 · Verify-on: change to SECURITY.md, src/services/agent_auth.py, src/cli/agent_keys.py, Dockerfile or .github/workflows/

# Security

Vulnerability reporting, secrets, the auth model and the supply chain.
Detail lives in the pages linked below; this page does not repeat it.

## Rules

1. Report a vulnerability privately, never in a public issue.
   (2026-11-22, [SECURITY.md](../../SECURITY.md#reporting-a-vulnerability))
2. Never print or commit a secret or key value: not in logs, commits, issues
   or PRs. `.env` and key files stay out of git. (2026-11-21, [AGENTS.md](../../AGENTS.md))
3. Each agent gets its own key. The server stores SHA-256 digests only, never
   keys; a key is shown once, when it is created.
   (2026-11-22, [README "Agent keys"](../../README.md#agent-keys-authentication))
4. Auth mode is `off` by default. Roll out `off`, then `warn`, then `enforce`;
   switch to `enforce` only when `/health` shows every client has a key.
   (2026-11-22, [INSTALLATION.md](../../INSTALLATION.md#4-roll-out-off-warn-enforce))
5. A default may change to a safer value in a minor release only under the
   security-hardening exception, with an `Upgrade note:` in the CHANGELOG.
   Any other behavior-changing default is a major release.
   (2026-11-18, [ADR-002](../decisions/ADR-002-semver-and-api-surface.md))
6. Supply chain: no npm; builds install from the committed `uv.lock` with
   hashes; service images are version-pinned; CI actions are SHA-pinned;
   pull-request workflows get a read-only token.
   (2026-11-22, [SECURITY.md](../../SECURITY.md#supply-chain-integrity))
7. A change of auth mode on an instance the maintainers run needs a
   maintainer's explicit yes. (2026-11-21, [AGENTS.md](../../AGENTS.md))

## Procedures

- Report a vulnerability: [SECURITY.md](../../SECURITY.md#reporting-a-vulnerability).
- Set up agent keys: [INSTALLATION.md "Agent keys"](../../INSTALLATION.md#agent-keys-authentication).
- Harden a deployment: [SECURITY.md "Security Best Practices for Operators"](../../SECURITY.md#security-best-practices-for-operators).
- Maintainers: secret storage and auth rollout on the live instance are in
  the overlay (see [AGENTS.md](../../AGENTS.md) section 5).

## Truth

- Policy and scope: [SECURITY.md](../../SECURITY.md).
- Auth middleware: [`src/services/agent_auth.py`](../../src/services/agent_auth.py).
- Key CLI: [`src/cli/agent_keys.py`](../../src/cli/agent_keys.py).
- Build: [`Dockerfile`](../../Dockerfile), [`uv.lock`](../../uv.lock).
- CI pinning and permissions: [`.github/workflows/`](../../.github/workflows/).
