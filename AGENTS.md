# AGENTS.md: Mnemodrome trailhead

Read this page first. It is the map. Find the row for your intent, read its
"Read first" source and its procedure before your first change, and state the
row and its gate in your first reply (or in the issue). If no row fits, stop and ask.

Status: active · Owner: the maintainers · Verified: 2026-11-21 · Verify-on: change to this table, CONTRIBUTING.md, CHANGELOG.md or .github/workflows/

## 1. What this is

- Mnemodrome: a multi-layer memory server for AI agents, exposed over MCP (Python 3.11, Neo4j, Qdrant, Redis, Postgres). Pitch and architecture: `README.md`.
- This repo is **public** (Apache-2.0). Maintainers also run a live instance; it is not described here (see section 5).
- Canonical remote: `https://github.com/Ballast-Research/mnemodrome` (`origin`, trunk `main`). Every change lands there by pull request.
- Fetch before you read: `git fetch origin && git status -sb`. If your checkout is behind `origin/main`, say so before acting.

## 2. Hard rules

1. Public repo: never commit internal hostnames, IPs, private paths, credentials or anything true only on a maintainer's network.
2. Never print a secret or key value: not in logs, commits, issues or PRs.
3. One issue, one branch from fresh `origin/main`, one PR. No force push to `main`.
4. The version lives only in `pyproject.toml` and changes only in a release commit. PRs add entries under `[Unreleased]` in `CHANGELOG.md`.
5. A change to an MCP tool, env var, HTTP route, CLI flag or stored-data shape is an API change: CHANGELOG entry under the right heading, stated in the PR.
6. Feature specs are maintained outside this repo; never add spec files to a PR (`docs/decisions/ADR-003-specs-maintained-privately.md`).
7. No release tag, deploy or auth-mode change without a maintainer's explicit yes.

## 3. Intent routing

Gate kinds: **check** (a command that must exit 0), **approval** (a named person says yes), **precondition** (a fact that must be true).
"(interim)" = the area page is not written yet; this is the best current source.

| Intent | Read first | Procedure | Gate (stop until) |
|---|---|---|---|
| idea: "it should also do X" | `docs/PHILOSOPHY.md` (interim) | `CONTRIBUTING.md` "Suggesting Features" (interim) | check: issue filed; approval: a maintainer before any code |
| bug: "I saw X misbehave" | `CONTRIBUTING.md` (interim) | `CONTRIBUTING.md` "Reporting Bugs", "Testing Strategy" (interim) | check: issue exists; lint + tests below exit 0; one PR per issue |
| change an MCP tool, env var, route, CLI or stored data | `README.md` "MCP Tools", "Configuration" (interim) | `docs/release/versioning-policy.md` | check: CHANGELOG entry under Added/Changed/Deprecated/Removed; approval: a maintainer for any removal |
| review a PR | `CONTRIBUTING.md` "Pull Requests" (interim) | overlay: review | one independent review per PR |
| push to GitHub | `CONTRIBUTING.md` "Pull Requests" (interim) | overlay: push | check: push gate prints PASSED |
| release: tag, version, changelog | `docs/release/README.md` | `docs/release/cut-a-release.md` | check: release check below exits 0 on the tagged commit; precondition: CI green; approval: a maintainer for the tag |
| deploy, rollback | overlay: operations | overlay runbook | approval: a maintainer says "deploy"; precondition: image digest exists for the tag |
| live service is misbehaving | overlay: operations | overlay incident runbook | precondition: read-only until a maintainer authorizes a change |
| secret, key, auth mode | `SECURITY.md` (interim) | overlay: security | never print a value; approval: a maintainer for any mode change |
| work that also changes another repo | `CONTRIBUTING.md` (interim) | overlay: cross-repo | check: an issue exists in that repo; you entered its trailhead |
| docs, including this map | `AGENTS.md` | edit in the same PR as the change it describes | check: `test "$(wc -l < AGENTS.md)" -le 80`; every path here exists (manual until check_map.py); approval: a maintainer |

Lint and tests (what CI runs): `uv run ruff check . && uv run ruff format --check . && uv run pytest tests/unit/`

Release check (tag == `pyproject.toml` version == `uv.lock` == CHANGELOG heading, legal next step after the last tag):
`python3 scripts/release_check.py`

## 4. Areas

Area pages (`docs/<area>/README.md`): only `release` is written. For the other areas the sources below are interim.

| Area | Current sources |
|---|---|
| product | `docs/PHILOSOPHY.md`, `README.md` |
| architecture | `README.md` "Architecture", "MCP Tools"; `src/` |
| development | `CONTRIBUTING.md` (setup, style, tests), `.github/workflows/ci.yml`, `.github/PULL_REQUEST_TEMPLATE.md` |
| release | `docs/release/README.md` (area page written); decisions in `docs/decisions/` |
| operations | Self-hosters: `INSTALLATION.md`. Maintainers: overlay. |
| security | `SECURITY.md`, `README.md` "Agent keys (authentication)" |

## 5. Maintainer overlay

Maintainers have a private overlay. If `.private/MAP.md` exists in this checkout, read it now.
In a worktree, look in the main checkout: the parent directory of `git rev-parse --path-format=absolute --git-common-dir`.
If it does not exist, you are working as an outside contributor: rows marked `overlay` need the
overlay and are not yours to perform. Stop at those rows and ask a maintainer.

## 6. Refocus rule

Every piece of work names the goal it serves: an issue.
A new finding does not join the running work: decide it now with a maintainer, or file it as an issue.
