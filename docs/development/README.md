Status: active · Owner: the maintainers · Verified: 2026-11-22 · Verify-on: change to CONTRIBUTING.md, .github/workflows/ci.yml, .github/PULL_REQUEST_TEMPLATE.md or pyproject.toml

# Development

Setup, branch and PR rules, tests and review.
Detail lives in the pages linked below; this page does not repeat it.

## Rules

1. One issue, one branch from fresh `origin/main`, one pull request. No force
   push to `main`. (2026-11-18, [ADR-001](../decisions/ADR-001-github-single-upstream.md))
2. A PR merges only when CI is green: lint, format check, unit tests on
   Python 3.11 and 3.12, and the map check. Run the same locally:
   `uv run ruff check . && uv run ruff format --check . && uv run pytest tests/unit/`
   and `python3 scripts/check_map.py`. (2026-11-22, [ci.yml](../../.github/workflows/ci.yml))
3. New features come with tests. (2026-11-22, [CONTRIBUTING.md "Testing"](../../CONTRIBUTING.md#testing))
4. Commit messages follow Conventional Commits.
   (2026-11-22, [CONTRIBUTING.md "Commit Messages"](../../CONTRIBUTING.md#commit-messages))
5. PRs add a `CHANGELOG.md` entry under `[Unreleased]` and state whether they
   change the public API surface. PRs never change the version.
   (2026-11-18, [ADR-002](../decisions/ADR-002-semver-and-api-surface.md))
6. Python 3.11+, dependencies through `uv` and the committed `uv.lock`. CI
   actions are pinned to a full commit SHA. Workflows that run on pull requests
   get a read-only token; only the tag-triggered release workflow may write.
   (2026-11-22, [SECURITY.md](../../SECURITY.md#supply-chain-integrity))
7. Feature specs stay outside this repo; no spec files in a PR.
   (2026-11-18, [ADR-003](../decisions/ADR-003-specs-maintained-privately.md))
8. Each PR gets one independent review before merge. (2026-11-21, [AGENTS.md](../../AGENTS.md))
9. A doc change goes in the same PR as the change it describes. Changes to
   `AGENTS.md`, `docs/release/` and `docs/decisions/` need a maintainer's approval.
   (2026-11-21, [AGENTS.md](../../AGENTS.md))

## Procedures

- Set up a checkout: [CONTRIBUTING.md "Development Setup"](../../CONTRIBUTING.md#development-setup).
- Fix a bug: [CONTRIBUTING.md "Reporting Bugs"](../../CONTRIBUTING.md#reporting-bugs),
  then [CONTRIBUTING.md "Testing Strategy"](../../CONTRIBUTING.md#testing-strategy).
- Add a feature: [CONTRIBUTING.md "Feature Development Process"](../../CONTRIBUTING.md#feature-development-process).
- Open a PR: [CONTRIBUTING.md "Pull Requests"](../../CONTRIBUTING.md#pull-requests)
  and the [PR template](../../.github/PULL_REQUEST_TEMPLATE.md).

## Truth

- CI: [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml).
- Tooling and lint rules: [`pyproject.toml`](../../pyproject.toml).
- Tests: [`tests/`](../../tests/) (unit tests in `tests/unit/` need no services).
- Map check: [`scripts/check_map.py`](../../scripts/check_map.py); machine
  index of the routing table: [`docs/map.yml`](../map.yml).
