Status: active · Owner: the maintainers · Verified: 2026-11-21 · Verify-on: change to an MCP tool, env var, HTTP route, CLI flag or stored-data shape; change to docs/decisions/ADR-002-semver-and-api-surface.md

# Versioning policy

Up: [Release](README.md). Decision: [ADR-002](../decisions/ADR-002-semver-and-api-surface.md).

Mnemodrome follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html)
over the public API surface below, written in [PEP 440](https://peps.python.org/pep-0440/)
form. SemVer promises apply from a tagged release onward.

## Public API surface

SemVer covers exactly these five categories:

1. **MCP tools**: tool names; parameter names, types and required-ness;
   documented response fields and status values.
2. **HTTP**: hook routes, `/health`, `/mcp` and the SSE endpoints; auth headers.
3. **Configuration**: environment variable names, meanings and defaults;
   `docker-compose.yml` service names and published ports.
4. **CLI**: `python -m src.cli.*` commands and flags; `install.sh` flags.
5. **Stored data**: Neo4j labels, relationship types and properties; Markdown
   file layout and frontmatter; event-log schema; Qdrant collection shape.
   The promise: a newer minor or patch starts on older data with no manual step.

## Not in the surface

Python module paths and internal APIs, Admin UI layout and HTML, log text,
test helpers, and anything marked `experimental`.

## Stability marks

- Each MCP tool and environment variable is `stable` or `experimental`.
- New tools and variables start `experimental` and stay so for at least one minor
  release before they can become `stable`.
- The mark will appear in the tool description and in a surface listing under
  `docs/architecture/`. Planned; until then, an item is `stable` if it was
  present in a tagged release, and a PR that adds an item states its mark.

## Bump rules

- **PATCH** (`1.1.0` → `1.1.1`): fixes, docs, dependency updates, internal
  refactors. No surface change.
- **MINOR** (`1.1.x` → `1.2.0`): additions to the surface; deprecations; any change
  or removal of an `experimental` item, listed under `### Changed` or
  `### Removed` with the mark stated.
- **MAJOR** (`1.x` → `2.0.0`): removal or incompatible change of a `stable` item;
  a changed default that alters behavior on upgrade (example: auth mode default
  `off` → `enforce`); a stored-data change that needs a manual migration.
- **Security-hardening default (exception, 2026-11-18).** A default may change
  to a safer value in a MINOR release when (a) the change closes an exposure,
  (b) the CHANGELOG entry carries an `Upgrade note:` with the exact setting that
  restores the old behavior, and (c) the release notes repeat that note. Any
  other changed default that alters behavior on upgrade stays MAJOR.
- **Removing a stable item takes two steps.** Deprecate it in a minor: it still
  works, says so in its response or log, and is listed under `### Deprecated`.
  Remove it in the next major.
- Items that were never in a tagged release are not part of any promise. Their
  removal is noted under `### Removed` but does not force a major.
  Example: the librarian admin tools were added and removed between releases,
  so their removal does not make 1.1.0 a major release.

## Keeping steps small

- The version changes only in the release commit. PRs never touch it;
  they add entries under `[Unreleased]` in `CHANGELOG.md`.
- One step per release: the next patch or the next minor. No skipped numbers.
  No bump "because it feels big". An open rc does not block a hotfix: the next
  patch of the last final tag may ship while an rc is open.
- Release when `main` holds something wanted live. A patch may carry one fix.
- Majors are rare and batched: collect deprecations, remove them together.
- Pre-releases: `X.Y.ZrcN` only (for example `1.1.0rc1`, tag `v1.1.0rc1`,
  image tag `1.1.0rc1`). No `a`, `b`, `.dev` or `.post` suffixes. Use an rc to
  rehearse a release; it must never move the `latest` image tag.

## API change in a pull request

A PR that changes the surface states it in the PR description ("API surface:
none" otherwise) and adds a CHANGELOG entry under `### Added`, `### Changed`,
`### Deprecated` or `### Removed`. Removing anything needs a maintainer's approval.

## One source, checked

- The version lives in `pyproject.toml` only. The MCP server, the Admin UI and
  its health route read it from the installed package metadata at runtime
  (`src/config.py`, `MCP_VERSION`); the README badge shows the latest GitHub
  release. `uv lock` regenerates the version entry in `uv.lock`.
- A release commit touches only `pyproject.toml`, `uv.lock` and the
  `CHANGELOG.md` heading. `tests/unit/test_version.py` fails if the installed
  version and `pyproject.toml` disagree.
- Enforced: `scripts/release_check.py` (run by a maintainer in
  [Cut a release](cut-a-release.md) and first in
  `.github/workflows/release.yml`, so a bad tag fails before an image is pushed)
  checks that the tag has the form `vX.Y.Z` or `vX.Y.ZrcN`, equals
  `v<pyproject version>`, has a `## [<version>] - YYYY-MM-DD` heading in
  `CHANGELOG.md`, matches the version in `uv.lock`, and is a legal next step
  after the existing `v*` tags (next patch or minor of the last final tag, or
  an rc of it; a major only with `--allow-major`; after `rcN` only `rc(N+1)` or
  the final; no tag reuse). One exception: an open rc never blocks a hotfix.
  While `v1.1.0rc1` is open, the next patch of the last final tag (`v1.0.1`)
  is still allowed (as `v1.0.1` or `v1.0.1rc1`), and nothing else is; a second
  hotfix `v1.0.2` waits until `v1.0.1` exists. The hotfix is cut from the last
  final tag, not from `main` (a manual rule: the script checks tags, not
  ancestry; see [Cut a release](cut-a-release.md#hotfix-path)). The hotfix's CHANGELOG entry must
  also land under `[Unreleased]` so the rc line carries the fix.
- A major release needs a reviewed change to `.github/workflows/release.yml`:
  `--allow-major` is deliberately not wired into CI, so a major tag fails the
  workflow until a maintainer-approved PR passes the flag.
- The workflow pushes the `latest` image tag only for final tags and marks the
  GitHub release as a pre-release for `rc` tags.
- Planned; until then, manual:
  - `scripts/release_check.py`: a check that `[Unreleased]` holds nothing
    already released.
  - `scripts/api_surface.py --check`: fails on a surface change that the
    committed snapshot does not declare.
