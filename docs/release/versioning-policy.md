Status: active · Owner: the maintainers · Verified: 2026-11-18 · Verify-on: change to an MCP tool, env var, HTTP route, CLI flag or stored-data shape; change to docs/decisions/ADR-002-semver-and-api-surface.md

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
  No bump "because it feels big".
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

- The version lives in `pyproject.toml` only. The MCP server reads it from
  package metadata at runtime. `uv.lock` records it too; `uv lock` regenerates
  that entry in the release commit. Known hand-written copies (the README
  version badge and the Admin UI version string) are to be removed; until then,
  update them in the release commit.
- Today: the release check in [Cut a release](cut-a-release.md) verifies that
  the tag equals `v<pyproject version>` and that `CHANGELOG.md` has a heading
  for that version. A maintainer runs it by hand.
- Planned; until then, manual:
  - `scripts/release_check.py`: the checks above, plus `[Unreleased]` holds
    nothing already released and the previous tag is the direct predecessor.
  - `scripts/api_surface.py --check`: fails on a surface change that the
    committed snapshot does not declare.
  - The same checks in `.github/workflows/release.yml`, so a bad tag fails
    before an image is pushed.
  - `.github/workflows/release.yml` skips the `latest` image tag and marks
    the GitHub release as a pre-release for tags that contain `rc`.
