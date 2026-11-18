Status: draft · Owner: the maintainers · Verified: 2026-11-18 · Verify-on: change to docs/release/versioning-policy.md or to the version scheme

# ADR-002: Semantic Versioning over a declared API surface

- Status: proposed
- Date: 2026-11-18
- Deciders: the maintainers

## Context

Mnemodrome is consumed by agents and integrations through MCP tools, HTTP routes,
environment variables, CLI commands and stored data. Users need to know which
upgrades are safe. SemVer only means something when the surface it covers is
named. Large, irregular version jumps make upgrades hard to judge, and a version
recorded in several files drifts.

## Decision

1. Versions follow Semantic Versioning 2.0.0 over the public API surface defined
   in `docs/release/versioning-policy.md` (MCP tools, HTTP, configuration, CLI,
   stored data). Anything outside it carries no compatibility promise.
2. Each MCP tool and environment variable is marked `stable` or `experimental`.
   New items start `experimental` for at least one minor release.
3. PATCH for fixes and internal changes; MINOR for additions, deprecations and
   changes to experimental items; MAJOR for incompatible changes to stable items,
   behavior-changing defaults, and data changes that need a manual migration.
   Removing a stable item takes two steps: deprecate in a minor, remove in the
   next major.
4. Steps are small: each release is the next patch or the next minor, with no
   skipped numbers. Majors are rare and batched.
5. Version strings use PEP 440 form. Pre-releases use only the `rcN` suffix, in
   the same form everywhere: `pyproject.toml` `1.1.0rc1`, git tag `v1.1.0rc1`,
   image tag `1.1.0rc1`. A pre-release never moves the `latest` image tag.
6. The version lives only in `pyproject.toml` and changes only in the release
   commit. Pull requests add entries under `[Unreleased]` in `CHANGELOG.md`.
7. A maintainer approves every release tag.

## Consequences

- Every PR states whether it changes the API surface, and records it in the
  CHANGELOG under the matching heading.
- Items that never appeared in a tagged release carry no promise; removing them
  is noted in the CHANGELOG but is not a major change. 1.1.0 is a minor release.
- PEP 440 form keeps Python packaging, git tags and image tags identical,
  at the cost of differing from the SemVer `-rc.N` spelling.
- Tooling to enforce this (release check script, API surface snapshot, checks in
  `.github/workflows/release.yml`, no `latest` for rc tags) is planned; until it exists, the checks
  are manual and listed in `docs/release/versioning-policy.md`.

## Supersedes

None. Replaces the undocumented practice described in the former "Release
Process" section of `CONTRIBUTING.md`.
