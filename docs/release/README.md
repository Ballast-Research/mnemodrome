Status: active · Owner: the maintainers · Verified: 2026-11-18 · Verify-on: change to the pyproject.toml version, CHANGELOG.md, .github/workflows/release.yml or docs/decisions/

# Release

How Mnemodrome is versioned, where changes land, and how a release is cut.
Level-2 detail lives in the pages linked below; this page does not repeat it.

## Rules

1. GitHub `Ballast-Research/mnemodrome` is the single upstream. Trunk is `main`.
   Every change lands by pull request. Releases come only from tags on that repo.
   (2026-11-18, [ADR-001](../decisions/ADR-001-github-single-upstream.md))
2. Versions follow Semantic Versioning over the declared public API surface.
   (2026-11-18, [ADR-002](../decisions/ADR-002-semver-and-api-surface.md),
   [versioning policy](versioning-policy.md))
3. The version lives only in `pyproject.toml` and changes only in the release
   commit. Pull requests add entries under `[Unreleased]` in `CHANGELOG.md`.
   (2026-11-18, [ADR-002](../decisions/ADR-002-semver-and-api-surface.md))
4. One step per release: the next patch or the next minor. No skipped numbers.
   Majors are rare and batched. (2026-11-18,
   [ADR-002](../decisions/ADR-002-semver-and-api-surface.md))
5. Pre-releases use the PEP 440 form `X.Y.ZrcN` in `pyproject.toml`, the git
   tag (`vX.Y.ZrcN`) and the image tag. Only `rc` is allowed. A pre-release
   never moves the `latest` image tag. (2026-11-18,
   [ADR-002](../decisions/ADR-002-semver-and-api-surface.md))
6. A maintainer approves every tag. No tag without that approval. (2026-11-18)
7. Feature specs are maintained by the core team outside this repo.
   Contributors open an issue; no spec files are needed in a PR.
   (2026-11-18, [ADR-003](../decisions/ADR-003-specs-maintained-privately.md))

## Procedures

- [Versioning policy](versioning-policy.md): what the public API surface is,
  stability marks, and which change needs which bump.
- [Cut a release](cut-a-release.md): release commit, check, tag, verify;
  includes the release-candidate path.
- Deploying a tagged image is a maintainer task outside this repo. Maintainers
  deploy tagged images by digest; see the maintainer overlay.

## Truth

- Version: [`pyproject.toml`](../../pyproject.toml) (`[project] version`).
- Change record: [`CHANGELOG.md`](../../CHANGELOG.md) (Keep a Changelog format).
- What a `v*` tag does: [`.github/workflows/release.yml`](../../.github/workflows/release.yml).
- Decisions: [`docs/decisions/`](../decisions/README.md).
- Release check (run on the tagged commit; must exit 0):

```bash
v=$(python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])") && test "$(git describe --tags --exact-match HEAD)" = "v$v" && grep -q "^## \[$v\]" CHANGELOG.md
```
