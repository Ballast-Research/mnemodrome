Status: active · Owner: the maintainers · Verified: 2026-11-21 · Verify-on: change to .github/workflows/release.yml, the release check in AGENTS.md, or docs/release/versioning-policy.md

# Cut a release

Up: [Release](README.md). Rules for choosing the number: [versioning policy](versioning-policy.md).
One outcome: a `vX.Y.Z` (or `vX.Y.ZrcN`) tag on `main`, with its image and GitHub release published.
Examples use `1.1.0`; replace with the version you are cutting.

## Preconditions

1. CI is green on the tip of `main`: `gh run list --branch main --workflow CI --limit 1`.
2. `[Unreleased]` in `CHANGELOG.md` has been reviewed: each entry sits under the
   right heading, and the surface changes it lists match the bump you plan.
3. The version follows the last final (non-rc) tag,
   `git tag --list 'v*' --sort=-v:refname | grep -v rc | head -1`, as its next
   patch or next minor, or as an rc of that version. No skipped numbers.
   After `vX.Y.ZrcN` the only allowed next tags are `vX.Y.Zrc(N+1)` or `vX.Y.Z`,
   plus one exception, the hotfix: the next patch of the last final tag (for
   example `v1.0.1` while `v1.1.0rc1` is open, or its rc `v1.0.1rc1`). Cut it
   from the last final tag, not from `main`: see [Hotfix path](#hotfix-path). A major (`X+1.0.0`) needs a reviewed change
   to `.github/workflows/release.yml` that passes `--allow-major`; the flag is
   deliberately not wired into CI.
4. For a release candidate only: `.github/workflows/release.yml` skips the
   `latest` image tag for `rc` tags and marks the GitHub release as a
   pre-release. Enforced by the workflow (see step 10).

## Steps

1. Start from fresh `main` on a release branch:
   `git fetch origin && git switch -c release/1.1.0 origin/main`
2. Set `version = "1.1.0"` in `pyproject.toml`, then run `uv lock` so `uv.lock`
   records the same version, and `uv sync` so the local install reports it
   (`tests/unit/test_version.py` fails on a stale install). No other file holds the version
   ([versioning policy](versioning-policy.md#one-source-checked)).
3. In `CHANGELOG.md`, rename the `## [Unreleased]` heading to
   `## [1.1.0] - YYYY-MM-DD` (today's date) and add a new empty `## [Unreleased]`
   above it. The file has no compare links today; if it gains them, update them here.
4. Run lint and tests:
   `uv run ruff check . && uv run ruff format --check . && uv run pytest tests/unit/`
5. Commit as `chore(release): 1.1.0`, push the branch, open a PR. The PR holds
   only the release commit.
6. One independent review, then merge. Nothing else may merge to `main` until
   the tag is pushed.
7. Ask a maintainer to approve the tag. State the version and the commit:
   `git fetch origin && git rev-parse origin/main`. Stop until the maintainer says yes.
8. Tag that commit locally. For a normal release it is the tip of `origin/main`;
   for a hotfix it is the tip of the hotfix branch (see [Hotfix path](#hotfix-path)),
   never `origin/main`:
   `git switch --detach origin/main && git tag -a v1.1.0 -m "Mnemodrome 1.1.0"`
9. Run the release check. It must exit 0. It compares the tag with the tags in
   your clone, so fetch first; a stale clone can pass here and still fail in the
   workflow.

   ```bash
   git fetch --tags origin && python3 scripts/release_check.py
   ```

   If it fails, run `git tag -d v1.1.0`, do not push, and see "Release check
   fails after merge" under Rollback.

   It checks the tag format, tag == `v<pyproject version>`, the CHANGELOG
   heading, the `uv.lock` version and the tag sequence, and prints one line per
   failed check. A major version needs `--allow-major`.

10. Push the tag: `git push origin v1.1.0`. This starts
    [`.github/workflows/release.yml`](../../.github/workflows/release.yml), which:
    - checks out the tag with full history, runs `python3 scripts/release_check.py`
      (a failure stops the run before any image is built), installs dependencies with `uv sync`, and runs the unit tests;
    - logs in to GHCR and builds and pushes `ghcr.io/ballast-research/mnemodrome:<version>`,
      where `<version>` is the tag without the leading `v`, and also `:latest`
      for a final tag only; an rc tag never moves `latest`;
    - creates a GitHub release for the tag with auto-generated notes, marked as
      a pre-release for an rc tag.

    Step 9 and the workflow run the same script, so a bad tag is caught twice.

## Verify

1. The workflow run succeeded: `gh run list --workflow Release --limit 1`.
2. The release exists: `gh release view v1.1.0`. Replace the generated notes
   with the `CHANGELOG.md` section if they differ: `gh release edit v1.1.0 --notes-file <file>`.
   Copy every `Upgrade note:` from that CHANGELOG section into the release notes.
3. The image exists and its digest is recorded:
   `docker buildx imagetools inspect ghcr.io/ballast-research/mnemodrome:1.1.0`.
   Maintainers deploy tagged images by digest; see the maintainer overlay.

## Release-candidate path

Use it to rehearse a release before the final tag. Only `rcN` suffixes are allowed.

1. Follow the steps with version `1.1.0rc1`: `pyproject.toml` `1.1.0rc1`,
   CHANGELOG heading `## [1.1.0rc1] - YYYY-MM-DD`, tag `v1.1.0rc1`.
   The image tag is `1.1.0rc1`; it must not move `latest` (precondition 4).
2. Rehearse the deploy from the rc image digest.
3. If the rehearsal needs a code change: fix on `main` by PR, then cut `1.1.0rc2`.
4. If nothing changed: cut `1.1.0` with a release commit that only sets the
   version to `1.1.0`, re-runs `uv lock`, and renames the CHANGELOG heading to
   `## [1.1.0] - YYYY-MM-DD`. Before tagging, confirm the code is the rc's:
   `git diff --stat v1.1.0rc1 origin/main` lists only `pyproject.toml`,
   `uv.lock` and `CHANGELOG.md`, and `git diff v1.1.0rc1 origin/main` shows
   only the version strings and the CHANGELOG heading changing.

## Hotfix path

Use it when a fix must ship while a release candidate is open, or when `main`
holds work you do not want in a patch. `main` then carries unreleased work, so
a patch tagged from it would ship all of that.

1. Branch from the last final tag: `git fetch --tags origin && git switch -c hotfix/1.0.1 v1.0.0`.
2. Cherry-pick the fix. It is already on `main`, merged there by its own PR.
3. Make the release commit on that branch (`pyproject.toml`, `uv.lock`, CHANGELOG
   heading `## [1.0.1] - YYYY-MM-DD`), and open a PR against the hotfix branch.
   Steps 4 to 7 apply as written.
4. Tag the tip of the hotfix branch after its review (step 8), not `origin/main`.
5. Forward-port the `1.0.1` CHANGELOG entry to `main` under `[Unreleased]` by PR,
   so the rc line carries the fix.

`scripts/release_check.py` checks tags, not ancestry: it cannot tell that a
patch was tagged from `main`. Branching from the last final tag is a manual rule.
The Release workflow triggers on `v*` tags regardless of the branch they sit on,
so a tag on the hotfix branch builds and publishes like any other.

## Rollback

- Before the tag is pushed: `git tag -d v1.1.0`, fix by PR, start again.
- Release check fails after merge (the release commit is on `main` but the
  version, tag or CHANGELOG heading disagree): delete the local tag
  (`git tag -d v1.1.0`); if it was pushed, a maintainer deletes the remote tag
  (`git push origin :refs/tags/v1.1.0`) before any image is published. Fix the
  mismatch by PR, then restart from step 1.
- Tag pushed, workflow failed before any image or release was published:
  a maintainer may delete the tag (`git push origin :refs/tags/v1.1.0`), and the
  tag is cut again after the fix.
- Re-running the workflow for an older tag after a newer tag exists fails the
  sequence check; that is intended. Re-tagging is never the fix: cut the next
  patch (or the next rc) instead.
- Image or release published: never move, delete or reuse the tag. Fix forward
  with the next patch (or the next rc).

## Last run

None under this procedure yet. `v1.0.0` (2026-06-15) predates it.

Done when: the release check exits 0 on the tagged commit, the Release workflow
run is green, `gh release view vX.Y.Z` shows the release, and the image digest
for `X.Y.Z` is recorded.
