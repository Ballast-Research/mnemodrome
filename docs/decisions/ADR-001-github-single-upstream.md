Status: proposed · Owner: the maintainers · Verified: 2026-11-18 · Verify-on: change of the canonical remote, the trunk branch, or where release tags are cut

# ADR-001: GitHub is the single upstream

- Status: proposed
- Date: 2026-11-18
- Deciders: the maintainers

## Context

Mnemodrome is a public Apache-2.0 project. Contributors, users and automated
agents need one unambiguous answer to three questions: where is the source of
truth, where do changes land, and where do releases come from. If more than one
remote can produce a release, a version number no longer names one exact
commit, and users cannot tell which build they run.

## Decision

1. `https://github.com/Ballast-Research/mnemodrome` is the single upstream.
   Its `main` branch is the trunk.
2. Every change lands on `main` by pull request. No direct pushes and no force
   pushes to `main`.
3. Release tags (`v*`) are cut only on this repository, on commits in `main`.
   Release images and GitHub releases are built only from those tags by
   `.github/workflows/release.yml`.
4. No other remote is a source of releases. Any other copy of the repository
   follows this one and is never the origin of a tag, image or version.

## Consequences

- A version number maps to exactly one commit on `main` and one image build.
- Contributors fork this repository and open pull requests against `main`.
- Agents start by `git fetch origin` and compare with `origin/main` before acting.
- Work done elsewhere reaches users only after it lands here by pull request.

## Supersedes

None.
