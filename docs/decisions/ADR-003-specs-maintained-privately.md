Status: proposed · Owner: the maintainers · Verified: 2026-11-18 · Verify-on: change to how features are proposed or specified

# ADR-003: Feature specs are maintained outside this repo

- Status: proposed
- Date: 2026-11-18
- Deciders: the maintainers

## Context

Feature work has used written specifications (problem, plan, tasks). Earlier
contributor guidance asked for spec files inside this repository. Those specs
carry planning detail that is not part of the product, and a requirement to
write them is a barrier for outside contributors who only want to report a
need or fix a bug.

## Decision

1. Feature specs are maintained by the core team outside this repo.
2. Contributors open an issue; no spec files are needed in a PR.
3. Pull requests do not add spec files. The issue and the PR description carry
   the public record of what changed and why.

## Consequences

- The public record of a feature is its issue, its PR, and its CHANGELOG entry.
- Contributors propose features with an issue; a maintainer agrees before code.
- `CONTRIBUTING.md` points here instead of describing a spec-file workflow.

## Supersedes

None. Replaces the former "Feature Development Process" section of `CONTRIBUTING.md`.
