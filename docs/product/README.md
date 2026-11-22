Status: active · Owner: the maintainers · Verified: 2026-11-22 · Verify-on: change to docs/product/PHILOSOPHY.md, LICENSE, the README.md pitch, or docs/decisions/

# Product

Why Mnemodrome exists, what it promises, and how a new idea becomes work.
Detail lives in the pages linked below; this page does not repeat it.

## Rules

1. Mnemodrome is open source under the Apache License 2.0. Contributions are
   licensed the same way. (2026-11-22, [LICENSE](../../LICENSE))
2. The design principles in [PHILOSOPHY.md](PHILOSOPHY.md) decide feature
   questions: never delete, only supersede; confidence is earned; silence is a
   failure mode; local by default; layers compose. A proposal that breaks one
   says so and why. (2026-11-22)
3. Before you suggest, recommend or propose anything, read the decisions that
   govern the area ([decisions index](../decisions/README.md)). A proposal cites
   the ADR it builds on or supersedes, or states that none applies.
   (2026-11-21, maintainer requirement)
4. An idea starts as a GitHub issue. No code until a maintainer agrees on the
   approach. (2026-11-18, [ADR-003](../decisions/ADR-003-specs-maintained-privately.md))
5. Feature specs are maintained by the core team outside this repo. Contributors
   open an issue; no spec files are needed in a PR.
   (2026-11-18, [ADR-003](../decisions/ADR-003-specs-maintained-privately.md))
6. Self-hosting comes first: every store runs locally and cloud providers are
   optional, never required. (2026-11-22, [PHILOSOPHY.md](PHILOSOPHY.md))

## Procedures

- Suggest a feature: [CONTRIBUTING.md "Suggesting Features"](../../CONTRIBUTING.md#suggesting-features),
  then [CONTRIBUTING.md "Feature Development Process"](../../CONTRIBUTING.md#feature-development-process).
- Roadmap: there is no roadmap file. Open and planned work is the issue list.

## Truth

- Thesis and design principles: [PHILOSOPHY.md](PHILOSOPHY.md).
- Pitch, comparison and positioning: [README.md](../../README.md).
- License: [LICENSE](../../LICENSE). Conduct: [CODE_OF_CONDUCT.md](../../CODE_OF_CONDUCT.md).
- Decisions: [docs/decisions/](../decisions/README.md).
- Work in progress: [GitHub issues](https://github.com/Ballast-Research/mnemodrome/issues).
