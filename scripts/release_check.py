#!/usr/bin/env python3
"""Release check: verify a release tag before an image is built.

Usage:
    python scripts/release_check.py [TAG] [--allow-major] [--root DIR]
    python scripts/release_check.py [TAG] --is-prerelease

TAG defaults to `git describe --tags --exact-match HEAD`. Checks:

  a. the tag has the form vX.Y.Z or vX.Y.ZrcN (PEP 440, rc only);
  b. the tag equals "v" + the pyproject.toml version;
  c. CHANGELOG.md has the heading "## [<version>] - YYYY-MM-DD";
  d. uv.lock, if present, records the same version for this package;
  e. the version is a legal next step after the existing v* tags
     (next patch or minor of the last final tag, next major only with
     --allow-major, or an rc of that; after rcN only rc(N+1) or the final;
     an existing tag is never reused).

One line is printed per failed check and the exit code is 1. With
--is-prerelease the script prints "true" or "false" and checks nothing else.
Standard library only (Python 3.11).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path

TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)(?:rc(\d+))?$")
PACKAGE = "mnemodrome"

# (major, minor, patch, rc) where rc is None for a final release.
Version = tuple[int, int, int, int | None]


def parse_tag(tag: str) -> Version | None:
    """Return the parsed tag, or None when it is not vX.Y.Z or vX.Y.ZrcN."""
    m = TAG_RE.match(tag)
    if not m:
        return None
    major, minor, patch, rc = m.groups()
    return int(major), int(minor), int(patch), None if rc is None else int(rc)


def is_prerelease(tag: str) -> bool:
    parsed = parse_tag(tag)
    return parsed is not None and parsed[3] is not None


def _fmt(base: tuple[int, int, int]) -> str:
    return "{}.{}.{}".format(*base)


def check_sequence(
    tag: str,
    existing: dict[str, str],
    head: str | None,
    allow_major: bool = False,
) -> list[str]:
    """Check the tag against existing v* tags. `existing` maps tag -> commit."""
    cand = parse_tag(tag)
    if cand is None:
        return []  # reported by the format check
    errors: list[str] = []

    if tag in existing and existing[tag] != head:
        errors.append(
            f"sequence: tag {tag} already exists on another commit; tags are never reused"
        )

    others: list[Version] = []
    for name in existing:
        if name == tag:
            continue
        parsed = parse_tag(name)
        if parsed is not None:
            others.append(parsed)

    base = cand[:3]
    finals = sorted(v[:3] for v in others if v[3] is None)
    last = finals[-1] if finals else None

    # An open rc line: rc tags for a base newer than the last final tag.
    open_bases = sorted(
        {v[:3] for v in others if v[3] is not None and (last is None or v[:3] > last)}
    )
    pending = open_bases[-1] if open_bases else None
    if pending is not None and base != pending:
        errors.append(
            f"sequence: rc line {_fmt(pending)} is open; the next tag must be an rc of "
            f"{_fmt(pending)} or the final {_fmt(pending)}, not {_fmt(base)}"
        )
        return errors

    if last is not None and base != pending:
        major, minor, patch = last
        allowed = [(major, minor, patch + 1), (major, minor + 1, 0)]
        if allow_major:
            allowed.append((major + 1, 0, 0))
        if base not in allowed:
            names = ", ".join(_fmt(a) for a in allowed)
            hint = "" if allow_major else " (a major needs --allow-major)"
            errors.append(
                f"sequence: {_fmt(base)} is not a legal step after last final v{_fmt(last)}; "
                f"allowed: {names}{hint}"
            )
            return errors

    if cand[3] is not None:
        prior = [v[3] for v in others if v[:3] == base and v[3] is not None]
        expected = (max(prior) + 1) if prior else 1
        if cand[3] != expected:
            errors.append(
                f"sequence: expected rc{expected} for {_fmt(base)}, got rc{cand[3]}"
            )
    return errors


def pyproject_version(root: Path) -> str | None:
    path = root / "pyproject.toml"
    if not path.is_file():
        return None
    with path.open("rb") as fh:
        return tomllib.load(fh).get("project", {}).get("version")


def lock_version(root: Path) -> tuple[bool, str | None]:
    """Return (lock file present, version recorded for this package)."""
    path = root / "uv.lock"
    if not path.is_file():
        return False, None
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    for pkg in data.get("package", []):
        if pkg.get("name") == PACKAGE:
            return True, pkg.get("version")
    return True, None


def run_checks(
    root: Path,
    tag: str,
    existing: dict[str, str],
    head: str | None,
    allow_major: bool = False,
) -> list[str]:
    """Run checks a-e and return one message per failure."""
    errors: list[str] = []

    if parse_tag(tag) is None:
        errors.append(
            f"format: tag '{tag}' must match vX.Y.Z or vX.Y.ZrcN (for example v1.1.0rc1)"
        )

    version = pyproject_version(root)
    if version is None:
        errors.append("version: pyproject.toml has no [project] version")
    elif tag != f"v{version}":
        errors.append(
            f"version: tag {tag} does not equal v{version} from pyproject.toml"
        )

    if version is not None:
        heading = re.compile(
            rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}\s*$", re.M
        )
        changelog = root / "CHANGELOG.md"
        if not changelog.is_file() or not heading.search(
            changelog.read_text(encoding="utf-8")
        ):
            errors.append(
                f"changelog: CHANGELOG.md has no heading '## [{version}] - YYYY-MM-DD'"
            )

        present, locked = lock_version(root)
        if present and locked != version:
            errors.append(
                f"lock: uv.lock records {locked or 'no entry'} for {PACKAGE}, pyproject.toml has {version}"
            )

    errors.extend(check_sequence(tag, existing, head, allow_major))
    return errors


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def git_tags(root: Path) -> dict[str, str]:
    """Map every v* tag to the commit it points at."""
    out = _git(
        root,
        "for-each-ref",
        "--format=%(refname:short) %(objectname) %(*objectname)",
        "refs/tags/v*",
    )
    tags: dict[str, str] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            tags[parts[0]] = parts[-1]  # peeled commit for annotated tags
    return tags


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check a release tag before publishing."
    )
    parser.add_argument(
        "tag", nargs="?", help="tag to check (default: the tag on HEAD)"
    )
    parser.add_argument(
        "--allow-major", action="store_true", help="accept the next major version"
    )
    parser.add_argument(
        "--is-prerelease", action="store_true", help="print true or false and exit"
    )
    parser.add_argument(
        "--root", default=".", help="repository root (default: current directory)"
    )
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()

    tag = args.tag
    if not tag:
        try:
            tag = _git(root, "describe", "--tags", "--exact-match", "HEAD")
        except RuntimeError as exc:
            print(f"tag: no tag given and HEAD has no exact tag ({exc})")
            return 1

    if args.is_prerelease:
        if parse_tag(tag) is None:
            print(f"format: tag '{tag}' must match vX.Y.Z or vX.Y.ZrcN")
            return 1
        print("true" if is_prerelease(tag) else "false")
        return 0

    try:
        existing = git_tags(root)
        head = _git(root, "rev-parse", "HEAD")
    except RuntimeError as exc:
        print(f"git: cannot read tags or HEAD ({exc})")
        return 1

    errors = run_checks(root, tag, existing, head, args.allow_major)
    for line in errors:
        print(line)
    if errors:
        return 1
    print(f"release check passed for {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
