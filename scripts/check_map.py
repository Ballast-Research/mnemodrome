#!/usr/bin/env python3
"""check_map.py v1: check the project map (AGENTS.md, docs/, docs/map.yml).

Usage:
    python3 scripts/check_map.py [--root DIR] [--today YYYY-MM-DD]

Fails (one line per problem, exit 1) when:
  1. AGENTS.md is missing, over 80 lines, or lacks a required section;
     docs/map.yml lacks a required intent;
  2. CLAUDE.md is not the stub "@AGENTS.md";
  3. a relative link (or a backticked path in AGENTS.md) does not resolve;
  4. a .md or .yml file under docs/ is not reachable from AGENTS.md;
  5. a map doc has no header line, or is over its cap (area page 60 lines,
     ADR 100, other docs 150);
  6. a map doc links into docs/transfers/;
  7. docs/map.yml disagrees with the routing table in AGENTS.md;
  8. docs/decisions/README.md does not link every ADR file;
  9. a link's #fragment names no heading in the target Markdown file.
Warns (exit code unchanged) when a Verified date is older than 120 days, or
older than the last commit touching a path named in its Verify-on field.
docs/transfers/ and docs/archive/ are excluded. Standard library only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

AGENTS_CAP, AREA_CAP, ADR_CAP, DOC_CAP, STALE_DAYS = 80, 60, 100, 150, 120
SECTIONS = ["What this is", "Hard rules", "Intent routing", "Areas"]
SECTIONS += ["Maintainer overlay", "Refocus rule"]
REQUIRED_INTENTS = ["idea", "bug", "change", "review", "release", "deploy"]
REQUIRED_INTENTS += ["live", "secret", "docs", "suggest"]
EXCLUDED = ("docs/transfers/", "docs/archive/")
STATUSES = ("active", "draft", "proposed", "accepted", "superseded by ")
HEADER_RE = re.compile(
    r"Status: (?P<status>.+?) · Owner: .+? · Verified: (?P<date>\d{4}-\d{2}-\d{2})"
    r" · Verify-on: (?P<on>.+)"
)
LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
TICK_RE = re.compile(r"`([A-Za-z0-9_./-]+)`")
HEADING_RE = re.compile(r"#{1,6}\s+(.*?)\s*#*\s*$")
PATH_EXT = (".md", ".py", ".yml", ".yaml", ".toml", ".json", ".lock", ".sh")


def rel(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def targets(root: Path, doc: Path) -> list[tuple[str, Path]]:
    """Return (written, resolved) for each local reference in a doc."""
    text = doc.read_text(encoding="utf-8")
    out = []
    for raw in LINK_RE.findall(text):
        if "://" in raw or raw.startswith("mailto:"):
            continue
        path = raw.split("#")[0]
        out.append((raw, (doc.parent / path).resolve() if path else doc.resolve()))
    if doc.name == "AGENTS.md" and doc.parent.resolve() == root.resolve():
        for raw in TICK_RE.findall(text):
            if raw.startswith(".private") or not (
                raw.endswith(PATH_EXT) or raw.endswith("/")
            ):
                continue
            out.append((raw, (root / raw).resolve()))
    return out


def slugs(path: Path) -> set[str]:
    """GitHub-style heading anchors of a Markdown file (duplicates get -1, -2)."""
    found: set[str] = set()
    counts: dict[str, int] = {}
    fenced = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
            continue
        m = None if fenced else HEADING_RE.match(line)
        if not m:
            continue
        text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", m[1])
        text = text.replace("`", "").strip().lower()
        base = "".join(c for c in text if c.isalnum() or c in " -_")
        base = base.replace(" ", "-")
        n = counts.get(base, 0)
        counts[base] = n + 1
        found.add(base if n == 0 else f"{base}-{n}")
    return found


def walk(root: Path, errors: list[str]) -> set[str]:
    """Follow links from AGENTS.md through Markdown files; return reached paths."""
    seen: set[str] = set()
    todo = [root / "AGENTS.md"]
    while todo:
        doc = todo.pop()
        name = rel(root, doc)
        if name in seen:
            continue
        seen.add(name)
        if doc.suffix != ".md":
            continue
        for raw, target in targets(root, doc):
            try:
                tname = rel(root, target)
            except ValueError:
                errors.append(f"link: {name} -> {raw} leaves the repo")
                continue
            if tname.startswith("docs/transfers/") and is_map_doc(name):
                errors.append(f"transfer: {name} links into docs/transfers/ ({raw})")
            if not target.exists():
                errors.append(f"link: {name} -> {raw} does not resolve")
            elif "#" in raw and target.suffix == ".md" and target.is_file():
                frag = raw.split("#", 1)[1].lower()
                if frag and frag not in slugs(target):
                    errors.append(f"anchor: {name} -> {raw} names no heading")
            if target.exists() and target.is_file() and tname not in seen:
                todo.append(target)
            elif target.is_dir():
                seen.add(tname.rstrip("/") + "/")
    return seen


def is_map_doc(name: str) -> bool:
    return name == "AGENTS.md" or (
        name.startswith("docs/") and not name.startswith(EXCLUDED)
    )


def map_docs(root: Path) -> list[str]:
    docs = root / "docs"
    if not docs.is_dir():
        return []
    names = [rel(root, p) for p in docs.rglob("*") if p.suffix in (".md", ".yml")]
    return sorted(n for n in names if not n.startswith(EXCLUDED))


def cap_for(name: str) -> int:
    if re.fullmatch(r"docs/[a-z-]+/README\.md", name):
        return AREA_CAP
    return ADR_CAP if Path(name).name.startswith("ADR-") else DOC_CAP


def git(root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:
        return ""


def last_commit(root: Path, path: str) -> dt.date | None:
    """Date of the last commit touching path; None without full git history."""
    if git(root, "rev-parse", "--is-shallow-repository") != "false":
        return None
    out = git(root, "log", "-1", "--format=%cs", "--", path)
    return dt.date.fromisoformat(out) if out else None


def check_headers(root: Path, today: dt.date, errors: list, warnings: list) -> None:
    for name in [n for n in map_docs(root) if n.endswith(".md")]:
        lines = (root / name).read_text(encoding="utf-8").splitlines()
        if len(lines) > cap_for(name):
            errors.append(f"cap: {name} has {len(lines)} lines (cap {cap_for(name)})")
        m = HEADER_RE.fullmatch(lines[0].strip()) if lines else None
        if not m or not m["status"].startswith(STATUSES):
            errors.append(f"header: {name} line 1 is not a valid header line")
            continue
        verified = dt.date.fromisoformat(m["date"])
        if (today - verified).days > STALE_DAYS:
            warnings.append(
                f"stale: {name} Verified {verified} is over {STALE_DAYS} days old"
            )
        for word in re.split(r"[\s,;()]+", m["on"]):
            word = word.strip(".`")
            if "/" not in word and "." not in word or not (root / word).exists():
                continue
            touched = last_commit(root, word)
            if touched and touched > verified:
                warnings.append(
                    f"stale: {name} Verified {verified}; {word} changed {touched}"
                )


def table_rows(agents: str) -> dict[str, dict]:
    """Parse the intent routing table: intent -> read, procedure paths, overlay."""
    rows = {}
    for line in agents.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 4 or cells[0] in ("Intent", "---") or not line.startswith("|"):
            continue
        intent = cells[0].split(":")[0].strip()
        read = TICK_RE.findall(cells[1])
        over = re.findall(r"overlay: ([a-z-]+)", cells[1] + " " + cells[2])
        rows[intent] = {
            "read": read[0] if read else None,
            "procedure": sorted(TICK_RE.findall(cells[2])),
            "overlay": sorted(set(over)),
        }
    return rows


def parse_map(text: str) -> dict[str, dict]:
    """Parse docs/map.yml: `intents:` with 2-space names and 4-space fields.

    A field value is a bare scalar or a JSON array (also valid YAML).
    """
    intents: dict[str, dict] = {}
    current = None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = re.fullmatch(r"  ([a-z-]+):\s*", line)
        if m:
            current = intents.setdefault(m[1], {})
            continue
        m = re.fullmatch(r"    ([a-z]+): (.+)", line)
        if m and current is not None:
            value = m[2].strip()
            current[m[1]] = json.loads(value) if value.startswith("[") else value
    return intents


def check_map_yml(root: Path, agents: str, errors: list[str]) -> None:
    path = root / "docs" / "map.yml"
    if not path.is_file():
        errors.append("map.yml: docs/map.yml is missing")
        return
    intents = parse_map(path.read_text(encoding="utf-8"))
    for key in REQUIRED_INTENTS:
        if key not in intents:
            errors.append(f"intent: required intent '{key}' missing from docs/map.yml")
    table = table_rows(agents)
    by_row = {i.get("row"): (k, i) for k, i in intents.items()}
    for row in sorted(set(table) - set(by_row)):
        errors.append(f"map.yml: table row '{row}' has no intent in docs/map.yml")
    for row, (key, intent) in sorted(by_row.items(), key=lambda kv: str(kv[0])):
        if row not in table:
            errors.append(f"map.yml: intent '{key}' row '{row}' is not in the table")
            continue
        want = table[row]
        got = {
            "read": intent.get("read"),
            "procedure": sorted(intent.get("procedure", [])),
            "overlay": sorted([intent["overlay"]] if "overlay" in intent else []),
        }
        for field in ("read", "procedure", "overlay"):
            if got[field] != want[field]:
                errors.append(
                    f"map.yml: intent '{key}' {field} {got[field]} != table {want[field]}"
                )


def check_decisions(root: Path, errors: list[str]) -> None:
    folder = root / "docs" / "decisions"
    if not folder.is_dir():
        return
    index = folder / "README.md"
    listed = (
        set(LINK_RE.findall(index.read_text(encoding="utf-8")))
        if index.is_file()
        else set()
    )
    for adr in sorted(folder.glob("ADR-*.md")):
        if adr.name not in listed:
            errors.append(
                f"decisions: docs/decisions/README.md does not list {adr.name}"
            )


def check(root: Path, today: dt.date) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    agents_path = root / "AGENTS.md"
    if not agents_path.is_file():
        return ["agents: AGENTS.md is missing"], warnings
    agents = agents_path.read_text(encoding="utf-8")
    if len(agents.splitlines()) > AGENTS_CAP:
        errors.append(
            f"cap: AGENTS.md has {len(agents.splitlines())} lines (cap {AGENTS_CAP})"
        )
    for n, title in enumerate(SECTIONS, start=1):
        if not re.search(rf"^## {n}\. {re.escape(title)}", agents, re.M):
            errors.append(f"agents: section '## {n}. {title}' missing")
    claude = root / "CLAUDE.md"
    if (
        not claude.is_file()
        or claude.read_text(encoding="utf-8").strip() != "@AGENTS.md"
    ):
        errors.append("stub: CLAUDE.md must contain only '@AGENTS.md'")
    reached = walk(root, errors)
    for name in map_docs(root):
        if name not in reached and not any(
            name.startswith(d) for d in reached if d.endswith("/")
        ):
            errors.append(f"orphan: {name} is not reachable from AGENTS.md")
    check_headers(root, today, errors, warnings)
    check_map_yml(root, agents, errors)
    check_decisions(root, errors)
    return errors, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--today", type=dt.date.fromisoformat, default=dt.date.today())
    args = parser.parse_args(argv)
    errors, warnings = check(args.root, args.today)
    for line in warnings:
        print(f"warning: {line}")
    for line in errors:
        print(line)
    print("map check: FAILED" if errors else "map check: OK")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
