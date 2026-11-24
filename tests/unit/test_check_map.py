"""Unit tests for scripts/check_map.py: a minimal map in a tmp dir, no network."""

import datetime as dt
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_map.py"
_spec = importlib.util.spec_from_file_location("check_map", SCRIPT)
cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cm)

TODAY = dt.date(2026, 11, 22)
HEADER = (
    "Status: active · Owner: the maintainers · Verified: 2026-11-20 · Verify-on: never"
)
ROWS = {
    "suggest": ("suggest", "`docs/decisions/README.md`", "overlay: decisions"),
    "idea": ('idea: "X"', "`docs/product/README.md`", "`CONTRIBUTING.md`"),
    "bug": ("bug: X", "`docs/product/README.md`", "`CONTRIBUTING.md`"),
    "change": ("change", "`docs/product/README.md`", "`CONTRIBUTING.md`"),
    "review": ("review", "`docs/product/README.md`", "overlay: review"),
    "release": ("release: tag", "`docs/product/README.md`", "`CONTRIBUTING.md`"),
    "deploy": ("deploy", "overlay: operations", "overlay runbook"),
    "live": ("live", "overlay: operations", "overlay runbook"),
    "secret": ("secret", "`docs/product/README.md`", "overlay: security"),
    "docs": ("docs", "`AGENTS.md`", "`docs/map.yml`"),
}


def agents_text(rows: dict = ROWS, extra: str = "") -> str:
    table = [
        "| Intent | Read first | Procedure | Gate (stop until) |",
        "|---|---|---|---|",
    ]
    table += [
        f"| {r} | {read} | {proc} | check: x |" for r, read, proc in rows.values()
    ]
    sections = [f"## {n}. {t}" for n, t in enumerate(cm.SECTIONS, start=1)]
    body = [sections[0], "See `README.md`.", sections[1], sections[2], *table]
    body += [sections[3], "Areas: `docs/product/README.md`, `docs/map.yml`.", extra]
    return "\n".join(["# AGENTS.md", *body, *sections[4:]]) + "\n"


def map_yml(rows: dict = ROWS) -> str:
    out = ["profile: full", "", "intents:"]
    for key, (row, read, proc) in rows.items():
        out += [f"  {key}:", f"    row: {row.split(':')[0]}"]
        out.append('    gates: ["check: x"]')
        if read.startswith("`"):
            out.append(f"    read: {read.strip('`')}")
        for cell in (read, proc):
            if cell.startswith("overlay: "):
                out.append(f"    overlay: {cell.split(': ')[1]}")
        if proc.startswith("`"):
            out.append(f'    procedure: ["{proc.strip("`")}"]')
    return "\n".join(out) + "\n"


def make_repo(tmp_path: Path) -> Path:
    (tmp_path / "AGENTS.md").write_text(agents_text())
    (tmp_path / "CLAUDE.md").write_text("@AGENTS.md\n")
    (tmp_path / "README.md").write_text("# Readme\n")
    (tmp_path / "CONTRIBUTING.md").write_text("# Contributing\n")
    (tmp_path / "docs" / "product").mkdir(parents=True)
    (tmp_path / "docs" / "decisions").mkdir()
    (tmp_path / "docs" / "map.yml").write_text(map_yml())
    (tmp_path / "docs" / "product" / "README.md").write_text(
        f"{HEADER}\n\n# Product\n\n[Decisions](../decisions/README.md)\n"
    )
    (tmp_path / "docs" / "decisions" / "README.md").write_text(
        f"{HEADER}\n\n# Decisions\n\n[ADR-001](ADR-001-x.md)\n"
    )
    (tmp_path / "docs" / "decisions" / "ADR-001-x.md").write_text(
        f"{HEADER.replace('active', 'proposed')}\n\n# ADR-001\n"
    )
    return tmp_path


def errors(root: Path) -> list[str]:
    return cm.check(root, TODAY)[0]


def test_valid_map_passes(tmp_path):
    assert errors(make_repo(tmp_path)) == []


def test_main_exit_codes(tmp_path, capsys):
    root = make_repo(tmp_path)
    assert cm.main(["--root", str(root), "--today", "2026-11-22"]) == 0
    (root / "CLAUDE.md").write_text("# full file\n")
    assert cm.main(["--root", str(root), "--today", "2026-11-22"]) == 1
    assert "map check: FAILED" in capsys.readouterr().out


def test_this_repo_map_passes():
    assert errors(SCRIPT.parents[1]) == []


def test_agents_over_cap_and_missing_section(tmp_path):
    root = make_repo(tmp_path)
    text = agents_text().replace("## 6. Refocus rule", "## 6. Something else")
    (root / "AGENTS.md").write_text(text + "x\n" * 80)
    found = errors(root)
    assert any(e.startswith("cap: AGENTS.md") for e in found)
    assert any("Refocus rule" in e for e in found)


def test_broken_link_and_backticked_path(tmp_path):
    root = make_repo(tmp_path)
    page = root / "docs" / "product" / "README.md"
    page.write_text(page.read_text() + "[gone](missing.md#anchor)\n")
    (root / "AGENTS.md").write_text(agents_text(extra="Also `scripts/nope.py`."))
    found = errors(root)
    assert "link: docs/product/README.md -> missing.md#anchor does not resolve" in found
    assert "link: AGENTS.md -> scripts/nope.py does not resolve" in found


def test_overlay_and_urls_are_not_links(tmp_path):
    root = make_repo(tmp_path)
    extra = "Read `.private/MAP.md`; see [site](https://example.com) and `origin/main`."
    (root / "AGENTS.md").write_text(agents_text(extra=extra))
    assert errors(root) == []


def _fresh(base):
    n = len(list(base.iterdir()))
    (base / f"r{n}").mkdir()
    return base / f"r{n}"


def _anchor_repo(tmp_path, link):
    root = make_repo(_fresh(tmp_path))
    (root / "docs" / "product" / "README.md").write_text(
        f"{HEADER}\n\n# Product\n\n## Same `Name`!\n\n## Same `Name`!\n\n"
        f"[Decisions](../decisions/README.md)\n[x]({link})\n"
    )
    return root


def test_good_anchors_pass(tmp_path):
    for link in ("#product", "#same-name", "../decisions/README.md#decisions"):
        assert errors(_anchor_repo(tmp_path, link)) == []


def test_duplicate_heading_suffix_passes(tmp_path):
    assert errors(_anchor_repo(tmp_path, "#same-name-1")) == []


def test_bad_anchor_fails(tmp_path):
    assert any("anchor:" in e for e in errors(_anchor_repo(tmp_path, "#nope")))
    bad = "../decisions/README.md#nope"
    assert any("anchor:" in e for e in errors(_anchor_repo(tmp_path, bad)))


def test_orphan_doc(tmp_path):
    root = make_repo(tmp_path)
    (root / "docs" / "product" / "lonely.md").write_text(f"{HEADER}\n\n# Lonely\n")
    assert "orphan: docs/product/lonely.md is not reachable from AGENTS.md" in errors(
        root
    )


def test_transfers_are_excluded_but_not_linkable(tmp_path):
    root = make_repo(tmp_path)
    (root / "docs" / "transfers").mkdir()
    (root / "docs" / "transfers" / "note.md").write_text("no header\n")
    assert errors(root) == []
    page = root / "docs" / "product" / "README.md"
    page.write_text(page.read_text() + "[t](../transfers/note.md)\n")
    assert any(e.startswith("transfer: docs/product/README.md") for e in errors(root))


def test_header_and_caps(tmp_path):
    root = make_repo(tmp_path)
    page = root / "docs" / "product" / "README.md"
    page.write_text(page.read_text() + "line\n" * 60)
    adr = root / "docs" / "decisions" / "ADR-001-x.md"
    adr.write_text("# ADR-001 without a header\n")
    found = errors(root)
    assert any(e.startswith("cap: docs/product/README.md") for e in found)
    assert (
        "header: docs/decisions/ADR-001-x.md line 1 is not a valid header line" in found
    )


def test_stale_verified_warns_only(tmp_path):
    root = make_repo(tmp_path)
    errs, warns = cm.check(root, dt.date(2027, 6, 1))
    assert errs == []
    assert any(w.startswith("stale: docs/product/README.md") for w in warns)


def test_map_yml_must_match_table(tmp_path):
    root = make_repo(tmp_path)
    rows = dict(ROWS)
    rows["bug"] = ("bug: X", "`docs/decisions/README.md`", "`CONTRIBUTING.md`")
    (root / "docs" / "map.yml").write_text(map_yml(rows))
    assert any(e.startswith("map.yml: intent 'bug' read") for e in errors(root))


def test_map_yml_gates_must_match_table(tmp_path):
    root = make_repo(tmp_path)
    text = (root / "docs" / "map.yml").read_text()
    (root / "docs" / "map.yml").write_text(text.replace("check: x", "check: y", 1))
    assert any("gates" in e and "!= table" in e for e in errors(root))


def test_gate_wording_ignores_quotes_backticks_and_spacing(tmp_path):
    root = make_repo(tmp_path)
    (root / "AGENTS.md").write_text(
        (root / "AGENTS.md").read_text().replace("check: x", 'check: `x`  "x"', 1)
    )
    text = (root / "docs" / "map.yml").read_text()
    (root / "docs" / "map.yml").write_text(text.replace("check: x", "check: x x", 1))
    assert errors(root) == []


def test_map_yml_required_intent_and_extra_row(tmp_path):
    root = make_repo(tmp_path)
    rows = {k: v for k, v in ROWS.items() if k != "suggest"}
    (root / "docs" / "map.yml").write_text(map_yml(rows))
    found = errors(root)
    assert "intent: required intent 'suggest' missing from docs/map.yml" in found
    assert "map.yml: table row 'suggest' has no intent in docs/map.yml" in found


def test_decisions_index_lists_every_adr(tmp_path):
    root = make_repo(tmp_path)
    (root / "docs" / "decisions" / "ADR-002-y.md").write_text(f"{HEADER}\n\n# 2\n")
    found = errors(root)
    assert "decisions: docs/decisions/README.md does not list ADR-002-y.md" in found


def test_parse_map_reads_lists_and_scalars():
    parsed = cm.parse_map(
        'profile: full\nintents:\n  a:\n    row: x, y\n    gates: ["check: z"]\n'
    )
    assert parsed == {"a": {"row": "x, y", "gates": ["check: z"]}}
