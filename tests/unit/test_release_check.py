"""Unit tests for scripts/release_check.py: tmp dirs and fake tag lists, no network."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "release_check.py"
_spec = importlib.util.spec_from_file_location("release_check", SCRIPT)
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)

HEAD = "a" * 40


def tags(*names: str) -> dict[str, str]:
    return {n: f"{i:040x}" for i, n in enumerate(names, start=1)}


def make_repo(
    tmp_path: Path,
    version: str = "1.1.0",
    heading: str | None = None,
    lock: str | None = "",
) -> Path:
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "mnemodrome"\nversion = "{version}"\n'
    )
    heading = heading if heading is not None else f"## [{version}] - 2026-11-21"
    (tmp_path / "CHANGELOG.md").write_text(
        f"# Changelog\n\n## [Unreleased]\n\n{heading}\n\n- thing\n"
    )
    if lock is not None:
        lock = lock or version
        (tmp_path / "uv.lock").write_text(
            f'version = 1\n\n[[package]]\nname = "other"\nversion = "9.9.9"\n\n'
            f'[[package]]\nname = "mnemodrome"\nversion = "{lock}"\nsource = {{ editable = "." }}\n'
        )
    return tmp_path


@pytest.mark.parametrize(
    "tag,expected",
    [
        ("v1.1.0", (1, 1, 0, None)),
        ("v1.1.0rc1", (1, 1, 0, 1)),
        ("v10.20.30rc12", (10, 20, 30, 12)),
        ("1.1.0", None),
        ("v1.1", None),
        ("v1.1.0-rc.1", None),
        ("v1.1.0a1", None),
        ("v1.1.0.dev1", None),
        ("v1.1.0.post1", None),
        ("v1.1.0rc", None),
        ("v1.1.0rc1x", None),
        ("v1.0.0\n", None),
        ("v01.0.0", None),
        ("v1.00.0", None),
        ("v1.1.0rc0", None),
        ("v1.1.0rc01", None),
        ("v\u0661.0.0", None),
    ],
)
def test_parse_tag(tag, expected):
    assert rc.parse_tag(tag) == expected


def test_is_prerelease():
    assert rc.is_prerelease("v1.1.0rc2") is True
    assert rc.is_prerelease("v1.1.0") is False


def test_all_checks_pass(tmp_path):
    root = make_repo(tmp_path)
    assert rc.run_checks(root, "v1.1.0", tags("v1.0.0"), HEAD) == []


def test_bad_format_reported(tmp_path):
    root = make_repo(tmp_path, version="1.1.0-rc.1")
    errors = rc.run_checks(root, "v1.1.0-rc.1", tags("v1.0.0"), HEAD)
    assert any(e.startswith("format:") for e in errors)


def test_tag_version_mismatch(tmp_path):
    root = make_repo(tmp_path, version="1.1.0")
    errors = rc.run_checks(root, "v1.1.1", tags("v1.0.0"), HEAD)
    assert any(e.startswith("version:") for e in errors)


def test_missing_changelog_heading(tmp_path):
    root = make_repo(tmp_path, heading="## [1.1.0]")
    errors = rc.run_checks(root, "v1.1.0", tags("v1.0.0"), HEAD)
    assert [e for e in errors if e.startswith("changelog:")]


def test_changelog_heading_needs_date_and_exact_version(tmp_path):
    root = make_repo(tmp_path, heading="## [1.1.0rc1] - 2026-11-21")
    assert any(
        e.startswith("changelog:")
        for e in rc.run_checks(root, "v1.1.0", tags("v1.0.0"), HEAD)
    )


def test_lock_mismatch(tmp_path):
    root = make_repo(tmp_path, lock="1.0.0")
    errors = rc.run_checks(root, "v1.1.0", tags("v1.0.0"), HEAD)
    assert [e for e in errors if e.startswith("lock:")]


def test_lock_absent_is_skipped(tmp_path):
    root = make_repo(tmp_path, lock=None)
    assert rc.run_checks(root, "v1.1.0", tags("v1.0.0"), HEAD) == []


@pytest.mark.parametrize("tag", ["v1.0.1", "v1.1.0", "v1.0.1rc1", "v1.1.0rc1"])
def test_sequence_legal_after_final(tag):
    assert rc.check_sequence(tag, tags("v1.0.0"), HEAD) == []


@pytest.mark.parametrize(
    "tag", ["v1.0.2", "v1.2.0", "v1.1.1", "v2.0.0", "v0.9.0", "v1.0.0rc1"]
)
def test_sequence_illegal_after_final(tag):
    errors = rc.check_sequence(tag, tags("v1.0.0"), HEAD)
    assert errors and all(e.startswith("sequence:") for e in errors)


def test_major_needs_flag():
    assert rc.check_sequence("v2.0.0", tags("v1.4.2"), HEAD)
    assert rc.check_sequence("v2.0.0", tags("v1.4.2"), HEAD, allow_major=True) == []
    assert rc.check_sequence("v2.0.0rc1", tags("v1.4.2"), HEAD, allow_major=True) == []
    assert rc.check_sequence("v3.0.0", tags("v1.4.2"), HEAD, allow_major=True)


def test_rc_must_follow_previous_rc():
    existing = tags("v1.0.0", "v1.1.0rc1")
    assert rc.check_sequence("v1.1.0rc2", existing, HEAD) == []
    assert rc.check_sequence("v1.1.0", existing, HEAD) == []
    assert rc.check_sequence("v1.1.0rc3", existing, HEAD)
    assert rc.check_sequence("v1.1.0rc1", existing, HEAD)  # reuse


def test_open_rc_line_blocks_other_versions():
    existing = tags("v1.0.0", "v1.1.0rc1")
    errors = rc.check_sequence("v1.2.0rc1", existing, HEAD)
    assert errors and "rc line 1.1.0 is open" in errors[0]
    assert rc.check_sequence("v1.0.2", existing, HEAD)  # skips 1.0.1
    assert rc.check_sequence("v1.2.0", existing, HEAD)


def test_hotfix_allowed_while_rc_open():
    existing = tags("v1.0.0", "v1.1.0rc1")
    assert rc.check_sequence("v1.0.1", existing, HEAD) == []
    assert rc.check_sequence("v1.0.1rc1", existing, HEAD) == []
    assert rc.check_sequence("v1.1.0rc2", existing, HEAD) == []


def test_next_hotfix_after_hotfix_while_rc_open():
    existing = tags("v1.0.0", "v1.1.0rc1", "v1.0.1")
    assert rc.check_sequence("v1.0.2", existing, HEAD) == []
    assert rc.check_sequence("v1.0.3", existing, HEAD)
    assert rc.check_sequence("v1.1.0", existing, HEAD) == []


def test_hotfix_rc_numbering_still_enforced():
    existing = tags("v1.0.0", "v1.1.0rc1", "v1.0.1rc1")
    assert rc.check_sequence("v1.0.1rc2", existing, HEAD) == []
    assert rc.check_sequence("v1.0.1rc3", existing, HEAD)


def test_rc_line_closed_after_final():
    existing = tags("v1.0.0", "v1.1.0rc1", "v1.1.0rc2", "v1.1.0")
    assert rc.check_sequence("v1.1.1", existing, HEAD) == []
    assert rc.check_sequence("v1.2.0rc1", existing, HEAD) == []


def test_tag_on_head_is_not_reuse():
    existing = {"v1.0.0": "b" * 40, "v1.1.0": HEAD}
    assert rc.check_sequence("v1.1.0", existing, HEAD) == []


def test_tag_on_other_commit_is_reuse():
    existing = {"v1.0.0": "b" * 40, "v1.1.0": "c" * 40}
    errors = rc.check_sequence("v1.1.0", existing, HEAD)
    assert any("already exists" in e for e in errors)


def test_no_existing_tags_accepts_first_tag():
    assert rc.check_sequence("v1.0.0", {}, HEAD) == []


def _git(root: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def test_main_against_real_git_repo(tmp_path, capsys):
    root = make_repo(tmp_path, version="1.1.0", lock=None)
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "x")
    _git(root, "tag", "v1.0.0")  # lightweight, on HEAD (fine for the prior final)
    _git(root, "tag", "-a", "v1.1.0", "-m", "m")
    assert rc.main(["--root", str(root)]) == 0
    assert "passed" in capsys.readouterr().out
    assert rc.main(["v1.1.0", "--root", str(root)]) == 0


def test_main_reports_failures_and_nonzero(tmp_path, capsys):
    root = make_repo(tmp_path, version="1.1.0", lock="1.0.0")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "x")
    _git(root, "tag", "-a", "v1.1.0", "-m", "m")
    assert rc.main(["--root", str(root)]) == 1
    assert capsys.readouterr().out.startswith("lock:")


def test_main_no_tag_on_head(tmp_path, capsys):
    root = make_repo(tmp_path)
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "x")
    assert rc.main(["--root", str(root)]) == 1
    assert capsys.readouterr().out.startswith("tag:")


def test_is_prerelease_mode(capsys):
    assert rc.main(["v1.1.0rc1", "--is-prerelease"]) == 0
    assert capsys.readouterr().out.strip() == "true"
    assert rc.main(["v1.1.0", "--is-prerelease"]) == 0
    assert capsys.readouterr().out.strip() == "false"
    assert rc.main(["v1.1.0-rc.1", "--is-prerelease"]) == 1


def test_is_prerelease_rejects_trailing_newline():
    assert rc.main(["v1.1.0\n", "--is-prerelease"]) == 1


def test_hotfix_rc_accepted_while_rc_open():
    assert rc.check_sequence("v1.0.1rc1", tags("v1.0.0", "v1.1.0rc1"), HEAD) == []


def test_second_hotfix_rejected_before_first_exists():
    errors = rc.check_sequence("v1.0.2", tags("v1.0.0", "v1.1.0rc1"), HEAD)
    assert errors and "hotfix 1.0.1" in errors[0]


def test_no_final_tag_has_no_hotfix_exception():
    errors = rc.check_sequence("v1.0.1", tags("v1.0.0rc1"), HEAD)
    assert errors and "hotfix" not in errors[0] and "rc line 1.0.0 is open" in errors[0]
    assert rc.check_sequence("v1.0.0rc2", tags("v1.0.0rc1"), HEAD) == []
