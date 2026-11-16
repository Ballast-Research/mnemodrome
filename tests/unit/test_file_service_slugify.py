"""Tests for slugify() and filename safety in FileService.write_file (H2/H3)."""

import re
import time
from uuid import uuid4

import pytest

from src.config import MemoryType
from src.services.file_service import FileService, slugify


def _service(tmp_path) -> FileService:
    svc = FileService.__new__(FileService)
    svc.memory_paths = {}
    for mt in MemoryType:
        d = tmp_path / mt.value
        d.mkdir()
        svc.memory_paths[mt] = d
    svc.storage_path = svc.memory_paths[MemoryType.FACT]
    return svc


def test_slugify_slash_and_backslash():
    assert slugify("CI/CD") == "ci-cd"
    assert slugify("a\\b") == "a-b"


def test_slugify_dotdot_has_no_separators():
    out = slugify("../../etc/passwd")
    assert "/" not in out and ".." not in out
    assert out == "etc-passwd"


def test_slugify_unicode_kept():
    assert slugify("Café Münch 日本") == "café-münch-日本"


@pytest.mark.parametrize("value", ["", "   ", "///", "..", "!!!"])
def test_slugify_never_empty(value):
    assert slugify(value) == "untitled"


def test_slugify_long_is_capped():
    out = slugify("a" * 500, max_length=50)
    assert len(out) == 50
    assert slugify("word " * 100, max_length=20) == "word-word-word-word"


def test_write_file_slash_name_stays_in_dir(tmp_path):
    svc = _service(tmp_path)
    path = svc.write_file("fact-CI/CD-20260101-000000.md", "x", MemoryType.FACT)
    assert path.parent == svc.memory_paths[MemoryType.FACT]
    assert path.exists()


def test_write_file_traversal_contained(tmp_path):
    svc = _service(tmp_path)
    path = svc.write_file("../../evil.md", "x", MemoryType.FACT)
    assert path.parent == svc.memory_paths[MemoryType.FACT]
    assert not (tmp_path.parent / "evil.md").exists()


def test_same_second_names_differ(tmp_path):
    """H3: names built like the call sites do not collide within one second."""
    svc = _service(tmp_path)
    ts = time.strftime("%Y%m%d-%H%M%S")
    names = [f"graph-{ts}-{uuid4().hex[:8]}.md" for _ in range(2)]
    paths = [svc.write_file(n, str(i), MemoryType.GRAPH) for i, n in enumerate(names)]
    assert paths[0] != paths[1]
    assert len(list(svc.memory_paths[MemoryType.GRAPH].glob("*.md"))) == 2
    assert re.fullmatch(r"graph-\d{8}-\d{6}-[0-9a-f]{8}\.md", paths[0].name)


def test_slugify_caps_bytes_not_characters():
    out = slugify("日本語" * 100, max_length=80)
    assert len(out.encode("utf-8")) <= 80
    assert out  # cut on a character boundary, still valid text


def test_long_non_ascii_name_writes(tmp_path):
    svc = _service(tmp_path)
    ts = time.strftime("%Y%m%d-%H%M%S")
    name = f"fact-{slugify('日本語' * 100)}-{ts}-{uuid4().hex[:8]}.md"
    path = svc.write_file(name, "x", MemoryType.FACT)
    assert path.exists()
    assert len(path.name.encode("utf-8")) <= 255
    assert path.name.endswith(f"-{name[-11:-3]}.md")  # uuid suffix survives

    # even a hostile over-long name is cut to a writable length
    path2 = svc.write_file("日本語" * 200, "x", MemoryType.FACT)
    assert path2.exists() and len(path2.name.encode("utf-8")) <= 255


def test_written_path_equals_precomputed_path(tmp_path):
    """Call sites (e.g. capture_learning) store a path before writing."""
    svc = _service(tmp_path)
    ts = time.strftime("%Y%m%d-%H%M%S")
    for title in ["CI/CD pipeline", "Plain title", "日本語 テスト", "a\\b..c"]:
        filename = f"learning-{slugify(title, max_length=50)}-{ts}-{uuid4().hex[:8]}.md"
        stored = svc._get_target_path(MemoryType.LEARNING) / filename
        written = svc.write_file(filename, "x", MemoryType.LEARNING)
        assert written == stored
        assert stored.exists()
