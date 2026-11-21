"""Embedding profile, fingerprint, templates, canary and model store (#134)."""

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from src.services.embedding import canary, model_store, templates
from src.services.embedding.profile import (
    ROLE_DOCUMENT,
    ROLE_QUERY,
    EmbedProfile,
    cosine,
)
from src.services.embedding.providers import LocalOnnxEmbedder


def _profile(**kw):
    base = dict(provider="local-onnx", model="m", dim=768, template_version=1)
    base.update(kw)
    return EmbedProfile(**base)


class TestFingerprint:
    def test_stable_for_equal_profiles(self):
        assert _profile().fingerprint == _profile().fingerprint
        assert len(_profile().fingerprint) == 64
        assert _profile().short_fingerprint == _profile().fingerprint[:8]

    @pytest.mark.parametrize(
        "change",
        [
            {"model": "other"},
            {"dim": 512},
            {"template_version": 2},
            {"document_prefix": "search_document: "},
            {"query_prefix": "search_query: "},
            {"max_tokens": 512},
            {"precision": "int8"},
            {"weights_sha256": "abc"},
            {"vocab_sha256": "abc"},
            {"pooling": "cls"},
            {"normalize": True},
            {"lowercase": True},
        ],
    )
    def test_every_pipeline_setting_changes_it(self, change):
        assert replace(_profile(), **change).fingerprint != _profile().fingerprint

    def test_library_versions_are_not_part_of_it(self):
        fields = set(_profile().to_dict())
        assert not {"onnxruntime", "numpy", "library_version"} & fields

    def test_local_onnx_profile_pins_artifacts(self):
        profile = LocalOnnxEmbedder("/nonexistent").profile
        assert profile.weights_sha256 == model_store.WEIGHTS_SHA256
        assert profile.vocab_sha256 == model_store.VOCAB_SHA256
        assert profile.precision == "fp32"
        assert profile.max_tokens == 2048
        assert profile.prefix(ROLE_DOCUMENT) == "search_document: "
        assert profile.prefix(ROLE_QUERY) == "search_query: "

    def test_unknown_role_rejected(self):
        with pytest.raises(ValueError):
            _profile().prefix("passage")


class TestTemplates:
    """Templates must reproduce the pre-#134 strings byte for byte."""

    def test_learning(self):
        assert templates.learning_text("T", "C") == "T\n\nC"

    def test_session(self):
        assert templates.session_text("S") == "S"
        assert templates.session_text("S", ["a", "b"]) == "S\n\nKey Insights:\n- a\n- b"

    def test_fact(self):
        assert templates.fact_text("A", "R", "B") == "A R B"
        assert templates.fact_text("A", "R", "B", "ctx") == "A R B\nContext: ctx"

    def test_procedure(self):
        assert templates.procedure_text("sym", "cause") == "sym cause"

    @pytest.mark.parametrize(
        "memory_type,payload,expected",
        [
            ("prompts", {"text": "hello"}, "hello"),
            ("learnings", {"title": "T", "content": "C"}, "T\n\nC"),
            (
                "sessions",
                {"summary": "S", "key_insights": ["i"]},
                "S\n\nKey Insights:\n- i",
            ),
            (
                "facts",
                {"entity1": "A", "relation": "R", "entity2": "B", "context": ""},
                "A R B",
            ),
            ("procedures", {"symptom": "s", "root_cause": "c"}, "s c"),
            ("prompts", {}, None),
            ("facts", {"entity1": "A"}, None),
        ],
    )
    def test_text_from_payload(self, memory_type, payload, expected):
        assert templates.text_from_payload(memory_type, payload) == expected


class TestCanary:
    def test_identical_vectors_pass(self):
        ok, worst = canary.compare([[1.0, 0.0]], [[1.0, 0.0]], canary.LOCAL_TOLERANCE)
        assert ok and worst == pytest.approx(1.0)

    def test_drift_fails(self):
        ok, worst = canary.compare([[1.0, 0.0]], [[0.9, 0.44]], canary.REMOTE_TOLERANCE)
        assert not ok and worst < 0.999

    def test_length_mismatch_fails(self):
        assert canary.compare([[1.0]], [], canary.LOCAL_TOLERANCE) == (False, 0.0)

    def test_golden_set_covers_case_and_unicode(self):
        texts = canary.CANARY_TEXTS
        assert any(t.isupper() for t in texts)
        assert any(any(ord(ch) > 0x3000 for ch in t) for t in texts)
        assert any(len(t) > 2000 for t in texts)

    def test_cosine(self):
        assert cosine([1, 0], [0, 1]) == 0.0
        assert cosine([0, 0], [1, 1]) == 0.0


class TestModelStore:
    @pytest.fixture
    def tiny_files(self, monkeypatch, tmp_path):
        blobs = {"model.onnx": b"weights", "vocab.txt": b"[PAD]\n"}
        specs = tuple(
            model_store.ModelFile(
                name=n,
                default_url=f"https://example.invalid/{n}",
                sha256=hashlib.sha256(b).hexdigest(),
                size=len(b),
            )
            for n, b in blobs.items()
        )
        monkeypatch.setattr(model_store, "MODEL_FILES", specs)
        return blobs

    def test_offline_dir_with_pinned_files(self, tiny_files, tmp_path):
        for name, blob in tiny_files.items():
            (tmp_path / name).write_bytes(blob)
        paths = model_store.ensure_model_files(tmp_path, allow_download=False)
        assert set(paths) == set(tiny_files)

    def test_missing_file_without_download_fails_closed(self, tiny_files, tmp_path):
        with pytest.raises(model_store.ModelIntegrityError, match="missing"):
            model_store.ensure_model_files(tmp_path, allow_download=False)

    def test_wrong_hash_fails_closed(self, tiny_files, tmp_path):
        for name in tiny_files:
            (tmp_path / name).write_bytes(b"tampered")
        with pytest.raises(model_store.ModelIntegrityError, match="sha256 mismatch"):
            model_store.ensure_model_files(tmp_path, allow_download=False)

    def test_download_uses_mirror_and_verifies(self, tiny_files, tmp_path, monkeypatch):
        seen = []

        def fake_download(url, dest: Path, expected):
            seen.append(url)
            dest.write_bytes(tiny_files[dest.name])

        monkeypatch.setattr(model_store, "_download", fake_download)
        model_store.ensure_model_files(tmp_path, base_url="https://mirror.invalid/m/")
        assert seen == [
            "https://mirror.invalid/m/model.onnx",
            "https://mirror.invalid/m/vocab.txt",
        ]

    def test_failed_download_fails_closed(self, tiny_files, tmp_path, monkeypatch):
        def broken(url, dest, expected):
            raise OSError("network down")

        monkeypatch.setattr(model_store, "_download", broken)
        with pytest.raises(model_store.ModelIntegrityError, match="download failed"):
            model_store.ensure_model_files(tmp_path)

    def test_default_dir_follows_xdg(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        assert model_store.default_model_dir() == tmp_path / "mnemodrome" / "models"

    def test_pins_are_full_sha256(self):
        for spec in model_store.MODEL_FILES:
            assert len(spec.sha256) == 64
            assert model_store.MODEL_REVISION in spec.default_url


@pytest.mark.asyncio
async def test_local_onnx_init_fails_closed_without_model(tmp_path):
    pytest.importorskip("onnxruntime")
    embedder = LocalOnnxEmbedder(tmp_path, allow_download=False)
    assert await embedder.initialize() is False


def test_model_store_cli_lists_and_verifies(tmp_path, capsys):
    assert model_store.main([]) == 0
    out = capsys.readouterr().out
    assert "model.onnx" in out and model_store.WEIGHTS_SHA256 in out
    assert model_store.main(["--verify", str(tmp_path)]) == 1
