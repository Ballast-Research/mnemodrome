"""
Pinned model files for the local ONNX embedder (#134).

The files are downloaded on first run into EMBED_MODEL_DIR and verified by
sha256. A file with the wrong hash is never used: startup fails closed.
Offline installs place the same files in EMBED_MODEL_DIR by hand (see
INSTALLATION.md, "Offline embedding model").
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx
from loguru import logger

MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
MODEL_REVISION = "e9b6763023c676ca8431644204f50c2b100d9aab"
_HF_BASE = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}"


@dataclass(frozen=True)
class ModelFile:
    name: str  # file name inside EMBED_MODEL_DIR (and under EMBED_MODEL_URL)
    default_url: str
    sha256: str
    size: int


MODEL_FILES: tuple[ModelFile, ...] = (
    ModelFile(
        name="model.onnx",
        default_url=f"{_HF_BASE}/onnx/model.onnx",
        sha256="147d5aa88c2101237358e17796cf3a227cead1ec304ec34b465bb08e9d952965",
        size=547_310_275,
    ),
    ModelFile(
        name="vocab.txt",
        default_url=f"{_HF_BASE}/vocab.txt",
        sha256="07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
        size=231_508,
    ),
)

WEIGHTS_SHA256 = MODEL_FILES[0].sha256
VOCAB_SHA256 = MODEL_FILES[1].sha256


class ModelIntegrityError(RuntimeError):
    """A model file is missing, could not be fetched, or has the wrong hash."""


def default_model_dir() -> Path:
    base = os.getenv("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "mnemodrome" / "models"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(
    url: str, dest: Path, expected_sha256: str, max_bytes: int | None = None
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dest.name}.", dir=dest.parent)
    tmp = Path(tmp_name)
    digest = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as out:
            with httpx.stream(
                "GET", url, follow_redirects=True, timeout=httpx.Timeout(60.0)
            ) as resp:
                resp.raise_for_status()
                received = 0
                for chunk in resp.iter_bytes(1 << 20):
                    received += len(chunk)
                    if max_bytes is not None and received > max_bytes:
                        raise ModelIntegrityError(
                            f"{dest.name}: download exceeds the pinned size "
                            f"({max_bytes} bytes); aborted"
                        )
                    digest.update(chunk)
                    out.write(chunk)
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise ModelIntegrityError(
                f"{dest.name}: sha256 mismatch after download "
                f"(expected {expected_sha256}, got {actual})"
            )
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()


def _remove_stale_temp_files(model_dir: Path) -> None:
    """Delete partial downloads (`.<name>.*`) left by a killed process."""
    if not model_dir.is_dir():
        return
    for spec in MODEL_FILES:
        for stale in model_dir.glob(f".{spec.name}.*"):
            try:
                stale.unlink()
                logger.info(f"Removed partial download {stale.name}")
            except OSError as e:
                logger.warning(f"Could not remove {stale}: {e}")


def ensure_model_files(
    model_dir: Path, base_url: str | None = None, allow_download: bool = True
) -> dict[str, Path]:
    """Return verified paths of all model files, downloading missing ones.

    Raises ModelIntegrityError when a file is missing (and download is off or
    fails) or when any file's sha256 differs from the pin.
    """
    paths: dict[str, Path] = {}
    _remove_stale_temp_files(model_dir)
    for spec in MODEL_FILES:
        path = model_dir / spec.name
        if not path.exists():
            if not allow_download:
                raise ModelIntegrityError(
                    f"{path} is missing and download is disabled. "
                    "Place the pinned model files there (INSTALLATION.md)."
                )
            url = (
                f"{base_url.rstrip('/')}/{spec.name}" if base_url else spec.default_url
            )
            logger.info(
                f"Downloading embedding model file {spec.name} ({spec.size} bytes)"
            )
            try:
                _download(url, path, spec.sha256, max_bytes=spec.size)
            except ModelIntegrityError:
                raise
            except Exception as e:
                raise ModelIntegrityError(f"{spec.name}: download failed: {e}") from e
        actual = sha256_file(path)
        if actual != spec.sha256:
            raise ModelIntegrityError(
                f"{path}: sha256 mismatch (expected {spec.sha256}, got {actual}). "
                "Refusing to load it. Delete the file to download it again."
            )
        paths[spec.name] = path
    return paths


def main(argv: list[str] | None = None) -> int:
    """Print the pinned files, or fetch/verify them into a folder.

    python -m src.services.embedding.model_store             # list URLs and sha256
    python -m src.services.embedding.model_store --verify DIR
    python -m src.services.embedding.model_store --fetch DIR
    """
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="python -m src.services.embedding.model_store"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--verify", metavar="DIR", help="check the files in DIR")
    group.add_argument("--fetch", metavar="DIR", help="download missing files into DIR")
    parser.add_argument("--url", help="mirror base URL (default: Hugging Face)")
    args = parser.parse_args(argv)
    if not args.verify and not args.fetch:
        for spec in MODEL_FILES:
            print(f"{spec.name}\t{spec.size}\t{spec.sha256}\t{spec.default_url}")
        return 0
    target = Path(args.verify or args.fetch).expanduser()
    try:
        ensure_model_files(target, base_url=args.url, allow_download=bool(args.fetch))
    except ModelIntegrityError as e:
        print(f"FAIL: {e}")
        return 1
    print(f"OK: pinned model files verified in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
