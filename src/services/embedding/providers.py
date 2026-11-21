"""
Embedding providers besides Ollama (#134).

- LocalOnnxEmbedder (`EMBED_PROVIDER=local-onnx`, experimental): runs the
  pinned nomic-embed-text-v1.5 fp32 ONNX model in-process. Mnemodrome owns
  the whole pipeline, so engine upgrades elsewhere never move its vectors.
- OpenAICompatibleEmbedder (`EMBED_PROVIDER=openai-compatible`,
  experimental): any `/v1/embeddings` endpoint. Mnemodrome cannot pin a
  remote engine, so it relies on the canary to detect drift.

The Ollama provider stays in `vector_service.OllamaEmbedder` (default).
All providers expose: `profile`, `is_remote`, `dimension`, `initialize()`,
`embed(text, role)`, `embed_batch(texts, role)`, `close()`.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from loguru import logger

from .model_store import (
    MODEL_ID,
    VOCAB_SHA256,
    WEIGHTS_SHA256,
    ModelIntegrityError,
    ensure_model_files,
)
from .profile import ROLE_QUERY, EmbedProfile
from .templates import TEMPLATE_VERSION
from .wordpiece import WordPieceTokenizer

NOMIC_DOCUMENT_PREFIX = "search_document: "
NOMIC_QUERY_PREFIX = "search_query: "
NOMIC_DIM = 768


class LocalOnnxEmbedder:
    """In-process nomic-embed-text-v1.5 (fp32) on ONNX Runtime, CPU only."""

    is_remote = False

    def __init__(
        self,
        model_dir: str | Path,
        base_url: str | None = None,
        max_tokens: int = 2048,
        threads: int = 2,
        allow_download: bool = True,
    ):
        self.model_dir = Path(model_dir).expanduser()
        self.base_url = base_url or None
        self.max_tokens = int(max_tokens)
        self.threads = max(1, int(threads))
        self.allow_download = allow_download
        self.model = MODEL_ID
        self._session = None
        self._tokenizer: WordPieceTokenizer | None = None
        self._input_names: set[str] = set()
        # One worker: concurrent requests queue instead of oversubscribing
        # the CPU; the async server loop never blocks on inference.
        self._executor: ThreadPoolExecutor | None = None
        self._profile = EmbedProfile(
            provider="local-onnx",
            model=MODEL_ID,
            dim=NOMIC_DIM,
            template_version=TEMPLATE_VERSION,
            document_prefix=NOMIC_DOCUMENT_PREFIX,
            query_prefix=NOMIC_QUERY_PREFIX,
            lowercase=True,
            max_tokens=self.max_tokens,
            pooling="mean",
            normalize=True,
            precision="fp32",
            weights_sha256=WEIGHTS_SHA256,
            vocab_sha256=VOCAB_SHA256,
        )

    @property
    def profile(self) -> EmbedProfile:
        return self._profile

    @property
    def dimension(self) -> int:
        return NOMIC_DIM

    async def initialize(self) -> bool:
        try:
            import onnxruntime as ort  # noqa: PLC0415
        except ImportError:
            logger.error(
                "EMBED_PROVIDER=local-onnx needs onnxruntime: install the "
                "'onnx' extra (uv sync --extra onnx)."
            )
            return False
        try:
            paths = await asyncio.to_thread(
                ensure_model_files, self.model_dir, self.base_url, self.allow_download
            )
        except ModelIntegrityError as e:
            logger.error(f"Local embedding model unavailable (fail closed): {e}")
            return False

        def _load():
            options = ort.SessionOptions()
            options.intra_op_num_threads = self.threads
            options.inter_op_num_threads = 1
            return ort.InferenceSession(
                str(paths["model.onnx"]),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )

        self._session = await asyncio.to_thread(_load)
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._tokenizer = WordPieceTokenizer.from_file(paths["vocab.txt"])
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mnemodrome-embed"
        )
        probe = await self.embed("test")
        if not probe or len(probe) != NOMIC_DIM:
            logger.error("Local ONNX embedder self-test failed")
            return False
        logger.success(
            f"Local ONNX embedder ready: {MODEL_ID} fp32, {NOMIC_DIM}D, "
            f"max_tokens={self.max_tokens}, threads={self.threads}, "
            f"fingerprint={self._profile.short_fingerprint}"
        )
        return True

    def _run(self, texts: list[str]) -> list[list[float]]:
        import numpy as np  # noqa: PLC0415

        assert self._tokenizer is not None and self._session is not None
        encoded = [self._tokenizer.encode(t, self.max_tokens) for t in texts]
        width = max(len(ids) for ids in encoded)
        input_ids = np.zeros((len(encoded), width), dtype=np.int64)
        mask = np.zeros((len(encoded), width), dtype=np.int64)
        for row, ids in enumerate(encoded):
            input_ids[row, : len(ids)] = ids
            mask[row, : len(ids)] = 1
        feeds = {"input_ids": input_ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids)
        hidden = self._session.run(None, feeds)[0]  # (batch, seq, dim)
        weights = mask[:, :, None].astype(np.float32)
        pooled = (hidden * weights).sum(axis=1) / np.clip(
            weights.sum(axis=1), 1e-9, None
        )
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled = pooled / np.clip(norms, 1e-12, None)
        return pooled.astype(np.float32).tolist()

    async def embed_batch(
        self, texts: list[str], role: str = ROLE_QUERY
    ) -> list[list[float] | None]:
        if not texts:
            return []
        prefix = self._profile.prefix(role)
        prepared = [prefix + t for t in texts]
        loop = asyncio.get_running_loop()
        try:
            vectors = await loop.run_in_executor(self._executor, self._run, prepared)
        except Exception as e:
            logger.error(f"Local ONNX embedding failed: {e}")
            return [None] * len(texts)
        return list(vectors)

    async def embed(self, text: str, role: str = ROLE_QUERY) -> list[float] | None:
        return (await self.embed_batch([text], role))[0]

    async def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None


class OpenAICompatibleEmbedder:
    """Any OpenAI-compatible `/v1/embeddings` endpoint (no task prefixes)."""

    is_remote = True

    def __init__(self, base_url: str, model: str, api_key: str | None = None):
        base = base_url.rstrip("/")
        self.url = (
            f"{base}/embeddings" if base.endswith("/v1") else f"{base}/v1/embeddings"
        )
        self.model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.AsyncClient(timeout=60.0, headers=headers)
        self._dim: int | None = None

    @property
    def dimension(self) -> int:
        return self._dim or 0

    @property
    def profile(self) -> EmbedProfile:
        return EmbedProfile(
            provider="openai-compatible",
            model=self.model,
            dim=self.dimension,
            template_version=TEMPLATE_VERSION,
        )

    async def initialize(self) -> bool:
        probe = await self.embed("test")
        if not probe:
            logger.error(f"openai-compatible embedder failed at {self.url}")
            return False
        self._dim = len(probe)
        logger.success(f"openai-compatible embedder ready: {self.model} ({self._dim}D)")
        return True

    async def embed_batch(
        self, texts: list[str], role: str = ROLE_QUERY
    ) -> list[list[float] | None]:
        if not texts:
            return []
        try:
            resp = await self.client.post(
                self.url, json={"model": self.model, "input": list(texts)}
            )
            resp.raise_for_status()
            data = sorted(resp.json().get("data", []), key=lambda d: d.get("index", 0))
            vectors = [d.get("embedding") for d in data]
            if len(vectors) != len(texts):
                raise ValueError(
                    f"expected {len(texts)} embeddings, got {len(vectors)}"
                )
            return vectors
        except Exception as e:
            logger.error(f"openai-compatible embedding failed: {e}")
            return [None] * len(texts)

    async def embed(self, text: str, role: str = ROLE_QUERY) -> list[float] | None:
        return (await self.embed_batch([text], role))[0]

    async def close(self):
        await self.client.aclose()
