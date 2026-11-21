"""
Vector Service (Feature 022: Semantic Vector Layer)

Provides semantic similarity search as the first stage in Mnemodrome's
retrieval cascade: Vector → Redis → Neo4j → Markdown

Constitutional Principle: Always-on observation
Every prompt is automatically embedded and searched for similar past interactions.

Embedding providers (#134), chosen by EMBED_PROVIDER:
- ollama (default): Ollama with nomic-embed-text (768 dimensions)
- local-onnx (experimental): pinned nomic-embed-text-v1.5 fp32, in-process
- openai-compatible (experimental): any /v1/embeddings endpoint

Every collection carries a fingerprint and a canary (src/services/embedding).
Code reads and writes through the `mnemodrome_<type>_live` aliases.
"""

import asyncio
import os
import time
import unicodedata
import uuid
from datetime import datetime
from typing import Any

import httpx
from loguru import logger

from src.services.embedding import canary as _canary
from src.services.embedding import registry as _registry
from src.services.embedding import templates as _templates
from src.services.embedding.model_store import default_model_dir
from src.services.embedding.profile import ROLE_DOCUMENT, ROLE_QUERY, EmbedProfile

try:
    from src import config as _config  # type: ignore[import]
except ImportError:
    _config = None  # type: ignore[assignment]

EMBED_PROVIDERS = ("ollama", "local-onnx", "openai-compatible")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        FieldCondition,
        Filter,
        IsEmptyCondition,
        MatchValue,
        PayloadField,
        PointStruct,
    )

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("Qdrant client not available. Vector layer disabled.")


class OllamaEmbedder:
    """
    Handles embedding generation via Ollama API.

    Uses nomic-embed-text model (768 dimensions) for semantic search.
    Legacy profile: no task prefixes, so `role` is accepted and ignored and
    vectors stay byte-identical to earlier releases.
    """

    is_remote = True

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        fallback_host: str | None = None,
    ):
        """
        Initialize Ollama embedder.

        Args:
            host: Ollama API host
            model: Embedding model name
            fallback_host: Fallback Ollama host for when primary is unavailable
        """
        self.host = host.rstrip("/")
        self.fallback_host = fallback_host.rstrip("/") if fallback_host else None
        self._using_fallback = False
        self.model = model
        self.client = httpx.AsyncClient(timeout=30.0)
        self._dimension: int | None = None

    async def initialize(self) -> bool:
        """
        Initialize and verify Ollama connection.
        Falls back to secondary host if primary is unavailable.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if model is available on primary
            response = await self.client.get(f"{self.host}/api/tags")
            response.raise_for_status()

            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]

            if (
                self.model not in model_names
                and f"{self.model}:latest" not in model_names
            ):
                logger.warning(
                    f"Model {self.model} not found on primary. Available: {model_names}. "
                    f"Run: ollama pull {self.model}"
                )
                if self.fallback_host:
                    return await self._try_fallback_init()
                return False

            # Test embedding to get dimension
            test_embedding = await self.embed("test")
            if test_embedding:
                self._dimension = len(test_embedding)
                logger.success(
                    f"Ollama embedder initialized: {self.model} ({self._dimension}D)"
                )
                if self.fallback_host:
                    logger.info(f"Ollama fallback available at {self.fallback_host}")
                return True

            return False

        except Exception as e:
            logger.warning(f"Primary Ollama embedder failed: {e}")
            if self.fallback_host:
                return await self._try_fallback_init()
            logger.error(
                "Failed to initialize Ollama embedder (no fallback configured)"
            )
            return False

    async def _try_fallback_init(self) -> bool:
        """Attempt initialization using fallback host."""
        logger.info(f"Trying fallback Ollama at {self.fallback_host}...")
        try:
            response = await self.client.get(f"{self.fallback_host}/api/tags")
            response.raise_for_status()

            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]

            if (
                self.model not in model_names
                and f"{self.model}:latest" not in model_names
            ):
                logger.error(
                    f"Model {self.model} not found on fallback. Available: {model_names}"
                )
                return False

            # Switch to fallback
            self.host = self.fallback_host
            self._using_fallback = True

            test_embedding = await self.embed("test")
            if test_embedding:
                self._dimension = len(test_embedding)
                logger.success(
                    f"Ollama embedder initialized via FALLBACK: {self.model} "
                    f"({self._dimension}D) at {self.fallback_host}"
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Fallback Ollama also failed: {e}")
            return False

    @property
    def profile(self) -> EmbedProfile:
        return EmbedProfile(
            provider="ollama",
            model=self.model,
            dim=self.dimension,
            template_version=_templates.TEMPLATE_VERSION,
        )

    async def embed_batch(
        self, texts: list[str], role: str = ROLE_QUERY
    ) -> list[list[float] | None]:
        return [await self.embed(t, role) for t in texts]

    async def _embed_at(self, host: str, text: str) -> list[float] | None:
        response = await self.client.post(
            f"{host}/api/embeddings", json={"model": self.model, "prompt": text}
        )
        response.raise_for_status()
        return response.json().get("embedding")

    async def check_fallback_parity(self) -> bool | None:
        """Embed the canary on primary and fallback; drop a fallback that differs.

        A fallback on another Ollama version would write vectors with different
        behaviour into the same collection (#134). Returns None when there is no
        fallback or a host is unreachable, True when they agree, False when the
        fallback was disabled.
        """
        if not self.fallback_host or self._using_fallback:
            return None
        try:
            worst = 1.0
            for text in _canary.CANARY_TEXTS:
                a = await self._embed_at(self.host, text)
                b = await self._embed_at(self.fallback_host, text)
                if not a or not b:
                    return None
                worst = min(worst, _canary.cosine(a, b))
        except Exception as e:
            logger.warning(f"Fallback parity check skipped: {e}")
            return None
        if worst < _canary.REMOTE_TOLERANCE:
            logger.error(
                f"Ollama fallback {self.fallback_host} embeds differently from the "
                f"primary (canary min cosine {worst:.4f}); fallback disabled."
            )
            self.fallback_host = None
            return False
        return True

    async def embed(self, text: str, role: str = ROLE_QUERY) -> list[float] | None:
        """
        Generate embedding for text. Falls back to secondary host on failure.

        Args:
            text: Input text
            role: "document" or "query" (ignored: the Ollama profile has no prefixes)

        Returns:
            Embedding vector or None if failed
        """
        try:
            response = await self.client.post(
                f"{self.host}/api/embeddings",
                json={"model": self.model, "prompt": text},
            )
            response.raise_for_status()

            result = response.json()
            embedding = result.get("embedding")

            if not embedding:
                logger.error(f"No embedding in Ollama response: {result}")
                return None

            return embedding

        except Exception as e:
            # Try fallback if available and not already using it
            if self.fallback_host and not self._using_fallback:
                logger.warning(f"Primary Ollama embed failed ({e}), trying fallback...")
                try:
                    response = await self.client.post(
                        f"{self.fallback_host}/api/embeddings",
                        json={"model": self.model, "prompt": text},
                    )
                    response.raise_for_status()
                    result = response.json()
                    embedding = result.get("embedding")
                    if embedding:
                        logger.info("Embedding generated via fallback Ollama")
                        # Switch to fallback for subsequent calls
                        self.host = self.fallback_host
                        self._using_fallback = True
                        return embedding
                except Exception as fallback_e:
                    logger.error(f"Fallback Ollama also failed: {fallback_e}")

            logger.error(f"Failed to generate embedding: {e}")
            return None

    @property
    def dimension(self) -> int:
        """Get embedding dimension.

        Returns the probed dimension from the live model when available.
        Falls back to EMBEDDING_DIM env var (default 768) so the value is
        consistent whether or not the Ollama endpoint was reachable at startup.
        """
        if self._dimension:
            return self._dimension
        default = int(_config.EMBEDDING_DIM) if _config is not None else 768
        return int(os.getenv("EMBEDDING_DIM", str(default)))

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


class VectorService:
    """
    Vector similarity search service for Mnemodrome semantic memory layer.

    Responsibilities:
    - Generate embeddings for prompts, learnings, facts, and sessions (EMBED_PROVIDER)
    - Store embeddings in Qdrant with metadata linking to Neo4j UUIDs
    - Search for semantically similar content
    - Maintain similarity threshold for greenfield vs. existing topic detection
    """

    def __init__(
        self,
        qdrant_host: str | None = None,
        qdrant_port: int = 6333,
        ollama_host: str | None = None,
        ollama_fallback_host: str | None = None,
        embedding_model: str = "nomic-embed-text",
        similarity_threshold: float = 0.45,
        enabled: bool = True,
        embed_provider: str | None = None,
    ):
        """
        Initialize Vector Service.

        Args:
            qdrant_host: Qdrant server host (default: from env QDRANT_HOST)
            qdrant_port: Qdrant HTTP port (default: 6333)
            ollama_host: Ollama API host (default: from env OLLAMA_HOST)
            ollama_fallback_host: Fallback Ollama host (default: from env OLLAMA_FALLBACK_URL)
            embedding_model: Ollama embedding model (default: nomic-embed-text)
            similarity_threshold: Minimum similarity score for context retrieval (0.0-1.0)
            enabled: Enable vector layer (default: from env QDRANT_ENABLED)
        """
        self.qdrant_host = qdrant_host or os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = qdrant_port
        # Prefer dedicated embedding GPU (OLLAMA_EMBEDDING_HOST) over general OLLAMA_HOST
        self.ollama_host = ollama_host or os.getenv(
            "OLLAMA_EMBEDDING_HOST", os.getenv("OLLAMA_HOST", "http://localhost:11434")
        )
        self.ollama_fallback_host = ollama_fallback_host or os.getenv(
            "OLLAMA_FALLBACK_URL", ""
        )
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.enabled = enabled and QDRANT_AVAILABLE
        self.embed_provider = (
            (embed_provider or os.getenv("EMBED_PROVIDER", "ollama")).strip().lower()
        )
        if self.embed_provider not in EMBED_PROVIDERS:
            raise ValueError(
                f"EMBED_PROVIDER must be one of {', '.join(EMBED_PROVIDERS)}; "
                f"got {self.embed_provider!r}"
            )

        self.client: QdrantClient | None = None
        self.embedder: Any = None
        self.registry: _registry.CollectionRegistry | None = None
        self._initialized = False

        # Logical collection names: aliases to the physical collections (#134).
        self.COLLECTION_PROMPTS = _registry.live_alias("prompts")
        self.COLLECTION_LEARNINGS = _registry.live_alias("learnings")
        self.COLLECTION_FACTS = _registry.live_alias("facts")
        self.COLLECTION_SESSIONS = _registry.live_alias("sessions")
        self.COLLECTION_PROCEDURES = _registry.live_alias("procedures")
        # memory type -> {"alias", "collection", "status", "detail"}
        self.collection_status: dict[str, dict[str, Any]] = {}
        self.blocked_writes: dict[str, int] = {}
        self._last_recheck: dict[str, float] = {}
        self._canary_cache: list[list[float]] | None = None
        self._registered_fp: dict[str, str | None] = {}

        if not self.enabled:
            logger.info("Vector layer disabled")
            return

        logger.info(
            f"Vector service configured (qdrant={self.qdrant_host}:{self.qdrant_port}, "
            f"provider={self.embed_provider}, ollama={self.ollama_host}, "
            f"fallback={self.ollama_fallback_host or 'none'}, "
            f"model={self.embedding_model}, threshold={self.similarity_threshold})"
        )

    def _build_embedder(self):
        """Create the embedder named by EMBED_PROVIDER (not yet initialized)."""
        if self.embed_provider == "local-onnx":
            from src.services.embedding.providers import LocalOnnxEmbedder

            return LocalOnnxEmbedder(
                model_dir=os.getenv("EMBED_MODEL_DIR") or default_model_dir(),
                base_url=os.getenv("EMBED_MODEL_URL") or None,
                max_tokens=int(os.getenv("EMBED_MAX_TOKENS", "2048")),
                threads=int(os.getenv("EMBED_THREADS", "2")),
                allow_download=os.getenv("EMBED_MODEL_DOWNLOAD", "true").lower()
                == "true",
            )
        if self.embed_provider == "openai-compatible":
            from src.services.embedding.providers import OpenAICompatibleEmbedder

            base_url = os.getenv("EMBED_BASE_URL", "")
            if not base_url:
                raise ValueError(
                    "EMBED_PROVIDER=openai-compatible needs EMBED_BASE_URL"
                )
            return OpenAICompatibleEmbedder(
                base_url=base_url,
                model=os.getenv("EMBED_MODEL", self.embedding_model),
                api_key=os.getenv("EMBED_API_KEY") or None,
            )
        return OllamaEmbedder(
            host=self.ollama_host,
            model=self.embedding_model,
            fallback_host=self.ollama_fallback_host or None,
        )

    def _active_profile(self) -> EmbedProfile | None:
        profile = getattr(self.embedder, "profile", None)
        return profile if isinstance(profile, EmbedProfile) else None

    def _active_fingerprint(self) -> str:
        profile = self._active_profile()
        return profile.fingerprint if profile else ""

    def _payload(
        self, base: dict[str, Any], metadata: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Build a point payload. `embed_*` keys are reserved: caller metadata
        cannot overwrite the fingerprint tag (#134)."""
        extra = {
            k: v
            for k, v in (metadata or {}).items()
            if not str(k).startswith(_registry.RESERVED_PREFIX)
        }
        return {
            **base,
            **extra,
            _registry.FP_KEY: self._active_fingerprint(),
            _registry.TEMPLATE_KEY: _templates.TEMPLATE_VERSION,
        }

    def tag_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Public helper for other writers (procedures)."""
        return self._payload(payload, None)

    def is_trusted(self, memory_type: str) -> bool:
        """True when scores in this collection come from the active embedder.

        `unknown` (the check could not run) is NOT trusted: dedup fails safe
        and writes are held back until `ensure_trusted()` re-checks. A type
        with no entry at all (the collection setup never ran, e.g. unit tests
        with mocks) keeps the earlier behaviour.
        """
        state = self.collection_status.get(memory_type)
        if state is None:
            return True
        return state.get("status") in _registry.TRUSTED

    async def ensure_trusted(self, memory_type: str) -> bool:
        """is_trusted(), re-running the check first when the state is unknown.

        The re-check is rate-limited (RECHECK_INTERVAL_S) so a broken embedder
        does not add a canary run to every capture.
        """
        state = self.collection_status.get(memory_type)
        if (
            state is not None
            and state.get("status") == _registry.STATUS_UNKNOWN
            and state.get("collection")
        ):
            now = time.monotonic()
            if (
                now - self._last_recheck.get(memory_type, 0.0)
                >= self.RECHECK_INTERVAL_S
            ):
                self._last_recheck[memory_type] = now
                try:
                    status, detail = await self._check_fingerprint(
                        memory_type, state["collection"], created=False
                    )
                    state.update(
                        status=status,
                        detail=detail,
                        fingerprint=self._registered_fp.get(state["collection"]),
                    )
                    logger.info(
                        f"Embedding re-check {memory_type}: {status} ({detail})"
                    )
                except Exception as e:
                    logger.warning(f"Embedding re-check failed for {memory_type}: {e}")
        return self.is_trusted(memory_type)

    async def _writable(self, memory_type: str) -> bool:
        """Block vector writes into a collection built by another embedder (#134).

        Writing anyway would mix two embedders in one collection. The capture
        itself (graph, files) is unaffected; only its vector is not stored.
        Rebuild with `python -m src.cli.reindex` to clear the block.
        """
        if await self.ensure_trusted(memory_type):
            return True
        self.blocked_writes[memory_type] = self.blocked_writes.get(memory_type, 0) + 1
        status = self.collection_status.get(memory_type, {}).get("status")
        logger.error(
            f"Vector write blocked for {memory_type}: collection status '{status}' "
            f"does not match the active embedder ({self.embed_provider}). "
            f"Blocked so far: {self.blocked_writes[memory_type]}. "
            "Rebuild with python -m src.cli.reindex (see vector_health)."
        )
        return False

    def trust_warning(self, memory_types: list[str] | tuple[str, ...]) -> str | None:
        """Warning text for search output when any searched collection is not trusted."""
        bad = [
            f"{t} ({self.collection_status[t]['status']})"
            for t in memory_types
            if not self.is_trusted(t)
        ]
        if not bad:
            return None
        return (
            "Embedding fingerprint mismatch in: "
            + ", ".join(bad)
            + ". Scores are not reliable until the collections are rebuilt "
            "(python -m src.cli.reindex). See vector_health."
        )

    async def initialize(self):
        """Initialize Qdrant client and Ollama embedder."""
        if not self.enabled:
            return

        if self._initialized:
            return

        try:
            logger.info(f"Initializing embedder (provider={self.embed_provider})...")
            self.embedder = self._build_embedder()

            embedder_ready = await self.embedder.initialize()
            if not embedder_ready:
                logger.error(f"Embedder initialization failed ({self.embed_provider})")
                self.enabled = False
                return
            fallback_check = getattr(self.embedder, "check_fallback_parity", None)
            if self.embed_provider == "ollama" and fallback_check is not None:
                await fallback_check()

            embedding_dim = self.embedder.dimension

            # Initialize Qdrant client
            logger.info(
                f"Connecting to Qdrant at {self.qdrant_host}:{self.qdrant_port}..."
            )
            self.client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)

            # Test connection
            collections = await asyncio.to_thread(self.client.get_collections)
            logger.info(
                f"Connected to Qdrant. Existing collections: {len(collections.collections)}"
            )

            # Ensure collections exist
            await self._ensure_collections(embedding_dim)

            self._initialized = True
            logger.success("Vector service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize vector service: {e}")
            self.enabled = False
            raise

    def _get_collection_dimension(self, collection_info) -> int | None:
        """Extract the vector dimension from an existing Qdrant collection info object.

        Handles both single-vector collections (VectorParams directly) and
        named-vector collections (dict of VectorParams).
        Returns None if the dimension cannot be determined.
        """
        try:
            vectors = collection_info.config.params.vectors
            if vectors is None:
                return None
            # Single-vector collection: vectors is a VectorParams with .size
            if hasattr(vectors, "size"):
                return int(vectors.size)
            # Named-vector collection: vectors is a dict
            if isinstance(vectors, dict):
                for v in vectors.values():
                    if hasattr(v, "size"):
                        return int(v.size)
        except Exception:
            pass
        return None

    RECHECK_INTERVAL_S = 30.0

    async def _ensure_collections(self, dimension: int):
        """Resolve every `_live` alias, create missing collections, and check
        each collection's dimension and embedding fingerprint (#134).

        Mnemodrome will NOT auto-delete or modify existing collections. A
        fingerprint mismatch is reported in `collection_status` and in
        `health_check()`; rebuild with `python -m src.cli.reindex`.
        """
        self.registry = _registry.CollectionRegistry(self.client)
        self._canary_cache = None
        logger.info(
            f"Unicode data version for the tokenizer: {unicodedata.unidata_version}"
        )
        try:
            await asyncio.to_thread(self.registry.ensure)
        except Exception as e:
            logger.warning(
                f"Embedding registry unavailable, fingerprint checks off: {e}"
            )
            self.registry = None

        for memory_type in _templates.MEMORY_TYPES:
            try:
                await self._ensure_collection(memory_type, dimension)
            except Exception as e:
                logger.warning(f"Collection setup check failed for {memory_type}: {e}")
                self.collection_status[memory_type] = {
                    "alias": _registry.live_alias(memory_type),
                    "collection": None,
                    "status": _registry.STATUS_UNKNOWN,
                    "detail": str(e),
                }

    async def _ensure_collection(self, memory_type: str, dimension: int):
        """Resolve alias -> physical collection, create if needed, check fingerprint."""
        alias = _registry.live_alias(memory_type)
        reg = self.registry
        if reg is None:
            self.collection_status[memory_type] = {
                "alias": alias,
                "collection": None,
                "status": _registry.STATUS_UNKNOWN,
                "detail": "registry unavailable",
            }
            return

        created = False
        physical = await asyncio.to_thread(reg.alias_target, alias)
        if physical is None:
            legacy = _registry.legacy_name(memory_type)
            if await asyncio.to_thread(reg.physical_exists, legacy):
                physical = legacy
                logger.info(f"Pointing alias {alias} at existing collection {legacy}")
            else:
                profile = self._active_profile()
                physical = (
                    _registry.physical_name(memory_type, profile) if profile else legacy
                )
                if not await asyncio.to_thread(reg.physical_exists, physical):
                    logger.info(f"Creating collection: {physical}")
                    await asyncio.to_thread(reg.create_physical, physical, dimension)
                    created = True
            await asyncio.to_thread(reg.point_alias, alias, physical)

        await self._warn_on_dimension_mismatch(physical, dimension)
        status, detail = await self._check_fingerprint(memory_type, physical, created)
        self.collection_status[memory_type] = {
            "alias": alias,
            "collection": physical,
            "status": status,
            "detail": detail,
            "fingerprint": self._registered_fp.get(physical),
        }
        if status == _registry.STATUS_UNKNOWN:
            self._last_recheck[memory_type] = time.monotonic()
        level = "info" if status in _registry.TRUSTED else "error"
        getattr(logger, level)(
            f"Embedding check {memory_type}: {physical} -> {status} ({detail})"
        )

    async def _embed_canary(self) -> list[list[float]] | None:
        """Canary vectors of the active embedder: computed once per start,
        reused for every collection (one retry on failure)."""
        if self._canary_cache is not None:
            return self._canary_cache
        for attempt in range(2):
            vectors = await self.embedder.embed_batch(
                list(_canary.CANARY_TEXTS), ROLE_DOCUMENT
            )
            if vectors and all(vectors):
                self._canary_cache = [list(map(float, v)) for v in vectors]
                return self._canary_cache
            if attempt == 0:
                await asyncio.sleep(1.0)
        return None

    async def _check_fingerprint(
        self, memory_type: str, physical: str, created: bool
    ) -> tuple[str, str]:
        reg = self.registry
        profile = self._active_profile()
        if reg is None or profile is None:
            return _registry.STATUS_UNKNOWN, "no registry or profile"
        record = await asyncio.to_thread(reg.get, physical)
        self._registered_fp[physical] = (record or {}).get("fingerprint")
        if record is None:
            count = (
                await asyncio.to_thread(
                    self.client.count, collection_name=physical, exact=True
                )
            ).count
            # Ollama wrote every legacy vector, so it may adopt them; any other
            # provider can only adopt an empty collection.
            if count and self.embed_provider != "ollama":
                return (
                    _registry.STATUS_UNTAGGED,
                    f"{count} legacy points not made by {self.embed_provider}",
                )
            canary_vectors = await self._embed_canary()
            if canary_vectors is None:
                return _registry.STATUS_UNKNOWN, "canary embedding failed"
            adopted = bool(count) and not created
            await asyncio.to_thread(
                reg.register,
                physical,
                memory_type,
                profile,
                canary_vectors,
                adopted,
                None,
                unicodedata.unidata_version,
            )
            await asyncio.to_thread(reg.ensure_fp_index, physical)
            self._registered_fp[physical] = profile.fingerprint
            return (
                _registry.STATUS_ADOPTED if adopted else _registry.STATUS_OK,
                f"registered fingerprint {profile.short_fingerprint}",
            )
        unicode_note = ""
        recorded_unicode = record.get("unicode_version")
        if recorded_unicode and recorded_unicode != unicodedata.unidata_version:
            # Not part of the fingerprint: it moves tokens only for rare code
            # points. Reported so a rebuild can be planned (see wordpiece.py).
            unicode_note = (
                f"; unicode {recorded_unicode} -> {unicodedata.unidata_version}"
            )
            logger.warning(
                f"{physical} was built with Unicode {recorded_unicode}, this Python "
                f"has {unicodedata.unidata_version}: rare code points may tokenize "
                "differently"
            )
        if record.get("fingerprint") != profile.fingerprint:
            return (
                _registry.STATUS_MISMATCH,
                f"collection {str(record.get('fingerprint'))[:8]} != active "
                f"{profile.short_fingerprint} ({self.embed_provider})",
            )
        current = await self._embed_canary()
        if current is None:
            return _registry.STATUS_UNKNOWN, "canary embedding failed"
        tolerance = (
            _canary.REMOTE_TOLERANCE
            if getattr(self.embedder, "is_remote", True)
            else _canary.LOCAL_TOLERANCE
        )
        ok, worst = _canary.compare(
            record.get("canary_vectors") or [], current, tolerance
        )
        status = (
            (_registry.STATUS_ADOPTED if record.get("adopted") else _registry.STATUS_OK)
            if ok
            else _registry.STATUS_CANARY_DRIFT
        )
        return (
            status,
            f"canary min cosine {worst:.6f} (tolerance {tolerance}){unicode_note}",
        )

    async def _warn_on_dimension_mismatch(self, collection_name: str, dimension: int):
        """Log a clear error when a collection's dimension differs from the model.

        Mnemodrome will NOT auto-delete or modify collections; rebuild them with
        `python -m src.cli.reindex`.
        """
        try:
            collection_info = await asyncio.to_thread(
                self.client.get_collection, collection_name
            )
            existing_dim = self._get_collection_dimension(collection_info)
            if existing_dim is not None and existing_dim != dimension:
                logger.error(
                    f"EMBEDDING DIMENSION MISMATCH detected in collection "
                    f"'{collection_name}': stored={existing_dim}D vs "
                    f"active model={dimension}D (provider: {self.embed_provider}, "
                    f"model: {self.embedding_model}, "
                    f"EMBEDDING_DIM={os.getenv('EMBEDDING_DIM', '768')}). "
                    f"Similarity searches will produce wrong results. "
                    f"TO FIX: rebuild the collection with the active embedder "
                    f"(python -m src.cli.reindex). Mnemodrome will NOT auto-delete "
                    f"collections."
                )
        except Exception as check_err:
            logger.warning(
                f"Could not verify dimension for existing collection "
                f"'{collection_name}': {check_err}"
            )

    async def embed_text(self, text: str, role: str = ROLE_QUERY) -> list[float]:
        """
        Generate embedding vector for text with the active embedder.

        Args:
            text: Input text to embed
            role: "query" (search input, default) or "document" (stored text).
                Only providers with task prefixes (local-onnx) use it.

        Returns:
            Embedding vector as list of floats
        """
        if not self.enabled or not self._initialized:
            return []

        # Input validation: strip and reject empty/trivial input
        text = text.strip()
        if not text or len(text) < 3:
            logger.warning(f"embed_text: rejected trivial input (len={len(text)})")
            return []

        try:
            embedding = await self.embedder.embed(text, role=role)
            if embedding and len(embedding) != self.embedder.dimension:
                logger.error(
                    f"Embedding dimension mismatch: got {len(embedding)}, "
                    f"expected {self.embedder.dimension}"
                )
                return []
            return embedding or []
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return []

    async def search_similar_prompts_by_embedding(
        self,
        query_embedding: list[float],
        limit: int = 5,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for semantically similar prompts using a pre-computed embedding.

        Task 2 Optimization: Reuse embedding for search operations
        Avoids regenerating embedding when already computed.

        Args:
            query_embedding: Pre-computed embedding vector
            limit: Maximum number of results
            min_score: Minimum similarity score (default: self.similarity_threshold)

        Returns:
            List of similar prompts with metadata and scores
        """
        if not self.enabled or not self._initialized:
            return []

        if not query_embedding:
            return []

        min_score = min_score or self.similarity_threshold

        try:
            # Search Qdrant using pre-computed embedding (no GPU call)
            results = await asyncio.to_thread(
                self.client.query_points,
                collection_name=self.COLLECTION_PROMPTS,
                query=query_embedding,
                limit=limit,
                score_threshold=min_score,
            )

            # Format results
            matches = []
            for hit in results.points:
                matches.append(
                    {
                        "id": hit.id,
                        "score": hit.score,
                        "payload": hit.payload,
                        "matched_text": hit.payload.get("text", ""),
                        "timestamp": hit.payload.get("timestamp"),
                        "neo4j_uuid": hit.payload.get("neo4j_uuid"),
                    }
                )

            if matches:
                scores_str = ", ".join([f"{m['score']:.3f}" for m in matches])
                logger.info(
                    f"Found {len(matches)} similar prompts via cached embedding (scores: {scores_str})"
                )
            else:
                logger.info(f"No similar prompts found (threshold: {min_score})")

            return matches

        except Exception as e:
            logger.error(f"Vector search by embedding failed: {e}")
            return []

    async def search_similar_learnings_by_embedding(
        self,
        query_embedding: list[float],
        limit: int = 5,
        category: str | None = None,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for semantically similar learnings using a pre-computed embedding.

        Task 2 Optimization: Reuse embedding across search operations

        Args:
            query_embedding: Pre-computed embedding vector
            limit: Maximum results
            category: Optional category filter
            min_score: Minimum similarity score

        Returns:
            List of similar learnings with scores
        """
        if not self.enabled or not self._initialized:
            return []

        if not query_embedding:
            return []

        min_score = min_score or self.similarity_threshold

        try:
            # Build filter if category specified
            query_filter = None
            if category:
                query_filter = Filter(
                    must=[
                        FieldCondition(key="category", match=MatchValue(value=category))
                    ]
                )

            results = await asyncio.to_thread(
                self.client.query_points,
                collection_name=self.COLLECTION_LEARNINGS,
                query=query_embedding,
                query_filter=query_filter,
                limit=limit,
                score_threshold=min_score,
            )

            matches = []
            for hit in results.points:
                matches.append(
                    {
                        "id": hit.id,
                        "score": hit.score,
                        "title": hit.payload.get("title"),
                        "content": hit.payload.get("content"),
                        "category": hit.payload.get("category"),
                        "timestamp": hit.payload.get("timestamp"),
                        "neo4j_uuid": hit.payload.get("neo4j_uuid"),
                    }
                )

            if matches:
                logger.info(
                    f"Found {len(matches)} similar learnings via cached embedding"
                )

            return matches

        except Exception as e:
            logger.error(f"Learning search by embedding failed: {e}")
            return []

    async def store_prompt_embedding_direct(
        self,
        text: str,
        embedding: list[float],
        prompt_id: str | uuid.UUID | None = None,
        neo4j_uuid: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | bool:
        """
        Store prompt embedding using a pre-computed embedding vector.

        Task 2 Optimization: Store without regenerating embedding
        Fire-and-forget operation for batched embedding reuse.

        Args:
            text: Prompt text
            embedding: Pre-computed embedding vector (no GPU call)
            prompt_id: Unique identifier for this prompt
            neo4j_uuid: Optional UUID linking to Neo4j node
            metadata: Additional metadata to store

        Returns:
            UUID string of stored point if successful, False otherwise
        """
        if not self.enabled or not self._initialized:
            return False

        if not await self._writable("prompts"):
            return False

        if not embedding:
            return False

        try:
            # Convert or generate UUID
            if prompt_id is None:
                point_id = uuid.uuid4()
            elif isinstance(prompt_id, uuid.UUID):
                point_id = prompt_id
            else:
                # Try to parse as UUID, otherwise generate new one
                try:
                    point_id = uuid.UUID(prompt_id)
                except ValueError:
                    # Generate deterministic UUID from string
                    point_id = uuid.uuid5(uuid.NAMESPACE_DNS, prompt_id)

            # Prepare payload
            payload = self._payload(
                {
                    "text": text,
                    "timestamp": datetime.utcnow().isoformat(),
                    "neo4j_uuid": neo4j_uuid,
                },
                metadata,
            )

            # Store in Qdrant using pre-computed embedding
            await asyncio.to_thread(
                self.client.upsert,
                collection_name=self.COLLECTION_PROMPTS,
                points=[
                    PointStruct(id=str(point_id), vector=embedding, payload=payload)
                ],
            )

            logger.debug(
                f"Stored prompt embedding (pre-computed): {str(point_id)[:8]}..."
            )
            return str(point_id)

        except Exception as e:
            logger.error(f"Failed to store prompt embedding directly: {e}")
            return False

    async def search_similar_prompts(
        self, query_text: str, limit: int = 5, min_score: float | None = None
    ) -> list[dict[str, Any]]:
        """
        Search for semantically similar prompts.

        Args:
            query_text: User prompt to find similar content for
            limit: Maximum number of results
            min_score: Minimum similarity score (default: self.similarity_threshold)

        Returns:
            List of similar prompts with metadata and scores
        """
        if not self.enabled or not self._initialized:
            return []

        min_score = min_score or self.similarity_threshold

        try:
            # Generate query embedding
            query_vector = await self.embed_text(query_text)
            if not query_vector:
                return []

            # Search Qdrant
            results = await asyncio.to_thread(
                self.client.query_points,
                collection_name=self.COLLECTION_PROMPTS,
                query=query_vector,
                limit=limit,
                score_threshold=min_score,
            )

            # Format results
            matches = []
            for hit in results.points:
                matches.append(
                    {
                        "id": hit.id,
                        "score": hit.score,
                        "payload": hit.payload,
                        "matched_text": hit.payload.get("text", ""),
                        "timestamp": hit.payload.get("timestamp"),
                        "neo4j_uuid": hit.payload.get("neo4j_uuid"),
                    }
                )

            if matches:
                scores_str = ", ".join([f"{m['score']:.3f}" for m in matches])
                logger.info(
                    f"Found {len(matches)} similar prompts (scores: {scores_str})"
                )
            else:
                # Diagnostic: show what scores exist below threshold
                try:
                    diag_results = await asyncio.to_thread(
                        self.client.query_points,
                        collection_name=self.COLLECTION_PROMPTS,
                        query=query_vector,
                        limit=3,
                    )
                    if diag_results.points:
                        top_scores = ", ".join(
                            [f"{h.score:.3f}" for h in diag_results.points]
                        )
                        logger.info(
                            f"No prompts above threshold {min_score}. "
                            f"Top scores below threshold: [{top_scores}]"
                        )
                    else:
                        logger.info(
                            "No similar prompts found (collection may be empty)"
                        )
                except Exception:
                    logger.info(f"No similar prompts found (threshold: {min_score})")

            return matches

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def store_prompt_embedding(
        self,
        text: str,
        prompt_id: str | uuid.UUID | None = None,
        neo4j_uuid: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | bool:
        """
        Store prompt embedding in Qdrant.

        Args:
            text: Prompt text
            prompt_id: Unique identifier for this prompt (UUID or str to convert, None to generate)
            neo4j_uuid: Optional UUID linking to Neo4j node
            metadata: Additional metadata to store

        Returns:
            UUID string of stored point if successful, False otherwise
        """
        if not self.enabled or not self._initialized:
            return False

        if not await self._writable("prompts"):
            return False

        try:
            # Prompts are compared with prompts: one role on both sides, so a
            # vector made for search can be stored as is.
            embedding = await self.embed_text(_templates.prompt_text(text), ROLE_QUERY)
            if not embedding:
                return False

            # Convert or generate UUID
            if prompt_id is None:
                point_id = uuid.uuid4()
            elif isinstance(prompt_id, uuid.UUID):
                point_id = prompt_id
            else:
                # Try to parse as UUID, otherwise generate new one
                try:
                    point_id = uuid.UUID(prompt_id)
                except ValueError:
                    # Generate deterministic UUID from string
                    point_id = uuid.uuid5(uuid.NAMESPACE_DNS, prompt_id)

            # Prepare payload
            payload = self._payload(
                {
                    "text": text,
                    "timestamp": datetime.utcnow().isoformat(),
                    "neo4j_uuid": neo4j_uuid,
                },
                metadata,
            )

            # Store in Qdrant
            await asyncio.to_thread(
                self.client.upsert,
                collection_name=self.COLLECTION_PROMPTS,
                points=[
                    PointStruct(id=str(point_id), vector=embedding, payload=payload)
                ],
            )

            logger.debug(f"Stored prompt embedding: {str(point_id)[:8]}...")
            return str(point_id)

        except Exception as e:
            logger.error(f"Failed to store prompt embedding: {e}")
            return False

    async def store_learning_embedding(
        self,
        title: str,
        content: str,
        learning_uuid: str | uuid.UUID,
        category: str = "general",
        metadata: dict[str, Any] | None = None,
    ) -> str | bool:
        """
        Store learning embedding in Qdrant.

        Args:
            title: Learning title
            content: Learning content
            learning_uuid: UUID from Neo4j Learning node (UUID or string)
            category: Learning category
            metadata: Additional metadata

        Returns:
            UUID string if successful, False otherwise
        """
        if not self.enabled or not self._initialized:
            return False

        if not await self._writable("learnings"):
            return False

        try:
            # Combine title and content for richer embedding
            combined_text = _templates.learning_text(title, content)
            embedding = await self.embed_text(combined_text, ROLE_DOCUMENT)
            if not embedding:
                return False

            # Convert to UUID
            if isinstance(learning_uuid, uuid.UUID):
                point_id = learning_uuid
            else:
                try:
                    point_id = uuid.UUID(learning_uuid)
                except ValueError:
                    point_id = uuid.uuid5(uuid.NAMESPACE_DNS, learning_uuid)

            payload = self._payload(
                {
                    "title": title,
                    "content": content,
                    "category": category,
                    "timestamp": datetime.utcnow().isoformat(),
                    "neo4j_uuid": str(point_id),
                    "type": "learning",
                },
                metadata,
            )

            await asyncio.to_thread(
                self.client.upsert,
                collection_name=self.COLLECTION_LEARNINGS,
                points=[
                    PointStruct(id=str(point_id), vector=embedding, payload=payload)
                ],
            )

            logger.debug(f"Stored learning embedding: {str(point_id)[:8]}...")
            return str(point_id)

        except Exception as e:
            logger.error(f"Failed to store learning embedding: {e}")
            return False

    async def store_document(
        self, content: str, metadata: dict[str, Any] | None = None
    ) -> str | bool:
        """
        Store document embedding in Qdrant (generic wrapper for backwards compatibility).

        This method wraps store_learning_embedding for capture_learning calls.

        Args:
            content: Document content (may include title)
            metadata: Metadata dict containing type, uuid, title, category, etc.

        Returns:
            UUID string if successful, False otherwise
        """
        if not self.enabled or not self._initialized:
            return False

        try:
            # Extract metadata
            meta = metadata or {}
            doc_type = meta.get("type", "document")
            doc_uuid = meta.get("uuid")
            title = meta.get("title", "")
            category = meta.get("category", "general")

            if not doc_uuid:
                logger.error("store_document called without uuid in metadata")
                return False

            # For learnings, use the learning-specific method
            if doc_type == "learning":
                return await self.store_learning_embedding(
                    title=title,
                    content=content,
                    learning_uuid=doc_uuid,
                    category=category,
                    metadata=meta,
                )

            # For other document types, store in learnings collection
            # (can be extended for other collections in the future)
            return await self.store_learning_embedding(
                title=title,
                content=content,
                learning_uuid=doc_uuid,
                category=category,
                metadata=meta,
            )

        except Exception as e:
            logger.error(f"Failed to store document embedding: {e}")
            return False

    async def store_session_embedding(
        self,
        summary: str,
        session_uuid: str | uuid.UUID,
        key_insights: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | bool:
        """
        Store session embedding in Qdrant.

        Args:
            summary: Session summary text
            session_uuid: UUID from Neo4j Session node (UUID or string)
            key_insights: Optional list of key insights
            metadata: Additional metadata

        Returns:
            UUID string if successful, False otherwise
        """
        if not self.enabled or not self._initialized:
            return False

        if not await self._writable("sessions"):
            return False

        try:
            # Combine summary and key insights for richer embedding
            combined_text = _templates.session_text(summary, key_insights)
            embedding = await self.embed_text(combined_text, ROLE_DOCUMENT)
            if not embedding:
                return False

            # Convert to UUID
            if isinstance(session_uuid, uuid.UUID):
                point_id = session_uuid
            else:
                try:
                    point_id = uuid.UUID(session_uuid)
                except ValueError:
                    point_id = uuid.uuid5(uuid.NAMESPACE_DNS, session_uuid)

            payload = self._payload(
                {
                    "summary": summary,
                    "key_insights": key_insights or [],
                    "timestamp": datetime.utcnow().isoformat(),
                    "neo4j_uuid": str(point_id),
                    "type": "session",
                },
                metadata,
            )

            await asyncio.to_thread(
                self.client.upsert,
                collection_name=self.COLLECTION_SESSIONS,
                points=[
                    PointStruct(id=str(point_id), vector=embedding, payload=payload)
                ],
            )

            logger.debug(f"Stored session embedding: {str(point_id)[:8]}...")
            return str(point_id)

        except Exception as e:
            logger.error(f"Failed to store session embedding: {e}")
            return False

    async def store_fact_embedding(
        self,
        entity1: str,
        relation: str,
        entity2: str,
        context: str,
        fact_uuid: str | uuid.UUID,
        metadata: dict[str, Any] | None = None,
    ) -> str | bool:
        """
        Store fact embedding in Qdrant.

        Args:
            entity1: Source entity name
            relation: Relationship type
            entity2: Target entity name
            context: Context or evidence for this fact
            fact_uuid: UUID from Neo4j relationship (UUID or string)
            metadata: Additional metadata (entity types, UUIDs, etc.)

        Returns:
            UUID string if successful, False otherwise
        """
        if not self.enabled or not self._initialized:
            return False

        if not await self._writable("facts"):
            return False

        try:
            # Combine fact components for embedding
            embed_text = _templates.fact_text(entity1, relation, entity2, context)
            embedding = await self.embed_text(embed_text, ROLE_DOCUMENT)
            if not embedding:
                return False

            # Convert to UUID
            if isinstance(fact_uuid, uuid.UUID):
                point_id = fact_uuid
            else:
                try:
                    point_id = uuid.UUID(fact_uuid)
                except ValueError:
                    point_id = uuid.uuid5(uuid.NAMESPACE_DNS, fact_uuid)

            payload = self._payload(
                {
                    "entity1": entity1,
                    "relation": relation,
                    "entity2": entity2,
                    "context": context,
                    "timestamp": datetime.utcnow().isoformat(),
                    "neo4j_uuid": str(point_id),
                    "type": "fact",
                },
                metadata,
            )

            await asyncio.to_thread(
                self.client.upsert,
                collection_name=self.COLLECTION_FACTS,
                points=[
                    PointStruct(id=str(point_id), vector=embedding, payload=payload)
                ],
            )

            logger.debug(f"Stored fact embedding: {str(point_id)[:8]}...")
            return str(point_id)

        except Exception as e:
            logger.error(f"Failed to store fact embedding: {e}")
            return False

    async def search_similar_learnings(
        self,
        query_text: str,
        limit: int = 5,
        category: str | None = None,
        min_score: float | None = None,
        role: str = ROLE_QUERY,
    ) -> list[dict[str, Any]]:
        """
        Search for semantically similar learnings.

        Args:
            query_text: Query text
            limit: Maximum results
            category: Optional category filter
            min_score: Minimum similarity score
            role: "query" (default) or "document" (dedup: stored text vs stored text)

        Returns:
            List of similar learnings with scores
        """
        if not self.enabled or not self._initialized:
            return []

        min_score = min_score or self.similarity_threshold

        try:
            query_vector = await self.embed_text(query_text, role)
            if not query_vector:
                return []

            # Build filter if category specified
            query_filter = None
            if category:
                query_filter = Filter(
                    must=[
                        FieldCondition(key="category", match=MatchValue(value=category))
                    ]
                )

            results = await asyncio.to_thread(
                self.client.query_points,
                collection_name=self.COLLECTION_LEARNINGS,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=min_score,
            )

            matches = []
            for hit in results.points:
                matches.append(
                    {
                        "id": hit.id,
                        "score": hit.score,
                        "title": hit.payload.get("title"),
                        "content": hit.payload.get("content"),
                        "category": hit.payload.get("category"),
                        "timestamp": hit.payload.get("timestamp"),
                        "neo4j_uuid": hit.payload.get("neo4j_uuid"),
                    }
                )

            if matches:
                scores_str = ", ".join([f"{m['score']:.3f}" for m in matches])
                logger.info(
                    f"Found {len(matches)} similar learnings (scores: {scores_str})"
                )
            else:
                # Diagnostic: show what scores exist below threshold
                try:
                    diag_results = await asyncio.to_thread(
                        self.client.query_points,
                        collection_name=self.COLLECTION_LEARNINGS,
                        query=query_vector,
                        limit=3,
                    )
                    if diag_results.points:
                        top_scores = ", ".join(
                            [f"{h.score:.3f}" for h in diag_results.points]
                        )
                        logger.info(
                            f"No learnings above threshold {min_score}. "
                            f"Top scores below threshold: [{top_scores}]"
                        )
                    else:
                        logger.info(
                            "No similar learnings found (collection may be empty)"
                        )
                except Exception:
                    logger.info(f"No similar learnings found (threshold: {min_score})")

            return matches

        except Exception as e:
            logger.error(f"Learning search failed: {e}")
            return []

    async def search_similar_sessions(
        self, query_text: str, limit: int = 5, min_score: float | None = None
    ) -> list[dict[str, Any]]:
        """
        Search for semantically similar sessions.

        Args:
            query_text: Query text
            limit: Maximum results
            min_score: Minimum similarity score

        Returns:
            List of similar sessions with scores
        """
        if not self.enabled or not self._initialized:
            return []

        min_score = min_score or self.similarity_threshold

        try:
            query_vector = await self.embed_text(query_text)
            if not query_vector:
                return []

            results = await asyncio.to_thread(
                self.client.query_points,
                collection_name=self.COLLECTION_SESSIONS,
                query=query_vector,
                limit=limit,
                score_threshold=min_score,
            )

            matches = []
            for hit in results.points:
                matches.append(
                    {
                        "id": hit.id,
                        "score": hit.score,
                        "summary": hit.payload.get("summary"),
                        "key_insights": hit.payload.get("key_insights", []),
                        "timestamp": hit.payload.get("timestamp"),
                        "neo4j_uuid": hit.payload.get("neo4j_uuid"),
                    }
                )

            if matches:
                logger.info(f"Found {len(matches)} similar sessions")

            return matches

        except Exception as e:
            logger.error(f"Session search failed: {e}")
            return []

    # Over-fetch factor: pull this many × `limit` raw vector hits so that, after
    # dropping facts the graph has invalidated/superseded, we can still return up
    # to `limit` live results.
    FACT_VALIDITY_OVERFETCH = 3

    async def search_similar_facts(
        self, query_text: str, limit: int = 5, min_score: float | None = None
    ) -> list[dict[str, Any]]:
        """
        Search for semantically similar facts.

        Args:
            query_text: Query text
            limit: Maximum results
            min_score: Minimum similarity score

        Returns:
            List of similar facts with scores
        """
        if not self.enabled or not self._initialized:
            return []

        min_score = min_score or self.similarity_threshold

        try:
            query_vector = await self.embed_text(query_text)
            if not query_vector:
                return []

            # Over-fetch so we can drop graph-invalidated facts and still return
            # up to `limit` live results (vector/graph temporal-validity sync).
            results = await asyncio.to_thread(
                self.client.query_points,
                collection_name=self.COLLECTION_FACTS,
                query=query_vector,
                limit=limit * self.FACT_VALIDITY_OVERFETCH,
                score_threshold=min_score,
            )

            matches = []
            for hit in results.points:
                matches.append(
                    {
                        "id": hit.id,
                        "score": hit.score,
                        "entity1": hit.payload.get("entity1"),
                        "relation": hit.payload.get("relation"),
                        "entity2": hit.payload.get("entity2"),
                        "context": hit.payload.get("context"),
                        "timestamp": hit.payload.get("timestamp"),
                        "neo4j_uuid": hit.payload.get("neo4j_uuid"),
                        "entity1_uuid": hit.payload.get("entity1_uuid"),
                        "entity2_uuid": hit.payload.get("entity2_uuid"),
                    }
                )

            # Drop facts whose relationship the graph has invalidated/superseded,
            # then trim back to the requested limit.
            matches = await self._filter_valid_facts(matches)
            matches = matches[:limit]

            if matches:
                logger.info(f"Found {len(matches)} similar facts")

            return matches

        except Exception as e:
            logger.error(f"Fact search failed: {e}")
            return []

    async def _filter_valid_facts(
        self, matches: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Remove facts whose underlying relationship has been invalidated or
        superseded in the graph. Qdrant is not updated on invalidation, so the
        vector layer would otherwise keep returning stale facts the graph has
        already retired.

        Fail-open: if the graph is unreachable, return matches unfiltered rather
        than break semantic search.
        """
        if not matches:
            return matches
        try:
            from .neo4j_service import neo4j_service

            valid_idx = await neo4j_service.filter_valid_facts(
                [
                    {
                        "idx": i,
                        "relation": m.get("relation"),
                        "e1_uuid": m.get("entity1_uuid"),
                        "e2_uuid": m.get("entity2_uuid"),
                        "e1_name": m.get("entity1"),
                        "e2_name": m.get("entity2"),
                    }
                    for i, m in enumerate(matches)
                ]
            )
        except Exception as e:
            logger.warning(
                f"Fact validity filter unavailable, returning unfiltered results: {e}"
            )
            return matches

        filtered = [m for i, m in enumerate(matches) if i in valid_idx]
        dropped = len(matches) - len(filtered)
        if dropped:
            logger.info(
                f"Dropped {dropped} invalidated/superseded fact(s) from vector search"
            )
        return filtered

    async def health_check(self) -> dict[str, Any]:
        """
        Check vector service health.

        Returns:
            Health status dictionary
        """
        if not self.enabled:
            return {"status": "disabled", "enabled": False}

        if not self._initialized:
            return {"status": "not_initialized", "enabled": True, "initialized": False}

        try:
            collections = await asyncio.to_thread(self.client.get_collections)
            collection_names = [c.name for c in collections.collections]

            # Get collection stats
            stats = {}
            for collection_name in collection_names:
                collection_info = await asyncio.to_thread(
                    self.client.get_collection, collection_name=collection_name
                )
                stats[collection_name] = {
                    "indexed_vectors_count": collection_info.indexed_vectors_count,
                    "points_count": collection_info.points_count,
                }

            return {
                "status": "healthy",
                "enabled": True,
                "initialized": True,
                "qdrant_host": f"{self.qdrant_host}:{self.qdrant_port}",
                "ollama_host": self.ollama_host,
                "embedding_model": self.embedding_model,
                "embedding_dimension": self.embedder.dimension,
                "similarity_threshold": self.similarity_threshold,
                "collections": collection_names,
                "stats": stats,
                **(await self._embedding_health()),
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "enabled": True,
                "initialized": self._initialized,
                "error": str(e),
            }

    async def _count(self, collection: str, flt) -> int | str:
        try:
            result = await asyncio.to_thread(
                self.client.count,
                collection_name=collection,
                count_filter=flt,
                exact=True,
            )
            return result.count
        except Exception as e:
            return f"unknown ({e})"

    async def _embedding_health(self) -> dict[str, Any]:
        """Embedding identity and per-collection fingerprint state (#134).

        For every collection (trusted or not): points not tagged with the
        active fingerprint, points without any tag (legacy), and mixed points
        (tagged with a fingerprint other than the collection's own).
        """
        fingerprint = self._active_fingerprint()
        per_type: dict[str, Any] = {}
        for memory_type, state in self.collection_status.items():
            entry = dict(state)
            physical = state.get("collection")
            entry["blocked_writes"] = self.blocked_writes.get(memory_type, 0)
            if physical:
                untagged = Filter(
                    must=[IsEmptyCondition(is_empty=PayloadField(key=_registry.FP_KEY))]
                )
                entry["points_untagged"] = await self._count(physical, untagged)
                if fingerprint:
                    entry["points_not_tagged_active"] = await self._count(
                        physical,
                        Filter(
                            must_not=[
                                FieldCondition(
                                    key=_registry.FP_KEY,
                                    match=MatchValue(value=fingerprint),
                                )
                            ]
                        ),
                    )
                own = state.get("fingerprint")
                if own:
                    entry["points_mixed"] = await self._count(
                        physical,
                        Filter(
                            must_not=[
                                FieldCondition(
                                    key=_registry.FP_KEY, match=MatchValue(value=own)
                                ),
                                IsEmptyCondition(
                                    is_empty=PayloadField(key=_registry.FP_KEY)
                                ),
                            ]
                        ),
                    )
            per_type[memory_type] = entry
        mismatched = [t for t in self.collection_status if not self.is_trusted(t)]
        return {
            "embed_provider": self.embed_provider,
            "embed_fingerprint": fingerprint,
            "embed_profile": (p.to_dict() if (p := self._active_profile()) else None),
            "embed_unicode_version": unicodedata.unidata_version,
            "embed_collections": per_type,
            "embed_mismatch": mismatched,
            "embed_blocked_writes": sum(self.blocked_writes.values()),
        }

    def is_enabled(self) -> bool:
        """Check if vector service is enabled and initialized."""
        return self.enabled and self._initialized

    async def search(
        self,
        query_text: str,
        search_type: str = "all",
        limit: int = 5,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Unified search across multiple collection types.

        Args:
            query_text: Query text
            search_type: "prompts", "learnings", "sessions", "facts", or "all"
            limit: Maximum results per type
            min_score: Minimum similarity score

        Returns:
            Combined results from requested collections
        """
        if not self.enabled or not self._initialized:
            return []

        results = []

        try:
            if search_type in ("prompts", "all"):
                prompt_results = await self.search_similar_prompts(
                    query_text=query_text, limit=limit, min_score=min_score
                )
                for result in prompt_results:
                    results.append(
                        {
                            **result,
                            "type": "prompt",
                            "content": result.get("matched_text", ""),
                            "created_at": result.get("timestamp"),
                        }
                    )

            if search_type in ("learnings", "all"):
                learning_results = await self.search_similar_learnings(
                    query_text=query_text, limit=limit, min_score=min_score
                )
                for result in learning_results:
                    results.append(
                        {
                            **result,
                            "type": "learning",
                            "created_at": result.get("timestamp"),
                        }
                    )

            if search_type in ("sessions", "all"):
                session_results = await self.search_similar_sessions(
                    query_text=query_text, limit=limit, min_score=min_score
                )
                for result in session_results:
                    results.append(
                        {
                            **result,
                            "type": "session",
                            "created_at": result.get("timestamp"),
                        }
                    )

            if search_type in ("facts", "all"):
                fact_results = await self.search_similar_facts(
                    query_text=query_text, limit=limit, min_score=min_score
                )
                for result in fact_results:
                    # Format fact as readable content
                    fact_text = f"{result.get('entity1')} {result.get('relation')} {result.get('entity2')}"
                    results.append(
                        {
                            **result,
                            "type": "fact",
                            "content": fact_text,
                            "created_at": result.get("timestamp"),
                        }
                    )

            # Sort by score descending
            results.sort(key=lambda x: x.get("score", 0), reverse=True)

            return results

        except Exception as e:
            logger.error(f"Unified search failed: {e}")
            return []

    async def close(self):
        """Cleanup resources."""
        if self.embedder:
            await self.embedder.close()


# Singleton instance
_vector_service: VectorService | None = None


def create_vector_service() -> VectorService:
    """
    Build a VectorService from environment configuration (not initialized).

    The Admin UI (Feature 039) uses this to own an instance on its own event
    loop: the embedder's httpx.AsyncClient is bound to the loop that made it.
    """
    enabled = os.getenv("QDRANT_ENABLED", "true").lower() == "true"
    embed_provider = os.getenv("EMBED_PROVIDER", "ollama")
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    ollama_host = os.getenv(
        "OLLAMA_EMBEDDING_HOST", os.getenv("OLLAMA_HOST", "http://localhost:11434")
    )
    ollama_fallback = os.getenv("OLLAMA_FALLBACK_URL", "")
    embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    threshold = float(os.getenv("QDRANT_SIMILARITY_THRESHOLD", "0.45"))

    return VectorService(
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        ollama_host=ollama_host,
        ollama_fallback_host=ollama_fallback,
        embedding_model=embedding_model,
        similarity_threshold=threshold,
        enabled=enabled,
        embed_provider=embed_provider,
    )


def get_vector_service() -> VectorService:
    """Get or create singleton vector service instance."""
    global _vector_service

    if _vector_service is None:
        _vector_service = create_vector_service()

    assert _vector_service is not None
    return _vector_service
