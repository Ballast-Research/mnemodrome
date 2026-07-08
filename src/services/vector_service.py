"""
Vector Service (Feature 022: Semantic Vector Layer)

Provides semantic similarity search as the first stage in Mnemodrome's
retrieval cascade: Vector → Redis → Neo4j → Markdown

Constitutional Principle: Always-on observation
Every prompt is automatically embedded and searched for similar past interactions.

Embedding Provider: Ollama with nomic-embed-text (768 dimensions)
"""

import asyncio
import os
import uuid
from datetime import datetime
from typing import Any

import httpx
from loguru import logger

try:
    from src import config as _config  # type: ignore[import]
except ImportError:
    _config = None  # type: ignore[assignment]

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
        VectorParams,
    )

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("Qdrant client not available. Vector layer disabled.")


class OllamaEmbedder:
    """
    Handles embedding generation via Ollama API.

    Uses nomic-embed-text model (768 dimensions) for semantic search.
    """

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

    async def embed(self, text: str) -> list[float] | None:
        """
        Generate embedding for text. Falls back to secondary host on failure.

        Args:
            text: Input text

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
    - Generate embeddings for prompts, learnings, facts, and sessions via Ollama
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

        self.client: QdrantClient | None = None
        self.embedder: OllamaEmbedder | None = None
        self._initialized = False

        # Collection names for different memory types
        self.COLLECTION_PROMPTS = "mnemodrome_prompts"
        self.COLLECTION_LEARNINGS = "mnemodrome_learnings"
        self.COLLECTION_FACTS = "mnemodrome_facts"
        self.COLLECTION_SESSIONS = "mnemodrome_sessions"
        self.COLLECTION_PROCEDURES = "mnemodrome_procedures"

        if not self.enabled:
            logger.info("Vector layer disabled")
            return

        logger.info(
            f"Vector service configured (qdrant={self.qdrant_host}:{self.qdrant_port}, "
            f"ollama={self.ollama_host}, fallback={self.ollama_fallback_host or 'none'}, "
            f"model={self.embedding_model}, threshold={self.similarity_threshold})"
        )

    async def initialize(self):
        """Initialize Qdrant client and Ollama embedder."""
        if not self.enabled:
            return

        if self._initialized:
            return

        try:
            # Initialize Ollama embedder
            logger.info(f"Initializing Ollama embedder ({self.embedding_model})...")
            self.embedder = OllamaEmbedder(
                host=self.ollama_host,
                model=self.embedding_model,
                fallback_host=self.ollama_fallback_host or None,
            )

            embedder_ready = await self.embedder.initialize()
            if not embedder_ready:
                logger.error("Ollama embedder initialization failed")
                self.enabled = False
                return

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

    async def _ensure_collections(self, dimension: int):
        """Create collections if they don't exist.

        For existing collections, compare the stored vector dimension against
        the active embedding dimension and log a clear error on mismatch.
        Mnemodrome will NOT auto-delete or modify existing collections — operator
        action is required (delete the mismatched collection(s) and restart so
        they are recreated with the correct dimension, then re-ingest data).
        """
        collections_to_create = [
            (self.COLLECTION_PROMPTS, "User prompts and queries"),
            (self.COLLECTION_LEARNINGS, "Captured learnings and insights"),
            (self.COLLECTION_FACTS, "Factual relationships from knowledge graph"),
            (self.COLLECTION_SESSIONS, "Session summaries and syntheses"),
            (
                self.COLLECTION_PROCEDURES,
                "Diagnostic procedures and problem-solving pathways",
            ),
        ]

        existing_collections = await asyncio.to_thread(self.client.get_collections)
        existing_names = {c.name for c in existing_collections.collections}

        for collection_name, description in collections_to_create:
            if collection_name not in existing_names:
                logger.info(f"Creating collection: {collection_name} ({description})")
                await asyncio.to_thread(
                    self.client.create_collection,
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=dimension, distance=Distance.COSINE
                    ),
                )
            else:
                # Collection already exists — check for a dimension mismatch that
                # would silently corrupt all future similarity searches.
                try:
                    collection_info = await asyncio.to_thread(
                        self.client.get_collection, collection_name
                    )
                    existing_dim = self._get_collection_dimension(collection_info)
                    if existing_dim is not None and existing_dim != dimension:
                        logger.error(
                            f"EMBEDDING DIMENSION MISMATCH detected in collection "
                            f"'{collection_name}': stored={existing_dim}D vs "
                            f"active model={dimension}D (model: {self.embedding_model}, "
                            f"EMBEDDING_DIM={os.getenv('EMBEDDING_DIM', '768')}). "
                            f"This means the embedding model changed after the collection "
                            f"was created. Similarity searches will produce wrong results. "
                            f"TO FIX: delete the affected Qdrant collection(s), restart "
                            f"Mnemodrome so they are recreated at the correct dimension, then "
                            f"re-run scripts/bootstrap_vector_embeddings.py to re-embed "
                            f"your existing data. Mnemodrome will NOT auto-delete collections."
                        )
                except Exception as check_err:
                    logger.warning(
                        f"Could not verify dimension for existing collection "
                        f"'{collection_name}': {check_err}"
                    )

    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding vector for text using Ollama.

        Args:
            text: Input text to embed

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
            embedding = await self.embedder.embed(text)
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
            payload = {
                "text": text,
                "timestamp": datetime.utcnow().isoformat(),
                "neo4j_uuid": neo4j_uuid,
                **(metadata or {}),
            }

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

        try:
            # Generate embedding
            embedding = await self.embed_text(text)
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
            payload = {
                "text": text,
                "timestamp": datetime.utcnow().isoformat(),
                "neo4j_uuid": neo4j_uuid,
                **(metadata or {}),
            }

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

        try:
            # Combine title and content for richer embedding
            combined_text = f"{title}\n\n{content}"
            embedding = await self.embed_text(combined_text)
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

            payload = {
                "title": title,
                "content": content,
                "category": category,
                "timestamp": datetime.utcnow().isoformat(),
                "neo4j_uuid": str(point_id),
                "type": "learning",
                **(metadata or {}),
            }

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

        try:
            # Combine summary and key insights for richer embedding
            insights_text = ""
            if key_insights:
                insights_text = "\n\nKey Insights:\n" + "\n".join(
                    f"- {insight}" for insight in key_insights
                )

            combined_text = f"{summary}{insights_text}"
            embedding = await self.embed_text(combined_text)
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

            payload = {
                "summary": summary,
                "key_insights": key_insights or [],
                "timestamp": datetime.utcnow().isoformat(),
                "neo4j_uuid": str(point_id),
                "type": "session",
                **(metadata or {}),
            }

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

        try:
            # Combine fact components for embedding
            embed_text = f"{entity1} {relation} {entity2}"
            if context:
                embed_text += f"\nContext: {context}"

            embedding = await self.embed_text(embed_text)
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

            payload = {
                "entity1": entity1,
                "relation": relation,
                "entity2": entity2,
                "context": context,
                "timestamp": datetime.utcnow().isoformat(),
                "neo4j_uuid": str(point_id),
                "type": "fact",
                **(metadata or {}),
            }

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
    ) -> list[dict[str, Any]]:
        """
        Search for semantically similar learnings.

        Args:
            query_text: Query text
            limit: Maximum results
            category: Optional category filter
            min_score: Minimum similarity score

        Returns:
            List of similar learnings with scores
        """
        if not self.enabled or not self._initialized:
            return []

        min_score = min_score or self.similarity_threshold

        try:
            query_vector = await self.embed_text(query_text)
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
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "enabled": True,
                "initialized": self._initialized,
                "error": str(e),
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


def get_vector_service() -> VectorService:
    """Get or create singleton vector service instance."""
    global _vector_service

    if _vector_service is None:
        # Read configuration from environment
        enabled = os.getenv("QDRANT_ENABLED", "true").lower() == "true"
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        ollama_host = os.getenv(
            "OLLAMA_EMBEDDING_HOST", os.getenv("OLLAMA_HOST", "http://localhost:11434")
        )
        ollama_fallback = os.getenv("OLLAMA_FALLBACK_URL", "")
        embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        threshold = float(os.getenv("QDRANT_SIMILARITY_THRESHOLD", "0.45"))

        _vector_service = VectorService(
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            ollama_host=ollama_host,
            ollama_fallback_host=ollama_fallback,
            embedding_model=embedding_model,
            similarity_threshold=threshold,
            enabled=enabled,
        )

    assert _vector_service is not None
    return _vector_service
