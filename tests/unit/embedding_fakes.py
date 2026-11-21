"""Test doubles for the embedding layer (#134). Not a test module."""

from __future__ import annotations

import hashlib

from src.services.embedding.profile import ROLE_QUERY, EmbedProfile

DIM = 8


class FakeEmbedder:
    """Deterministic embedder: a vector derived from sha256(prefix + text).

    `variant` changes the output (simulates an engine change with the same
    declared profile); `provider`/`model` change the fingerprint.
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "fake-model",
        variant: str = "",
        is_remote: bool = True,
        prefixes: tuple[str, str] = ("", ""),
    ):
        self.provider = provider
        self.model = model
        self.variant = variant
        self.is_remote = is_remote
        self.prefixes = prefixes
        self.calls: list[tuple[str, str]] = []

    @property
    def dimension(self) -> int:
        return DIM

    @property
    def profile(self) -> EmbedProfile:
        return EmbedProfile(
            provider=self.provider,
            model=self.model,
            dim=DIM,
            template_version=1,
            document_prefix=self.prefixes[0],
            query_prefix=self.prefixes[1],
        )

    def vector(self, text: str, role: str = ROLE_QUERY) -> list[float]:
        prefix = self.profile.prefix(role)
        digest = hashlib.sha256(f"{self.variant}|{prefix}{text}".encode()).digest()
        return [b / 255.0 + 0.01 for b in digest[:DIM]]

    async def initialize(self) -> bool:
        return True

    async def embed(self, text: str, role: str = ROLE_QUERY) -> list[float]:
        self.calls.append((text, role))
        return self.vector(text, role)

    async def embed_batch(self, texts: list[str], role: str = ROLE_QUERY):
        return [await self.embed(t, role) for t in texts]

    async def close(self):
        pass
