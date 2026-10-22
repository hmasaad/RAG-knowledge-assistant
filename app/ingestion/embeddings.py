"""Embedding providers.

OpenAI is the production provider. ``LocalEmbedder`` is a deterministic hashed
bag-of-words used by tests and offline demos. It is not a semantic model.
"""

import hashlib
import math
import re
from typing import Protocol

from app.ingestion.errors import IngestionError

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "where",
    "which",
    "with",
}


class Embedder(Protocol):
    model: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class LocalEmbedder:
    """Signed feature hashing so overlapping words rank closer together."""

    model = "local-hash-bow"

    def __init__(self, dimension: int = 384) -> None:
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = _TOKEN.findall(text.lower())
        content_tokens = [token for token in tokens if token not in _STOPWORDS]
        if not content_tokens:
            content_tokens = tokens
        if not content_tokens:
            vector[0] = 1.0
            return vector
        for token in content_tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class OpenAIEmbedder:
    """Batch calls to the OpenAI embeddings API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "text-embedding-3-small",
        client: object | None = None,
        batch_size: int = 64,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
        self.model = model
        self._client = client
        self._batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if any(not text.strip() for text in texts):
            raise IngestionError("Refusing to embed empty text.", status_code=422)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            response = self._client.embeddings.create(model=self.model, input=batch)
            ordered = sorted(response.data, key=lambda item: item.index)
            if len(ordered) != len(batch):
                raise IngestionError(
                    "Embedding provider returned an unexpected number of vectors.",
                    status_code=502,
                )
            vectors.extend(item.embedding for item in ordered)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_documents([text])
        return vectors[0]


def build_embedder(
    *,
    provider: str,
    api_key: str,
    model: str,
    dimension: int,
) -> Embedder:
    if provider == "local":
        return LocalEmbedder(dimension=dimension)
    if provider == "openai":
        if not api_key:
            raise IngestionError(
                "OPENAI_API_KEY is not set. Set it, or use EMBEDDING_PROVIDER=local.",
                status_code=503,
            )
        return OpenAIEmbedder(api_key=api_key, model=model)
    raise IngestionError(f"Unknown embedding provider: {provider}.", status_code=500)
