"""Remember embeddings so repeated text is not sent to the model again."""

import hashlib
import threading
from collections import OrderedDict

from app.ingestion.embeddings import Embedder
from app.ingestion.metadata_store import MetadataStore
from app.observability.metrics import Metrics


class CachingEmbedder:
    """LRU in memory, with the same entries stored in SQLite for the next process."""

    def __init__(
        self,
        inner: Embedder,
        metadata: MetadataStore,
        metrics: Metrics,
        *,
        memory_size: int,
    ) -> None:
        self._inner = inner
        self._metadata = metadata
        self._metrics = metrics
        self._memory_size = memory_size
        self._memory: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def model(self) -> str:
        return getattr(self._inner, "model", "unknown")

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_documents([text])
        return vectors[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self.model
        keys = [_cache_key(model, text) for text in texts]
        found: dict[int, list[float]] = {}
        missing: list[int] = []
        for index, key in enumerate(keys):
            cached = self._read(key)
            if cached is None:
                missing.append(index)
                self._metrics.inc("rag_embedding_cache_misses_total")
            else:
                found[index] = cached
                self._metrics.inc("rag_embedding_cache_hits_total")
        if missing:
            fresh = self._inner.embed_documents([texts[index] for index in missing])
            if len(fresh) != len(missing):
                raise RuntimeError("Embedding provider returned an unexpected number of vectors.")
            for index, vector in zip(missing, fresh, strict=True):
                found[index] = vector
                self._write(keys[index], vector)
        return [found[index] for index in range(len(texts))]

    def _read(self, key: str) -> list[float] | None:
        with self._lock:
            cached = self._memory.get(key)
            if cached is not None:
                self._memory.move_to_end(key)
                return list(cached)
        stored = self._metadata.get_embedding(key)
        if stored is None:
            return None
        self._remember(key, stored)
        return stored

    def _write(self, key: str, vector: list[float]) -> None:
        self._remember(key, vector)
        self._metadata.put_embedding(key, vector, max_entries=self._memory_size)

    def _remember(self, key: str, vector: list[float]) -> None:
        if self._memory_size < 1:
            return
        with self._lock:
            self._memory[key] = list(vector)
            self._memory.move_to_end(key)
            while len(self._memory) > self._memory_size:
                self._memory.popitem(last=False)


def _cache_key(model: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{model}:{digest}"
