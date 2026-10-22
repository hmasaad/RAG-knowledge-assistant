"""Chroma collection of indexed chunks.

Embeddings are always supplied by the caller. The collection does not download
or run a default embedding model.
"""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.types import Documents, Embeddings, Space

from app.ingestion.models import Chunk

_PAGE_MISSING = -1


class _ExternalEmbeddings(chromadb.EmbeddingFunction[Documents]):
    """Placeholder so Chroma will not load its default ONNX embedder."""

    def __init__(self) -> None:
        """Embeddings are supplied by the caller, so there is nothing to configure."""

    def __call__(self, input: Documents) -> Embeddings:
        raise RuntimeError("Pass embeddings explicitly to the vector store.")

    @staticmethod
    def name() -> str:
        return "external-embeddings"

    def get_config(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "_ExternalEmbeddings":
        return _ExternalEmbeddings()

    def default_space(self) -> Space:
        return "cosine"

    def supported_spaces(self) -> list[Space]:
        return ["cosine", "l2", "ip"]

    def embed_query(self, input: Documents) -> Embeddings:
        raise RuntimeError("Pass query embeddings explicitly to the vector store.")


class VectorHit:
    def __init__(
        self,
        *,
        chunk_id: str,
        text: str,
        score: float,
        metadata: dict[str, Any],
    ) -> None:
        self.chunk_id = chunk_id
        self.text = text
        self.score = score
        self.metadata = metadata


class VectorStore:
    def __init__(self, path: Path, collection_name: str) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(path),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=_ExternalEmbeddings(),
        )

    def count(self) -> int:
        return int(self._collection.count())

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk needs one embedding.")
        batch_size = 100
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = embeddings[start : start + batch_size]
            self._collection.add(
                ids=[chunk.id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                embeddings=vectors,
                metadatas=[_metadata(chunk) for chunk in batch],
            )

    def query(self, embedding: list[float], top_k: int) -> list[VectorHit]:
        total = self.count()
        if total == 0 or top_k < 1:
            return []
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, total),
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids") or [[]]
        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]
        hits: list[VectorHit] = []
        for chunk_id, text, metadata, distance in zip(
            ids[0],
            documents[0],
            metadatas[0],
            distances[0],
            strict=False,
        ):
            hits.append(
                VectorHit(
                    chunk_id=chunk_id,
                    text=text or "",
                    score=_distance_to_score(distance),
                    metadata=dict(metadata or {}),
                )
            )
        return hits

    def chunks_for_document(self, document_id: str) -> list[dict[str, Any]]:
        result = self._collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"],
        )
        rows: list[dict[str, Any]] = []
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        for chunk_id, text, metadata in zip(ids, documents, metadatas, strict=False):
            meta = dict(metadata or {})
            rows.append(
                {
                    "id": chunk_id,
                    "text": text or "",
                    "index": int(meta.get("chunk_index", 0)),
                    "start_char": int(meta.get("start_char", 0)),
                    "end_char": int(meta.get("end_char", 0)),
                    "page": _optional_page(meta.get("page")),
                    "heading": _optional_text(meta.get("heading")),
                }
            )
        rows.sort(key=lambda row: row["index"])
        return rows

    def delete_document(self, document_id: str) -> None:
        existing = self._collection.get(where={"document_id": document_id})
        ids = existing.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)


def _metadata(chunk: Chunk) -> dict[str, Any]:
    return {
        "document_id": chunk.document_id,
        "filename": chunk.filename,
        "content_type": chunk.content_type,
        "chunk_index": chunk.index,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
        "page": chunk.page if chunk.page is not None else _PAGE_MISSING,
        "heading": chunk.heading or "",
    }


def _optional_page(value: Any) -> int | None:
    if value is None:
        return None
    page = int(value)
    return None if page == _PAGE_MISSING else page


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _distance_to_score(distance: float | None) -> float:
    """Convert a cosine distance into a higher-is-better similarity."""
    if distance is None:
        return 0.0
    score = 1.0 - float(distance)
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score
