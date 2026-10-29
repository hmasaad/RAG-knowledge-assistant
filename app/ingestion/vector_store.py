"""Chroma collection of indexed chunks.

Embeddings are always supplied by the caller. The collection does not download
or run a default embedding model.
"""

import threading
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
    def __init__(
        self,
        path: Path,
        collection_name: str,
        *,
        host: str = "",
        port: int = 8000,
    ) -> None:
        self._lock = threading.RLock()
        self._collection_name = collection_name
        client_settings = chromadb.Settings(anonymized_telemetry=False)
        if host:
            self._client = chromadb.HttpClient(host=host, port=port, settings=client_settings)
        else:
            path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(path), settings=client_settings)
        self._collection = self._open_collection()

    def recreate(self) -> None:
        """Drop the collection so the next vectors can use a new width.

        Chroma keeps the dimension of the first embedding after every row is
        deleted. A model migration has to create a fresh collection.
        """
        with self._lock:
            self._client.delete_collection(self._collection_name)
            self._collection = self._open_collection()

    def _open_collection(self):
        return self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=_ExternalEmbeddings(),
        )

    def count(self) -> int:
        with self._lock:
            return int(self._collection.count())

    def list_chunks(self, *, document_ids: set[str] | None = None) -> list[VectorHit]:
        """Return indexed chunks, optionally limited to the caller's documents."""
        if document_ids is not None and not document_ids:
            return []
        with self._lock:
            if self._collection.count() == 0:
                return []
            where = _document_filter(document_ids)
            result = self._collection.get(where=where, include=["documents", "metadatas"])
        hits: list[VectorHit] = []
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        for chunk_id, text, metadata in zip(ids, documents, metadatas, strict=False):
            hits.append(
                VectorHit(
                    chunk_id=chunk_id,
                    text=text or "",
                    score=0.0,
                    metadata=dict(metadata or {}),
                )
            )
        return hits

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk needs one embedding.")
        with self._lock:
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

    def query(
        self,
        embedding: list[float],
        top_k: int,
        *,
        document_ids: set[str] | None = None,
    ) -> list[VectorHit]:
        if top_k < 1 or (document_ids is not None and not document_ids):
            return []
        with self._lock:
            where = _document_filter(document_ids)
            available = self._match_count(where)
            if available == 0:
                return []
            result = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(top_k, available),
                include=["documents", "metadatas", "distances"],
                where=where,
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

    def chunks_for_document(
        self,
        document_id: str,
        *,
        version: int | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
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
            if version is not None and chunk_version(meta) != version:
                continue
            rows.append(
                {
                    "id": chunk_id,
                    "text": text or "",
                    "index": int(meta.get("chunk_index", 0)),
                    "start_char": int(meta.get("start_char", 0)),
                    "end_char": int(meta.get("end_char", 0)),
                    "page": _optional_page(meta.get("page")),
                    "heading": _optional_text(meta.get("heading")),
                    "context": str(meta.get("context") or ""),
                }
            )
        rows.sort(key=lambda row: row["index"])
        return rows

    def delete_document(self, document_id: str) -> None:
        self._delete_where(document_id, keep_version=None, drop_version=None)

    def delete_version(self, document_id: str, version: int) -> None:
        self._delete_where(document_id, keep_version=None, drop_version=version)

    def delete_other_versions(self, document_id: str, keep_version: int) -> None:
        self._delete_where(document_id, keep_version=keep_version, drop_version=None)

    def _delete_where(
        self,
        document_id: str,
        *,
        keep_version: int | None,
        drop_version: int | None,
    ) -> None:
        with self._lock:
            existing = self._collection.get(
                where={"document_id": document_id},
                include=["metadatas"],
            )
            ids = existing.get("ids") or []
            metadatas = existing.get("metadatas") or []
            stale: list[str] = []
            for chunk_id, metadata in zip(ids, metadatas, strict=False):
                version = chunk_version(metadata or {})
                if drop_version is not None and version != drop_version:
                    continue
                if keep_version is not None and version == keep_version:
                    continue
                stale.append(chunk_id)
            if stale:
                self._collection.delete(ids=stale)

    def _match_count(self, where: dict[str, Any] | None) -> int:
        if where is None:
            return self.count()
        result = self._collection.get(where=where, include=[])
        return len(result.get("ids") or [])


def _document_filter(document_ids: set[str] | None) -> dict[str, Any] | None:
    if document_ids is None:
        return None
    return {"document_id": {"$in": sorted(document_ids)}}


def _metadata(chunk: Chunk) -> dict[str, Any]:
    payload = {
        "document_id": chunk.document_id,
        "filename": chunk.filename,
        "content_type": chunk.content_type,
        "chunk_index": chunk.index,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
        "page": chunk.page if chunk.page is not None else _PAGE_MISSING,
        "heading": chunk.heading or "",
        "version": chunk.version,
        "section_id": chunk.section_id,
        "section_title": chunk.section_title,
        "section_number": chunk.section_number,
        "parent_section": chunk.parent_section,
        "chunk_type": chunk.chunk_type,
        "bullet_index": chunk.bullet_index,
    }
    if chunk.parent_id:
        payload["parent_id"] = chunk.parent_id
        payload["parent_text"] = chunk.parent_text
        payload["parent_index"] = chunk.parent_index
        payload["parent_start"] = chunk.parent_start
        payload["parent_end"] = chunk.parent_end
    if chunk.context:
        payload["context"] = chunk.context
    return payload


def chunk_version(metadata: dict[str, Any] | None) -> int:
    """Version stored on a chunk. Chunks written before versioning count as 1."""
    if not metadata:
        return 1
    value = metadata.get("version")
    if value is None or value == "":
        return 1
    return int(value)


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
    """Convert a cosine distance into a higher-is-better similarity.

    Chroma's cosine distance matches ``cosine_distance`` for normalized
    vectors, so this score matches ``cosine_similarity`` when it is positive.
    """
    if distance is None:
        return 0.0
    score = 1.0 - float(distance)
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score
