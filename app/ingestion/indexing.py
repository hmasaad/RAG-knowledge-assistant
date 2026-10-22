"""Index one stored document version."""

import time

from app.ingestion.chunking import build_chunks
from app.ingestion.cleaning import clean_text
from app.ingestion.embeddings import Embedder
from app.ingestion.errors import IngestionError
from app.ingestion.metadata_store import MetadataStore
from app.ingestion.models import Chunk, DocumentRecord, ParsedPage
from app.ingestion.parsers import parse_document
from app.ingestion.vector_store import VectorStore
from app.observability.logging import log_event
from app.storage.blobs import BlobStore


def index_stored_version(
    *,
    metadata: MetadataStore,
    vectors: VectorStore,
    blobs: BlobStore,
    embedder: Embedder,
    document_id: str,
    version: int,
    chunk_size: int,
    chunk_overlap: int,
) -> DocumentRecord:
    """Embed the stored bytes for one version and publish that version."""
    current = metadata.get(document_id)
    if current is None:
        raise IngestionError("Document not found.", status_code=404)
    if current.version != version:
        raise IngestionError("Document version is no longer current.", status_code=409)
    stored = metadata.get_version(document_id, version)
    if stored is None:
        raise IngestionError("Document version not found.", status_code=404)

    started = time.perf_counter()
    metadata.mark_indexing(document_id, version)
    data = blobs.get(stored.checksum)
    try:
        chunks = chunks_for_bytes(
            document_id=document_id,
            version=version,
            filename=stored.filename,
            data=data,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        embeddings = embedder.embed_documents([chunk.text for chunk in chunks])
        _require_embeddings(chunks, embeddings)
        vectors.delete_version(document_id, version)
        vectors.add(chunks, embeddings)
    except Exception as exc:
        vectors.delete_version(document_id, version)
        message = exc.message if isinstance(exc, IngestionError) else "Failed to index the document."
        metadata.mark_version_failed(document_id, version, message)
        log_event(
            "rag.ingestion",
            "document_index_failed",
            document_id=document_id,
            version=version,
            filename=stored.filename,
            error=message,
        )
        if isinstance(exc, IngestionError):
            raise
        raise IngestionError(message, status_code=500) from exc

    record = metadata.complete_version(document_id, version, len(chunks))
    vectors.delete_other_versions(document_id, version)
    log_event(
        "rag.ingestion",
        "document_indexed",
        document_id=record.id,
        version=version,
        filename=record.filename,
        content_type=record.content_type,
        chunk_count=record.chunk_count,
        byte_size=record.byte_size,
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
        embedding_model=getattr(embedder, "model", "unknown"),
    )
    return record


def chunks_for_bytes(
    *,
    document_id: str,
    version: int,
    filename: str,
    data: bytes,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    parsed = parse_document(filename, data)
    chunks = _chunks_for_pages(
        document_id=document_id,
        version=version,
        filename=filename,
        content_type=parsed.content_type,
        pages=parsed.pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        raise IngestionError(
            "No text could be extracted from the document.",
            status_code=422,
        )
    return chunks


def _chunks_for_pages(
    *,
    document_id: str,
    version: int,
    filename: str,
    content_type: str,
    pages: list[ParsedPage],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        cleaned = clean_text(page.text)
        drafts = build_chunks(
            cleaned,
            content_type=content_type,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
        )
        for draft in drafts:
            index = len(chunks)
            chunks.append(
                Chunk(
                    id=f"{document_id}:{version}:{index}",
                    document_id=document_id,
                    filename=filename,
                    content_type=content_type,
                    text=draft.text,
                    index=index,
                    start_char=draft.start,
                    end_char=draft.end,
                    page=page.number,
                    heading=draft.heading,
                    version=version,
                )
            )
    return chunks


def _require_embeddings(chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    if len(embeddings) != len(chunks):
        raise IngestionError(
            "Embedding provider returned an unexpected number of vectors.",
            status_code=502,
        )
    width = len(embeddings[0]) if embeddings else 0
    if width == 0 or any(len(vector) != width for vector in embeddings):
        raise IngestionError(
            "Embedding provider returned malformed vectors.",
            status_code=502,
        )
