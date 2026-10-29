"""Index one stored document version."""

import time
from dataclasses import replace

from app.ingestion.chunking import build_chunks
from app.ingestion.cleaning import clean_text
from app.ingestion.context import embedding_text, situate
from app.ingestion.embeddings import Embedder
from app.ingestion.errors import IngestionError, index_failure_message
from app.ingestion.graph import GraphEdge, extract_edges
from app.ingestion.metadata_store import MetadataStore
from app.ingestion.models import Chunk, DocumentRecord, ParsedPage
from app.ingestion.parents import split_children
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
    chunk_strategy: str = "structure",
    semantic_similarity: float = 0.35,
    parent_retrieval: bool = False,
    child_chunk_size: int = 240,
    contextual_retrieval: bool = False,
    knowledge_graph: bool = False,
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
            chunk_strategy=chunk_strategy,
            semantic_similarity=semantic_similarity,
            embedder=embedder,
            parent_retrieval=parent_retrieval,
            child_chunk_size=child_chunk_size,
            contextual_retrieval=contextual_retrieval,
        )
        embeddings = embedder.embed_documents([embedding_text(chunk.text, chunk.context) for chunk in chunks])
        _require_embeddings(chunks, embeddings)
        metadata.assert_embedding_compatible(getattr(embedder, "model", "unknown"), len(embeddings[0]))
        vectors.delete_version(document_id, version)
        vectors.add(chunks, embeddings)
    except Exception as exc:
        vectors.delete_version(document_id, version)
        message = index_failure_message(exc)
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
    metadata.record_embedding(getattr(embedder, "model", "unknown"), len(embeddings[0]))
    metadata.replace_graph(
        document_id,
        version,
        _graph_edges(chunks) if knowledge_graph else [],
    )
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
        chunk_strategy=chunk_strategy,
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
    chunk_strategy: str = "structure",
    semantic_similarity: float = 0.35,
    embedder: Embedder | None = None,
    parent_retrieval: bool = False,
    child_chunk_size: int = 240,
    contextual_retrieval: bool = False,
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
        chunk_strategy=chunk_strategy,
        semantic_similarity=semantic_similarity,
        embedder=embedder,
        parent_retrieval=parent_retrieval,
        child_chunk_size=child_chunk_size,
        contextual_retrieval=contextual_retrieval,
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
    chunk_strategy: str,
    semantic_similarity: float,
    embedder: Embedder | None,
    parent_retrieval: bool,
    child_chunk_size: int,
    contextual_retrieval: bool,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        cleaned = clean_text(page.text)
        drafts = build_chunks(
            cleaned,
            content_type=content_type,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            strategy=chunk_strategy,
            embedder=embedder,
            semantic_similarity=semantic_similarity,
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
                    section_id=draft.section_id,
                    section_title=draft.section_title,
                    section_number=draft.section_number,
                    parent_section=draft.parent_section,
                    chunk_type=draft.chunk_type,
                    bullet_index=draft.bullet_index,
                    context=situate(
                        filename,
                        draft.heading,
                        section_title=draft.section_title,
                    ),
                )
            )
    if parent_retrieval:
        chunks = split_children(chunks, child_chunk_size=child_chunk_size, overlap=chunk_overlap)
        chunks = [
            replace(
                chunk,
                context=chunk.context
                or situate(chunk.filename, chunk.heading, section_title=chunk.section_title),
            )
            for chunk in chunks
        ]
    if contextual_retrieval:
        chunks = [
            replace(
                chunk,
                context=situate(chunk.filename, chunk.heading, section_title=chunk.section_title),
            )
            for chunk in chunks
        ]
    return chunks


def _graph_edges(chunks: list[Chunk]) -> list[GraphEdge]:
    edges: list[GraphEdge] = []
    seen_parents: set[str] = set()
    for chunk in chunks:
        if chunk.parent_text:
            if chunk.parent_id in seen_parents:
                continue
            seen_parents.add(chunk.parent_id)
            edges.extend(extract_edges(chunk.parent_text, chunk.parent_id))
        else:
            edges.extend(extract_edges(chunk.text, chunk.id))
    return edges


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
