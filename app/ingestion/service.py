"""Turn an uploaded file into indexed chunks."""

import hashlib
import time
import uuid

from app.ingestion.chunking import build_chunks
from app.ingestion.cleaning import clean_text
from app.ingestion.embeddings import Embedder
from app.ingestion.errors import IngestionError
from app.ingestion.metadata_store import MetadataStore
from app.ingestion.models import Chunk, DocumentRecord, ParsedPage
from app.ingestion.parsers import parse_document
from app.ingestion.vector_store import VectorStore
from app.observability.logging import log_event
from app.security.uploads import validate_upload


class IngestionService:
    def __init__(
        self,
        *,
        metadata: MetadataStore,
        vectors: VectorStore,
        embedder: Embedder,
        chunk_size: int,
        chunk_overlap: int,
        max_upload_bytes: int,
    ) -> None:
        self._metadata = metadata
        self._vectors = vectors
        self._embedder = embedder
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._max_upload_bytes = max_upload_bytes

    def ingest(self, filename: str | None, data: bytes) -> tuple[DocumentRecord, bool]:
        """Index a document.

        The boolean is True when this call wrote a new index entry. The same
        file bytes return the existing indexed document without embedding again.
        """
        started = time.perf_counter()
        upload = validate_upload(filename, data, max_bytes=self._max_upload_bytes)
        checksum = hashlib.sha256(upload.data).hexdigest()
        existing = self._metadata.find_by_checksum(checksum)
        if existing is not None and existing.status == "indexed":
            log_event(
                "rag.ingestion",
                "document_already_indexed",
                document_id=existing.id,
                filename=existing.filename,
                checksum=checksum,
            )
            return existing, False

        parsed = parse_document(upload.filename, upload.data)
        document_id = existing.id if existing is not None else uuid.uuid4().hex
        chunks = _chunks_for_pages(
            document_id=document_id,
            filename=upload.filename,
            content_type=parsed.content_type,
            pages=parsed.pages,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        if not chunks:
            raise IngestionError(
                "No text could be extracted from the document.",
                status_code=422,
            )

        if existing is None:
            self._metadata.insert_indexing(
                document_id=document_id,
                filename=upload.filename,
                content_type=parsed.content_type,
                checksum=checksum,
                byte_size=len(upload.data),
            )
        else:
            self._vectors.delete_document(document_id)
            self._metadata.mark_indexing(document_id)

        try:
            embeddings = self._embedder.embed_documents([chunk.text for chunk in chunks])
            _require_embeddings(chunks, embeddings)
            self._vectors.delete_document(document_id)
            self._vectors.add(chunks, embeddings)
            record = self._metadata.mark_indexed(document_id, len(chunks))
        except Exception as exc:
            self._vectors.delete_document(document_id)
            message = exc.message if isinstance(exc, IngestionError) else "Failed to index the document."
            self._metadata.mark_failed(document_id, message)
            log_event(
                "rag.ingestion",
                "document_index_failed",
                document_id=document_id,
                filename=upload.filename,
                error=message,
            )
            if isinstance(exc, IngestionError):
                raise
            raise IngestionError(message, status_code=500) from exc

        log_event(
            "rag.ingestion",
            "document_indexed",
            document_id=record.id,
            filename=record.filename,
            content_type=record.content_type,
            chunk_count=record.chunk_count,
            byte_size=record.byte_size,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            embedding_model=getattr(self._embedder, "model", "unknown"),
        )
        return record, True

    def list_documents(self) -> list[DocumentRecord]:
        return self._metadata.list_documents()

    def get_document(self, document_id: str) -> DocumentRecord:
        record = self._metadata.get(document_id)
        if record is None:
            raise IngestionError("Document not found.", status_code=404)
        return record

    def chunks_for(self, document_id: str) -> list[dict]:
        self.get_document(document_id)
        return self._vectors.chunks_for_document(document_id)

    def delete_document(self, document_id: str) -> None:
        record = self.get_document(document_id)
        self._vectors.delete_document(record.id)
        self._metadata.delete(record.id)
        log_event(
            "rag.ingestion",
            "document_deleted",
            document_id=record.id,
            filename=record.filename,
        )


def _chunks_for_pages(
    *,
    document_id: str,
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
                    id=f"{document_id}:{index}",
                    document_id=document_id,
                    filename=filename,
                    content_type=content_type,
                    text=draft.text,
                    index=index,
                    start_char=draft.start,
                    end_char=draft.end,
                    page=page.number,
                    heading=draft.heading,
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
