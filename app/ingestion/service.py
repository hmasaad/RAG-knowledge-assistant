"""Turn an uploaded file into a versioned, indexed document."""

import hashlib
import time
import uuid

from dataclasses import dataclass

from app.ingestion.errors import IngestionError
from app.ingestion.indexing import chunks_for_bytes, index_stored_version
from app.ingestion.metadata_store import MetadataStore
from app.ingestion.models import DocumentRecord, VersionRecord
from app.ingestion.vector_store import VectorStore
from app.observability.logging import log_event
from app.observability.metrics import Metrics
from app.security.access import normalize_visibility
from app.security.uploads import validate_upload
from app.storage.blobs import BlobStore, MemoryBlobStore


@dataclass(frozen=True)
class RecoveryReport:
    requeued_failed: int
    requeued_missing: int
    reindexed: int


class IngestionService:
    def __init__(
        self,
        *,
        metadata: MetadataStore,
        vectors: VectorStore,
        embedder: object,
        chunk_size: int,
        chunk_overlap: int,
        max_upload_bytes: int,
        blobs: BlobStore | None = None,
        indexing_mode: str = "inline",
        metrics: Metrics | None = None,
        job_timeout_seconds: float = 900,
        chunk_strategy: str = "structure",
        semantic_similarity: float = 0.35,
        parent_retrieval: bool = False,
        child_chunk_size: int = 240,
        contextual_retrieval: bool = False,
        knowledge_graph: bool = False,
    ) -> None:
        self._metadata = metadata
        self._vectors = vectors
        self._embedder = embedder
        self._blobs = blobs if blobs is not None else MemoryBlobStore()
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._max_upload_bytes = max_upload_bytes
        self._indexing_mode = indexing_mode
        self._metrics = metrics
        self._job_timeout_seconds = job_timeout_seconds
        self._chunk_strategy = chunk_strategy
        self._semantic_similarity = semantic_similarity
        self._parent_retrieval = parent_retrieval
        self._child_chunk_size = child_chunk_size
        self._contextual_retrieval = contextual_retrieval
        self._knowledge_graph = knowledge_graph

    @property
    def embedder(self) -> object:
        return self._embedder

    def ingest(
        self,
        filename: str | None,
        data: bytes,
        *,
        owner: str = "local",
        visibility: str = "private",
    ) -> tuple[DocumentRecord, bool]:
        """Index a document.

        The boolean is True when this call accepted new bytes. The same file
        bytes return the existing indexed document without embedding again.
        """
        upload, checksum, parsed_type = self._prepare(filename, data)
        audience = normalize_visibility(visibility)
        existing = self._metadata.find_by_checksum(checksum, owner)
        if existing is not None and existing.status == "indexed":
            log_event(
                "rag.ingestion",
                "document_already_indexed",
                document_id=existing.id,
                filename=existing.filename,
                checksum=checksum,
            )
            return existing, False
        if existing is not None and existing.status in {"failed", "queued", "indexing"}:
            self._blobs.put(checksum, upload.data)
            record = self._metadata.requeue(existing.id, existing.version)
            return self._submit(record), True

        self._blobs.put(checksum, upload.data)
        record = self._metadata.create_document(
            document_id=uuid.uuid4().hex,
            filename=upload.filename,
            content_type=parsed_type,
            checksum=checksum,
            byte_size=len(upload.data),
            owner=owner,
            visibility=audience,
        )
        return self._submit(record), True

    def update_document(
        self,
        document_id: str,
        filename: str | None,
        data: bytes,
        *,
        owner: str = "local",
        visibility: str | None = None,
    ) -> tuple[DocumentRecord, bool]:
        """Replace a document with a new version, or retry the current one.

        The previous indexed version stays searchable until the new one succeeds.
        """
        current = self._owned(document_id, owner)
        changed_visibility = False
        if visibility is not None and visibility.strip():
            audience = normalize_visibility(visibility)
            if audience != current.visibility:
                self._metadata.set_visibility(document_id, audience)
                changed_visibility = True
                current = self._metadata.get(document_id)
                if current is None:
                    raise IngestionError("Document not found.", status_code=404)
        upload, checksum, parsed_type = self._prepare(filename, data)
        if checksum == current.checksum and current.status == "indexed":
            return current, changed_visibility
        if checksum == current.checksum:
            self._blobs.put(checksum, upload.data)
            record = self._metadata.requeue(current.id, current.version)
            return self._submit(record), True
        other = self._metadata.find_by_checksum(checksum, owner)
        if other is not None and other.id != current.id:
            raise IngestionError(
                "A document with these contents already exists.",
                status_code=409,
            )
        self._blobs.put(checksum, upload.data)
        record = self._metadata.add_version(
            current.id,
            filename=upload.filename,
            content_type=parsed_type,
            checksum=checksum,
            byte_size=len(upload.data),
            version=current.version + 1,
        )
        return self._submit(record), True

    def process_next_job(self) -> bool:
        """Index one queued document. Returns False when the queue is empty."""
        job = self._metadata.claim_job(self._job_timeout_seconds)
        if job is None:
            return False
        self._execute(job.document_id, job.version, raise_errors=False)
        return True

    def recover(self) -> RecoveryReport:
        """Requeue failed versions and indexed versions whose vectors are gone.

        Inline mode rebuilds those versions before returning. Background mode
        leaves them on the queue for the worker. A version that is still
        indexed stays searchable until the replacement finishes.
        """
        failed = 0
        missing = 0
        for record in self._metadata.list_documents():
            if record.status == "failed":
                self._metadata.requeue(record.id, record.version)
                failed += 1
                continue
            if record.indexed_version < 1 or record.version != record.indexed_version:
                continue
            chunks = self._vectors.chunks_for_document(
                record.id,
                version=record.indexed_version,
            )
            if chunks:
                continue
            self._metadata.requeue(record.id, record.indexed_version)
            missing += 1
        reindexed = 0
        if self._indexing_mode == "inline":
            while self.process_next_job():
                reindexed += 1
        log_event(
            "rag.ingestion",
            "index_recovered",
            requeued_failed=failed,
            requeued_missing=missing,
            reindexed=reindexed,
        )
        return RecoveryReport(
            requeued_failed=failed,
            requeued_missing=missing,
            reindexed=reindexed,
        )

    def list_documents(self, *, user_id: str = "local") -> list[DocumentRecord]:
        return self._metadata.list_visible(user_id)

    def get_document(self, document_id: str, *, user_id: str = "local") -> DocumentRecord:
        record = self._metadata.get(document_id)
        if record is None or not _can_read(record, user_id):
            raise IngestionError("Document not found.", status_code=404)
        return record

    def list_versions(self, document_id: str, *, user_id: str = "local") -> list[VersionRecord]:
        self.get_document(document_id, user_id=user_id)
        return self._metadata.list_versions(document_id)

    def read_version(
        self,
        document_id: str,
        version: int,
        *,
        user_id: str = "local",
    ) -> tuple[VersionRecord, bytes]:
        self.get_document(document_id, user_id=user_id)
        stored = self._metadata.get_version(document_id, version)
        if stored is None:
            raise IngestionError("Document version not found.", status_code=404)
        return stored, self._blobs.get(stored.checksum)

    def chunks_for(self, document_id: str, *, user_id: str = "local") -> list[dict]:
        record = self.get_document(document_id, user_id=user_id)
        if record.indexed_version < 1:
            return []
        return self._vectors.chunks_for_document(document_id, version=record.indexed_version)

    def delete_document(self, document_id: str, *, user_id: str = "local") -> None:
        record = self._metadata.get(document_id)
        if record is None or record.owner != user_id:
            raise IngestionError("Document not found.", status_code=404)
        checksums = {version.checksum for version in self._metadata.list_versions(document_id)}
        checksums.add(record.checksum)
        self._vectors.delete_document(record.id)
        if self._vectors.count() == 0:
            self._vectors.recreate()
        self._metadata.delete(record.id)
        for checksum in checksums:
            if not self._metadata.checksum_in_use(checksum):
                self._blobs.delete(checksum)
        if self._metrics is not None:
            self._metrics.inc("rag_documents_deleted_total")
        log_event(
            "rag.ingestion",
            "document_deleted",
            document_id=record.id,
            filename=record.filename,
            version=record.version,
        )

    def _prepare(self, filename: str | None, data: bytes):
        upload = validate_upload(filename, data, max_bytes=self._max_upload_bytes)
        checksum = hashlib.sha256(upload.data).hexdigest()
        chunks = chunks_for_bytes(
            document_id="pending",
            version=0,
            filename=upload.filename,
            data=upload.data,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            chunk_strategy=self._chunk_strategy,
            semantic_similarity=self._semantic_similarity,
            embedder=self._embedder,
            parent_retrieval=self._parent_retrieval,
            child_chunk_size=self._child_chunk_size,
            contextual_retrieval=self._contextual_retrieval,
        )
        return upload, checksum, chunks[0].content_type

    def _submit(self, record: DocumentRecord) -> DocumentRecord:
        if self._indexing_mode == "background":
            return record
        self._metadata.mark_job_running(record.id, record.version)
        return self._execute(record.id, record.version, raise_errors=True)

    def _execute(self, document_id: str, version: int, *, raise_errors: bool) -> DocumentRecord:
        started = time.perf_counter()
        try:
            record = index_stored_version(
                metadata=self._metadata,
                vectors=self._vectors,
                blobs=self._blobs,
                embedder=self._embedder,
                document_id=document_id,
                version=version,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                chunk_strategy=self._chunk_strategy,
                semantic_similarity=self._semantic_similarity,
                parent_retrieval=self._parent_retrieval,
                child_chunk_size=self._child_chunk_size,
                contextual_retrieval=self._contextual_retrieval,
                knowledge_graph=self._knowledge_graph,
            )
        except Exception as exc:
            message = exc.message if isinstance(exc, IngestionError) else "Failed to index the document."
            self._metadata.finish_job(document_id, version, "failed", message)
            if self._metrics is not None:
                self._metrics.inc("rag_documents_failed_total")
                self._metrics.observe("rag_index_seconds", time.perf_counter() - started, status="failed")
            if raise_errors:
                if isinstance(exc, IngestionError):
                    raise
                raise IngestionError(message, status_code=500) from exc
            log_event(
                "rag.worker",
                "job_failed",
                document_id=document_id,
                version=version,
                error=message,
            )
            failed = self._metadata.get(document_id)
            if failed is None:
                raise IngestionError("Document not found.", status_code=404) from exc
            return failed
        self._metadata.finish_job(document_id, version, "succeeded", None)
        if self._metrics is not None:
            self._metrics.inc("rag_documents_indexed_total")
            self._metrics.observe("rag_index_seconds", time.perf_counter() - started, status="succeeded")
        log_event(
            "rag.worker",
            "job_completed",
            document_id=document_id,
            version=version,
            chunk_count=record.chunk_count,
        )
        return record

    def _owned(self, document_id: str, owner: str) -> DocumentRecord:
        record = self._metadata.get(document_id)
        if record is None or record.owner != owner:
            raise IngestionError("Document not found.", status_code=404)
        return record


def _can_read(record: DocumentRecord, user_id: str) -> bool:
    return record.owner == user_id or record.visibility == "shared"
