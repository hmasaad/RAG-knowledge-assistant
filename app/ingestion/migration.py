"""Re-embed the collection when the embedding model changes.

Vectors from different models, or from the same model at a different width,
are not comparable. Migration replaces the stored vectors and records the
active model.
"""

from dataclasses import dataclass

from app.ingestion.context import embedding_text
from app.ingestion.errors import IngestionError
from app.ingestion.metadata_store import MetadataStore
from app.ingestion.models import Chunk
from app.ingestion.vector_store import VectorStore


@dataclass(frozen=True)
class MigrationResult:
    model: str
    dimension: int
    documents: int
    chunks: int


def migrate_embeddings(
    *,
    metadata: MetadataStore,
    vectors: VectorStore,
    embedder: object,
) -> MigrationResult:
    """Replace every indexed version's vectors with the active embedder.

    The vector collection is rebuilt so a new width can replace the old one.
    """
    model = str(getattr(embedder, "model", "unknown"))
    documents = 0
    chunk_count = 0
    dimension: int | None = None
    pending: list[tuple[list[Chunk], list[list[float]]]] = []
    for record in metadata.list_documents():
        if record.indexed_version < 1:
            continue
        rows = vectors.chunks_for_document(record.id, version=record.indexed_version)
        if not rows:
            continue
        texts = [
            embedding_text(str(row["text"]), str(row.get("context") or ""))
            for row in rows
        ]
        embedded = embedder.embed_documents(texts)
        if len(embedded) != len(rows):
            raise IngestionError(
                "Embedding provider returned an unexpected number of vectors.",
                status_code=502,
            )
        width = len(embedded[0]) if embedded else 0
        if width == 0 or any(len(vector) != width for vector in embedded):
            raise IngestionError(
                "Embedding provider returned malformed vectors.",
                status_code=502,
            )
        if dimension is None:
            dimension = width
        elif dimension != width:
            raise IngestionError(
                "Embedding provider changed vector width during migration.",
                status_code=502,
            )
        rebuilt = [
            Chunk(
                id=str(row["id"]),
                document_id=record.id,
                filename=record.filename,
                content_type=record.content_type,
                text=str(row["text"]),
                index=int(row["index"]),
                start_char=int(row["start_char"]),
                end_char=int(row["end_char"]),
                page=row["page"],
                heading=row["heading"],
                version=record.indexed_version,
                context=str(row.get("context") or ""),
            )
            for row in rows
        ]
        pending.append((rebuilt, embedded))
        documents += 1
        chunk_count += len(rebuilt)

    vectors.recreate()
    for rebuilt, embedded in pending:
        vectors.add(rebuilt, embedded)

    if dimension is None:
        probe = embedder.embed_documents(["embedding migration"])
        if len(probe) != 1 or not probe[0]:
            raise IngestionError(
                "Embedding provider returned malformed vectors.",
                status_code=502,
            )
        dimension = len(probe[0])
    metadata.record_embedding(model, dimension, replace=True)
    return MigrationResult(
        model=model,
        dimension=dimension,
        documents=documents,
        chunks=chunk_count,
    )
