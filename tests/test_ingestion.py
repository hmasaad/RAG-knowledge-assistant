import hashlib
from pathlib import Path

import pymupdf
import pytest

from app.config import Settings
from app.ingestion.embeddings import LocalEmbedder
from app.ingestion.errors import IngestionError
from app.ingestion.service import IngestionService
from app.ingestion.vector_store import VectorStore
from app.retrieval.search import SearchService
from app.wiring import Services
from evaluation.dataset import load_retrieval_cases

ROOT = Path(__file__).resolve().parents[1]


def _read_sample(name: str) -> bytes:
    return (ROOT / "samples" / name).read_bytes()


def test_ingest_search_and_duplicate(services: Services) -> None:
    created, was_created = services.ingestion.ingest("api-errors.md", _read_sample("api-errors.md"))
    services.ingestion.ingest("database.md", _read_sample("database.md"))
    indexed_chunks = services.vectors.count()
    again, was_created_again = services.ingestion.ingest(
        "copy-of-api-errors.md",
        _read_sample("api-errors.md"),
    )

    assert was_created is True
    assert created.status == "indexed"
    assert created.chunk_count >= 2
    assert was_created_again is False
    assert again.id == created.id
    assert services.vectors.count() == indexed_chunks

    hits = services.search.search("which file implements the retry logic", top_k=3)
    assert hits[0].filename == "api-errors.md"
    assert "app/http/retry.py" in hits[0].text
    assert hits[0].heading == "Retry logic"
    assert hits[0].score > hits[-1].score

    reopened = VectorStore(
        services.settings.chroma_path,
        services.settings.collection_name,
    )
    persisted = SearchService(
        vectors=reopened,
        embedder=LocalEmbedder(dimension=services.settings.embedding_dimension),
    )
    persisted_hits = persisted.search("which file implements the retry logic", top_k=1)
    assert "app/http/retry.py" in persisted_hits[0].text


def test_pdf_pages_are_searchable(services: Services) -> None:
    document = pymupdf.open()
    first = document.new_page()
    first.insert_text((72, 72), "Alpha page mentions zephyr-token-one.")
    second = document.new_page()
    second.insert_text((72, 72), "Beta page mentions quillon-token-two.")
    payload = document.tobytes()
    document.close()

    record, _ = services.ingestion.ingest("spec.pdf", payload)
    detail_chunks = services.ingestion.chunks_for(record.id)
    hits = services.search.search("quillon-token-two", top_k=2)

    assert {chunk["page"] for chunk in detail_chunks} == {1, 2}
    assert hits[0].page == 2
    assert "quillon-token-two" in hits[0].text


def test_failed_index_can_be_retried(settings: Settings, services: Services) -> None:
    class BrokenEmbedder:
        model = "broken"

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("network down")

        def embed_query(self, text: str) -> list[float]:
            raise RuntimeError("network down")

    broken = IngestionService(
        metadata=services.metadata,
        vectors=services.vectors,
        embedder=BrokenEmbedder(),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_upload_bytes=settings.max_upload_bytes,
    )
    payload = b"# Notes\n\nA searchable sentence about cobalt-widget.\n"
    with pytest.raises(IngestionError, match="Failed to index"):
        broken.ingest("notes.md", payload)

    failed = services.metadata.find_by_checksum(hashlib.sha256(payload).hexdigest())
    assert failed is not None
    assert failed.status == "failed"
    assert services.vectors.count() == 0

    record, created = services.ingestion.ingest("notes.md", payload)
    assert created is True
    assert record.status == "indexed"
    assert services.search.search("cobalt-widget", top_k=1)[0].text.startswith("# Notes")


def test_empty_and_unreadable_documents_are_rejected(services: Services) -> None:
    with pytest.raises(IngestionError, match="No text") as empty:
        services.ingestion.ingest("blank.md", b"   \n\n")
    assert empty.value.status_code == 422

    with pytest.raises(IngestionError, match="Could not read"):
        services.ingestion.ingest("broken.pdf", b"%PDF-1.4 not a real pdf")

    assert services.ingestion.list_documents() == []


def test_delete_removes_chunks(services: Services) -> None:
    record, _ = services.ingestion.ingest("database.md", _read_sample("database.md"))
    services.ingestion.delete_document(record.id)

    with pytest.raises(IngestionError, match="not found") as missing:
        services.ingestion.get_document(record.id)
    assert missing.value.status_code == 404
    assert services.search.search("schema migrations", top_k=3) == []


def test_retrieval_dataset(services: Services) -> None:
    for path in (ROOT / "samples").glob("*.md"):
        services.ingestion.ingest(path.name, path.read_bytes())
    cases = load_retrieval_cases(ROOT / "evaluation" / "datasets" / "phase1_retrieval.json")

    assert cases
    for case in cases:
        hits = services.search.search(case.query, top_k=3)
        assert hits, case.id
        assert case.expected_substring in hits[0].text
        assert hits[0].filename == case.document
