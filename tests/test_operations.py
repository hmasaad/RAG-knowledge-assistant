"""Updates, versions, background indexing, caches, and monitoring."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.ingestion.embeddings import LocalEmbedder
from app.ingestion.errors import IngestionError
from app.ingestion.service import IngestionService
from app.main import create_app
from app.wiring import Services, build_services

ROOT = Path(__file__).resolve().parents[1]
QUESTION = "how many retry attempts are allowed"


def _policy(limit: str) -> bytes:
    return (
        f"# Policy\n\nThe retry limit is {limit} attempts.\n"
        f"Callers are allowed {limit} attempts.\n"
    ).encode()


def _background_settings(tmp_path: Path) -> Settings:
    return Settings(
        embedding_provider="local",
        embedding_dimension=128,
        chroma_path=tmp_path / "chroma",
        sqlite_path=tmp_path / "metadata.db",
        blob_path=tmp_path / "blobs",
        chunk_size=500,
        chunk_overlap=40,
        max_upload_bytes=100_000,
        log_level="WARNING",
        llm_provider="local",
        indexing_mode="background",
        worker_in_process=False,
    )


def test_update_keeps_history_and_replaces_searchable_text(client: TestClient) -> None:
    created = client.post(
        "/api/documents",
        files={"file": ("policy.md", _policy("three"), "text/markdown")},
    )
    document_id = created.json()["id"]
    updated = client.put(
        f"/api/documents/{document_id}",
        files={"file": ("policy.md", _policy("nine"), "text/markdown")},
    )
    versions = client.get(f"/api/documents/{document_id}/versions")
    previous = client.get(f"/api/documents/{document_id}/versions/1/file")
    found = client.post("/api/search", json={"query": QUESTION, "top_k": 3})

    assert created.status_code == 201
    assert created.json()["version"] == 1
    assert updated.status_code == 200
    assert updated.json()["version"] == 2
    assert updated.json()["indexed_version"] == 2
    assert updated.json()["already_indexed"] is False
    assert [item["status"] for item in versions.json()["versions"]] == ["superseded", "indexed"]
    assert b"three attempts" in previous.content
    assert "nine" in found.json()["hits"][0]["text"]
    assert "three" not in found.json()["hits"][0]["text"]


def test_same_bytes_do_not_create_a_version(client: TestClient) -> None:
    payload = _policy("three")
    created = client.post(
        "/api/documents",
        files={"file": ("policy.md", payload, "text/markdown")},
    )
    document_id = created.json()["id"]
    again = client.put(
        f"/api/documents/{document_id}",
        files={"file": ("policy.md", payload, "text/markdown")},
    )

    assert again.status_code == 200
    assert again.json()["already_indexed"] is True
    assert again.json()["version"] == 1


def test_failed_update_leaves_the_previous_version_searchable(services: Services) -> None:
    record, _ = services.ingestion.ingest("policy.md", _policy("three"))

    class Boom:
        model = "boom"

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("down")

        def embed_query(self, text: str) -> list[float]:
            raise RuntimeError("down")

    broken = IngestionService(
        metadata=services.metadata,
        vectors=services.vectors,
        embedder=Boom(),
        blobs=services.blobs,
        chunk_size=services.settings.chunk_size,
        chunk_overlap=services.settings.chunk_overlap,
        max_upload_bytes=services.settings.max_upload_bytes,
    )
    with pytest.raises(IngestionError, match="Failed to index"):
        broken.update_document(record.id, "policy.md", _policy("nine"), owner="local")

    current = services.metadata.get(record.id)
    hits = services.search.search(QUESTION, top_k=2)
    assert current is not None
    assert current.status == "failed"
    assert current.version == 2
    assert current.indexed_version == 1
    assert "three" in hits[0].text
    assert "nine" not in hits[0].text


def test_delete_removes_bytes_versions_and_search_hits(services: Services) -> None:
    record, _ = services.ingestion.ingest("policy.md", _policy("three"))
    blob = (
        services.settings.blob_path
        / record.checksum[:2]
        / record.checksum[2:4]
        / record.checksum
    )

    services.ingestion.delete_document(record.id)
    hits = services.search.search(QUESTION, top_k=3)

    assert blob.is_file() is False
    assert services.metadata.get(record.id) is None
    assert services.metadata.list_versions(record.id) == []
    assert hits == []


def test_duplicate_contents_cannot_replace_another_document(services: Services) -> None:
    services.ingestion.ingest("policy.md", _policy("three"))
    other, _ = services.ingestion.ingest("limits.md", _policy("nine"))

    with pytest.raises(IngestionError, match="already exists") as error:
        services.ingestion.update_document(other.id, "limits.md", _policy("three"), owner="local")

    assert error.value.status_code == 409


def test_background_job_is_searchable_after_the_worker_runs(tmp_path: Path) -> None:
    settings = _background_settings(tmp_path)
    application = create_app(settings, embedder=LocalEmbedder(dimension=128))
    with TestClient(application) as client:
        created = client.post(
            "/api/documents",
            files={"file": ("api-errors.md", (ROOT / "samples" / "api-errors.md").read_bytes(), "text/markdown")},
        )
        waiting = client.post(
            "/api/search",
            json={"query": "which file implements the retry logic", "top_k": 3},
        )
        worked = application.state.services.ingestion.process_next_job()
        found = client.post(
            "/api/search",
            json={"query": "which file implements the retry logic", "top_k": 3},
        )

    assert created.status_code == 202
    assert created.json()["status"] == "queued"
    assert waiting.json()["hits"] == []
    assert worked is True
    assert "app/http/retry.py" in found.json()["hits"][0]["text"]


def test_stale_running_job_is_requeued(tmp_path: Path) -> None:
    services = build_services(
        _background_settings(tmp_path),
        embedder=LocalEmbedder(dimension=128),
    )
    record, _ = services.ingestion.ingest("policy.md", _policy("three"))
    with services.metadata._connect() as connection:
        connection.execute(
            """
            UPDATE index_jobs
            SET status = 'running', started_at = '2000-01-01T00:00:00+00:00'
            WHERE document_id = ?
            """,
            (record.id,),
        )

    assert services.ingestion.process_next_job() is True
    stored = services.metadata.get(record.id)
    assert stored is not None
    assert stored.status == "indexed"
    assert "three" in services.search.search(QUESTION, top_k=1)[0].text


def test_embedding_and_answer_caches_reuse_work_until_the_catalog_changes(services: Services) -> None:
    services.ingestion.ingest("policy.md", _policy("three"))
    services.search.search(QUESTION, top_k=2)
    misses = services.metrics.value("rag_embedding_cache_misses_total")
    services.search.search(QUESTION, top_k=2)
    first = services.generation.answer(QUESTION, top_k=2)
    second = services.generation.answer(QUESTION, top_k=2)

    assert services.metrics.value("rag_embedding_cache_hits_total") >= 1
    assert services.metrics.value("rag_embedding_cache_misses_total") == misses
    assert first.answer == second.answer
    assert services.metrics.value("rag_answer_cache_hits_total") == 1
    assert "three" in first.answer

    document_id = services.ingestion.list_documents()[0].id
    services.ingestion.update_document(document_id, "policy.md", _policy("nine"), owner="local")
    third = services.generation.answer(QUESTION, top_k=2)

    assert "nine" in third.answer
    assert services.metrics.value("rag_answer_cache_hits_total") == 1


def test_ready_and_metrics_report_the_service(client: TestClient) -> None:
    ready = client.get("/ready")
    body = client.get("/metrics")

    assert ready.status_code == 200
    assert ready.json()["status"] == "ok"
    assert "rag_chunks" in body.text
    assert "rag_catalog_generation" in body.text
    assert "rag_http_requests_total" in body.text
