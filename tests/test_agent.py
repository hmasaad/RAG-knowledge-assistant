"""Decisions to clarify, search, or retry, plus trace and index recovery."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.generation.agent import (
    CLARIFICATION_TEXT,
    is_document_overview,
    needs_clarification,
    reformulate,
)
from app.ingestion.embeddings import LocalEmbedder
from app.wiring import build_services

ROOT = Path(__file__).resolve().parents[1]


def test_vague_questions_ask_for_a_topic() -> None:
    assert needs_clarification("help")
    assert needs_clarification("help please")
    assert not needs_clarification("which file implements the retry logic")
    assert is_document_overview("what is this document about")
    assert is_document_overview("summarize the file")
    assert not is_document_overview("what is the office wifi password")
    assert not is_document_overview("which file implements the retry logic")
    assert reformulate("Can you tell me which file implements the retry logic?") == (
        "file implements retry logic"
    )


def test_agent_clarifies_searches_and_retries(client: TestClient) -> None:
    payload = (ROOT / "samples" / "api-errors.md").read_bytes()
    uploaded = client.post(
        "/api/documents",
        files={"file": ("api-errors.md", payload, "text/markdown")},
    )
    vague = client.post("/api/answer", json={"question": "help", "top_k": 3})
    found = client.post(
        "/api/answer",
        json={"question": "which file implements the retry logic", "top_k": 3},
    )
    retried = client.post(
        "/api/answer",
        json={
            "question": "please tell me something about app/http/retry.py now",
            "top_k": 3,
        },
    )

    assert uploaded.status_code == 201
    assert vague.status_code == 200
    assert vague.json()["action"] == "clarify"
    assert vague.json()["answer"] == CLARIFICATION_TEXT
    assert vague.json()["passages"] == []
    clarify_trace = client.get(f"/api/traces/{vague.json()['trace_id']}")
    assert clarify_trace.status_code == 200
    assert clarify_trace.json()["steps"][0]["action"] == "clarify"

    assert found.json()["action"] == "answer"
    assert "app/http/retry.py" in found.json()["answer"]
    answer_trace = client.get(f"/api/traces/{found.json()['trace_id']}")
    assert [step["action"] for step in answer_trace.json()["steps"]][:2] == ["search", "answer"]

    assert retried.json()["action"] == "retry"
    assert "app/http/retry.py" in retried.json()["answer"]
    missing = client.get("/api/traces/missing-trace")
    assert missing.status_code == 404


def test_recovery_rebuilds_a_missing_index(tmp_path: Path) -> None:
    services = _services(tmp_path)
    record, _ = services.ingestion.ingest(
        "notes.md",
        b"# Retry logic\n\nThe file that implements retry logic is app/http/retry.py.\n",
    )
    services.vectors.delete_document(record.id)

    report = services.ingestion.recover()
    hits = services.search.search("which file implements the retry logic", top_k=2)

    assert report.requeued_missing == 1
    assert report.reindexed == 1
    assert "retry.py" in hits[0].text


def test_unchanged_section_reuses_cached_embeddings(tmp_path: Path) -> None:
    services = _services(tmp_path, chunk_size=180)
    stable = "# Stable\n\nThe file that implements retry logic is app/http/retry.py.\n\n"
    created, _ = services.ingestion.ingest(
        "policy.md",
        (stable + "# Limit\n\nThe retry limit is three attempts.\n").encode(),
    )
    before = services.metrics.value("rag_embedding_cache_hits_total")
    services.ingestion.update_document(
        created.id,
        "policy.md",
        (stable + "# Limit\n\nThe retry limit is nine attempts.\n").encode(),
    )

    assert services.metrics.value("rag_embedding_cache_hits_total") > before


def _services(tmp_path: Path, **overrides: object):
    values: dict[str, object] = {
        "embedding_provider": "local",
        "embedding_dimension": 128,
        "chroma_path": tmp_path / "chroma",
        "sqlite_path": tmp_path / "metadata.db",
        "blob_path": tmp_path / "blobs",
        "chunk_size": 500,
        "chunk_overlap": 20,
        "max_upload_bytes": 100_000,
        "log_level": "WARNING",
        "llm_provider": "local",
    }
    values.update(overrides)
    return build_services(Settings(**values), embedder=LocalEmbedder(dimension=128))
