"""A halted cycle resumes from disk after the application process is replaced."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.ingestion.embeddings import LocalEmbedder
from app.main import create_app

ROOT = Path(__file__).resolve().parents[1]
QUESTION = "which file implements the retry logic"


def _settings(tmp_path: Path) -> Settings:
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
    )


def test_halted_session_resumes_after_application_restart(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    payload = (ROOT / "samples" / "api-errors.md").read_bytes()

    first = create_app(settings, embedder=LocalEmbedder(dimension=128))
    with TestClient(first) as client:
        uploaded = client.post(
            "/api/documents",
            files={"file": ("api-errors.md", payload, "text/markdown")},
        )
        halted = client.post(
            "/api/research/cycle",
            json={"question": QUESTION, "top_k": 3, "halt_after": 1},
        )
        body = halted.json()
        session_id = body["session_id"]
        search_runs = [item["runs"] for item in body["side_effects"] if item["action"] == "search"]

        assert uploaded.status_code == 201
        assert halted.status_code == 200
        assert body["status"] == "halted"
        assert body["decisions"] == ["search"]
        assert search_runs == [1]
        assert "reverify" not in body["decisions"]
        assert "synthesize" not in body["decisions"]

    restarted = create_app(settings, embedder=LocalEmbedder(dimension=128))
    with TestClient(restarted) as client:
        stored = client.get(f"/api/research/sessions/{session_id}")
        audit = client.get(f"/api/research/sessions/{session_id}/audit")
        trail = audit.json()
        by_action = {item["action"]: item["status"] for item in trail["plan"]}
        resumed = client.post(f"/api/research/sessions/{session_id}/resume")
        finished = resumed.json()
        search_again = [item["runs"] for item in finished["side_effects"] if item["action"] == "search"]

        assert stored.json()["status"] == "halted"
        assert stored.json()["decisions"] == ["search"]
        assert audit.status_code == 200
        assert trail["status"] == "halted"
        assert trail["events"][0]["action"] == "search"
        assert trail["evidence"]
        assert by_action["search"] == "done"
        assert by_action["reverify"] == "pending"
        assert by_action["synthesize"] == "pending"
        assert resumed.status_code == 200
        assert finished["status"] == "completed"
        assert finished["decisions"][0] == "search"
        assert "reverify" in finished["decisions"]
        assert "synthesize" in finished["decisions"]
        assert finished["decisions"].count("search") == 1
        assert search_again == [1]
        assert "retry.py" in finished["report"]
        assert "[1]" in finished["report"]
