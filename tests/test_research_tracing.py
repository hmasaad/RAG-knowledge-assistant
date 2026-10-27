"""Research traces persist across processes and keep a parent/child span tree."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.ingestion.embeddings import LocalEmbedder
from app.main import create_app
from app.observability.tracing import TraceLog

ROOT = Path(__file__).resolve().parents[1]
QUESTION = "which file implements the retry logic"


def test_trace_log_reloads_spans_from_sqlite(tmp_path: Path) -> None:
    path = tmp_path / "metadata.db"
    first = TraceLog(path)
    trace_id = first.start(question=QUESTION, user_id="ada")
    root = first.step(trace_id, action="search", query=QUESTION, duration_ms=12.5, cost=1, status="ok")
    child = first.step(trace_id, action="reverify", parent_id=root, duration_ms=4.0, cost=0, status="ok")

    reloaded = TraceLog(path).get(trace_id)

    assert reloaded is not None
    assert reloaded["question"] == QUESTION
    assert reloaded["user_id"] == "ada"
    assert [step["action"] for step in reloaded["steps"]] == ["search", "reverify"]
    assert reloaded["steps"][0]["span_id"] == root
    assert reloaded["steps"][0]["parent_id"] == ""
    assert reloaded["steps"][1]["span_id"] == child
    assert reloaded["steps"][1]["parent_id"] == root
    assert reloaded["steps"][0]["duration_ms"] == 12.5
    assert reloaded["steps"][0]["cost"] == 1


def test_halted_cycle_keeps_the_same_trace_after_restart(tmp_path: Path) -> None:
    settings = Settings(
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
        trace_id = body["trace_id"]
        session_id = body["session_id"]
        mid = client.get(f"/api/traces/{trace_id}")

        assert uploaded.status_code == 201
        assert halted.status_code == 200
        assert mid.status_code == 200
        assert mid.json()["id"] == trace_id
        assert mid.json()["session_id"] == session_id
        assert mid.json()["steps"][0]["action"] == "search"
        assert mid.json()["steps"][0]["span_id"]
        assert {step["action"] for step in mid.json()["steps"]} >= {"search", "cycle"}

    restarted = create_app(settings, embedder=LocalEmbedder(dimension=128))
    with TestClient(restarted) as client:
        stored = client.get(f"/api/traces/{trace_id}")
        resumed = client.post(f"/api/research/sessions/{session_id}/resume")
        finished = client.get(f"/api/traces/{trace_id}")
        metrics = client.get("/metrics")
        actions = [step["action"] for step in finished.json()["steps"]]

        assert stored.status_code == 200
        assert stored.json()["id"] == trace_id
        assert stored.json()["session_id"] == session_id
        assert stored.json()["steps"][0]["action"] == "search"
        assert resumed.status_code == 200
        assert resumed.json()["trace_id"] == trace_id
        assert finished.json()["session_id"] == session_id
        assert actions[0] == "search"
        assert "reverify" in actions
        assert "synthesize" in actions
        assert finished.json()["steps"][1]["parent_id"] == finished.json()["steps"][0]["span_id"]
        assert any(step.get("duration_ms") is not None for step in finished.json()["steps"])
        assert "rag_research_seconds_count" in metrics.text
        assert "rag_research_action_seconds_count" in metrics.text
        assert "rag_research_reliability_count" in metrics.text


def test_research_run_records_quality_cost_and_latency(client: TestClient) -> None:
    payload = (ROOT / "samples" / "api-errors.md").read_bytes()
    client.post("/api/documents", files={"file": ("api-errors.md", payload, "text/markdown")})
    researched = client.post(
        "/api/research",
        json={"question": QUESTION, "top_k": 3},
    )
    cycled = client.post(
        "/api/research/cycle",
        json={"question": QUESTION, "top_k": 3},
    )
    planned = client.get(f"/api/traces/{researched.json()['trace_id']}")
    cycle = client.get(f"/api/traces/{cycled.json()['trace_id']}")
    metrics = client.get("/metrics")

    assert researched.status_code == 200
    assert cycled.status_code == 200
    assert planned.status_code == 200
    assert cycle.status_code == 200
    assert planned.json()["reliability"] is not None
    assert planned.json()["quality_confident"] is True
    assert planned.json()["duration_ms"] >= 0
    assert planned.json()["cost"] >= 1
    assert cycle.json()["reliability"] is not None
    assert cycle.json()["duration_ms"] >= 0
    assert cycle.json()["cost"] >= 1
    assert cycle.json()["status"] in {"completed", "research"}
    assert "rag_research_reliability_count" in metrics.text
