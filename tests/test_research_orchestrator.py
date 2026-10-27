"""The production orchestrator claims, prioritizes, caps, and cancels jobs."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.ingestion.embeddings import LocalEmbedder
from app.main import create_app
from app.wiring import build_services

ROOT = Path(__file__).resolve().parents[1]


def _background_settings(tmp_path: Path, **extra) -> Settings:
    values = dict(
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
        research_worker_concurrency=2,
        research_user_concurrency=2,
    )
    values.update(extra)
    return Settings(**values)


def test_orchestrator_claims_priority_jobs_and_respects_user_limits(tmp_path: Path) -> None:
    services = build_services(
        _background_settings(tmp_path, research_user_concurrency=1),
        embedder=LocalEmbedder(dimension=128),
    )
    low = services.research.submit_cycle("which file implements the retry logic", priority=1)
    high = services.research.submit_cycle(
        "where are timeouts configured",
        user_id="ada",
        priority=9,
    )
    same_user = services.research.submit_cycle(
        "what is the office wifi password",
        user_id="ada",
        priority=8,
    )

    claimed = services.research._jobs.claim_batch()
    ids = [session.id for session in claimed]

    assert [session.priority for session in claimed] == [9, 1]
    assert high.session_id in ids
    assert low.session_id in ids
    assert same_user.session_id not in ids
    assert all(session.status == "running" for session in claimed)
    assert claimed[0].lease_owner.startswith("worker-")


def test_cancel_stops_a_queued_job_and_a_running_job(tmp_path: Path) -> None:
    settings = _background_settings(tmp_path)
    application = create_app(settings, embedder=LocalEmbedder(dimension=128))
    with TestClient(application) as client:
        client.post(
            "/api/documents",
            files={"file": ("api-errors.md", (ROOT / "samples" / "api-errors.md").read_bytes(), "text/markdown")},
        )
        application.state.services.ingestion.process_next_job()
        queued = client.post(
            "/api/research/cycle",
            json={"question": "which file implements the retry logic", "top_k": 3, "priority": 4},
        )
        cancelled = client.post(f"/api/research/sessions/{queued.json()['session_id']}/cancel")
        skipped = application.state.services.research.process_next_job()
        listed = client.get("/api/research/sessions")

        running = application.state.services.research.submit_cycle(
            "where are timeouts configured",
            top_k=3,
        )
        claimed = application.state.services.research._state.claim()
        application.state.services.research.cancel(running.session_id)
        stopped = application.state.services.research._run_session(claimed.id, claimed=True)

    assert queued.status_code == 202
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert skipped is False
    assert listed.json()["sessions"][0]["priority"] == 4
    assert claimed.id == running.session_id
    assert stopped.status == "cancelled"


def test_orchestrator_finishes_the_batch_when_one_session_fails(tmp_path: Path) -> None:
    services = build_services(
        _background_settings(tmp_path),
        embedder=LocalEmbedder(dimension=128),
    )
    first = services.research.submit_cycle("which file implements the retry logic", priority=2)
    second = services.research.submit_cycle(
        "where are timeouts configured",
        user_id="ada",
        priority=1,
    )
    seen: list[str] = []

    def boom(session_id: str, *, claimed: bool = False, **_kwargs):
        seen.append(session_id)
        if session_id == first.session_id:
            raise RuntimeError("adapter exploded")

    services.research._jobs._run_session = boom
    processed = services.research.process_available()
    stored = services.research._sessions.get(first.session_id)
    metrics = services.metrics

    assert processed == 2
    assert seen == [first.session_id, second.session_id]
    assert stored is not None
    assert stored.status == "failed"
    assert "adapter exploded" in (stored.error or "")
    assert metrics.value("rag_research_jobs_claimed_total", priority="2") == 1
    assert metrics.value("rag_research_jobs_claimed_total", priority="1") == 1
    assert metrics.value("rag_research_jobs_finished_total", status="failed") == 1
    assert metrics.summary("rag_research_jobs_queue_wait_seconds", priority="2")[1] == 1
