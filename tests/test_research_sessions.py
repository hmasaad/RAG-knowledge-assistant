"""Research sessions persist cycle state and resume on the worker queue."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.generation.agent import reformulate
from app.ingestion.embeddings import LocalEmbedder
from app.main import create_app
from app.research.contracts import (
    FILL_GAPS,
    REVERIFY,
    SEARCH,
    SYNTHESIZE,
    ActionOutcome,
    ActionRequest,
    ClaimVerdict,
    EvidenceRef,
    ResearchState,
)
from app.research.cycle import run_cycle
from app.research.dispatcher import ActionDispatcher
from app.research.planner import subquestions
from app.wiring import build_services

ROOT = Path(__file__).resolve().parents[1]
QUESTION = "which file implements the retry logic and what is the office wifi password"


class _CycleAdapter:
    def __init__(self, action: str) -> None:
        self.action = action
        self.calls: list[ActionRequest] = []

    def run(self, request: ActionRequest, state: ResearchState | None = None) -> ActionOutcome:
        self.calls.append(request)
        if self.action == SEARCH:
            return ActionOutcome(
                action=SEARCH,
                ok=True,
                evidence=(EvidenceRef("e1", "d1", "api-errors.md", "retry logic is app/http/retry.py"),),
                provenance=("e1",),
                cost=1,
            )
        if self.action == REVERIFY:
            supported = "wifi" not in request.claim
            return ActionOutcome(
                action=REVERIFY,
                ok=True,
                verdicts=(
                    ClaimVerdict(
                        claim=request.claim,
                        verdict="supported" if supported else "insufficient",
                        evidence_ids=("e1",) if supported else (),
                    ),
                ),
                gaps=() if supported else (request.claim,),
                cost=0,
            )
        if self.action == FILL_GAPS:
            return ActionOutcome(action=FILL_GAPS, ok=True, gaps=(request.query,), cost=1)
        return ActionOutcome(action=SYNTHESIZE, ok=True, report=request.query, cost=0)


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


def test_run_cycle_skips_finished_steps_when_resuming() -> None:
    adapters = {name: _CycleAdapter(name) for name in (SEARCH, REVERIFY, FILL_GAPS, SYNTHESIZE)}
    dispatcher = ActionDispatcher(list(adapters.values()), budget=16, max_iterations=8)
    retry, wifi = subquestions(QUESTION)
    state = ResearchState.from_payload(
        {
            "evidence": [
                {
                    "evidence_id": "e1",
                    "document_id": "d1",
                    "source": "api-errors.md",
                    "text": "retry logic is app/http/retry.py",
                }
            ],
            "verdicts": [
                {"claim": retry, "verdict": "supported", "evidence_ids": ["e1"]},
                {"claim": wifi, "verdict": "insufficient", "evidence_ids": []},
            ],
            "gaps": [wifi],
            "executed": [
                [SEARCH, retry, ""],
                [REVERIFY, "", retry],
                [SEARCH, wifi, ""],
                [REVERIFY, "", wifi],
            ],
            "steps": [SEARCH, REVERIFY, SEARCH, REVERIFY],
            "cost": 2,
            "iterations": 4,
        }
    )

    result = run_cycle(dispatcher, QUESTION, state=state)
    alternative = reformulate(wifi) or wifi

    assert [item.query for item in adapters[SEARCH].calls] == []
    assert [item.query for item in adapters[FILL_GAPS].calls] == [alternative]
    assert result.steps[-3:] == (FILL_GAPS, REVERIFY, SYNTHESIZE)
    assert any("wifi" in gap for gap in result.gaps)
    assert "retry.py" in result.report


def test_research_cycle_persists_and_can_be_fetched(client: TestClient) -> None:
    payload = (ROOT / "samples" / "api-errors.md").read_bytes()
    uploaded = client.post(
        "/api/documents",
        files={"file": ("api-errors.md", payload, "text/markdown")},
    )
    created = client.post(
        "/api/research/cycle",
        json={"question": "which file implements the retry logic", "top_k": 3},
    )
    body = created.json()
    fetched = client.get(f"/api/research/sessions/{body['session_id']}")
    listed = client.get("/api/research/sessions")
    resumed = client.post(f"/api/research/sessions/{body['session_id']}/resume")
    stranger = client.get(
        f"/api/research/sessions/{body['session_id']}",
        headers={"X-User-Id": "other"},
    )

    assert uploaded.status_code == 201
    assert created.status_code == 200
    assert body["status"] == "completed"
    assert body["session_id"]
    assert fetched.status_code == 200
    assert fetched.json()["report"] == body["report"]
    assert fetched.json()["session_id"] == body["session_id"]
    assert listed.json()["sessions"][0]["session_id"] == body["session_id"]
    assert resumed.status_code == 200
    assert resumed.json()["report"] == body["report"]
    assert stranger.status_code == 404

    removed = client.delete(f"/api/research/sessions/{body['session_id']}")
    listed_after = client.get("/api/research/sessions")
    missing = client.get(f"/api/research/sessions/{body['session_id']}")
    stranger_delete = client.delete(
        f"/api/research/sessions/{body['session_id']}",
        headers={"X-User-Id": "other"},
    )

    assert removed.status_code == 204
    assert listed_after.json()["sessions"] == []
    assert missing.status_code == 404
    assert stranger_delete.status_code == 404


def test_background_cycle_runs_on_the_worker(tmp_path: Path) -> None:
    settings = _background_settings(tmp_path)
    application = create_app(settings, embedder=LocalEmbedder(dimension=128))
    with TestClient(application) as client:
        uploaded = client.post(
            "/api/documents",
            files={"file": ("api-errors.md", (ROOT / "samples" / "api-errors.md").read_bytes(), "text/markdown")},
        )
        application.state.services.ingestion.process_next_job()
        created = client.post(
            "/api/research/cycle",
            json={"question": "which file implements the retry logic", "top_k": 3},
        )
        waiting = client.get(f"/api/research/sessions/{created.json()['session_id']}")
        worked = application.state.services.research.process_next_job()
        finished = client.get(f"/api/research/sessions/{created.json()['session_id']}")
        health = client.get("/health")

    assert uploaded.status_code == 202
    assert created.status_code == 202
    assert created.json()["status"] == "queued"
    assert waiting.json()["status"] == "queued"
    assert waiting.json()["report"] == ""
    assert worked is True
    assert finished.status_code == 200
    assert finished.json()["status"] == "completed"
    assert "retry.py" in finished.json()["report"]
    assert "[1]" in finished.json()["report"]
    assert health.json()["research_jobs_queued"] == 0


def test_stale_running_research_session_is_requeued(tmp_path: Path) -> None:
    services = build_services(
        _background_settings(tmp_path),
        embedder=LocalEmbedder(dimension=128),
    )
    services.ingestion.ingest("api-errors.md", (ROOT / "samples" / "api-errors.md").read_bytes())
    while services.ingestion.process_next_job():
        pass
    report = services.research.submit_cycle("which file implements the retry logic", top_k=3)
    with services.research._sessions._connect() as connection:
        connection.execute(
            """
            UPDATE research_sessions
            SET status = 'running', started_at = '2000-01-01T00:00:00+00:00'
            WHERE id = ?
            """,
            (report.session_id,),
        )

    assert services.research.process_next_job() is True
    stored = services.research.get_session(report.session_id)
    assert stored.status == "completed"
    assert "retry.py" in stored.report
