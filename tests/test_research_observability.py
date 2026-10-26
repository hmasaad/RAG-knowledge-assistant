"""Research observability records latency, cost, retries, and failures."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.observability.metrics import Metrics
from app.research.contracts import SEARCH, ActionRequest, ResearchState
from app.research.dispatcher import ActionDispatcher
from app.research.observability import ResearchObserver, failure_kind
from app.research.tools import ToolContext
from app.research.worker import ResearchWorker

from tests.test_research_dispatcher import FakeAdapter

ROOT = Path(__file__).resolve().parents[1]


def test_dispatcher_records_action_cost_and_classified_failures() -> None:
    metrics = Metrics()
    dispatcher = ActionDispatcher(
        [FakeAdapter(SEARCH)],
        budget=1,
        max_iterations=4,
        observer=ResearchObserver(metrics),
    )
    state = ResearchState()
    dispatcher.dispatch(ActionRequest(SEARCH, query="retry limit"), state)
    dispatcher.dispatch(ActionRequest(SEARCH, query="timeouts"), state)
    dispatcher.dispatch(ActionRequest("browse", query="retry"), state)

    assert metrics.value("rag_research_cost_total", action=SEARCH, kind="action") == 1.0
    assert metrics.summary("rag_research_action_seconds", action=SEARCH, kind="action", status="ok")[1] == 1
    assert metrics.value("rag_research_failures_total", action=SEARCH, kind="action", error="budget") == 1.0
    assert metrics.value("rag_research_failures_total", action="browse", kind="action", error="invalid") == 1.0
    assert failure_kind("Caller is not permitted.") == "permission"
    assert failure_kind("Action timed out.") == "timeout"


def test_research_metrics_include_latency_cost_retries_and_failures(client: TestClient) -> None:
    payload = (ROOT / "samples" / "api-errors.md").read_bytes()
    client.post("/api/documents", files={"file": ("api-errors.md", payload, "text/markdown")})
    researched = client.post(
        "/api/research",
        json={"question": "which file implements the retry logic", "top_k": 3},
    )
    cycled = client.post(
        "/api/research/cycle",
        json={"question": "what is the office wifi password", "top_k": 3},
    )
    worker = ResearchWorker(observer=ResearchObserver(client.app.state.services.metrics))
    denied = worker.execute("web", "retry", ToolContext(user_id="ada"))
    metrics = client.get("/metrics")
    services = client.app.state.services

    assert researched.status_code == 200
    assert cycled.status_code == 200
    assert denied.ok is False
    assert services.metrics.summary("rag_research_seconds", path="research", status="completed")[1] >= 1
    assert services.metrics.summary("rag_research_seconds", path="cycle", status="completed")[1] >= 1
    assert services.metrics.value("rag_research_cost_total", action="research", kind="request") >= 1
    assert services.metrics.value("rag_research_retries_total", kind="gap") >= 1
    assert services.metrics.value("rag_research_failures_total", action="web", kind="tool", error="invalid") == 1
    assert "rag_research_seconds_count" in metrics.text
    assert "rag_research_retries_total" in metrics.text
    assert "rag_research_failures_total" in metrics.text
