"""The research cycle retrieves, verifies, fills gaps, and cites an answer."""

from pathlib import Path

from fastapi.testclient import TestClient

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

ROOT = Path(__file__).resolve().parents[1]


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


def test_run_cycle_searches_verifies_fills_and_synthesizes() -> None:
    adapters = {name: _CycleAdapter(name) for name in (SEARCH, REVERIFY, FILL_GAPS, SYNTHESIZE)}
    dispatcher = ActionDispatcher(list(adapters.values()), budget=16, max_iterations=8)

    result = run_cycle(
        dispatcher,
        "which file implements the retry logic and what is the office wifi password",
    )

    assert list(result.steps) == [
        SEARCH,
        REVERIFY,
        SEARCH,
        REVERIFY,
        FILL_GAPS,
        REVERIFY,
        SYNTHESIZE,
    ]
    assert any("wifi" in gap for gap in result.gaps)
    assert "[1]" in result.report
    assert "retry.py" in result.report


def test_research_cycle_cites_supported_evidence(client: TestClient) -> None:
    payload = (ROOT / "samples" / "api-errors.md").read_bytes()
    uploaded = client.post(
        "/api/documents",
        files={"file": ("api-errors.md", payload, "text/markdown")},
    )
    response = client.post(
        "/api/research/cycle",
        json={"question": "which file implements the retry logic", "top_k": 3},
    )
    planned = client.post(
        "/api/research",
        json={
            "question": "which file implements the retry logic and where are timeouts configured",
            "top_k": 3,
        },
    )
    body = response.json()
    planned_body = planned.json()
    planned_trace = client.get(f"/api/traces/{planned_body['trace_id']}")

    assert uploaded.status_code == 201
    assert response.status_code == 200
    assert SEARCH in body["decisions"]
    assert REVERIFY in body["decisions"]
    assert SYNTHESIZE in body["decisions"]
    assert "[1]" in body["report"]
    assert "retry.py" in body["report"]
    assert body["action"] == "research"
    assert [step["action"] for step in planned_trace.json()["steps"]].count("search") == 2


def test_research_cycle_fills_gaps_then_synthesizes(client: TestClient) -> None:
    payload = (ROOT / "samples" / "api-errors.md").read_bytes()
    uploaded = client.post(
        "/api/documents",
        files={"file": ("api-errors.md", payload, "text/markdown")},
    )
    response = client.post(
        "/api/research/cycle",
        json={
            "question": "which file implements the retry logic and what is the office wifi password",
            "top_k": 3,
        },
    )
    body = response.json()

    assert uploaded.status_code == 201
    assert response.status_code == 200
    assert FILL_GAPS in body["decisions"]
    assert SYNTHESIZE in body["decisions"]
    assert any("wifi" in gap for gap in body["gaps"])
    assert "retry.py" in body["report"]
