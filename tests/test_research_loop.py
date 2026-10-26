"""The adaptive loop picks the next step from quality, gaps, and cost."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.research.loop import PROBE_CONFLICT, REPORT, RETRY, STOP, LoopState, decide

ROOT = Path(__file__).resolve().parents[1]


def test_loop_reports_when_evidence_is_complete() -> None:
    choice = decide(
        LoopState(
            complete=True,
            supported=2,
            gaps=(),
            unresolved_conflicts=(),
            remaining_budget=8,
        )
    )
    assert choice.action == REPORT
    assert "sufficient" in choice.reason


def test_loop_stops_when_the_budget_is_gone() -> None:
    choice = decide(
        LoopState(
            complete=False,
            supported=0,
            gaps=("what is the office wifi password",),
            unresolved_conflicts=(),
            remaining_budget=0,
        )
    )
    assert choice.action == STOP
    assert "budget" in choice.reason


def test_loop_retries_a_gap_and_probes_a_one_sided_conflict() -> None:
    retry = decide(
        LoopState(
            complete=False,
            supported=1,
            gaps=("please tell me something about app/http/retry.py now",),
            unresolved_conflicts=(),
            remaining_budget=4,
        )
    )
    probe = decide(
        LoopState(
            complete=False,
            supported=1,
            gaps=(),
            unresolved_conflicts=('The documents disagree on "what is the retry limit".',),
            remaining_budget=4,
        )
    )
    costly = decide(
        LoopState(
            complete=False,
            supported=1,
            gaps=("what is the office wifi password",),
            unresolved_conflicts=(),
            remaining_budget=3,
            gap_retries=1,
        )
    )

    assert retry.action == RETRY
    assert "retry" in retry.query
    assert probe.action == PROBE_CONFLICT
    assert costly.action == REPORT
    assert "not worth" in costly.reason


def test_research_records_the_adaptive_decision(client: TestClient) -> None:
    payload = (ROOT / "samples" / "api-errors.md").read_bytes()
    client.post(
        "/api/documents",
        files={"file": ("api-errors.md", payload, "text/markdown")},
    )
    complete = client.post(
        "/api/research",
        json={"question": "which file implements the retry logic", "top_k": 3},
    ).json()
    partial = client.post(
        "/api/research",
        json={
            "question": "which file implements the retry logic and what is the office wifi password",
            "top_k": 3,
        },
    ).json()
    complete_trace = client.get(f"/api/traces/{complete['trace_id']}").json()
    decide_steps = [step for step in complete_trace["steps"] if step["action"] == "decide"]

    assert complete["decision"] == "report"
    assert complete["decisions"] == ["report"]
    assert complete["remaining_budget"] >= 0
    assert decide_steps
    assert partial["decision"] in {"report", "retry", "stop"}
    assert any("wifi" in gap for gap in partial["gaps"])
    assert partial["action"] == "research"
