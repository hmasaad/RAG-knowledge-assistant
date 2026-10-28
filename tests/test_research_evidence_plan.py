"""Evidence quality, gaps, and contradiction severity drive DAG changes."""

from app.research.autonomous import AutonomousPlanner
from app.research.contracts import (
    FILL_GAPS,
    RESOLVE_CONTRADICTIONS,
    REVERIFY,
    ClaimVerdict,
    ResearchState,
)
from app.research.evidence_plan import EvidenceDrivenPlanner
from fastapi.testclient import TestClient


def test_high_severity_contradiction_adds_resolve() -> None:
    decision = EvidenceDrivenPlanner().decide(
        ResearchState(
            contradictions=["retry-limit-three.md states 3; retry-limit-five.md states 5"],
            gaps=["what is the office wifi password"],
        )
    )

    assert decision.action == RESOLVE_CONTRADICTIONS
    assert decision.verb == "add"
    assert decision.severity == "high"
    assert decision.signals["contradiction_severity"] == "high"


def test_gaps_add_fill_and_supported_claims_remove_it() -> None:
    planner = EvidenceDrivenPlanner()
    missing = planner.decide(ResearchState(gaps=["what is the office wifi password"]))
    cleared = planner.decide(
        ResearchState(
            verdicts=[ClaimVerdict("what is the office wifi password", "supported", ("e1",))],
        ),
        pending_actions=(FILL_GAPS,),
    )

    assert missing.action == FILL_GAPS
    assert missing.verb == "add"
    assert cleared.action == "stop"
    assert cleared.verb == "remove"


def test_unresolved_claim_repeats_reverify() -> None:
    decision = EvidenceDrivenPlanner().decide(
        ResearchState(
            verdicts=[ClaimVerdict("which file implements the retry logic", "insufficient", ())],
        ),
        pending_actions=(REVERIFY,),
    )

    assert decision.action == REVERIFY
    assert decision.verb == "repeat"
    assert decision.quality == 0.0


def test_replan_reprioritizes_resolve_ahead_of_a_pending_fill() -> None:
    planner = AutonomousPlanner()
    plan = planner.generate("what is the retry limit", budget=8)
    plan.dag.mark_done("search:0")
    plan.dag.mark_done("reverify:0")
    with_gap = planner.replan(
        plan,
        state=ResearchState(gaps=["what is the office wifi password"]),
        remaining_budget=4,
        spent_cost=1,
    )
    both = planner.replan(
        with_gap,
        state=ResearchState(
            gaps=["what is the office wifi password"],
            contradictions=["retry-limit-three.md states 3; retry-limit-five.md states 5"],
        ),
        remaining_budget=3,
        spent_cost=2,
    )
    pending = [
        both.dag.tasks[task_id].action
        for task_id in both.dag.order
        if both.dag.tasks[task_id].status == "pending"
        and both.dag.tasks[task_id].action != "synthesize"
    ]

    assert FILL_GAPS in pending
    assert RESOLVE_CONTRADICTIONS in pending
    assert pending.index(RESOLVE_CONTRADICTIONS) < pending.index(FILL_GAPS)
    assert both.decision["verb"] in {"add", "reprioritize"}
    assert both.signals["contradiction_severity"] == "high"


def test_evidence_plan_endpoint_returns_the_decision(client: TestClient) -> None:
    response = client.post(
        "/api/research/evidence-plan",
        json={
            "contradictions": ["retry-limit-three.md states 3; retry-limit-five.md states 5"],
            "gaps": ["office wifi password"],
            "pending_actions": ["fill_gaps", "resolve_contradictions"],
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["action"] == RESOLVE_CONTRADICTIONS
    assert body["verb"] == "reprioritize"
    assert body["severity"] == "high"
    assert body["signals"]["gaps"] == ["office wifi password"]
