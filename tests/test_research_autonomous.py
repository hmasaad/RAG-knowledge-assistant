"""The autonomous planner generates, revises, and tightens research DAGs."""

from app.research.autonomous import AutonomousPlanner
from app.research.contracts import FILL_GAPS, RESOLVE_CONTRADICTIONS, SEARCH, ClaimVerdict, ResearchState
from app.research.guardrails import GuardrailPolicy, ResearchGuardrails
from fastapi.testclient import TestClient


def test_planner_generates_search_then_reverify_for_each_clause() -> None:
    plan = AutonomousPlanner().generate(
        "which file implements the retry logic and where are timeouts configured",
        budget=16,
    )

    assert plan.actions() == ("search", "reverify", "search", "reverify")
    assert plan.dag.tasks["search:1"].depends_on == ()
    assert {task.id for task in plan.dag.ready()} == {"search:0", "search:1"}
    assert plan.dag.next_ready() is not None
    assert plan.dag.next_ready().id == "search:0"
    assert plan.version == 1


def test_revise_adds_gap_fills_and_skips_them_when_budget_is_gone() -> None:
    planner = AutonomousPlanner()
    funded = planner.generate("which file implements the retry logic", budget=8)
    funded.dag.mark_done("search:0")
    funded.dag.mark_done("reverify:0")
    revised = planner.revise(
        funded,
        quality_action=FILL_GAPS,
        quality_reason="The answer is missing evidence.",
        gaps=("what is the office wifi password",),
        remaining_budget=2,
        spent_cost=1,
    )
    empty = planner.generate("which file implements the retry logic", budget=8)
    empty.dag.mark_done("search:0")
    empty.dag.mark_done("reverify:0")
    stopped = planner.revise(
        empty,
        quality_action=FILL_GAPS,
        gaps=("what is the office wifi password",),
        remaining_budget=0,
        spent_cost=8,
    )

    assert FILL_GAPS in revised.actions()
    assert "synthesize" in revised.dag.tasks
    assert revised.version == 2
    assert FILL_GAPS not in stopped.actions()
    assert "synthesize" in stopped.dag.tasks
    assert "budget" in stopped.reason.casefold() or "Stopped" in stopped.revisions[-1]


def test_optimize_drops_denied_actions_and_repeat_fingerprints() -> None:
    planner = AutonomousPlanner(
        ResearchGuardrails(GuardrailPolicy(denied_actions=frozenset({FILL_GAPS})))
    )
    plan = planner.generate("which file implements the retry logic", budget=8)
    plan.dag.mark_done("search:0")
    plan.dag.mark_done("reverify:0")
    revised = planner.revise(
        plan,
        quality_action=FILL_GAPS,
        gaps=("office wifi password",),
        executed={(SEARCH, "which file implements the retry logic", "")},
        remaining_budget=4,
        spent_cost=1,
    )

    assert FILL_GAPS not in revised.actions()
    assert SEARCH not in {
        revised.dag.tasks[task_id].action
        for task_id in revised.dag.order
        if revised.dag.tasks[task_id].status == "pending"
        and revised.dag.tasks[task_id].query.casefold() == "which file implements the retry logic"
    }


def test_replan_adds_gap_work_and_drops_it_when_evidence_arrives() -> None:
    planner = AutonomousPlanner()
    plan = planner.generate("which file implements the retry logic", budget=8)
    plan.dag.mark_done("search:0")
    plan.dag.mark_done("reverify:0")
    missing = ResearchState(gaps=["what is the office wifi password"])
    grown = planner.replan(plan, state=missing, remaining_budget=4, spent_cost=1)
    assert FILL_GAPS in grown.actions()
    assert "synthesize" in grown.dag.tasks

    filled = ResearchState(
        verdicts=[ClaimVerdict("what is the office wifi password", "supported", ("e1",))],
    )
    trimmed = planner.replan(grown, state=filled, remaining_budget=3, spent_cost=2)
    assert FILL_GAPS not in {
        trimmed.dag.tasks[task_id].action
        for task_id in trimmed.dag.order
        if trimmed.dag.tasks[task_id].status == "pending"
    }


def test_replan_adds_a_resolve_step_when_contradictions_appear() -> None:
    planner = AutonomousPlanner()
    plan = planner.generate("what is the retry limit", budget=8)
    plan.dag.mark_done("search:0")
    plan.dag.mark_done("reverify:0")
    conflicted = ResearchState(contradictions=["retry-limit-three.md states 3; retry-limit-five.md states 5"])
    revised = planner.replan(plan, state=conflicted, remaining_budget=4, spent_cost=1)

    assert RESOLVE_CONTRADICTIONS in revised.actions()
    assert set(revised.dag.tasks["resolve:1"].depends_on) == {"reverify:0"}
    assert "synthesize" in revised.dag.tasks
    assert "resolve:1" in revised.dag.tasks["synthesize"].depends_on


def test_plan_endpoint_returns_the_generated_dag(client: TestClient) -> None:
    response = client.post(
        "/api/research/plan",
        json={
            "question": "which file implements the retry logic and where are timeouts configured",
            "top_k": 3,
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["actions"] == ["search", "reverify", "search", "reverify"]
    assert body["tasks"][0]["id"] == "search:0"
    assert body["tasks"][1]["depends_on"] == ["search:0"]
    assert body["version"] == 1
    assert "knowledge" in body["tools"]
