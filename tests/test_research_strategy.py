"""Remember which research actions worked and reuse them on the next similar case."""

from fastapi.testclient import TestClient

from app.research.autonomous import AutonomousPlanner
from app.research.contracts import FILL_GAPS, RESOLVE_CONTRADICTIONS
from app.research.loop import PROBE_CONFLICT, RETRY, LoopState, decide
from app.research.strategy import (
    AMPLE,
    CATALOG,
    COMPARE,
    CONFLICT,
    FACT,
    GAP,
    TIGHT,
    StrategyMemory,
    classify_situation,
    question_kind,
)


def test_empty_memory_keeps_the_fallback_action() -> None:
    memory = StrategyMemory()

    assert memory.recommend(GAP, FILL_GAPS) == FILL_GAPS
    assert memory.recommend(CONFLICT, RESOLVE_CONTRADICTIONS) == RESOLVE_CONTRADICTIONS
    assert classify_situation(gaps=("wifi",)) == GAP
    assert classify_situation(conflicts=("disagree",)) == CONFLICT
    assert question_kind("which file implements the retry logic") == FACT
    assert question_kind("which file implements the retry logic and where are timeouts configured") == COMPARE
    assert question_kind("which documents or files do I have") == CATALOG
    assert classify_situation(
        gaps=("wifi",),
        question="what is the office wifi password",
        remaining_budget=1,
    ) == f"{GAP}|{FACT}|{TIGHT}"
    assert classify_situation(
        gaps=("wifi",),
        question="which file implements the retry logic and where are timeouts configured",
        remaining_budget=4,
    ) == f"{GAP}|{COMPARE}|{AMPLE}"


def test_two_wins_override_the_fallback() -> None:
    memory = StrategyMemory()
    memory.observe(GAP, RESOLVE_CONTRADICTIONS, 1.0)
    memory.observe(GAP, RESOLVE_CONTRADICTIONS, 1.0)
    memory.observe(GAP, FILL_GAPS, 0.0)
    memory.observe(GAP, FILL_GAPS, 0.0)

    assert memory.recommend(GAP, FILL_GAPS) == RESOLVE_CONTRADICTIONS


def test_specific_budget_beats_the_generic_gap_action() -> None:
    memory = StrategyMemory()
    memory.observe(GAP, FILL_GAPS, 1.0)
    memory.observe(GAP, FILL_GAPS, 1.0)
    tight = f"{GAP}|{FACT}|{TIGHT}"
    memory.observe(tight, "stop", 1.0)
    memory.observe(tight, "stop", 1.0)

    assert memory.recommend(tight, FILL_GAPS) == "stop"
    assert memory.recommend(f"{GAP}|{FACT}|{AMPLE}", FILL_GAPS) == FILL_GAPS


def test_denied_actions_are_never_recommended() -> None:
    memory = StrategyMemory()
    memory.observe(CONFLICT, RESOLVE_CONTRADICTIONS, 1.0)
    memory.observe(CONFLICT, RESOLVE_CONTRADICTIONS, 1.0)

    assert memory.recommend(CONFLICT, FILL_GAPS, denied={RESOLVE_CONTRADICTIONS}) == FILL_GAPS


def test_planner_revise_uses_the_learned_action() -> None:
    memory = StrategyMemory()
    memory.observe(GAP, FILL_GAPS, 1.0)
    memory.observe(GAP, FILL_GAPS, 1.0)
    planner = AutonomousPlanner(strategy=memory)
    plan = planner.generate("which file implements the retry logic", budget=8)
    plan.dag.mark_done("search:0")
    plan.dag.mark_done("reverify:0")
    revised = planner.revise(
        plan,
        quality_action="reverify",
        gaps=("office wifi password",),
        remaining_budget=4,
        spent_cost=1,
    )

    assert FILL_GAPS in revised.actions()


def test_loop_uses_a_learned_conflict_action() -> None:
    memory = StrategyMemory()
    memory.observe(CONFLICT, RESOLVE_CONTRADICTIONS, 1.0)
    memory.observe(CONFLICT, RESOLVE_CONTRADICTIONS, 1.0)
    choice = decide(
        LoopState(
            complete=False,
            supported=1,
            gaps=("please tell me something about app/http/retry.py now",),
            unresolved_conflicts=('The documents disagree on "what is the retry limit".',),
            remaining_budget=4,
            quality_action=FILL_GAPS,
        ),
        strategy=memory,
    )

    assert choice.action == PROBE_CONFLICT
    empty = decide(
        LoopState(
            complete=False,
            supported=1,
            gaps=("please tell me something about app/http/retry.py now",),
            unresolved_conflicts=(),
            remaining_budget=4,
        )
    )
    assert empty.action == RETRY


def test_strategy_api_records_and_recommends(client: TestClient) -> None:
    first = client.post(
        "/api/research/strategy",
        json={"situation": "gap", "action": "fill_gaps", "reward": 1.0},
    )
    client.post(
        "/api/research/strategy",
        json={"situation": "gap", "action": "fill_gaps", "reward": 1.0},
    )
    recommended = client.post(
        "/api/research/strategy/recommend",
        json={"situation": "gap", "fallback": "stop"},
    )
    snapshot = client.get("/api/research/strategy")

    assert first.status_code == 200
    assert first.json()["wins"] == 1
    assert recommended.status_code == 200
    assert recommended.json()["action"] == FILL_GAPS
    assert any(item["situation"] == "gap" and item["wins"] == 2 for item in snapshot.json())
