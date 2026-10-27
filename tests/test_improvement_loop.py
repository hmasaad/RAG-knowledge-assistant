"""Continuous evaluation loop: record planner scores and block a weaker revision."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.engineering.loop import ContinuousImprovementLoop, compare_metrics
from app.engineering.service import EngineeringService

ROOT = Path(__file__).resolve().parents[1]

FLOORS = {
    "correctness": 1.0,
    "safety": 1.0,
    "budget": 1.0,
    "recovery": 1.0,
}


def test_loop_blocks_a_mandatory_regression(tmp_path: Path) -> None:
    loop = ContinuousImprovementLoop(tmp_path / "metadata.db")
    first = loop.record("planner-a", FLOORS, promote=True)
    weaker = dict(FLOORS)
    weaker["safety"] = 0.0
    blocked = loop.promote("planner-b", weaker)

    assert first.promoted is True
    assert first.allowed is True
    assert blocked.allowed is False
    assert blocked.promoted is False
    assert "safety" in blocked.regressions
    assert loop.baseline() is not None
    assert loop.baseline().planner_version == "planner-a"
    assert compare_metrics(FLOORS, weaker).allowed is False


def test_loop_api_refuses_a_weaker_promotion(client: TestClient) -> None:
    first = client.post(
        "/api/engineering/loop",
        json={"planner_version": "planner-a", "metrics": FLOORS, "promote": True},
    )
    blocked = client.post(
        "/api/engineering/loop/promote",
        json={"planner_version": "planner-b", "metrics": {**FLOORS, "budget": 0.5}},
    )
    state = client.get("/api/engineering/loop")

    assert first.status_code == 200
    assert first.json()["promoted"] is True
    assert blocked.status_code == 200
    assert blocked.json()["allowed"] is False
    assert "budget" in blocked.json()["regressions"]
    assert blocked.json()["promoted"] is False
    assert state.status_code == 200
    assert state.json()["baseline"]["planner_version"] == "planner-a"
    assert len(state.json()["history"]) >= 2


def test_improvement_loop_suite_stays_at_floor() -> None:
    from evaluation.improvement_loop import run_improvement_loop

    result = run_improvement_loop(ROOT, planner_version="0.1.0")
    snapshot = result["snapshot"]

    assert snapshot.allowed is True
    assert snapshot.promoted is True
    for name in result["mandatory"]:
        assert result["metrics"][name] >= result["floors"][name]
    assert result["metrics"]["pass_rate"] >= result["floors"]["pass_rate"]


def test_engineering_service_records_loop_history(tmp_path: Path) -> None:
    service = EngineeringService(tmp_path / "metadata.db")
    service.record_loop("planner-a", FLOORS, promote=True)
    service.record_loop("planner-b", {**FLOORS, "recovery": 0.0}, promote=True)

    state = service.loop_state()
    assert state.baseline is not None
    assert state.baseline.planner_version == "planner-a"
    assert [item.planner_version for item in state.history] == ["planner-a", "planner-b"]
    assert state.history[-1].allowed is False
