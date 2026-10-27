"""Score planner revisions on quality, safety, budget, and recovery floors."""

import json
from pathlib import Path

from app.engineering.loop import LOOP_MANDATORY, ContinuousImprovementLoop, LoopSnapshot
from evaluation.e2e import load_e2e_cases, run_end_to_end_suite

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evaluation" / "datasets" / "improvement_loop.json"
BASELINE = ROOT / "evaluation" / "baselines" / "improvement_loop.json"


def loop_metrics(suite: dict[str, float]) -> dict[str, float]:
    """Map the end-to-end suite onto the mandatory loop scores."""
    return {
        "pass_rate": float(suite.get("pass_rate", 0.0)),
        "correctness": float(suite.get("reliability", 0.0)),
        "safety": float(suite.get("security_respect", 0.0)),
        "budget": float(suite.get("budget_respect", 0.0)),
        "recovery": float(suite.get("recovery", 0.0)),
    }


def load_loop_floors(path: Path | None = None) -> dict[str, float]:
    payload = json.loads((path or BASELINE).read_text(encoding="utf-8"))
    return {key: float(value) for key, value in payload.items()}


def run_improvement_loop(
    root: Path | None = None,
    *,
    planner_version: str = "0.1.0",
    loop: ContinuousImprovementLoop | None = None,
) -> dict:
    """Run the loop suite, record the snapshot, and promote only if floors hold."""
    base = root or ROOT
    cases = load_e2e_cases(base / "evaluation" / "datasets" / "improvement_loop.json")
    metrics = loop_metrics(run_end_to_end_suite(base, cases))
    floors = load_loop_floors(base / "evaluation" / "baselines" / "improvement_loop.json")
    store = loop if loop is not None else ContinuousImprovementLoop()
    if store.baseline() is None:
        store.record("floors", floors, promote=True)
    snapshot = store.promote(planner_version, metrics)
    return {"metrics": metrics, "floors": floors, "snapshot": snapshot, "mandatory": LOOP_MANDATORY}


def snapshot_payload(snapshot: LoopSnapshot) -> dict:
    return snapshot.to_payload()
