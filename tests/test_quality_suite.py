"""Quality suite: hard questions, citations, conflicts, and retrieval failures."""

from pathlib import Path

from evaluation.harness import load_baseline, run_quality_suite

ROOT = Path(__file__).resolve().parents[1]


def test_phase4_quality_meets_baseline() -> None:
    report = run_quality_suite(ROOT)
    baseline = load_baseline(ROOT / "evaluation" / "baselines" / "phase4_quality.json")
    failures = [f"{result.id}: {result.detail}" for result in report.failures()]

    assert failures == [], report.metrics
    for name, floor in baseline.items():
        assert report.metrics[name] >= floor, f"{name}={report.metrics[name]} floor={floor}"
