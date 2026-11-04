"""Acceptance criteria for evidence, hypotheses, agents, and traces."""

from app.research.acceptance import measure_acceptance
from app.research.discovery import discover
from app.research.learning import WASTE


def _report(**overrides) -> dict:
    report = {
        "question": "What are people saying about NVIDIA?",
        "asset": "NVDA",
        "answer": "Retail investors stayed bullish.",
        "stop_reason": "sufficient",
        "elapsed_seconds": 1.25,
        "search_budget": 4,
        "token_budget": 2500,
        "time_budget": 15,
        "metrics": {
            "factual_support": 1.0,
            "confidence": 0.8,
            "citation_validity": 1.0,
            "freshness": 1.0,
            "source_diversity": 1.0,
        },
        "contradictions": [],
        "counterfactual": {"verdict": "none"},
        "collaboration": {
            "agents": [
                {"name": "Lead agent", "searches": 1, "useful": 1},
                {"name": "Corroboration agent", "searches": 1, "useful": 1},
            ]
        },
        "steps": [
            {"action": "search", "status": "ok", "detail": "A new passage.", "agent": "Lead agent"},
            {"action": "search", "status": "ok", "detail": "A new passage.", "agent": "Corroboration agent"},
        ],
        "scene": {
            "claims": [
                {
                    "id": "claim:0",
                    "short": "Retail bullish",
                    "text": "Retail bullish",
                    "provenance": {"filename": "nvda-news.md", "checksum": "abc123def456"},
                },
                {
                    "id": "claim:1",
                    "short": "Data center demand",
                    "text": "Data center demand",
                    "provenance": {"filename": "nvda-social.md", "checksum": "fff999eee888"},
                },
            ],
            "markers": [{"claim_id": "claim:0"}, {"claim_id": "claim:1"}],
        },
    }
    report.update(overrides)
    return report


def _criteria(report: dict) -> dict[str, dict]:
    acceptance = measure_acceptance(report, discover(report))
    return acceptance, {item["criterion"]: item for item in acceptance["criteria"]}


def test_a_cited_path_meets_every_acceptance_criterion() -> None:
    acceptance, criteria = _criteria(_report())

    assert acceptance["met"] is True
    assert acceptance["passed"] == 6
    assert acceptance["summary"] == "The path meets all 6 acceptance criteria."
    assert criteria["evidence"]["result"] == "pass"
    assert criteria["hypothesis"]["result"] == "pass"
    assert criteria["hypothesis"]["reason"].startswith("The hypothesis is relevant")
    assert criteria["reliability"]["result"] == "pass"
    assert criteria["reliability"]["score"] == 1.0
    assert criteria["coordination"]["result"] == "pass"
    assert "2.00 times shorter" in criteria["coordination"]["reason"]
    assert "1 search across" in criteria["coordination"]["reason"]
    assert "1.25s" in criteria["coordination"]["reason"]
    assert criteria["conflicts"]["result"] == "pass"
    assert criteria["reproducibility"]["result"] == "pass"


def test_an_open_contradiction_fails_evidence_and_resolution() -> None:
    acceptance, criteria = _criteria(
        _report(contradictions=[{"status": "pending", "left": "Retail bullish", "right": "Bridgewater cuts"}])
    )

    assert acceptance["met"] is False
    assert criteria["evidence"]["result"] == "fail"
    assert criteria["hypothesis"]["result"] == "pass"
    assert "fail" in criteria["hypothesis"]["reason"]
    assert criteria["conflicts"]["result"] == "fail"
    assert "Retail bullish versus Bridgewater cuts" in criteria["conflicts"]["reason"]
    assert criteria["conflicts"]["score"] == 0.0


def test_a_wasted_search_lowers_reliability_without_failing_the_schedule() -> None:
    report = _report(
        steps=[
            {"action": "search", "status": "ok", "detail": "A new passage.", "agent": "Lead agent"},
            {"action": "search", "status": "ok", "detail": WASTE, "agent": "Corroboration agent"},
        ]
    )
    _acceptance, criteria = _criteria(report)

    assert criteria["reliability"]["result"] == "partial"
    assert criteria["reliability"]["reason"] == "6 of 7 assigned tasks completed correctly."
    assert criteria["coordination"]["result"] == "pass"


def test_a_refusal_fails_the_hypothesis_and_stores_a_partial_trace() -> None:
    acceptance, criteria = _criteria(
        _report(
            answer="I don't have enough evidence to answer.",
            metrics={"factual_support": 0.0, "confidence": 0.1, "citation_validity": 0.0},
            collaboration={"agents": []},
            scene={"claims": [], "markers": []},
            steps=[{"action": "search", "status": "ok", "detail": WASTE, "agent": "Library agent"}],
            contradictions=[],
        )
    )

    assert acceptance["met"] is False
    assert criteria["evidence"]["result"] == "fail"
    assert criteria["hypothesis"]["result"] == "fail"
    assert criteria["reliability"]["result"] == "fail"
    assert criteria["coordination"]["result"] == "fail"
    assert criteria["conflicts"]["result"] == "pass"
    assert criteria["reproducibility"]["result"] == "partial"
    assert "No passage checksum" in criteria["reproducibility"]["reason"]
