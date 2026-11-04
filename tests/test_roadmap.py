"""Eight-phase scientific discovery roadmap."""

from app.research.discovery import discover
from app.research.learning import WASTE


def _report(**overrides) -> dict:
    claims = [
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
    ]
    report = {
        "question": "What are people saying about NVIDIA?",
        "asset": "NVDA",
        "answer": "Retail investors stayed bullish.",
        "stop_reason": "sufficient",
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
            {"action": "search", "status": "ok", "strategy": "initial", "top_k": 4, "detail": "A new passage.", "agent": "Lead agent"},
            {"action": "investigate", "status": "ok", "strategy": "corroborate", "detail": "Searching documents outside the current graph."},
            {"action": "search", "status": "ok", "strategy": "corroborate", "top_k": 6, "detail": "A new passage.", "agent": "Corroboration agent"},
        ],
        "scene": {
            "claims": claims,
            "markers": [{"claim_id": "claim:0"}, {"claim_id": "claim:1"}],
        },
    }
    report.update(overrides)
    return report


def _phases(report: dict) -> dict[str, dict]:
    return {phase["component"]: phase for phase in discover(report)["phases"]}


def test_a_supported_path_runs_all_eight_stages() -> None:
    phases = _phases(_report())

    assert list(phases) == [
        "orchestration",
        "hypotheses",
        "collaboration",
        "debate",
        "experiment",
        "consensus",
        "reproducibility",
        "evaluation",
    ]
    assert phases["orchestration"]["status"] == "complete"
    assert "Investigator done" in phases["orchestration"]["summary"]
    assert phases["hypotheses"]["status"] == "ranked"
    assert phases["hypotheses"]["records"][0]["detail"].startswith("1.00 ·")
    alternatives = " ".join(record["detail"] for record in phases["hypotheses"]["records"][1:])
    assert "Retail bullish" in alternatives
    assert "Data center demand" in alternatives
    assert phases["collaboration"]["status"] == "gathered"
    assert "Lead agent" in phases["collaboration"]["summary"]
    assert phases["experiment"]["status"] == "pass"
    assert phases["consensus"]["status"] == "accept"
    assert phases["reproducibility"]["status"] == "reproducible"
    assert "nvda-news.md (abc123def456)" in phases["reproducibility"]["records"][1]["detail"]
    assert "corroborate top_k 6" in phases["reproducibility"]["records"][3]["detail"]
    assert phases["evaluation"]["status"] == "scored"
    assert phases["evaluation"]["records"][-1]["label"] == "overall"


def test_the_same_path_keeps_the_same_fingerprint() -> None:
    report = _report()
    first = _phases(report)["reproducibility"]["records"][0]["detail"]
    second = _phases(report)["reproducibility"]["records"][0]["detail"]

    assert first == second
    assert len(first) == 12


def test_duplicate_claims_are_normalized_before_ranking() -> None:
    report = _report()
    report["scene"]["claims"].append(
        {"id": "claim:2", "short": "retail   bullish", "text": "retail bullish", "provenance": {}}
    )
    report["scene"]["markers"].append({"claim_id": "claim:2"})

    records = _phases(report)["hypotheses"]["records"]
    alternatives = [record["detail"] for record in records[1:]]

    assert sum("retail bullish" in item.casefold() for item in alternatives) == 1


def test_an_open_contradiction_fails_the_experiment_and_the_vote() -> None:
    report = _report(
        contradictions=[{"status": "pending", "left": "Retail bullish", "right": "Bridgewater cuts"}],
        counterfactual={"verdict": "missed"},
    )
    phases = _phases(report)

    assert phases["debate"]["status"] == "challenged"
    assert any("Retail bullish and Bridgewater cuts" in record["detail"] for record in phases["debate"]["records"])
    assert phases["experiment"]["status"] == "fail"
    assert phases["consensus"]["status"] == "dissent"
    assert phases["evaluation"]["records"][2]["detail"] == "0.40"


def test_a_refusal_withholds_consensus_and_stores_a_partial_manifest() -> None:
    report = _report(
        answer="I don't have enough evidence to answer.",
        metrics={"factual_support": 0.0, "confidence": 0.2, "citation_validity": 0.0},
        steps=[{"action": "search", "status": "ok", "detail": WASTE, "agent": "Library agent"}],
        collaboration={"agents": []},
        scene={"claims": [], "markers": []},
    )
    phases = _phases(report)

    assert phases["orchestration"]["status"] == "partial"
    assert "Investigator skipped" in phases["orchestration"]["summary"]
    assert phases["hypotheses"]["status"] == "empty"
    assert phases["collaboration"]["status"] == "partial"
    assert phases["experiment"]["status"] == "not_run"
    assert phases["consensus"]["status"] == "withhold"
    assert phases["reproducibility"]["status"] == "partial"
    assert "no passage checksum" in phases["reproducibility"]["records"][1]["detail"]
