"""Agent society: hierarchy, blackboard, messages, quorum, and traces."""

from app.research.discovery import discover
from app.research.learning import WASTE


def _report(**overrides) -> dict:
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
        "steps": [],
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


def _notes(report: dict) -> dict[str, dict]:
    return {note["component"]: note for note in discover(report)["society"]["notes"]}


def test_a_supported_path_forms_a_society_and_the_chair_accepts() -> None:
    notes = _notes(_report())

    assert list(notes) == [
        "mas",
        "hierarchy",
        "blackboard",
        "protocol",
        "allocation",
        "planning",
        "debate",
        "consensus",
        "distributed",
        "observability",
    ]
    assert notes["mas"]["decision"] == "formed"
    assert "7 agents" in notes["mas"]["reason"]
    assert "Level 1 is the Chair" in notes["hierarchy"]["reason"]
    assert "Lead agent" in notes["hierarchy"]["reason"]
    assert notes["blackboard"]["decision"] == "posted"
    assert "5 posts" in notes["blackboard"]["reason"]
    assert "replies accept" in notes["protocol"]["reason"]
    assert notes["allocation"]["decision"] == "closed"
    assert notes["planning"]["decision"] == "coordinated"
    assert notes["debate"]["decision"] == "upheld"
    assert notes["consensus"]["decision"] == "accept"
    assert "Investigator yes" in notes["consensus"]["reason"]
    assert notes["distributed"]["decision"] == "partitioned"
    assert notes["observability"]["decision"] == "observed"
    assert "5 protocol messages" in notes["observability"]["reason"]


def test_a_caution_amends_the_claim_without_reopening_the_task_list() -> None:
    report = _report(steps=[{"action": "search", "status": "ok", "detail": WASTE, "agent": "Social agent"}])
    notes = _notes(report)

    assert notes["debate"]["decision"] == "amended"
    assert "kept the Challenger's limit" in notes["debate"]["reason"]
    assert notes["allocation"]["decision"] == "closed"
    assert notes["consensus"]["decision"] == "accept"
    assert "Challenger abstain" in notes["consensus"]["reason"]


def test_an_open_contradiction_assigns_the_next_task_and_rejects_the_claim() -> None:
    notes = _notes(
        _report(contradictions=[{"status": "pending", "left": "Retail bullish", "right": "Bridgewater cuts"}])
    )

    assert notes["protocol"]["reason"].endswith("The Chair replies reject to the team.")
    assert notes["allocation"]["decision"] == "assigned"
    assert "Corroboration agent" in notes["allocation"]["reason"]
    assert notes["debate"]["decision"] == "rejected"
    assert notes["consensus"]["decision"] == "dissent"
    assert notes["planning"]["decision"] == "coordinated"


def test_a_refusal_blocks_the_plan_and_asks_the_lead_to_retrieve() -> None:
    notes = _notes(
        _report(
            answer="I don't have enough evidence to answer.",
            metrics={"factual_support": 0.0, "confidence": 0.2, "citation_validity": 0.0},
            collaboration={"agents": []},
            scene={"claims": [], "markers": []},
            steps=[{"action": "search", "status": "ok", "detail": WASTE, "agent": "Library agent"}],
        )
    )

    assert notes["allocation"]["decision"] == "assigned"
    assert "Lead agent" in notes["allocation"]["reason"]
    assert "retrieve a cited answer" in notes["allocation"]["reason"]
    assert notes["planning"]["decision"] == "blocked"
    assert notes["debate"]["decision"] == "unresolved"
    assert notes["consensus"]["decision"] == "withhold"
    assert notes["distributed"]["decision"] == "single"
    assert "objects to" in notes["protocol"]["reason"]
    assert "replies withhold" in notes["protocol"]["reason"]
