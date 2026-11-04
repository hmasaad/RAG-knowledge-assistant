"""Discovery team: investigate, challenge, test, and conclude."""

from app.research.discovery import discover
from app.research.learning import WASTE


def _report(
    *,
    answer: str = "Retail investors stayed bullish.",
    support: float = 1.0,
    confidence: float = 0.72,
    contradictions: list | None = None,
    steps: list | None = None,
    counterfactual: str = "none",
    shorts: list[str] | None = None,
) -> dict:
    claims = []
    markers = []
    for index, short in enumerate(shorts or ["Retail bullish"]):
        claim_id = f"claim:{index}"
        claims.append({"id": claim_id, "short": short, "text": short})
        markers.append({"claim_id": claim_id})
    return {
        "question": "What are people saying about NVIDIA?",
        "answer": answer,
        "metrics": {"factual_support": support, "confidence": confidence},
        "contradictions": contradictions or [],
        "counterfactual": {"verdict": counterfactual},
        "steps": steps or [],
        "scene": {"claims": claims, "markers": markers},
    }


def test_a_grounded_answer_with_no_conflict_is_concluded() -> None:
    result = discover(_report(shorts=["Retail bullish", "Data center demand"]))

    assert result["hypothesis"]["status"] == "support"
    assert "Retail bullish and Data center demand" in result["hypothesis"]["statement"]
    assert result["reviews"][0]["stance"] == "found"
    assert result["reviews"][1]["stance"] == "clear"
    assert result["conclusion"]["verdict"] == "concluded"
    assert result["conclusion"]["text"].startswith("The team concludes:")


def test_an_open_contradiction_blocks_the_conclusion() -> None:
    result = discover(
        _report(
            contradictions=[
                {"status": "pending", "left": "Retail bullish", "right": "Bridgewater cuts"}
            ]
        )
    )

    assert result["hypothesis"]["status"] == "contest"
    assert result["reviews"][1]["stance"] == "challenge"
    assert "Retail bullish" in result["reviews"][1]["finding"]
    assert "Bridgewater cuts" in result["reviews"][1]["finding"]
    assert result["conclusion"]["verdict"] == "disputed"
    assert result["conclusion"]["text"].startswith("The team does not conclude.")


def test_a_refusal_stays_untested() -> None:
    result = discover(
        _report(answer="I don't have enough evidence to answer.", support=0.0, confidence=0.2, shorts=[])
    )

    assert result["reviews"][0]["stance"] == "empty"
    assert result["hypothesis"]["status"] == "untested"
    assert result["conclusion"]["verdict"] == "insufficient"


def test_a_wasted_search_is_a_caution_and_does_not_block() -> None:
    result = discover(
        _report(
            steps=[{"action": "search", "detail": WASTE}],
            counterfactual="missed",
        )
    )

    assert result["reviews"][1]["stance"] == "caution"
    assert "no new evidence" in result["reviews"][1]["finding"]
    assert "higher expected value" in result["reviews"][1]["finding"]
    assert result["conclusion"]["verdict"] == "concluded"
    assert "Challenger noted a limit" in result["conclusion"]["text"]
