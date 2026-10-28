"""Bayesian design ranks experiments by expected information gain."""

from app.research.bayes import SequentialDesigner
from app.research.contracts import (
    FILL_GAPS,
    RESOLVE_CONTRADICTIONS,
    REVERIFY,
    ClaimVerdict,
    ResearchState,
)
from app.research.voi import ValueCandidate, ValueOfInformation
from fastapi.testclient import TestClient


def test_uncertain_claim_has_higher_entropy_than_a_supported_one() -> None:
    designer = SequentialDesigner()
    uncertain = designer.beliefs(
        ResearchState(verdicts=[ClaimVerdict("retry limit", "insufficient", ())])
    )
    settled = designer.beliefs(
        ResearchState(
            verdicts=[
                ClaimVerdict("retry limit", "supported", ("e1",)),
                ClaimVerdict("retry limit", "supported", ("e2",)),
            ]
        )
    )

    assert uncertain.uncertainty() > settled.uncertainty()
    assert uncertain.claims[0].entropy() > settled.claims[0].entropy()


def test_gap_fill_has_more_information_gain_than_rechecking_a_settled_claim() -> None:
    designer = SequentialDesigner()
    state = ResearchState(
        gaps=["what is the office wifi password"],
        verdicts=[ClaimVerdict("which file implements the retry logic", "supported", ("e1",))],
    )
    beliefs = designer.beliefs(state)
    fill = ValueCandidate(FILL_GAPS, "search", "what is the office wifi password", 0.4, 1, 0.4, "gap")
    reverify = ValueCandidate(REVERIFY, "verification", "which file implements the retry logic", 0.3, 1, 0.3, "check")

    assert designer.expected_information_gain(fill, beliefs) > designer.expected_information_gain(reverify, beliefs)


def test_sequence_looks_ahead_then_stops_when_beliefs_are_settled() -> None:
    designer = SequentialDesigner(horizon=3)
    open_state = ResearchState(
        gaps=["what is the office wifi password"],
        contradictions=["retry-limit-three.md states 3; retry-limit-five.md states 5"],
    )
    ranked = ValueOfInformation().candidates(open_state, remaining_budget=4)
    opened = designer.plan(open_state, ranked, remaining_budget=4)
    settled = designer.plan(
        ResearchState(verdicts=[ClaimVerdict("retry limit", "supported", ("e1",))]),
        [],
        remaining_budget=4,
    )

    assert opened.stop is False
    assert opened.sequence
    assert opened.sequence[0]["action"] in {RESOLVE_CONTRADICTIONS, FILL_GAPS}
    assert settled.stop is True
    assert settled.beliefs.uncertainty() < opened.beliefs.uncertainty()


def test_design_endpoint_returns_beliefs_and_a_sequence(client: TestClient) -> None:
    response = client.post(
        "/api/research/design",
        json={
            "gaps": ["office wifi password"],
            "contradictions": ["retry-limit-three.md states 3; retry-limit-five.md states 5"],
            "remaining_budget": 3,
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["stop"] is False
    assert body["beliefs"]["uncertainty"] > 0
    assert body["sequence"]
    assert body["ranked"][0]["eig"] >= 0
