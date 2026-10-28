"""Value of information ranks research actions under a limited budget."""

from app.research.contracts import (
    FILL_GAPS,
    RESOLVE_CONTRADICTIONS,
    REVERIFY,
    ClaimVerdict,
    EvidenceRef,
    ResearchState,
)
from app.research.voi import (
    ValueOfInformation,
    can_stop,
    observe_result,
    record_selection,
    source_reliability,
)
from fastapi.testclient import TestClient


def test_high_severity_contradiction_outvalues_a_gap_search() -> None:
    ranked = ValueOfInformation().candidates(
        ResearchState(
            contradictions=["retry-limit-three.md states 3; retry-limit-five.md states 5"],
            gaps=["what is the office wifi password"],
        ),
        remaining_budget=2,
    )

    assert ranked[0].action == RESOLVE_CONTRADICTIONS
    assert ranked[0].kind == "document"
    assert ranked[0].mandatory is True
    assert ranked[0].voi > next(item.voi for item in ranked if item.action == FILL_GAPS)


def test_zero_budget_selects_nothing() -> None:
    selected = ValueOfInformation().select(
        ResearchState(gaps=["what is the office wifi password"]),
        remaining_budget=0,
    )

    assert selected is None


def test_tool_call_limit_blocks_optional_search() -> None:
    selected = ValueOfInformation().select(
        ResearchState(gaps=["what is the office wifi password"], iterations=8),
        remaining_budget=4,
        iterations=8,
        max_iterations=8,
    )

    assert selected is None


def test_deadline_blocks_selection() -> None:
    selected = ValueOfInformation().select(
        ResearchState(gaps=["what is the office wifi password"]),
        remaining_budget=4,
        deadline="2020-01-01T00:00:00+00:00",
        now="2026-01-01T00:00:00+00:00",
    )

    assert selected is None


def test_duplicate_sources_lower_search_reliability() -> None:
    duplicate = ResearchState(
        evidence=[
            EvidenceRef("e1", "d1", "notes.md", "The retry limit is three attempts."),
            EvidenceRef("e2", "d1", "notes.md", "The retry limit is three attempts."),
        ],
        gaps=["what is the office wifi password"],
    )
    diverse = ResearchState(
        evidence=[
            EvidenceRef("e1", "d1", "notes.md", "The retry limit is three attempts."),
            EvidenceRef("e2", "d2", "guide.md", "Timeouts live in client.py."),
        ],
        gaps=["what is the office wifi password"],
    )

    assert source_reliability(duplicate) < source_reliability(diverse)


def test_mandatory_reverify_is_kept_when_raw_voi_is_low() -> None:
    selected = ValueOfInformation(min_voi=0.99).select(
        ResearchState(
            verdicts=[ClaimVerdict("which file implements the retry logic", "insufficient", ())],
        ),
        remaining_budget=1,
    )

    assert selected is not None
    assert selected.action == REVERIFY
    assert selected.mandatory is True
    assert selected.calibrated_voi < 0.99


def test_low_confidence_does_not_inflate_optional_value() -> None:
    ranked = ValueOfInformation(min_voi=0.4, min_confidence=0.9).candidates(
        ResearchState(gaps=["what is the office wifi password"]),
        remaining_budget=4,
    )

    assert ranked[0].action == FILL_GAPS
    assert ranked[0].confidence < 0.9
    assert ranked[0].calibrated_voi < ranked[0].voi or ranked[0].calibrated_voi < 0.4


def test_stop_requires_mandatory_checks_to_be_done() -> None:
    conflicted = ResearchState(
        contradictions=["retry-limit-three.md states 3; retry-limit-five.md states 5"],
    )
    ranked = ValueOfInformation().candidates(conflicted, remaining_budget=2)
    complete = ResearchState(
        verdicts=[ClaimVerdict("retry limit", "supported", ("e1",))],
    )

    assert can_stop(conflicted, ranked) is False
    assert can_stop(complete, []) is True


def test_audit_records_estimate_and_whether_quality_improved() -> None:
    state = ResearchState()
    selected = ValueOfInformation().select(
        ResearchState(gaps=["what is the office wifi password"]),
        remaining_budget=2,
    )
    assert selected is not None
    record_selection(state, selected, 0.2)
    observed = observe_result(state, 0.6)

    assert observed is not None
    assert observed["estimated_voi"] == selected.voi
    assert observed["quality_before"] == 0.2
    assert observed["quality_after"] == 0.6
    assert observed["improved"] is True
    assert observed["reason"]


def test_verification_is_chosen_when_it_is_the_only_valuable_work() -> None:
    selected = ValueOfInformation().select(
        ResearchState(
            verdicts=[ClaimVerdict("which file implements the retry logic", "insufficient", ())],
        ),
        remaining_budget=1,
    )

    assert selected is not None
    assert selected.action == REVERIFY
    assert selected.kind == "verification"
    assert selected.cost == 1


def test_voi_endpoint_returns_the_ranked_selection(client: TestClient) -> None:
    response = client.post(
        "/api/research/voi",
        json={
            "contradictions": ["retry-limit-three.md states 3; retry-limit-five.md states 5"],
            "gaps": ["office wifi password"],
            "remaining_budget": 1,
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["selected"]["action"] == RESOLVE_CONTRADICTIONS
    assert body["selected"]["mandatory"] is True
    assert body["selected"]["voi"] >= body["candidates"][-1]["voi"]
    assert body["remaining_budget"] == 1
