"""Specialized agents share a board and a referee resolves disagreements."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.research.contracts import RESOLVE_CONTRADICTIONS, REVERIFY, SEARCH, SYNTHESIZE
from app.research.coordination import (
    CITE_BOTH,
    PREFER_SUPPORTED,
    UNRESOLVED,
    AgentFinding,
    ConflictReferee,
    MultiAgentCoordinator,
)
from app.research.planner import build_plan
from app.research.verification import Contradiction

ROOT = Path(__file__).resolve().parents[1]


def test_coordinator_assigns_a_searcher_and_verifier_per_clause() -> None:
    plan = build_plan(
        "which file implements the retry logic and where are timeouts configured",
        budget=16,
    )
    assignments = MultiAgentCoordinator().assign(plan)
    roles = [item.role for item in assignments]
    actions = [item.action for item in assignments]

    assert roles.count("searcher") == 2
    assert roles.count("verifier") == 2
    assert roles[-2:] == ["referee", "synthesizer"]
    assert actions.count(SEARCH) == 2
    assert actions.count(REVERIFY) == 2
    assert actions[-2:] == [RESOLVE_CONTRADICTIONS, SYNTHESIZE]
    assert assignments[-1].depends_on == ("referee",)


def test_denied_referee_lets_the_synthesizer_follow_verifiers() -> None:
    plan = build_plan("which file implements the retry logic", budget=8)
    assignments = MultiAgentCoordinator(denied_actions={RESOLVE_CONTRADICTIONS}).assign(plan)

    assert "referee" not in {item.agent_id for item in assignments}
    assert assignments[-1].role == "synthesizer"
    assert assignments[-1].depends_on == ("verifier:q1",)


def test_referee_cites_both_supported_numeric_sides() -> None:
    left = AgentFinding(
        "searcher:a",
        "searcher",
        "The retry limit is three attempts.",
        "supported",
        source="retry-limit-three.md",
        values=(3,),
    )
    right = AgentFinding(
        "searcher:b",
        "searcher",
        "The retry limit is five attempts.",
        "supported",
        source="retry-limit-five.md",
        values=(5,),
    )
    resolution = ConflictReferee().resolve((left, right))[0]

    assert resolution.decision == CITE_BOTH
    assert left.claim in resolution.kept_claims
    assert right.claim in resolution.kept_claims
    assert resolution.dropped_claims == ()


def test_referee_prefers_supported_and_leaves_two_weak_findings_unresolved() -> None:
    supported = AgentFinding("a", "searcher", "Timeouts live in client.py", "supported", source="api-errors.md")
    weak = AgentFinding("b", "searcher", "Timeouts live in nowhere.py", "insufficient", source="missing.md")
    empty = AgentFinding("c", "searcher", "The office wifi password is violet", "insufficient")
    other = AgentFinding("d", "searcher", "The office wifi password is indigo", "unsupported")

    preferred = ConflictReferee().resolve((supported, weak))[0]
    stalled = ConflictReferee().resolve((empty, other))[0]

    assert preferred.decision == PREFER_SUPPORTED
    assert preferred.kept_claims == (supported.claim,)
    assert preferred.dropped_claims == (weak.claim,)
    assert stalled.decision == UNRESOLVED
    assert stalled.kept_claims == ()


def test_coordinate_endpoint_resolves_submitted_findings(client: TestClient) -> None:
    response = client.post(
        "/api/research/coordinate",
        json={
            "question": "what is the retry limit",
            "findings": [
                {
                    "agent_id": "searcher:a",
                    "role": "searcher",
                    "claim": "The retry limit is three attempts.",
                    "verdict": "supported",
                    "source": "retry-limit-three.md",
                    "values": [3],
                },
                {
                    "agent_id": "searcher:b",
                    "role": "searcher",
                    "claim": "The retry limit is five attempts.",
                    "verdict": "supported",
                    "source": "retry-limit-five.md",
                    "values": [5],
                },
            ],
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert {item["role"] for item in body["assignments"]} >= {"searcher", "verifier", "referee", "synthesizer"}
    assert body["resolutions"][0]["decision"] == CITE_BOTH
    assert body["resolved"] is True
    assert len(body["consensus"]) == 2


def test_research_report_includes_cite_both_coordination(client: TestClient) -> None:
    for name in ("retry-limit-three.md", "retry-limit-five.md"):
        payload = (ROOT / "evaluation" / "fixtures" / name).read_bytes()
        client.post("/api/documents", files={"file": (name, payload, "text/markdown")})
    body = client.post("/api/research", json={"question": "what is the retry limit", "top_k": 4}).json()
    coordination = body["coordination"]

    assert body["conflict"] is True
    assert coordination is not None
    assert coordination["resolved"] is True
    assert any(item["decision"] == CITE_BOTH for item in coordination["resolutions"])
    assert {item["role"] for item in coordination["assignments"]} >= {"searcher", "referee", "synthesizer"}


def test_document_conflict_object_is_cited_on_both_sides() -> None:
    conflict = Contradiction(
        summary="retry-limit-three.md states 3; retry-limit-five.md states 5",
        left_filename="retry-limit-three.md",
        right_filename="retry-limit-five.md",
        left_values=(3,),
        right_values=(5,),
    )
    left = AgentFinding(
        "searcher:1",
        "searcher",
        "The retry limit is 3.",
        "supported",
        source="retry-limit-three.md",
        values=(3,),
    )
    right = AgentFinding(
        "searcher:2",
        "searcher",
        "The retry limit is 5.",
        "supported",
        source="retry-limit-five.md",
        values=(5,),
    )
    report = MultiAgentCoordinator().coordinate(
        "what is the retry limit",
        findings=(left, right),
        conflicts=(conflict,),
    )

    assert report.resolutions[0].decision == CITE_BOTH
    assert report.resolved is True
    assert left.claim in report.consensus
    assert right.claim in report.consensus
