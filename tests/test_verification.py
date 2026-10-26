"""Claim verification scores a citation, and contradictions name both sources."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.research.verification import assess_claim, find_contradictions
from app.retrieval.search import RankedPassage, SourceMetadata

ROOT = Path(__file__).resolve().parents[1]
THREE = "The retry limit is three attempts."
FIVE = "The retry limit is five attempts."


def test_claim_verification_scores_support() -> None:
    supported = assess_claim(THREE, _passage(THREE, filename="retry-limit-three.md"))
    unsupported = assess_claim(
        "The office wifi password is violet",
        _passage(THREE, filename="retry-limit-three.md"),
    )
    unverifiable = assess_claim(
        "The documents disagree.",
        _passage(THREE, filename="retry-limit-three.md"),
    )

    assert supported.verdict == "supported"
    assert supported.support == 1.0
    assert supported.values == (3,)
    assert unsupported.verdict == "unsupported"
    assert unsupported.support is not None and unsupported.support < 0.6
    assert unverifiable.verdict == "unverifiable"
    assert unverifiable.support is None


def test_contradictions_require_different_files_and_numbers() -> None:
    three = assess_claim(THREE, _passage(THREE, filename="retry-limit-three.md"))
    five = assess_claim(FIVE, _passage(FIVE, filename="retry-limit-five.md"))
    same_file = assess_claim(FIVE, _passage(FIVE, filename="retry-limit-three.md"))
    same_number = assess_claim(THREE, _passage(THREE, filename="retry-limit-five.md"))

    conflict = find_contradictions("what is the retry limit", [three, five])
    assert len(conflict) == 1
    assert {conflict[0].left_filename, conflict[0].right_filename} == {
        "retry-limit-three.md",
        "retry-limit-five.md",
    }
    assert set(conflict[0].left_values + conflict[0].right_values) == {3, 5}
    assert find_contradictions("what is the retry limit", [three, same_file]) == []
    assert find_contradictions("what is the retry limit", [three, same_number]) == []


def test_research_reports_verified_claims_and_the_conflict(client: TestClient) -> None:
    for name in ("retry-limit-three.md", "retry-limit-five.md"):
        payload = (ROOT / "evaluation" / "fixtures" / name).read_bytes()
        uploaded = client.post(
            "/api/documents",
            files={"file": (name, payload, "text/markdown")},
        )
        assert uploaded.status_code == 201

    body = client.post(
        "/api/research",
        json={"question": "what is the retry limit", "top_k": 4},
    ).json()
    conflict = body["conflicts"][0]

    assert body["subquestions"][0]["status"] == "contradicted"
    assert {conflict["left_filename"], conflict["right_filename"]} == {
        "retry-limit-three.md",
        "retry-limit-five.md",
    }
    assert set(conflict["left_values"] + conflict["right_values"]) == {3, 5}
    assert conflict["left_claim_id"]
    assert conflict["right_claim_id"]
    assert {record["verdict"] for record in body["evidence"]} == {"supported"}
    assert all(record["support"] >= 0.6 for record in body["evidence"])
    assert "retry-limit-three.md" in conflict["summary"]
    assert "retry-limit-five.md" in conflict["summary"]


def _passage(text: str, *, filename: str) -> RankedPassage:
    return RankedPassage(
        rank=1,
        text=text,
        score=0.9,
        vector_score=0.9,
        lexical_score=0.9,
        answers_question=True,
        matched_terms=("retry", "limit"),
        source=SourceMetadata(
            chunk_id=filename,
            document_id=filename,
            filename=filename,
            content_type="text/markdown",
            chunk_index=0,
            start_char=0,
            end_char=len(text),
            page=None,
            heading="Retry policy",
        ),
    )
