from pathlib import Path

from fastapi.testclient import TestClient

from app.security.prompts import REFUSAL_TEXT
from app.wiring import Services
from evaluation.dataset import load_answer_cases

ROOT = Path(__file__).resolve().parents[1]


def _index_samples(services: Services) -> None:
    for path in sorted((ROOT / "samples").glob("*.md")):
        services.ingestion.ingest(path.name, path.read_bytes())


def test_retrieve_reports_ranked_sources_and_missing_evidence(services: Services) -> None:
    _index_samples(services)

    found = services.search.retrieve("which file implements the retry logic", top_k=3)
    missing = services.search.retrieve("what is the office wifi password", top_k=3)

    assert found.sufficient_evidence is True
    assert found.passages[0].source.filename == "api-errors.md"
    assert found.passages[0].source.heading == "Retry logic"
    assert found.passages[0].source.document_id
    assert found.passages[0].answers_question is True
    assert "app/http/retry.py" in found.passages[0].text
    assert "file" in found.passages[0].matched_terms
    assert missing.sufficient_evidence is False
    assert missing.passages
    assert all(not passage.answers_question for passage in missing.passages)


def test_answer_dataset(services: Services) -> None:
    _index_samples(services)
    cases = load_answer_cases(ROOT / "evaluation" / "datasets" / "phase3_answers.json")

    assert cases
    for case in cases:
        result = services.generation.answer(case.question, top_k=4)
        assert result.sufficient_evidence is case.expect_evidence, case.id
        if case.expect_evidence:
            assert case.expected_substring in result.answer
            assert result.citations
            assert result.citations[0].filename == case.document
            assert result.citations[0].marker in result.answer
        else:
            assert result.answer == REFUSAL_TEXT
            assert result.citations == []


def test_answer_api(client: TestClient) -> None:
    payload = (ROOT / "samples" / "api-errors.md").read_bytes()
    client.post("/api/documents", files={"file": ("api-errors.md", payload, "text/markdown")})

    found = client.post(
        "/api/retrieve",
        json={"query": "which file implements the retry logic", "top_k": 3},
    )
    answered = client.post(
        "/api/answer",
        json={"question": "which file implements the retry logic", "top_k": 3},
    )
    refused = client.post(
        "/api/answer",
        json={"question": "what is the office wifi password"},
    )

    assert found.status_code == 200
    body = found.json()
    assert body["sufficient_evidence"] is True
    assert body["passages"][0]["source"]["filename"] == "api-errors.md"
    assert body["passages"][0]["source"]["heading"] == "Retry logic"
    assert answered.status_code == 200
    assert "app/http/retry.py" in answered.json()["answer"]
    assert answered.json()["citations"][0]["filename"] == "api-errors.md"
    assert refused.status_code == 200
    assert refused.json()["answer"] == REFUSAL_TEXT
    assert refused.json()["sufficient_evidence"] is False
    assert refused.json()["citations"] == []
