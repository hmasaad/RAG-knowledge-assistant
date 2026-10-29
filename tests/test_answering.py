from pathlib import Path

from fastapi.testclient import TestClient

from app.generation.answerers import best_sentence, topic_excerpt
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

    overview = client.post(
        "/api/answer",
        json={"question": "what is this document about", "top_k": 3},
    )
    assert overview.status_code == 200
    assert overview.json()["sufficient_evidence"] is True
    assert overview.json()["citations"]
    assert overview.json()["citations"][0]["filename"] == "api-errors.md"
    assert overview.json()["answer"] != REFUSAL_TEXT


def test_best_sentence_skips_pdf_title_page() -> None:
    text = (
        "Salon Booking App — Requirements Specification Page 1 Salon Booking App "
        "Software Requirements Specification (SRS) Document Version 1.0 Date 25 August 2026 "
        "Status Initial Requirements Primary Platforms iOS, Android, Web Admin 1. "
        "Executive Summary The Salon Booking App is a customer-facing mobile application "
        "and salon-facing management platform that allows customers to discover salons."
    )

    chosen = best_sentence(text, ("overview", "requirements"))

    assert "customer-facing" in chosen
    assert "Page 1" not in chosen


def test_topic_excerpt_skips_repeated_pdf_headers() -> None:
    text = (
        "Salon Booking App — Requirements Specification Page 3 4.8 Notifications "
        "Push notifications for booking confirmation. "
        "4.3 Salon management portal Staff can manage appointments, services, "
        "schedules, and daily operations from the management portal."
    )

    chosen = topic_excerpt(text, ("salon", "management", "portal"))

    assert "management portal" in chosen
    assert "Notifications" not in chosen
    assert "Page 3" not in chosen


def test_topic_excerpt_prefers_admin_section_over_title_platforms() -> None:
    text = (
        "Salon Booking App Requirements Specification Page 1 Primary Platforms "
        "iOS, Android, Web Admin 1. Executive Summary The app is customer-facing. "
        "3. Roles Platform Admin Manage salons, users, moderation, payments, "
        "and platform analytics."
    )

    chosen = topic_excerpt(text, ("admin", "portal", "requirements"))

    assert "Platform Admin" in chosen
    assert "analytics" in chosen


def test_requirements_overview_quotes_the_indexed_file(client: TestClient) -> None:
    payload = (
        b"Salon Booking App - Requirements Specification\nPage 1\n"
        b"Document Version 1.0\nDate 25 August 2026\nStatus Initial Requirements\n"
        b"1. Executive Summary\n"
        b"The Salon Booking App is a customer-facing mobile application "
        b"that allows customers to discover salons and book services.\n"
    )
    uploaded = client.post(
        "/api/documents",
        files={"file": ("salon_booking_app_requirements.md", payload, "text/markdown")},
    )
    asked = client.post(
        "/api/answer",
        json={"question": "Give me overview of all the requirements?", "top_k": 4},
    )

    assert uploaded.status_code == 201
    assert asked.status_code == 200
    assert asked.json()["sufficient_evidence"] is True
    assert "customer-facing" in asked.json()["answer"]
    assert "Page 1" not in asked.json()["answer"]
    assert asked.json()["citations"][0]["filename"] == "salon_booking_app_requirements.md"


def test_section_breakdown_uses_topic_words(client: TestClient) -> None:
    payload = (
        b"# Customer App Requirements\n\n"
        b"## 4.1 Authentication & Profile\n\n"
        b"Sign up and sign in with email or phone. Edit name, photo, and saved salons.\n\n"
        b"## 4.2 Booking\n\nCustomers pick a salon, service, and time slot.\n"
    )
    uploaded = client.post(
        "/api/documents",
        files={"file": ("salon-requirements.md", payload, "text/markdown")},
    )
    asked = client.post(
        "/api/answer",
        json={
            "question": "Can you breakdown Authentication & Profile so that i can understand it properly?",
            "top_k": 4,
        },
    )

    assert uploaded.status_code == 201
    assert asked.status_code == 200
    assert asked.json()["sufficient_evidence"] is True
    assert "Sign up and sign in" in asked.json()["answer"]
    assert "saved salons" in asked.json()["answer"]
    assert "1." in asked.json()["answer"]
    assert "\n" in asked.json()["answer"]
    assert asked.json()["citations"]


def test_any_overlapping_topic_is_answered_from_the_file(client: TestClient) -> None:
    payload = (
        b"# Billing\n\n"
        b"Customers complete payment with a card after the appointment is confirmed.\n"
    )
    client.post(
        "/api/documents",
        files={"file": ("payments.md", payload, "text/markdown")},
    )
    asked = client.post(
        "/api/answer",
        json={"question": "How does the payment process work in simple words?", "top_k": 3},
    )
    missing = client.post(
        "/api/answer",
        json={"question": "what is the office wifi password"},
    )

    assert asked.json()["sufficient_evidence"] is True
    assert "card" in asked.json()["answer"] or "wallet" in asked.json()["answer"]
    assert missing.json()["sufficient_evidence"] is False
    assert missing.json()["answer"] == REFUSAL_TEXT
