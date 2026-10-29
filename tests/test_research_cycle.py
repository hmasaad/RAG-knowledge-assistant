"""The research cycle retrieves, verifies, fills gaps, and cites an answer."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.research.contracts import (
    FILL_GAPS,
    REVERIFY,
    SEARCH,
    SYNTHESIZE,
    ActionOutcome,
    ActionRequest,
    ClaimVerdict,
    EvidenceRef,
    ResearchState,
)
from app.research.adapters import cited_answer
from app.research.cycle import run_cycle
from app.research.dispatcher import ActionDispatcher

ROOT = Path(__file__).resolve().parents[1]


class _CycleAdapter:
    def __init__(self, action: str) -> None:
        self.action = action
        self.calls: list[ActionRequest] = []

    def run(self, request: ActionRequest, state: ResearchState | None = None) -> ActionOutcome:
        self.calls.append(request)
        if self.action == SEARCH:
            return ActionOutcome(
                action=SEARCH,
                ok=True,
                evidence=(EvidenceRef("e1", "d1", "api-errors.md", "retry logic is app/http/retry.py"),),
                provenance=("e1",),
                cost=1,
            )
        if self.action == REVERIFY:
            supported = "wifi" not in request.claim
            return ActionOutcome(
                action=REVERIFY,
                ok=True,
                verdicts=(
                    ClaimVerdict(
                        claim=request.claim,
                        verdict="supported" if supported else "insufficient",
                        evidence_ids=("e1",) if supported else (),
                    ),
                ),
                gaps=() if supported else (request.claim,),
                cost=0,
            )
        if self.action == FILL_GAPS:
            return ActionOutcome(action=FILL_GAPS, ok=True, gaps=(request.query,), cost=1)
        return ActionOutcome(action=SYNTHESIZE, ok=True, report=request.query, cost=0)


def test_run_cycle_searches_verifies_fills_and_synthesizes() -> None:
    adapters = {name: _CycleAdapter(name) for name in (SEARCH, REVERIFY, FILL_GAPS, SYNTHESIZE)}
    dispatcher = ActionDispatcher(list(adapters.values()), budget=16, max_iterations=8)

    result = run_cycle(
        dispatcher,
        "which file implements the retry logic and what is the office wifi password",
    )

    assert list(result.steps) == [
        SEARCH,
        SEARCH,
        REVERIFY,
        REVERIFY,
        FILL_GAPS,
        REVERIFY,
        SYNTHESIZE,
    ]
    assert any("wifi" in gap for gap in result.gaps)
    assert "[1]" in result.report
    assert "retry.py" in result.report


def test_research_cycle_cites_supported_evidence(client: TestClient) -> None:
    payload = (ROOT / "samples" / "api-errors.md").read_bytes()
    uploaded = client.post(
        "/api/documents",
        files={"file": ("api-errors.md", payload, "text/markdown")},
    )
    response = client.post(
        "/api/research/cycle",
        json={"question": "which file implements the retry logic", "top_k": 3},
    )
    planned = client.post(
        "/api/research",
        json={
            "question": "which file implements the retry logic and where are timeouts configured",
            "top_k": 3,
        },
    )
    body = response.json()
    planned_body = planned.json()
    planned_trace = client.get(f"/api/traces/{planned_body['trace_id']}")

    assert uploaded.status_code == 201
    assert response.status_code == 200
    assert SEARCH in body["decisions"]
    assert REVERIFY in body["decisions"]
    assert SYNTHESIZE in body["decisions"]
    assert "[1]" in body["report"]
    assert "retry.py" in body["report"]
    assert body["action"] == "research"
    assert [step["action"] for step in planned_trace.json()["steps"]].count("search") == 2


def test_research_cycle_fills_gaps_then_synthesizes(client: TestClient) -> None:
    payload = (ROOT / "samples" / "api-errors.md").read_bytes()
    uploaded = client.post(
        "/api/documents",
        files={"file": ("api-errors.md", payload, "text/markdown")},
    )
    response = client.post(
        "/api/research/cycle",
        json={
            "question": "which file implements the retry logic and what is the office wifi password",
            "top_k": 3,
        },
    )
    body = response.json()

    assert uploaded.status_code == 201
    assert response.status_code == 200
    assert FILL_GAPS in body["decisions"]
    assert SYNTHESIZE in body["decisions"]
    assert any("wifi" in gap for gap in body["gaps"])
    assert "retry.py" in body["report"]


def test_research_cycle_answers_a_section_question(client: TestClient) -> None:
    payload = (
        b"# Customer App Requirements\n\n"
        b"## 4.1 Authentication & Profile\n\n"
        b"Sign up and sign in with email or phone. Edit name, photo, and saved salons.\n"
    )
    uploaded = client.post(
        "/api/documents",
        files={"file": ("salon-requirements.md", payload, "text/markdown")},
    )
    response = client.post(
        "/api/research/cycle",
        json={
            "question": "Can you breakdown Authentication & Profile so that i can understand it properly?",
            "top_k": 4,
        },
    )
    body = response.json()

    assert uploaded.status_code == 201
    assert response.status_code == 200
    assert body["action"] == "research"
    assert "Sign up and sign in" in body["report"]
    assert body["report"] != "I don't have enough evidence to answer."
    assert not any("Authentication" in gap for gap in body["gaps"])


def test_research_cycle_answers_authentication_flow(client: TestClient) -> None:
    payload = (
        b"## 4.1 Authentication & Profile\n\n"
        b"Sign up and sign in using email/phone and password or OTP. "
        b"Optional social login. Password reset and account recovery.\n\n"
        b"## Non-Functional Requirements\n\n"
        b"Observability. Scalability Backend should support growth in salons.\n\n"
        b"## Key User Flow\n\n"
        b"Customer: Open App then Search then Select Salon.\n"
    )
    uploaded = client.post(
        "/api/documents",
        files={"file": ("auth.md", payload, "text/markdown")},
    )
    response = client.post(
        "/api/research/cycle",
        json={"question": "authentication flow", "top_k": 4},
    )
    body = response.json()

    assert uploaded.status_code == 201
    assert body["action"] == "research"
    assert "Sign up" in body["report"] or "sign in" in body["report"].lower()
    assert body["report"] != "I don't have enough evidence to answer."
    assert not any("authentication" in gap.casefold() for gap in body["gaps"])


def test_research_cycle_answers_mvp_scope(client: TestClient) -> None:
    payload = (
        b"10. Notifications Push booking notifications. Payment details and invoices.\n"
        b"11. MVP Scope\n"
        b"1. Customer registration/login and profile.\n"
        b"2. Salon discovery and search.\n"
        b"3. Salon details and service catalog.\n"
        b"4. Stylist selection.\n"
        b"5. Availability and appointment booking.\n"
        b"6. Booking history and cancellation/rescheduling.\n"
        b"7. Push booking notifications.\n"
        b"8. Basic salon dashboard and calendar.\n"
        b"9. Service and staff management.\n"
        b"10. Basic admin management.\n"
    )
    uploaded = client.post(
        "/api/documents",
        files={"file": ("mvp.txt", payload, "text/plain")},
    )
    response = client.post(
        "/api/research/cycle",
        json={"question": "what is MVP scope?", "top_k": 4},
    )
    body = response.json()

    assert uploaded.status_code == 201
    assert body["action"] == "research"
    report = body["report"]
    assert "registration" in report
    assert "Stylist selection" in report
    assert "Basic admin management" in report
    assert report.count("\n") >= 9
    assert report != "I don't have enough evidence to answer."
    assert not any("MVP" in gap for gap in body["gaps"])


def test_research_cycle_answers_admin_portal_requirements(client: TestClient) -> None:
    payload = (
        b"5. Admin Portal\n"
        b"1. Platform Admin Manage salons, users, moderation, payments, and analytics.\n"
        b"2. Web Admin Configure salon listings and support tools.\n"
        b"3. Salon owners can view bookings from the admin dashboard.\n"
    )
    uploaded = client.post(
        "/api/documents",
        files={"file": ("admin-portal.txt", payload, "text/plain")},
    )
    response = client.post(
        "/api/research/cycle",
        json={"question": "admin portal requirements", "top_k": 4},
    )
    body = response.json()

    assert uploaded.status_code == 201
    assert body["action"] == "research"
    assert "Admin" in body["report"] or "admin" in body["report"].lower()
    assert body["report"] != "I don't have enough evidence to answer."


def test_cited_answer_keeps_a_compact_on_topic_report() -> None:
    state = ResearchState(
        evidence=[
            EvidenceRef(
                "e1",
                "d1",
                "salon.pdf",
                "Salon Booking App — Requirements Specification Page 3 4.8 Notifications "
                "Push notifications for booking confirmation.",
            ),
            EvidenceRef(
                "e2",
                "d1",
                "salon.pdf",
                "4.3 Salon management portal Staff can manage appointments, services, "
                "and schedules from the management portal.",
            ),
        ],
        verdicts=[
            ClaimVerdict("Explain Salon management portal?", "supported", ("e1", "e2")),
        ],
    )

    report = cited_answer("Explain Salon management portal?", state)

    assert report.count("[") <= 8
    assert "management portal" in report
    assert "Notifications" not in report
    assert "Page 3" not in report
