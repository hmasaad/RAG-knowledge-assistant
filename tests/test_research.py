"""Research decomposes a question, checks claims, and reports gaps."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.generation.agent import CLARIFICATION_TEXT
from app.ingestion.embeddings import LocalEmbedder
from app.research.planner import select_tools, subquestions
from app.wiring import build_services

ROOT = Path(__file__).resolve().parents[1]


def test_planner_splits_questions_and_picks_tools() -> None:
    question = "which file implements the retry logic and where are timeouts configured"

    assert subquestions(question) == (
        "which file implements the retry logic",
        "where are timeouts configured",
    )
    assert select_tools(question) == ("knowledge",)
    assert select_tools("which documents mention retry") == ("knowledge", "catalog")
    assert select_tools("what does app/http/retry.py call") == ("knowledge", "graph")
    assert subquestions("please tell me something about app/http/retry.py now") == (
        "please tell me something about app/http/retry.py now",
    )


def test_research_answers_admin_portal_requirements(client: TestClient) -> None:
    payload = (
        b"# Roles\n\n"
        b"Platform Admin Manage salons, users, moderation, payments, and platform analytics.\n"
        b"Web Admin Configure salon listings and support tools.\n"
    )
    uploaded = client.post(
        "/api/documents",
        files={"file": ("salon-admin.md", payload, "text/markdown")},
    )
    response = client.post(
        "/api/research",
        json={"question": "What are admin portal requirements?", "top_k": 4},
    )
    body = response.json()

    assert uploaded.status_code == 201
    assert response.status_code == 200
    assert body["action"] == "research"
    assert "I don't have enough evidence" not in body["report"]
    assert "Admin" in body["report"] or "admin" in body["report"].lower()


def test_research_reports_each_supported_subquestion(client: TestClient) -> None:
    payload = (ROOT / "samples" / "api-errors.md").read_bytes()
    uploaded = client.post(
        "/api/documents",
        files={"file": ("api-errors.md", payload, "text/markdown")},
    )
    response = client.post(
        "/api/research",
        json={
            "question": "which file implements the retry logic and where are timeouts configured",
            "top_k": 3,
        },
    )
    again = client.post(
        "/api/research",
        json={
            "question": "which file implements the retry logic and where are timeouts configured",
            "top_k": 3,
        },
    )
    body = response.json()
    trace = client.get(f"/api/traces/{body['trace_id']}")

    assert uploaded.status_code == 201
    assert response.status_code == 200
    assert body["action"] == "research"
    assert body["sufficient_evidence"] is True
    assert body["conflict"] is False
    assert body["tools"] == ["knowledge"]
    assert body["gaps"] == []
    assert {step["status"] for step in body["subquestions"]} == {"supported"}
    assert "app/http/retry.py" in body["report"]
    assert "app/http/client.py" in body["report"]
    assert len(body["citations"]) >= 2
    assert again.json()["trace_id"] == body["trace_id"]
    actions = [step["action"] for step in trace.json()["steps"]]
    assert actions[0] == "plan"
    assert actions.count("search") == 2
    assert "verify" in actions
    assert actions[-1] == "research"


def test_research_investigates_a_contradiction(client: TestClient) -> None:
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

    assert body["action"] == "research"
    assert body["conflict"] is True
    assert body["subquestions"][0]["status"] == "contradicted"
    assert "three" in body["report"]
    assert "five" in body["report"]
    assert "retry-limit-three.md" in body["contradictions"][0]
    assert "retry-limit-five.md" in body["contradictions"][0]
    assert "disagree" in body["report"].lower()


def test_research_records_an_evidence_gap_and_retries(client: TestClient) -> None:
    payload = (ROOT / "samples" / "api-errors.md").read_bytes()
    client.post(
        "/api/documents",
        files={"file": ("api-errors.md", payload, "text/markdown")},
    )
    vague = client.post("/api/research", json={"question": "help", "top_k": 3}).json()
    partial = client.post(
        "/api/research",
        json={
            "question": "which file implements the retry logic and what is the office wifi password",
            "top_k": 3,
        },
    ).json()
    retried = client.post(
        "/api/research",
        json={"question": "please tell me something about app/http/retry.py now", "top_k": 3},
    ).json()
    retry_trace = client.get(f"/api/traces/{retried['trace_id']}").json()

    assert vague["action"] == "clarify"
    assert vague["report"] == CLARIFICATION_TEXT
    assert vague["subquestions"] == []
    assert partial["sufficient_evidence"] is False
    assert partial["action"] == "research"
    assert "app/http/retry.py" in partial["report"]
    assert any("wifi" in gap for gap in partial["gaps"])
    assert "Evidence gaps:" in partial["report"]
    assert retried["action"] == "research"
    assert "app/http/retry.py" in retried["report"]
    assert any(step["action"] in {"retry", "knowledge", "search"} for step in retry_trace["steps"])


def test_research_uses_the_catalog_and_hides_other_owners(client: TestClient) -> None:
    client.post(
        "/api/documents",
        files={"file": ("api-errors.md", (ROOT / "samples" / "api-errors.md").read_bytes(), "text/markdown")},
        headers={"X-User-Id": "ada"},
    )
    client.post(
        "/api/documents",
        files={
            "file": (
                "secret.md",
                b"The canary: violet-heron-1942 must stay private.\n",
                "text/markdown",
            )
        },
        headers={"X-User-Id": "grace"},
    )
    body = client.post(
        "/api/research",
        json={"question": "which documents are indexed", "top_k": 3},
        headers={"X-User-Id": "ada"},
    ).json()

    assert body["tools"] == ["knowledge", "catalog"]
    assert body["catalog"] == ["api-errors.md"]
    assert "violet-heron-1942" not in body["report"]
    assert "Visible documents: api-errors.md." in body["report"]


def test_research_follows_a_graph_neighbor(tmp_path: Path) -> None:
    services = build_services(
        Settings(
            embedding_provider="local",
            embedding_dimension=128,
            chroma_path=tmp_path / "chroma",
            sqlite_path=tmp_path / "metadata.db",
            blob_path=tmp_path / "blobs",
            chunk_size=500,
            chunk_overlap=40,
            max_upload_bytes=100_000,
            log_level="WARNING",
            llm_provider="local",
            knowledge_graph=True,
        ),
        embedder=LocalEmbedder(dimension=128),
    )
    services.ingestion.ingest(
        "calls.md",
        b"alpha/retry.py calls beta/client.py before it returns.\n",
    )

    report = services.research.research("what does alpha/retry.py call", top_k=3)

    assert report.tools == ("knowledge", "graph")
    assert any(step.tool == "graph" and step.status == "supported" for step in report.subquestions)
    assert "beta/client.py" in report.report
