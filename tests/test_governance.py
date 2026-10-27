"""Production readiness, operator-approved releases, and rollback."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.engineering.governance import GovernancePolicy, ProductionGovernance
from app.engineering.service import EngineeringService
from app.ingestion.embeddings import LocalEmbedder
from app.ingestion.errors import KnowledgeError
from app.main import create_app

FLOORS = {
    "correctness": 1.0,
    "safety": 1.0,
    "budget": 1.0,
    "recovery": 1.0,
}


def _settings(tmp_path: Path, **overrides) -> Settings:
    values = dict(
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
    )
    values.update(overrides)
    return Settings(**values)


def test_readiness_is_not_production_until_a_release() -> None:
    board = ProductionGovernance()
    report = board.readiness()

    assert report.ready is True
    assert report.production_ready is False
    assert report.serving == ""
    names = {item.name: item.passed for item in report.checks}
    assert names["metadata"] is True
    assert names["vectors"] is True
    assert names["guardrails"] is True
    assert names["loop_floors"] is False


def test_release_blocks_a_score_below_the_floor() -> None:
    service = EngineeringService()
    decision = service.release("planner-b", {**FLOORS, "safety": 0.0}, actor="ada")

    assert decision.allowed is False
    assert decision.promoted is False
    assert "loop_floors" in decision.regressions
    assert service.loop_state().baseline is None
    assert service.audit()[0].allowed is False


def test_release_and_rollback_keep_an_audit_trail() -> None:
    service = EngineeringService()
    first = service.release("planner-a", FLOORS, actor="ada")
    second = service.release("planner-b", FLOORS, actor="ada")
    restored = service.rollback(actor="ada")

    assert first.allowed is True
    assert second.serving == "planner-b"
    assert restored.allowed is True
    assert restored.serving == "planner-a"
    assert service.readiness().production_ready is True
    assert service.readiness().serving == "planner-a"
    actions = [item.action for item in service.audit()]
    assert actions == ["release", "release", "rollback"]


def test_hold_pauses_releases() -> None:
    service = EngineeringService(policy=GovernancePolicy(hold=True))
    report = service.readiness(FLOORS)
    decision = service.release("planner-a", FLOORS, actor="ada")

    assert report.hold is True
    assert report.production_ready is False
    assert decision.allowed is False
    assert "hold" in decision.regressions


def test_unknown_operator_cannot_release(tmp_path: Path) -> None:
    settings = _settings(tmp_path, governance_operators="ada")
    application = create_app(settings, embedder=LocalEmbedder(dimension=128))
    with TestClient(application) as client:
        denied = client.post(
            "/api/engineering/release",
            json={"planner_version": "planner-a", "metrics": FLOORS},
            headers={"X-User-Id": "grace"},
        )
        released = client.post(
            "/api/engineering/release",
            json={"planner_version": "planner-a", "metrics": FLOORS},
            headers={"X-User-Id": "ada"},
        )
        readiness = client.get("/api/engineering/readiness")
        audit = client.get("/api/engineering/audit")

    assert denied.status_code == 403
    assert released.status_code == 200
    assert released.json()["allowed"] is True
    assert released.json()["serving"] == "planner-a"
    assert readiness.json()["production_ready"] is True
    assert audit.json()[0]["actor"] == "ada"
    assert audit.json()[0]["action"] == "release"


def test_empty_operator_is_rejected() -> None:
    board = ProductionGovernance(policy=GovernancePolicy(require_approval=True))
    try:
        board.release("planner-a", FLOORS, actor=" ")
    except KnowledgeError as error:
        assert error.status_code == 403
        assert error.message == "Caller is not permitted."
    else:
        raise AssertionError("expected a forbidden operator")
