"""Repository intelligence reads a tree before any code is generated."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.engineering.repository import inspect_repository


def _fixture(tmp_path: Path) -> Path:
    package = tmp_path / "pkg"
    (package / "api").mkdir(parents=True)
    (package / "web").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / ".venv" / "pkg").mkdir(parents=True)
    (package / "service.py").write_text(
        '"""Greeting service."""\n\n\ndef greet(name: str) -> str:\n    return f"Hello {name}"\n',
        encoding="utf-8",
    )
    (package / "contracts.py").write_text(
        '"""Contracts."""\nfrom typing import Protocol\n\n\nclass GreeterProtocol(Protocol):\n'
        '    def greet(self, name: str) -> str:\n        """Greet a person."""\n        ...\n',
        encoding="utf-8",
    )
    (package / "api" / "routes.py").write_text(
        '"""HTTP routes."""\nfrom fastapi import APIRouter\nfrom ..service import greet\n\n'
        "router = APIRouter()\n\n\nclass Greeter:\n"
        "    def greet(self, name: str) -> str:\n        return greet(name)\n",
        encoding="utf-8",
    )
    (package / "web" / "page.html").write_text("<p>Greeting</p>\n", encoding="utf-8")
    (tmp_path / "tests" / "test_service.py").write_text(
        '"""Service checks."""\nfrom pkg.service import greet\n\n\n'
        'def test_greet() -> None:\n    assert greet("Ada") == "Hello Ada"\n',
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        '[project]\ndependencies = ["fastapi>=0.115.0", "pydantic>=2"]\n',
        encoding="utf-8",
    )
    (tmp_path / ".venv" / "pkg" / "hidden.py").write_text(
        "def hidden():\n    return 1\n",
        encoding="utf-8",
    )
    return tmp_path


def test_a_repository_is_scanned_before_code_is_written(tmp_path: Path) -> None:
    report = inspect_repository(_fixture(tmp_path), "the greeting service")
    notes = {note["component"]: note for note in report["notes"]}

    assert [note["component"] for note in report["notes"]] == [
        "structure",
        "stack",
        "symbols",
        "dependencies",
        "layers",
        "files",
        "patterns",
        "report",
    ]
    assert notes["structure"]["decision"] == "scanned"
    assert "pkg (4 files)" in notes["structure"]["reason"]
    assert ".venv" not in notes["structure"]["reason"]
    assert notes["stack"]["decision"] == "identified"
    assert "Python (4)" in notes["stack"]["reason"]
    assert "FastAPI" in notes["stack"]["reason"]
    assert "fastapi" in notes["stack"]["reason"]
    assert report["symbols"] == {"classes": 1, "functions": 4, "interfaces": 1, "imports": 4}
    assert notes["symbols"]["reason"] == (
        "Parsed 1 class, 4 functions, 1 interface, and 4 imports."
    )
    assert {"source": "pkg.api.routes", "target": "pkg.service"} in report["dependencies"]
    assert "pkg.service is imported by 2 modules" in notes["dependencies"]["reason"]
    assert notes["layers"]["decision"] == "layered"
    assert "HTTP routes" in next(layer["role"] for layer in report["layers"] if layer["name"] == "api")
    assert report["relevant"][0]["path"] == "pkg/service.py"
    assert report["relevant"][0]["reason"] == "Matches greet, service."
    assert notes["files"]["decision"] == "matched"
    assert "APIRouter" in notes["patterns"]["reason"]
    assert "test_*" in notes["patterns"]["reason"]
    assert notes["report"]["decision"] == "written"
    assert "hidden" not in notes["symbols"]["reason"]


def test_a_missing_requirement_does_not_choose_files(tmp_path: Path) -> None:
    report = inspect_repository(_fixture(tmp_path), "   ")
    files = next(note for note in report["notes"] if note["component"] == "files")

    assert files["decision"] == "open"
    assert report["relevant"] == []


def test_an_unrecognized_requirement_selects_nothing(tmp_path: Path) -> None:
    report = inspect_repository(_fixture(tmp_path), "Explain the warehouse roster.")
    files = next(note for note in report["notes"] if note["component"] == "files")

    assert files["decision"] == "unmatched"
    assert report["relevant"] == []


def test_this_repository_reports_its_delivery_module() -> None:
    report = inspect_repository(requirement="Find the delivery function that prepares a pull request.")

    assert "Python" in report["languages"]
    assert "FastAPI" in report["frameworks"]
    assert "fastapi" in report["packages"]
    assert any(layer["name"] == "engineering" for layer in report["layers"])
    assert {"source": "app.api.engineering", "target": "app.engineering.delivery"} in report["dependencies"]
    assert any(item["path"] == "app/engineering/delivery.py" for item in report["relevant"])
    assert report["symbols"]["functions"] > 10
    assert report["notes"][-1]["component"] == "report"


def test_repository_api_returns_the_architecture_report(client: TestClient) -> None:
    response = client.post(
        "/api/engineering/repository",
        json={"requirement": "Find the delivery function that prepares a pull request."},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["notes"][0]["component"] == "structure"
    assert "FastAPI" in body["frameworks"]
    assert any(item["path"] == "app/engineering/delivery.py" for item in body["relevant"])
    assert body["dependency_count"] == len(body["dependencies"])
