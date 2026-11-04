"""Delivery pipeline: requirement, code, tests, review, and a pull request draft."""

from fastapi.testclient import TestClient

from app.engineering.delivery import deliver


GREETING = """Add a function that greets a person by name.
- The greeting includes the name.
- An empty name is rejected.
"""


def test_a_greeting_requirement_is_implemented_tested_and_drafted() -> None:
    report = deliver(GREETING)
    source = next(item["content"] for item in report["files"] if item["path"] == "change.py")

    assert "def greet(" in source
    assert report["tests"]["status"] == "passed"
    assert report["tests"]["passed"] >= 2
    assert report["review"]["decision"] == "approve"
    assert report["pull_request"]["ready"] is True
    assert report["pull_request"]["opened"] is False
    assert report["pull_request"]["head"] == "delivery/greet"
    assert [stage["stage"] for stage in report["stages"]] == [
        "understand",
        "plan",
        "code",
        "test",
        "review",
        "pull_request",
    ]
    assert [note["component"] for note in report["repository"]["notes"]] == [
        "structure",
        "stack",
        "symbols",
        "dependencies",
        "layers",
        "files",
        "patterns",
        "report",
    ]
    assert "tests/test_delivery.py" in report["stages"][1]["detail"]
    assert any(item["path"] == "tests/test_delivery.py" for item in report["repository"]["relevant"])


def test_a_requirement_without_a_behavior_is_recorded_and_held() -> None:
    report = deliver("Document the billing policy for overdue invoices.")

    assert report["review"]["decision"] == "recorded"
    assert report["tests"]["status"] == "passed"
    assert report["pull_request"]["ready"] is False
    assert report["pull_request"]["opened"] is False
    assert "billing policy" in report["files"][0]["content"]


def test_requirement_text_is_stored_as_data() -> None:
    report = deliver('Document this note: "); import os; os.system("echo unsafe")')
    source = report["files"][0]["content"]

    compile(source, "change.py", "exec")
    assert report["tests"]["status"] == "passed"
    assert "os.system" not in source.split("REQUIREMENT = ", 1)[0]


def test_blank_requirement_is_rejected(client: TestClient) -> None:
    response = client.post("/api/engineering/deliver", json={"requirement": "   "})

    assert response.status_code == 400


def test_delivery_api_returns_the_draft(client: TestClient) -> None:
    response = client.post("/api/engineering/deliver", json={"requirement": GREETING})
    body = response.json()

    assert response.status_code == 200
    assert body["review"]["decision"] == "approve"
    assert body["pull_request"]["opened"] is False
    assert body["tests"]["passed"] >= 2
