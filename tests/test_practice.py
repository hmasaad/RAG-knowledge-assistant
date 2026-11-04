"""Repository graph, test-driven repair, review votes, and governance."""

from app.engineering.delivery import deliver, repair_module


GREETING = """Add a function that greets a person by name.
- The greeting includes the name.
- An empty name is rejected.
"""


def _notes(requirement: str) -> dict[str, dict]:
    report = deliver(requirement)
    return report, {note["component"]: note for note in report["practice"]["notes"]}


def test_a_greeting_is_repaired_from_a_failing_test_and_held_for_a_person() -> None:
    report, notes = _notes(GREETING)

    assert list(notes) == [
        "repository",
        "graph",
        "agents",
        "specification",
        "planning",
        "ast",
        "tdd",
        "healing",
        "review",
        "policy",
        "debug",
        "governance",
    ]
    assert "greet" in notes["repository"]["reason"]
    assert "calls greet" in notes["graph"]["reason"]
    assert notes["specification"]["decision"] == "generated"
    assert notes["ast"]["decision"] == "parsed"
    assert "1 raise" in notes["ast"]["reason"]
    assert notes["tdd"]["decision"] == "red_then_green"
    assert notes["healing"]["decision"] == "healed"
    assert notes["review"]["decision"] == "accept"
    assert notes["policy"]["decision"] == "enforced"
    assert notes["debug"]["decision"] == "clear"
    assert notes["governance"]["decision"] == "hold"
    assert report["pull_request"]["opened"] is False
    assert report["pull_request"]["ready"] is True
    assert report["tests"]["status"] == "passed"
    assert "ValueError" in report["files"][0]["content"]


def test_a_specification_is_stable_and_stays_held() -> None:
    report, notes = _notes("Document the billing policy for overdue invoices.")

    assert notes["specification"]["decision"] == "recorded"
    assert notes["healing"]["decision"] == "stable"
    assert notes["tdd"]["decision"] == "red_then_green"
    assert notes["governance"]["decision"] == "hold"
    assert report["pull_request"]["ready"] is False
    assert report["pull_request"]["opened"] is False


def test_an_ast_edit_inserts_the_missing_raise() -> None:
    source = "def greet(name: str) -> str:\n    cleaned = ' '.join(str(name).split())\n    return cleaned\n"
    repaired = repair_module(source, "Failed: DID NOT RAISE <class 'ValueError'>")

    assert repaired is not None
    assert "raise ValueError" in repaired
    compile(repaired, "change.py", "exec")


def test_a_banned_call_blocks_the_draft() -> None:
    from app.engineering.practice import observe_practice

    practice = observe_practice(
        understood={"goal": "Run a command", "checks": ["Run it"], "behaviors": [{"kind": "include_argument"}], "function": "greet"},
        files=[
            {"path": "change.py", "content": "import os\n\ndef greet(name: str) -> str:\n    os.system(name)\n    return name\n"},
            {"path": "test_change.py", "content": "from change import greet\n\ndef test_greet():\n    greet('Ada')\n"},
        ],
        red={"status": "failed", "passed": 0, "summary": "0 passed, 1 failed.", "output": ""},
        tests={"status": "passed", "passed": 1, "summary": "1 passed, 0 failed.", "output": ""},
        healed=False,
        review={"decision": "approve", "reason": "ok"},
        pull={"ready": True, "opened": False, "head": "delivery/greet", "title": "Run a command"},
    )
    notes = {note["component"]: note for note in practice["notes"]}

    assert notes["policy"]["decision"] == "blocked"
    assert "os.system" in notes["policy"]["reason"]
    assert notes["review"]["decision"] == "dissent"
    assert practice["ready"] is False
    assert notes["governance"]["decision"] == "hold"
