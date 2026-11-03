"""The autonomous loop stops on evidence or a budget, and the dashboard reports it."""

from fastapi.testclient import TestClient

from app.research.intelligence import run_research_loop
from app.research.trace import trace_scene


def _grounded(scene: dict) -> dict:
    return {
        "asset": scene["asset"],
        "answer": scene["answer"],
        "markers": scene["markers"],
        "scene": scene,
    }


def _sufficient_scene() -> dict:
    scene = trace_scene("NVDA")
    scene["plan"] = {"summary": "Evidence is sufficient. No follow-up research.", "next_id": "", "items": []}
    scene["conflicts"] = [item for item in scene["conflicts"] if item["status"] != "pending"]
    return scene


def test_loop_stops_when_evidence_is_sufficient() -> None:
    scene = _sufficient_scene()

    def search(question: str, asset: str) -> dict:
        return _grounded(scene)

    report = run_research_loop(
        "What are people saying about NVIDIA?",
        search,
        max_searches=4,
        max_tokens=5000,
        max_seconds=5,
    )

    assert report["stop_reason"] == "sufficient"
    assert report["sufficient"] is True
    assert report["searches_used"] == 1
    assert report["steps"][-1]["detail"] == "Evidence is sufficient."
    assert 0 < report["metrics"]["confidence"] <= 1
    assert report["metrics"]["quality"] > 0


def test_loop_respects_search_time_and_token_budgets() -> None:
    def search(question: str, asset: str) -> dict:
        return _grounded(trace_scene("NVDA"))

    limited = run_research_loop(
        "What are people saying about NVIDIA?",
        search,
        max_searches=1,
        max_tokens=5000,
        max_seconds=5,
    )
    assert limited["stop_reason"] == "search"
    assert limited["searches_used"] == 1
    assert limited["steps"][-1]["detail"] == "Search budget exhausted before the next investigation."

    clock = {"t": 0.0}

    def tick() -> float:
        return clock["t"]

    def slow(question: str, asset: str) -> dict:
        clock["t"] += 10
        return _grounded(trace_scene("NVDA"))

    timed = run_research_loop(
        "What are people saying about NVIDIA?",
        slow,
        max_searches=4,
        max_tokens=5000,
        max_seconds=5,
        clock=tick,
    )
    assert timed["stop_reason"] == "time"
    assert timed["searches_used"] == 1

    def verbose(question: str, asset: str) -> dict:
        scene = trace_scene("NVDA")
        scene["answer"] = " ".join(["token"] * 80)
        return _grounded(scene)

    spent = run_research_loop(
        "What are people saying about NVIDIA?",
        verbose,
        max_searches=4,
        max_tokens=20,
        max_seconds=5,
    )
    assert spent["stop_reason"] == "token"
    assert spent["tokens_used"] >= 20
    assert spent["searches_used"] == 1


def test_loop_retries_a_failed_search_and_skips_a_second_failure() -> None:
    state = {"n": 0}

    def flaky(question: str, asset: str) -> dict:
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("index timeout")
        return _grounded(_sufficient_scene())

    recovered = run_research_loop(
        "What are people saying about NVIDIA?",
        flaky,
        max_searches=3,
        max_tokens=5000,
        max_seconds=5,
    )
    assert any(step["status"] == "recovered" for step in recovered["steps"])
    assert recovered["stop_reason"] == "sufficient"
    assert recovered["searches_used"] == 2

    def down(question: str, asset: str) -> dict:
        raise RuntimeError("index down")

    failed = run_research_loop("What is missing?", down, max_searches=2, max_tokens=500, max_seconds=5)
    assert failed["stop_reason"] == "failed"
    assert any(step["status"] == "recovered" for step in failed["steps"])
    assert any(step["status"] == "failed" for step in failed["steps"])
    assert failed["scene"] is None


def test_dashboard_api_reports_progress_contradictions_and_quality(client: TestClient) -> None:
    page = client.get("/")
    overview = client.post(
        "/api/trace/loop",
        json={
            "question": "What are people saying about NVIDIA?",
            "max_searches": 1,
            "max_tokens": 2500,
            "max_seconds": 15,
        },
    )
    deeper = client.post(
        "/api/trace/loop",
        json={
            "question": "What are people saying about NVIDIA?",
            "max_searches": 4,
            "max_tokens": 2500,
            "max_seconds": 15,
        },
    )
    report = overview.json()
    follow = deeper.json()
    pending = [row for row in report["contradictions"] if row["status"] == "pending"]

    assert page.status_code == 200
    assert 'data-tab="dashboard"' in page.text
    assert "Contradiction report" in page.text
    assert "Research intelligence" in page.text
    assert overview.status_code == 200
    assert report["stop_reason"] == "search"
    assert report["metrics"]["sufficient"] is False
    assert report["scene"]["asset"] == "NVDA"
    assert pending and "Bridgewater" in f"{pending[0]['left']} {pending[0]['right']}"
    assert follow["stop_reason"] == "unchanged"
    assert follow["searches_used"] == 2
    assert follow["scene"]["asset"] == "NVDA"
    assert follow["steps"][-1]["detail"] == "The search added no new evidence."
    assert any(
        step["action"] == "investigate" and "outside" in step["detail"] for step in follow["steps"]
    )


def test_loop_stops_when_a_search_adds_nothing() -> None:
    scene = trace_scene("NVDA")

    def search(question: str, asset: str) -> dict:
        return _grounded(scene)

    report = run_research_loop(
        "What are people saying about NVIDIA?",
        search,
        max_searches=4,
        max_tokens=5000,
        max_seconds=5,
    )

    assert report["stop_reason"] == "unchanged"
    assert report["searches_used"] == 2
    assert report["scene"]["asset"] == "NVDA"
    assert report["steps"][-1]["detail"] == "The search added no new evidence."
    assert any(row["status"] == "pending" for row in report["contradictions"])


def test_corroborate_searches_outside_the_current_graph() -> None:
    calls: list[tuple[str, str]] = []

    def search(question: str, asset: str) -> dict:
        calls.append(("graph", question))
        return _grounded(trace_scene("NVDA"))

    def outside(question: str, asset: str) -> dict:
        calls.append(("library", question))
        scene = {
            "asset": "DOCS",
            "question": question,
            "answer": "A filing shows Bridgewater cut the NVIDIA position.",
            "markers": [{"n": 1, "claim_id": "claim:ext"}],
            "claims": [
                {
                    "id": "claim:ext",
                    "text": "A filing shows Bridgewater cut the NVIDIA position.",
                    "short": "Filing",
                    "agent_id": "agent:DOCS:news",
                    "band": "above",
                    "upstream": [],
                    "provenance": {
                        "filename": "filing.md",
                        "checksum": "abc",
                        "version": 1,
                        "start_char": 0,
                        "end_char": 48,
                    },
                }
            ],
            "conflicts": [],
            "plan": {"summary": "Evidence is sufficient. No follow-up research.", "next_id": "", "items": []},
        }
        return {"asset": "DOCS", "answer": scene["answer"], "markers": scene["markers"], "scene": scene}

    report = run_research_loop(
        "What are people saying about NVIDIA?",
        search,
        max_searches=4,
        max_tokens=5000,
        max_seconds=5,
        outside_search=outside,
    )

    assert calls[0][0] == "graph"
    assert calls[1] == ("library", "Which claim about NVIDIA holds: Retail bullish or Bridgewater cuts?")
    assert report["stop_reason"] == "sufficient"
    assert report["asset"] == "DOCS"
    assert report["searches_used"] == 2
    assert any(step["detail"] == "Searching documents outside the current graph." for step in report["steps"])
