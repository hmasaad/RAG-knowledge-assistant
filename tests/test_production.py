"""The production review gates a finished answer before it can be released."""

from app.research.intelligence import run_research_loop
from app.research.production import evaluate_production
from app.research.trace import trace_scene
from app.security.prompts import REFUSAL_TEXT


def _claim(claim_id: str, short: str, text: str, filename: str, checksum: str, **provenance) -> dict:
    source = {"filename": filename, "checksum": checksum, "version": 2}
    source.update(provenance)
    return {
        "id": claim_id,
        "short": short,
        "text": text,
        "provenance": source,
    }


def _report(**overrides) -> dict:
    retry = "The retry logic is implemented in app/http/retry.py."
    report = {
        "question": "Where is the retry logic implemented?",
        "answer": retry,
        "user_id": "local",
        "sufficient": True,
        "stop_reason": "sufficient",
        "searches_used": 1,
        "search_budget": 4,
        "tokens_used": 120,
        "token_budget": 2500,
        "elapsed_seconds": 0.4,
        "time_budget": 15,
        "metrics": {
            "completeness": 1.0,
            "factual_support": 1.0,
            "citation_validity": 1.0,
            "source_diversity": 1.0,
            "freshness": 1.0,
            "unresolved_contradictions": 0.0,
            "confidence": 0.9,
        },
        "contradictions": [],
        "steps": [
            {
                "action": "search",
                "status": "ok",
                "question": "Where is the retry logic implemented?",
                "strategy": "initial",
                "top_k": 4,
                "detail": retry,
            }
        ],
        "scene": {
            "claims": [
                _claim("claim:retry", "Retry logic", retry, "api-errors.md", "abc123def456"),
            ],
            "markers": [{"n": 1, "claim_id": "claim:retry"}],
            "plan": {"items": []},
        },
        "benchmark": {"trend": "baseline", "latest_score": 0.8},
    }
    report.update(overrides)
    return report


def _areas(report: dict) -> dict[str, dict]:
    return {area["component"]: area for area in evaluate_production(report)["areas"]}


def test_a_complete_cited_answer_is_released() -> None:
    production = evaluate_production(_report())

    assert production["release"] == "release"
    assert production["revision"] == ""
    assert production["summary"] == "Release. Completeness 1.00. 8 of 8 production areas passed."
    assert [area["component"] for area in production["areas"]] == [
        "completeness",
        "retrieval",
        "citation",
        "conflict",
        "adaptive",
        "evaluation",
        "security",
        "observability",
    ]
    areas = {area["component"]: area for area in production["areas"]}
    assert areas["completeness"]["status"] == "complete"
    assert areas["retrieval"]["status"] == "ranked"
    assert "api-errors.md (abc123def456)" in areas["citation"]["records"][0]["detail"]
    assert areas["conflict"]["status"] == "current"
    assert areas["security"]["status"] == "secure"
    assert areas["observability"]["records"][-1]["detail"] == "No user feedback is attached to this run."


def test_the_same_answer_keeps_the_same_review() -> None:
    report = _report()

    assert evaluate_production(report) == evaluate_production(report)


def test_a_missing_clause_is_kept_for_revision() -> None:
    retry = "The retry logic is implemented in app/http/retry.py."
    timeouts = "Timeouts are configured in app/http/client.py."
    report = _report(
        question="Which file implements the retry logic and where are timeouts configured?",
        answer=retry,
    )
    report["scene"]["claims"].append(
        _claim("claim:timeouts", "Timeouts", timeouts, "database.md", "fff999eee888")
    )
    report["scene"]["markers"].append({"n": 2, "claim_id": "claim:timeouts"})

    production = evaluate_production(report)
    completeness = _areas(report)["completeness"]

    assert production["release"] == "revise"
    assert completeness["status"] == "partial"
    assert any(record["detail"] == "where are timeouts configured" for record in completeness["records"])
    assert "Timeouts are configured in app/http/client.py. (database.md)" in production["revision"]
    assert "Still required:" in production["revision"]
    assert production["revision"].startswith(retry)


def test_an_open_contradiction_is_named_in_the_revision() -> None:
    report = _report(
        contradictions=[{"status": "pending", "left": "Retry limit 3", "right": "Retry limit 5"}]
    )

    production = evaluate_production(report)

    assert production["release"] == "revise"
    assert _areas(report)["conflict"]["status"] == "contested"
    assert "pending · Retry limit 3 vs Retry limit 5" in production["revision"]


def test_an_older_cited_version_is_not_released() -> None:
    report = _report()
    report["scene"]["claims"][0]["provenance"]["version"] = 1
    report["scene"]["claims"][0]["provenance"]["history"] = [{"version": 2, "checksum": "new"}]

    production = evaluate_production(report)

    assert _areas(report)["conflict"]["status"] == "stale"
    assert production["release"] == "revise"
    assert "api-errors.md v1 is older than a later version." in production["revision"]


def test_a_secret_in_the_answer_is_withheld() -> None:
    report = _report(answer="The retry logic is implemented in app/http/retry.py. api_key: purple-otter")

    production = evaluate_production(report)

    assert _areas(report)["security"]["status"] == "blocked"
    assert production["release"] == "withhold"


def test_a_private_source_from_another_owner_is_withheld() -> None:
    report = _report()
    report["scene"]["claims"][0]["provenance"]["visibility"] = "private"
    report["scene"]["claims"][0]["provenance"]["owner"] = "ada"

    production = evaluate_production(report)

    assert _areas(report)["security"]["status"] == "blocked"
    assert production["release"] == "withhold"
    assert "another owner" in _areas(report)["security"]["records"][2]["detail"]


def test_a_refusal_is_withheld_without_a_revision() -> None:
    report = _report(answer=REFUSAL_TEXT, question="What is the office wifi password?")
    report["scene"]["markers"] = []

    production = evaluate_production(report)

    assert production["release"] == "withhold"
    assert production["revision"] == ""
    assert _areas(report)["completeness"]["status"] == "missing"
    assert "asks for a secret" in _areas(report)["security"]["records"][1]["detail"]


def test_hybrid_retrieval_and_a_stalled_plan_are_visible() -> None:
    report = _report(
        question="Which file implements the retry logic and where are timeouts configured?",
        answer=(
            "The retry logic is implemented in app/http/retry.py. "
            "Timeouts are configured in app/http/client.py."
        ),
        sufficient=False,
    )
    report["metrics"]["confidence"] = 0.4
    report["scene"]["claims"].append(
        _claim(
            "claim:timeouts",
            "Timeouts",
            "Timeouts are configured in app/http/client.py.",
            "database.md",
            "fff999eee888",
        )
    )
    report["scene"]["markers"].append({"n": 2, "claim_id": "claim:timeouts"})
    report["steps"] = [
        {
            "action": "search",
            "status": "ok",
            "question": "Which file implements the retry logic?",
            "strategy": "initial",
            "top_k": 4,
            "detail": "A new passage.",
        },
        {
            "action": "search",
            "status": "failed",
            "question": "Where are timeouts configured?",
            "strategy": "decompose",
            "top_k": 6,
            "detail": "Index down.",
        },
    ]

    areas = _areas(report)

    assert areas["retrieval"]["status"] == "hybrid"
    assert "top_k 4, 6" in areas["retrieval"]["records"][2]["detail"]
    assert areas["adaptive"]["status"] == "adapted"
    assert "Index down." in areas["observability"]["records"][3]["detail"]


def test_a_compound_question_that_never_adapts_stays_unreleased() -> None:
    report = _report(
        question="Which file implements the retry logic and where are timeouts configured?",
        answer=(
            "The retry logic is implemented in app/http/retry.py. "
            "Timeouts are configured in app/http/client.py."
        ),
        sufficient=False,
    )
    report["metrics"]["confidence"] = 0.4
    report["scene"]["claims"].append(
        _claim(
            "claim:timeouts",
            "Timeouts",
            "Timeouts are configured in app/http/client.py.",
            "database.md",
            "fff999eee888",
        )
    )
    report["scene"]["markers"].append({"n": 2, "claim_id": "claim:timeouts"})

    production = evaluate_production(report)

    assert _areas(report)["adaptive"]["status"] == "stalled"
    assert production["release"] == "revise"


def test_an_open_research_gap_keeps_the_answer_incomplete() -> None:
    report = _report()
    report["scene"]["plan"] = {
        "items": [{"missing": "No independent source settles the retry limit.", "question": "What is the retry limit?"}]
    }

    areas = _areas(report)

    assert areas["completeness"]["status"] == "partial"
    assert any(record["label"] == "gap" for record in areas["completeness"]["records"])
    assert evaluate_production(report)["release"] == "revise"


def test_the_research_loop_attaches_the_production_review() -> None:
    scene = trace_scene("NVDA")
    scene["plan"] = {"summary": "Evidence is sufficient. No follow-up research.", "next_id": "", "items": []}
    scene["conflicts"] = [item for item in scene["conflicts"] if item["status"] != "pending"]

    def search(question: str, asset: str) -> dict:
        return {"asset": scene["asset"], "answer": scene["answer"], "markers": scene["markers"], "scene": scene}

    report = run_research_loop(
        "What are people saying about NVIDIA?",
        search,
        max_searches=4,
        max_tokens=5000,
        max_seconds=5,
    )

    assert [area["component"] for area in report["production"]["areas"]] == [
        "completeness",
        "retrieval",
        "citation",
        "conflict",
        "adaptive",
        "evaluation",
        "security",
        "observability",
    ]
    assert report["production"]["release"] in {"release", "revise", "withhold"}
    assert report["production"]["areas"][0]["priority"] == 1
