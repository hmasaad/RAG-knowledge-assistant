"""Trajectory scores and failure predictions for a finished research path."""

from fastapi.testclient import TestClient

from app.research.intelligence import run_research_loop
from app.research.learning import (
    ResearchMemory,
    choose_next_step,
    choose_retrieval,
    evaluate_trajectory,
    learn_strategies,
    predict_failure,
)
from app.research.trace import trace_scene


def _report(
    *,
    stop: str,
    steps: list[dict],
    sufficient: bool = False,
    confidence: float = 0.6,
    factual: float = 1.0,
    unresolved: float = 0.0,
    searches: int | None = None,
    budget: int = 4,
    answer: str = "A cited answer.",
) -> dict:
    used = searches if searches is not None else sum(1 for step in steps if step["action"] == "search")
    return {
        "answer": answer,
        "stop_reason": stop,
        "sufficient": sufficient,
        "searches_used": used,
        "search_budget": budget,
        "steps": steps,
        "metrics": {
            "confidence": confidence,
            "factual_support": factual,
            "unresolved_contradictions": unresolved,
        },
    }


def _search(detail: str, *, status: str = "ok", confidence: float = 0.6, question: str = "What changed?") -> dict:
    return {
        "action": "search",
        "status": status,
        "question": question,
        "strategy": "",
        "detail": detail,
        "confidence": confidence,
    }


def _stop(detail: str) -> dict:
    return {"action": "stop", "status": "stopped", "question": "", "strategy": "", "detail": detail, "confidence": 0.6}


def test_successful_path_scores_higher_than_a_wasted_search() -> None:
    success = _report(
        stop="sufficient",
        sufficient=True,
        confidence=0.8,
        steps=[_search("Cited answer.", confidence=0.8), _stop("Evidence is sufficient.")],
        searches=1,
    )
    wasted = _report(
        stop="unchanged",
        confidence=0.6,
        unresolved=1.0,
        steps=[
            _search("Cited answer.", confidence=0.6, question="Overview"),
            _search("No new claim, settled conflict, or higher confidence.", confidence=0.6, question="Follow up"),
            _stop("The search added no new evidence."),
        ],
    )

    good = evaluate_trajectory(success)
    poor = evaluate_trajectory(wasted)
    warning = predict_failure(wasted)

    assert good["outcome"] == "success"
    assert good["decisions"][0]["verdict"] == "useful"
    assert good["score"] > poor["score"]
    assert poor["outcome"] == "partial"
    assert any(item["verdict"] == "waste" for item in poor["decisions"])
    assert any(item["kind"] == "waste" for item in poor["findings"])
    assert warning["label"] == "likely_fail"
    assert warning["warned_at_search"] == 2
    assert "A search added no new evidence." in warning["reasons"]


def test_failed_search_is_predicted_on_the_first_search() -> None:
    report = _report(
        stop="failed",
        confidence=0.0,
        factual=0.0,
        answer="",
        steps=[_search("index down", status="failed", confidence=0.0), _stop("A research step failed.")],
        searches=1,
    )

    evaluation = evaluate_trajectory(report)
    prediction = predict_failure(report)

    assert evaluation["outcome"] == "failed"
    assert evaluation["decisions"][0]["verdict"] == "failed"
    assert any(item["kind"] == "failure" for item in evaluation["findings"])
    assert prediction["label"] == "likely_fail"
    assert prediction["warned_at_search"] == 1


def test_past_runs_raise_the_fail_probability_for_the_same_signals() -> None:
    report = _report(
        stop="search",
        confidence=0.4,
        searches=3,
        budget=4,
        steps=[_search("Cited answer.", confidence=0.4), _stop("Search budget exhausted.")],
    )
    memory = ResearchMemory()
    alone = predict_failure(report, memory)
    for _ in range(3):
        memory.remember(report)
    learned = predict_failure(report, memory)

    assert alone["basis"] == "signals"
    assert alone["label"] == "likely_succeed"
    assert learned["basis"] == "history"
    assert learned["history_runs"] == 3
    assert learned["probability"] > alone["probability"]
    assert learned["label"] == "likely_fail"
    assert any(reason.startswith("Past runs with these signals failed") for reason in learned["reasons"])


def test_loop_attaches_a_score_and_a_prediction() -> None:
    scene = trace_scene("NVDA")
    scene["plan"] = {"summary": "Evidence is sufficient. No follow-up research.", "next_id": "", "items": []}
    scene["conflicts"] = [item for item in scene["conflicts"] if item["status"] != "pending"]

    def search(question: str, asset: str) -> dict:
        return {"asset": scene["asset"], "answer": scene["answer"], "markers": scene["markers"], "scene": scene}

    report = run_research_loop("What are people saying about NVIDIA?", search, max_searches=4, max_tokens=5000, max_seconds=5)

    assert report["evaluation"]["outcome"] == "success"
    assert report["evaluation"]["score"] > 0.5
    assert report["prediction"]["label"] == "likely_succeed"
    assert report["prediction"]["warned_at_search"] is None
    assert report["learning"]["retrieval"]["top_k"] == 4
    assert report["learning"]["policy"]["action"] == "follow"
    assert report["learning"]["strategies"][0]["strategy"] == "initial"


def test_dashboard_shows_trajectory_evaluation_and_failure_prediction(client: TestClient) -> None:
    page = client.get("/")
    response = client.post(
        "/api/trace/loop",
        json={
            "question": "What are people saying about NVIDIA?",
            "max_searches": 4,
            "max_tokens": 2500,
            "max_seconds": 15,
        },
    )
    body = response.json()

    assert page.status_code == 200
    assert "Trajectory evaluation" in page.text
    assert "Failure prediction" in page.text
    assert response.status_code == 200
    assert body["evaluation"]["outcome"] == "partial"
    assert any(item["verdict"] == "waste" for item in body["evaluation"]["decisions"])
    assert any(item["verdict"] == "outside" for item in body["evaluation"]["decisions"])
    assert body["prediction"]["label"] == "likely_fail"
    assert body["prediction"]["warned_at_search"] == 2
    assert "Strategy learning" in page.text
    assert "<h2>Retrieval</h2>" in page.text
    assert "<h2>Policy</h2>" in page.text
    assert body["learning"]["retrieval"]["strategy"] == "corroborate"
    assert body["learning"]["retrieval"]["top_k"] == 6
    assert any(row["strategy"] == "corroborate" for row in body["learning"]["strategies"])
    assert "<h2>Counterfactual</h2>" in page.text
    assert "<h2>Research agents</h2>" in page.text
    assert "<h2>Benchmark</h2>" in page.text
    assert "<h2>Scientific discovery</h2>" in page.text
    assert "<h2>Agent society</h2>" in page.text
    assert "<h2>Advanced evaluation</h2>" in page.text
    society = {note["component"]: note for note in body["discovery"]["society"]["notes"]}
    assert list(society) == [
        "mas",
        "hierarchy",
        "blackboard",
        "protocol",
        "allocation",
        "planning",
        "debate",
        "consensus",
        "distributed",
        "observability",
    ]
    assert society["consensus"]["decision"] == "dissent"
    assert "<h2>Acceptance criteria</h2>" in page.text
    acceptance = {item["criterion"]: item for item in body["acceptance"]["criteria"]}
    assert list(acceptance) == [
        "evidence",
        "hypothesis",
        "reliability",
        "coordination",
        "conflicts",
        "reproducibility",
    ]
    assert body["acceptance"]["met"] is False
    assert acceptance["evidence"]["result"] == "fail"
    assert acceptance["conflicts"]["result"] == "fail"
    agents = [review["agent"] for review in body["discovery"]["reviews"]]
    assert agents == ["Investigator", "Challenger", "Hypothesis tester", "Conclusion"]
    assert body["discovery"]["conclusion"]["verdict"] == "disputed"
    assert [phase["component"] for phase in body["discovery"]["phases"]] == [
        "orchestration",
        "hypotheses",
        "collaboration",
        "debate",
        "experiment",
        "consensus",
        "reproducibility",
        "evaluation",
    ]
    assert body["discovery"]["phases"][5]["status"] == "dissent"
    components = {note["component"] for note in body["advanced"]["notes"]}
    assert components == {
        "failure",
        "bandit",
        "bayesian_retrieval",
        "offline_policy",
        "causal",
        "consensus",
        "safety",
        "governance",
    }
    assert body["counterfactual"]["verdict"] in {"none", "matched", "missed"}
    assert any(agent["name"] == "Lead agent" for agent in body["collaboration"]["agents"])
    assert any(agent["name"] == "Corroboration agent" for agent in body["collaboration"]["agents"])
    assert body["collaboration"]["aggregator"]["decision"] == "unchanged"
    assert body["benchmark"]["trend"] in {"baseline", "improving", "steady", "regressing"}


def _strategy_report(strategy: str, *, useful: bool, top_k: int) -> dict:
    detail = "A new passage." if useful else "No new claim, settled conflict, or higher confidence."
    return {
        "sufficient": False,
        "stop_reason": "unchanged",
        "answer": "A cited answer.",
        "searches_used": 1,
        "search_budget": 4,
        "metrics": {"confidence": 0.5, "factual_support": 1.0, "unresolved_contradictions": 0.0},
        "steps": [
            {
                "action": "investigate",
                "strategy": strategy,
                "status": "ok",
                "detail": "follow",
                "question": "follow up",
            },
            {
                "action": "search",
                "status": "ok",
                "detail": detail,
                "question": "follow up",
                "confidence": 0.5,
                "top_k": top_k,
            },
        ],
    }


def test_strategy_learning_ranks_the_strategy_that_adds_evidence() -> None:
    memory = ResearchMemory()
    for _ in range(3):
        memory.remember(_strategy_report("broaden", useful=True, top_k=8))
        memory.remember(_strategy_report("corroborate", useful=False, top_k=4))

    learned = learn_strategies(memory)

    assert learned["best"] == "broaden"
    assert learned["strategies"][0]["strategy"] == "broaden"
    assert learned["strategies"][0]["useful"] == 3
    assert "broaden is the strongest strategy" in learned["summary"]


def test_history_prefers_a_strategy_and_a_passage_count_that_worked() -> None:
    memory = ResearchMemory()
    for _ in range(3):
        memory.remember(_strategy_report("entity", useful=True, top_k=4))
        memory.remember(_strategy_report("corroborate", useful=False, top_k=6))
    for _ in range(2):
        memory.remember(_strategy_report("broaden", useful=False, top_k=4))
        memory.remember(_strategy_report("broaden", useful=True, top_k=8))
    scene = {
        "plan": {
            "items": [
                {"id": "c", "question": "Which claim holds?", "strategy": "corroborate", "value": 0.45},
                {"id": "e", "question": "Which source mentions Bridgewater?", "strategy": "entity", "value": 0.30},
            ]
        }
    }

    decision = choose_next_step(scene, {}, memory)
    retrieval = choose_retrieval("broaden", memory)

    assert decision["policy"]["action"] == "reorder"
    assert decision["item"]["strategy"] == "entity"
    assert retrieval["basis"] == "history"
    assert retrieval["top_k"] == 8
    assert choose_retrieval("broaden", None)["top_k"] == 8
    assert choose_retrieval("initial", None)["top_k"] == 4


def test_history_stops_a_strategy_that_rarely_adds_evidence() -> None:
    memory = ResearchMemory()
    for _ in range(3):
        memory.remember(_strategy_report("corroborate", useful=False, top_k=6))
    scene = {
        "asset": "NVDA",
        "answer": "People are divided on the stock.",
        "markers": [{"n": 1, "claim_id": "claim:a"}],
        "claims": [
            {
                "id": "claim:a",
                "text": "People are divided on the stock.",
                "short": "Divided",
                "agent_id": "agent:NVDA:news",
                "band": "above",
                "upstream": [],
                "provenance": {
                    "filename": "note.md",
                    "checksum": "abc",
                    "version": 2,
                    "start_char": 0,
                    "end_char": 20,
                },
            }
        ],
        "conflicts": [],
        "plan": {
            "summary": "Next",
            "next_id": "c",
            "items": [
                {
                    "id": "c",
                    "question": "Which claim about NVIDIA holds?",
                    "strategy": "corroborate",
                    "value": 0.45,
                }
            ],
        },
    }
    calls: list[str] = []

    def search(question: str, asset: str) -> dict:
        calls.append("graph")
        return {"asset": "NVDA", "answer": scene["answer"], "markers": scene["markers"], "scene": scene}

    def outside(question: str, asset: str) -> dict:
        calls.append("library")
        return {"asset": "NVDA", "answer": "", "markers": [], "scene": None}

    report = run_research_loop(
        "What are people saying about NVIDIA?",
        search,
        max_searches=4,
        max_tokens=5000,
        max_seconds=5,
        outside_search=outside,
        memory=memory,
    )

    assert calls == ["graph"]
    assert report["stop_reason"] == "policy"
    assert report["searches_used"] == 1
    assert report["learning"]["policy"]["action"] == "stop"
    assert "0 of 3" in report["learning"]["policy"]["reason"]
