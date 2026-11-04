"""Bandits, Bayesian retrieval, offline evaluation, and the safety gate."""

from app.research.advanced import analyze_run
from app.research.learning import ResearchMemory, WASTE


def _logged(strategy: str, useful: bool, *, top_k: int = 4) -> dict:
    detail = "A new passage." if useful else WASTE
    return {
        "stop_reason": "unchanged",
        "sufficient": False,
        "answer": "A cited answer.",
        "searches_used": 1,
        "search_budget": 4,
        "metrics": {"confidence": 0.6, "factual_support": 1.0, "unresolved_contradictions": 0.0},
        "contradictions": [],
        "steps": [
            {"action": "investigate", "strategy": strategy, "status": "ok", "detail": "follow", "question": "q"},
            {
                "action": "search",
                "status": "ok",
                "detail": detail,
                "question": "q",
                "confidence": 0.6,
                "top_k": top_k,
                "agent": "Social agent",
            },
        ],
    }


def _notes(report: dict, memory: ResearchMemory | None = None) -> dict[str, dict]:
    return {note["component"]: note for note in analyze_run(report, memory)["notes"]}


def _waste_report() -> dict:
    return {
        "stop_reason": "unchanged",
        "metrics": {"confidence": 0.6, "factual_support": 1.0, "unresolved_contradictions": 0.0},
        "contradictions": [],
        "collaboration": {"agents": [{"name": "Social agent"}]},
        "steps": [
            {
                "action": "search",
                "status": "ok",
                "detail": WASTE,
                "agent": "Social agent",
                "top_k": 4,
                "strategy": "entity",
            }
        ],
    }


def test_failure_analysis_names_the_agent_that_added_nothing() -> None:
    notes = _notes(_waste_report())

    assert notes["failure"]["decision"] == "waste"
    assert notes["failure"]["reason"].startswith("Social agent:")
    assert notes["safety"]["decision"] == "withhold"
    assert notes["offline_policy"]["decision"] == "insufficient"


def test_bandit_and_causal_effect_prefer_the_strategy_that_worked() -> None:
    memory = ResearchMemory()
    for _ in range(3):
        memory.remember(_logged("entity", True))
        memory.remember(_logged("corroborate", False))

    notes = _notes(_waste_report(), memory)

    assert notes["bandit"]["decision"] == "entity"
    assert "clear:mid" in notes["bandit"]["reason"]
    assert notes["causal"]["decision"] == "identified"
    assert "1.00 more often than corroborate" in notes["causal"]["reason"]
    assert notes["offline_policy"]["decision"] == "doubly_robust"
    assert notes["safety"]["decision"] == "allow"


def test_bayesian_retrieval_explores_but_the_gate_keeps_the_proven_count() -> None:
    memory = ResearchMemory()
    for _ in range(2):
        memory.remember(_logged("broaden", True, top_k=8))
        memory.remember(_logged("broaden", False, top_k=4))

    notes = _notes(_waste_report(), memory)

    assert notes["bayesian_retrieval"]["decision"] == "withhold"
    assert "top_k 6" in notes["bayesian_retrieval"]["reason"]
    assert "keeps top_k 8" in notes["bayesian_retrieval"]["reason"]


def test_an_open_contradiction_is_a_dissent() -> None:
    report = _waste_report()
    report["contradictions"] = [{"status": "pending", "left": "Retail bullish", "right": "Bridgewater cuts"}]

    notes = _notes(report)

    assert notes["consensus"]["decision"] == "dissent"
    assert "Corroboration agent" in notes["consensus"]["reason"]
    assert notes["governance"]["decision"] == "withhold"
