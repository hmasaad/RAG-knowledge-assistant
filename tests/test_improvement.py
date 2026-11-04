"""Counterfactuals, specialist handoffs, and the running benchmark."""

from app.research.intelligence import run_research_loop
from app.research.learning import ResearchMemory, benchmark_run


def _claim(text: str, claim_id: str = "claim:a") -> dict:
    return {
        "id": claim_id,
        "text": text,
        "short": text[:24],
        "agent_id": "agent:NVDA:news",
        "band": "above",
        "upstream": [],
        "provenance": {
            "filename": "note.md",
            "checksum": "abc",
            "version": 2,
            "start_char": 0,
            "end_char": len(text),
        },
    }


def _scene(text: str, items: list[dict]) -> dict:
    return {
        "asset": "NVDA",
        "answer": text,
        "markers": [{"n": 1, "claim_id": "claim:a"}],
        "claims": [_claim(text)],
        "conflicts": [],
        "plan": {"summary": "Next", "next_id": items[0]["id"] if items else "", "items": items},
    }


def _item(item_id: str, strategy: str, question: str, value: float) -> dict:
    return {"id": item_id, "strategy": strategy, "question": question, "value": value}


def test_a_wasted_search_is_scored_against_the_question_that_was_skipped() -> None:
    scene = _scene(
        "People are divided on the stock.",
        [
            _item("c", "corroborate", "Which claim holds?", 0.45),
            _item("e", "entity", "Which source mentions Bridgewater?", 0.30),
        ],
    )
    calls: list[str] = []

    def search(question: str, asset: str) -> dict:
        calls.append(question)
        return {"asset": "NVDA", "answer": scene["answer"], "markers": scene["markers"], "scene": scene}

    report = run_research_loop(
        "What are people saying about NVIDIA?",
        search,
        max_searches=4,
        max_tokens=5000,
        max_seconds=5,
    )

    fork = report["counterfactual"]["forks"][0]
    assert report["counterfactual"]["verdict"] == "missed"
    assert "entity" in report["counterfactual"]["summary"]
    assert fork["taken_strategy"] == "corroborate"
    assert fork["useful"] is False
    assert fork["alternatives"][0]["strategy"] == "entity"
    assert fork["alternatives"][0]["expected"] == 0.15
    assert report["collaboration"]["handoffs"][0]["from_agent"] == "Lead agent"
    assert report["collaboration"]["handoffs"][0]["to_agent"] == "Corroboration agent"
    assert report["collaboration"]["aggregator"]["decision"] == "unchanged"
    assert report["benchmark"]["trend"] == "baseline"
    assert calls[1] == "Which claim holds?"


def test_the_chosen_question_beats_a_weaker_alternative() -> None:
    first = _scene(
        "People are divided on the stock.",
        [
            _item("c", "corroborate", "Which claim holds?", 0.45),
            _item("e", "entity", "Which source mentions Bridgewater?", 0.30),
        ],
    )
    done = _scene("A filing settles the claim.", [])
    done["claims"] = [_claim("A filing settles the claim.", "claim:b")]
    done["markers"] = [{"n": 1, "claim_id": "claim:b"}]
    seen = {"n": 0}

    def search(question: str, asset: str) -> dict:
        seen["n"] += 1
        scene = first if seen["n"] == 1 else done
        return {"asset": "NVDA", "answer": scene["answer"], "markers": scene["markers"], "scene": scene}

    report = run_research_loop(
        "What are people saying about NVIDIA?",
        search,
        max_searches=4,
        max_tokens=5000,
        max_seconds=5,
    )

    assert report["stop_reason"] == "sufficient"
    assert report["counterfactual"]["verdict"] == "matched"
    assert report["counterfactual"]["forks"][0]["useful"] is True
    assert any(agent["name"] == "Corroboration agent" and agent["useful"] == 1 for agent in report["collaboration"]["agents"])


def test_a_skipped_question_with_the_same_strategy_is_named() -> None:
    from app.research.learning import close_counterfactual

    fork = close_counterfactual(
        {
            "taken_strategy": "entity",
            "taken_question": "Which source mentions Retail?",
            "taken_value": 0.55,
            "alternatives": [
                {
                    "strategy": "entity",
                    "question": "Which source mentions Bridgewater?",
                    "expected": 0.28,
                    "basis": "half the planner value, with no matching history",
                }
            ],
        },
        useful=False,
    )

    assert fork["verdict"] == "missed"
    assert "Which source mentions Bridgewater?" in fork["summary"]


def test_later_runs_are_benchmarked_against_earlier_scores() -> None:
    memory = ResearchMemory()
    earlier = {
        "evaluation": {"score": 0.4, "outcome": "partial"},
        "steps": [],
        "stop_reason": "unchanged",
        "sufficient": False,
        "answer": "A cited answer.",
        "searches_used": 1,
        "search_budget": 4,
        "metrics": {"confidence": 0.4, "factual_support": 1.0, "unresolved_contradictions": 0.0},
    }
    memory.remember(earlier)
    memory.remember(earlier)
    improved = {"evaluation": {"score": 0.8, "outcome": "success"}}
    missed = {
        "forks": [
            {
                "verdict": "missed",
                "alternatives": [{"strategy": "entity", "expected": 0.3}],
            }
        ]
    }

    rising = benchmark_run(memory, improved)
    falling = benchmark_run(
        memory,
        {"evaluation": {"score": 0.2, "outcome": "partial"}},
        missed,
    )

    assert rising["trend"] == "improving"
    assert rising["runs"] == 3
    assert rising["delta"] == 0.4
    assert "above the earlier mean" in rising["summary"]
    assert falling["trend"] == "regressing"
    assert "Prefer entity" in falling["improvement"]
