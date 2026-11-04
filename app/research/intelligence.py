"""Run a research question until the evidence is sufficient or a budget stops it.

Each pass searches, scores the scene, and either stops or investigates the
highest-value follow-up. A failed search is retried once. A second failure is
skipped so the loop can continue with another question.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from app.research.acceptance import measure_acceptance
from app.research.advanced import analyze_run
from app.research.discovery import discover
from app.research.learning import (
    ResearchMemory,
    assign_agent,
    benchmark_run,
    choose_next_step,
    choose_retrieval,
    close_counterfactual,
    collaboration_report,
    evaluate_trajectory,
    learn_strategies,
    open_counterfactual,
    predict_failure,
    stop_counterfactual,
    summarize_counterfactuals,
)

REFUSAL = "I don't have enough evidence to answer."
_ROLES = ("news", "social", "aggregator")
_OUTSIDE = frozenset({"corroborate", "broaden"})
_UNCHANGED = "The search added no new evidence."


def run_research_loop(
    question: str,
    search: Callable[[str, str], dict],
    *,
    asset: str = "NVDA",
    max_searches: int = 4,
    max_tokens: int = 2500,
    max_seconds: float = 15.0,
    clock: Callable[[], float] | None = None,
    outside_search: Callable[[str, str], dict] | None = None,
    memory: ResearchMemory | None = None,
    retrieval: dict | None = None,
) -> dict:
    """Search, evaluate, and investigate again inside the three budgets."""
    now = clock or time.monotonic
    started = now()
    deadline = started + max_seconds
    asked = question.strip()
    current = asked
    current_asset = asset
    searches_used = 0
    tokens_used = 0
    tried: dict[str, int] = {}
    steps: list[dict] = []
    scene: dict | None = None
    answer = ""
    metrics = _empty_metrics()
    stop_reason = ""
    seek_outside = False
    active_strategy = "initial"
    retrieval_choice = choose_retrieval(active_strategy, memory)
    applied_retrieval = retrieval_choice
    policy_note = {
        "action": "follow",
        "strategy": "initial",
        "reason": "The first search follows the question as asked.",
    }
    forks: list[dict] = []
    open_fork: dict | None = None

    def elapsed() -> float:
        return round(now() - started, 3)

    def record(
        action: str,
        status: str,
        detail: str,
        *,
        query: str = "",
        strategy: str = "",
        top_k: int = 0,
        agent: str = "",
    ) -> None:
        steps.append(
            {
                "action": action,
                "status": status,
                "question": query or current,
                "strategy": strategy,
                "detail": detail,
                "searches_used": searches_used,
                "tokens_used": tokens_used,
                "elapsed_seconds": elapsed(),
                "confidence": metrics["confidence"],
                "top_k": top_k,
                "agent": agent,
            }
        )

    def settle_fork(useful: bool) -> None:
        nonlocal open_fork
        if open_fork is None:
            return
        forks.append(close_counterfactual(open_fork, useful=useful))
        open_fork = None

    def apply_retrieval() -> None:
        nonlocal applied_retrieval
        applied_retrieval = retrieval_choice
        if retrieval is not None:
            retrieval["top_k"] = int(retrieval_choice["top_k"])

    while True:
        if now() >= deadline:
            stop_reason = "time"
            record("stop", "stopped", "Time budget exhausted.")
            break
        if searches_used >= max_searches:
            stop_reason = "search"
            record("stop", "stopped", "Search budget exhausted.")
            break
        if tokens_used >= max_tokens:
            stop_reason = "token"
            record("stop", "stopped", "Token budget exhausted.")
            break

        attempts = tried.get(current, 0)
        finder = outside_search if seek_outside and outside_search is not None else search
        running_agent = assign_agent(active_strategy)["name"]
        apply_retrieval()
        try:
            result = finder(current, current_asset)
        except Exception as exc:
            searches_used += 1
            tried[current] = attempts + 1
            message = str(exc).strip() or "Search failed."
            can_retry = attempts < 1 and searches_used < max_searches and now() < deadline
            if can_retry:
                record(
                    "search",
                    "recovered",
                    f"Recovered from a failed search: {message}",
                    top_k=applied_retrieval["top_k"],
                    agent=running_agent,
                )
                continue
            settle_fork(False)
            record("search", "failed", message, top_k=applied_retrieval["top_k"], agent=running_agent)
            decision = choose_next_step(scene, tried, memory)
            if decision["item"] is None:
                stop_reason = "failed"
                record("stop", "stopped", "A research step failed and there is no alternate question.")
                break
            if decision["abandon"]:
                forks.append(stop_counterfactual(decision, memory))
                policy_note = decision["policy"]
                stop_reason = "policy"
                record("stop", "stopped", decision["policy"]["reason"])
                break
            item = decision["item"]
            current = str(item.get("question") or "")
            active_strategy = str(item.get("strategy") or "")
            retrieval_choice = choose_retrieval(active_strategy, memory)
            policy_note = decision["policy"]
            open_fork = open_counterfactual(decision, memory)
            seek_outside = active_strategy in _OUTSIDE and outside_search is not None
            record(
                "investigate",
                "skipped",
                "Skipped the failed step and continued with the next question.",
                query=current,
                strategy=active_strategy,
                agent=assign_agent(active_strategy)["name"],
            )
            continue

        searches_used += 1
        tried[current] = attempts + 1
        found_answer = str(result.get("answer") or "")
        found_scene = result.get("scene") or None
        found_metrics = score_scene(found_scene or {})
        tokens_used += _tokens(current) + _tokens(found_answer)
        plan_items = ((found_scene or {}).get("plan") or {}).get("items") or []
        strategy = plan_items[0]["strategy"] if plan_items else ""
        if scene is not None and not _gained(scene, metrics, found_scene, found_metrics):
            settle_fork(False)
            record(
                "search",
                "ok",
                "No new claim, settled conflict, or higher confidence.",
                strategy=strategy,
                top_k=applied_retrieval["top_k"],
                agent=running_agent,
            )
            stop_reason = "unchanged"
            record("stop", "stopped", _UNCHANGED)
            break
        answer = found_answer
        scene = found_scene or {}
        current_asset = str(result.get("asset") or current_asset)
        metrics = found_metrics
        seek_outside = False
        settle_fork(True)
        record("search", "ok", answer[:240], strategy=strategy, top_k=applied_retrieval["top_k"], agent=running_agent)

        if tokens_used >= max_tokens:
            stop_reason = "token"
            record("stop", "stopped", "Token budget exhausted.")
            break
        if metrics["sufficient"]:
            stop_reason = "sufficient"
            record("stop", "stopped", "Evidence is sufficient.")
            break
        decision = choose_next_step(scene, tried, memory)
        if decision["item"] is None:
            stop_reason = "unresolved"
            record(
                "stop",
                "stopped",
                "Open gaps remain, and another search would repeat a question already tried.",
            )
            break
        if decision["abandon"]:
            forks.append(stop_counterfactual(decision, memory))
            policy_note = decision["policy"]
            stop_reason = "policy"
            record("stop", "stopped", decision["policy"]["reason"])
            break
        if searches_used >= max_searches:
            stop_reason = "search"
            record("stop", "stopped", "Search budget exhausted before the next investigation.")
            break
        item = decision["item"]
        current = str(item.get("question") or "")
        active_strategy = str(item.get("strategy") or "")
        retrieval_choice = choose_retrieval(active_strategy, memory)
        policy_note = decision["policy"]
        open_fork = open_counterfactual(decision, memory)
        seek_outside = active_strategy in _OUTSIDE and outside_search is not None
        record(
            "investigate",
            "ok",
            "Searching documents outside the current graph."
            if seek_outside
            else "Investigating the highest-value follow-up.",
            query=current,
            strategy=active_strategy,
            agent=assign_agent(active_strategy)["name"],
        )

    report = {
        "question": asked,
        "asset": current_asset,
        "answer": answer,
        "stop_reason": stop_reason,
        "sufficient": metrics["sufficient"],
        "searches_used": searches_used,
        "search_budget": max_searches,
        "tokens_used": tokens_used,
        "token_budget": max_tokens,
        "elapsed_seconds": elapsed(),
        "time_budget": max_seconds,
        "steps": steps,
        "metrics": metrics,
        "contradictions": contradiction_report(scene or {}),
        "scene": scene,
    }
    report["evaluation"] = evaluate_trajectory(report)
    report["prediction"] = predict_failure(report, memory)
    stop_detail = steps[-1]["detail"] if steps and steps[-1]["action"] == "stop" else ""
    report["counterfactual"] = summarize_counterfactuals(forks)
    report["collaboration"] = collaboration_report(steps, stop_reason, stop_detail)
    report["benchmark"] = benchmark_run(memory, report, report["counterfactual"])
    report["advanced"] = analyze_run(report, memory)
    report["discovery"] = discover(report)
    report["acceptance"] = measure_acceptance(report, report["discovery"])
    learned_from = memory
    if memory is None:
        learned_from = ResearchMemory()
    learned_from.remember(report)
    learned = learn_strategies(learned_from)
    learned["policy"] = policy_note
    learned["retrieval"] = applied_retrieval
    report["learning"] = learned
    return report


def score_scene(scene: dict) -> dict:
    """Confidence and quality from citations, source mix, freshness, and conflicts."""
    claims = {item["id"]: item for item in scene.get("claims") or []}
    support = _support(scene, claims)
    conflicts = list(scene.get("conflicts") or [])
    pending = [item for item in conflicts if item.get("status") == "pending"]
    gaps = (scene.get("plan") or {}).get("items") or []
    answer = str(scene.get("answer") or "").strip()
    factual = 1.0 if support and answer and answer != REFUSAL else 0.0
    cited = [
        claims[marker["claim_id"]]
        for marker in scene.get("markers") or []
        if marker.get("claim_id") in claims
    ]
    valid = [
        claim
        for claim in cited
        if (claim.get("provenance") or {}).get("filename") and (claim.get("provenance") or {}).get("checksum")
    ]
    citation_validity = round(len(valid) / len(cited), 2) if cited else 0.0
    roles = {_role(claim) for claim in support if _role(claim) in _ROLES}
    source_diversity = round(len(roles) / len(_ROLES), 2)
    versions = [int((claim.get("provenance") or {}).get("version") or 0) for claim in support]
    fresh = [version for version in versions if version >= 2]
    freshness = round(len(fresh) / len(versions), 2) if versions else 1.0
    unresolved = round(len(pending) / len(conflicts), 2) if conflicts else 0.0
    completeness = round(1.0 - min(1.0, len(gaps) / 4), 2)
    settled = 1.0 - unresolved
    confidence = _mean((factual, citation_validity, source_diversity, freshness, settled))
    quality = _mean((completeness, factual, citation_validity, source_diversity, freshness, settled))
    sufficient = not gaps and factual == 1.0 and not pending
    return {
        "confidence": confidence,
        "quality": quality,
        "completeness": completeness,
        "factual_support": factual,
        "citation_validity": citation_validity,
        "source_diversity": source_diversity,
        "freshness": freshness,
        "unresolved_contradictions": unresolved,
        "sufficient": sufficient,
    }


def contradiction_report(scene: dict) -> list[dict]:
    """One row for every detected contradiction, settled or still open."""
    claims = {item["id"]: item for item in scene.get("claims") or []}
    rows = []
    for conflict in scene.get("conflicts") or []:
        left = claims.get(conflict.get("left_id"), {})
        right = claims.get(conflict.get("right_id"), {})
        rows.append(
            {
                "id": conflict.get("id", ""),
                "status": conflict.get("status", ""),
                "reason": conflict.get("reason", ""),
                "left": left.get("short", ""),
                "right": right.get("short", ""),
                "winner_id": conflict.get("winner_id", ""),
            }
        )
    return rows


def _gained(previous: dict, previous_metrics: dict, scene: dict | None, metrics: dict) -> bool:
    """True when this search added a claim, settled a conflict, or raised confidence."""
    if not scene:
        return False
    if _evidence_keys(scene) - _evidence_keys(previous):
        return True
    if _resolved_pending(previous, scene):
        return True
    return metrics["confidence"] > previous_metrics["confidence"]


def _evidence_keys(scene: dict) -> set[tuple]:
    keys = set()
    for claim in scene.get("claims") or []:
        provenance = claim.get("provenance") or {}
        filename = provenance.get("filename") or ""
        text = claim.get("text") or ""
        if filename:
            keys.add(
                (
                    "source",
                    filename,
                    provenance.get("checksum") or "",
                    provenance.get("start_char"),
                    provenance.get("end_char"),
                    text,
                )
            )
        elif text:
            keys.add(("text", text))
    return keys


def _resolved_pending(previous: dict, scene: dict) -> bool:
    current = {item.get("id"): item for item in scene.get("conflicts") or []}
    for conflict in previous.get("conflicts") or []:
        if conflict.get("status") != "pending":
            continue
        later = current.get(conflict.get("id"))
        if later is not None and later.get("status") != "pending":
            return True
    return False


def _support(scene: dict, claims: dict[str, dict]) -> list[dict]:
    found: list[str] = []
    for marker in scene.get("markers") or []:
        claim_id = marker.get("claim_id")
        if claim_id not in claims or claim_id in found:
            continue
        found.append(claim_id)
        for upstream in claims[claim_id].get("upstream") or []:
            if upstream in claims and upstream not in found:
                found.append(upstream)
    return [claims[item] for item in found]


def _role(claim: dict) -> str:
    return str(claim.get("agent_id") or "").rsplit(":", 1)[-1]


def _tokens(text: str) -> int:
    words = [word for word in text.split() if word]
    return len(words)


def _mean(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def _empty_metrics() -> dict:
    return {
        "confidence": 0.0,
        "quality": 0.0,
        "completeness": 0.0,
        "factual_support": 0.0,
        "citation_validity": 0.0,
        "source_diversity": 0.0,
        "freshness": 1.0,
        "unresolved_contradictions": 0.0,
        "sufficient": False,
    }
