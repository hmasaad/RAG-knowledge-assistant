"""Run a research question until the evidence is sufficient or a budget stops it.

Each pass searches, scores the scene, and either stops or investigates the
highest-value follow-up. A failed search is retried once. A second failure is
skipped so the loop can continue with another question.
"""

from __future__ import annotations

import time
from collections.abc import Callable

REFUSAL = "I don't have enough evidence to answer."
_ROLES = ("news", "social", "aggregator")


def run_research_loop(
    question: str,
    search: Callable[[str, str], dict],
    *,
    asset: str = "NVDA",
    max_searches: int = 4,
    max_tokens: int = 2500,
    max_seconds: float = 15.0,
    clock: Callable[[], float] | None = None,
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

    def elapsed() -> float:
        return round(now() - started, 3)

    def record(action: str, status: str, detail: str, *, query: str = "", strategy: str = "") -> None:
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
            }
        )

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
        try:
            result = search(current, current_asset)
        except Exception as exc:
            searches_used += 1
            tried[current] = attempts + 1
            message = str(exc).strip() or "Search failed."
            can_retry = attempts < 1 and searches_used < max_searches and now() < deadline
            if can_retry:
                record("search", "recovered", f"Recovered from a failed search: {message}")
                continue
            record("search", "failed", message)
            nxt = _next_question(scene, tried)
            if nxt is None:
                stop_reason = "failed"
                record("stop", "stopped", "A research step failed and there is no alternate question.")
                break
            current = nxt
            record(
                "investigate",
                "skipped",
                "Skipped the failed step and continued with the next question.",
                query=current,
            )
            continue

        searches_used += 1
        tried[current] = attempts + 1
        answer = str(result.get("answer") or "")
        scene = result.get("scene") or {}
        current_asset = str(result.get("asset") or current_asset)
        tokens_used += _tokens(current) + _tokens(answer)
        metrics = score_scene(scene)
        plan_items = (scene.get("plan") or {}).get("items") or []
        strategy = plan_items[0]["strategy"] if plan_items else ""
        record("search", "ok", answer[:240], strategy=strategy)

        if tokens_used >= max_tokens:
            stop_reason = "token"
            record("stop", "stopped", "Token budget exhausted.")
            break
        if metrics["sufficient"]:
            stop_reason = "sufficient"
            record("stop", "stopped", "Evidence is sufficient.")
            break
        nxt = _next_question(scene, tried)
        if nxt is None:
            stop_reason = "unresolved"
            record(
                "stop",
                "stopped",
                "Open gaps remain, and another search would repeat a question already tried.",
            )
            break
        if searches_used >= max_searches:
            stop_reason = "search"
            record("stop", "stopped", "Search budget exhausted before the next investigation.")
            break
        current = nxt
        record(
            "investigate",
            "ok",
            "Investigating the highest-value follow-up.",
            query=current,
            strategy=strategy,
        )

    return {
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


def _next_question(scene: dict | None, tried: dict[str, int]) -> str | None:
    if not scene:
        return None
    for item in (scene.get("plan") or {}).get("items") or []:
        question = str(item.get("question") or "").strip()
        if question and question not in tried:
            return question
    return None


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
