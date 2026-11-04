"""Deeper evaluation of a research run, using the logged decisions already stored.

The bandit, retrieval model, offline estimate, and causal contrast are fit on
past searches. A safety gate withholds any change that is untried or that the
offline estimate says would be worse than the logged policy. The notes are the
audit trail for that decision.
"""

from __future__ import annotations

import math

from app.research.learning import ResearchMemory, WASTE, observation_context, strategy_observations

_ARMS = ("corroborate", "authority", "broaden", "entity", "decompose")
_TOP_K = (4, 6, 8)


def analyze_run(report: dict, memory: ResearchMemory | None = None) -> dict:
    """Eight notes: failure, bandit, retrieval, offline policy, cause, consensus, safety, governance."""
    strategy_rows = list(memory.strategy_log()) if memory is not None else []
    strategy_rows.extend(strategy_observations(report))
    retrieval_rows = list(memory.retrieval_log()) if memory is not None else []
    retrieval_rows.extend(_retrieval_rows(report))
    context = observation_context(report)
    failure = _failure(report)
    bandit = _bandit(strategy_rows, context)
    retrieval = _bayes_retrieval(retrieval_rows, bandit["arm"] or "corroborate")
    offline = _offline_policy(strategy_rows, bandit["arm"])
    causal = _causal(strategy_rows, bandit["arm"], context)
    consensus = _consensus(report)
    safety = _safety(bandit, offline)
    governance = _governance(safety, bandit, retrieval)
    notes = [failure, bandit_note(bandit), retrieval, offline, causal, consensus, safety, governance]
    return {"summary": governance["reason"], "notes": notes}


def bandit_note(bandit: dict) -> dict:
    return {"component": "bandit", "decision": bandit["arm"] or "hold", "reason": bandit["summary"]}


def _failure(report: dict) -> dict:
    stop = report.get("stop_reason") or ""
    steps = report.get("steps") or []
    failed = next(
        (
            step
            for step in steps
            if step.get("action") == "search" and (step.get("status") == "failed" or step.get("detail") == WASTE)
        ),
        None,
    )
    if stop == "sufficient":
        label, detail = "none", "The trajectory reached sufficient evidence."
    elif stop == "failed" or (failed and failed.get("status") == "failed"):
        label, detail = "tool", "A research step failed."
    elif stop in {"search", "token", "time"}:
        label, detail = "budget", f"The {stop} budget ended the trajectory."
    elif stop == "policy":
        label, detail = "policy", "The policy stopped a strategy that rarely adds evidence."
    elif stop == "unresolved":
        label, detail = "gap", "Open gaps remained after the questions the loop could try."
    elif failed or stop == "unchanged":
        label, detail = "waste", "A search added no new evidence."
    else:
        label, detail = "none", "No failure class matched this trajectory."
    agent = str((failed or {}).get("agent") or "")
    if agent:
        detail = f"{agent}: {detail}"
    return {"component": "failure", "decision": label, "reason": detail}


def _bandit(rows: list[dict], context: str) -> dict:
    named = [row for row in rows if row.get("strategy") not in {"", "initial"}]
    scoped = [row for row in named if row.get("context") == context]
    pool = scoped if len(scoped) >= 2 else named
    total = len(pool)
    if total == 0:
        return {
            "arm": "",
            "ucb": 0.0,
            "runs": 0,
            "context": context,
            "summary": "No logged searches, so the bandit does not change the planner.",
        }
    ranked = []
    for arm in _ARMS:
        matched = [row for row in pool if row.get("strategy") == arm]
        useful = sum(1 for row in matched if row.get("useful"))
        ranked.append((_ucb(useful, len(matched), total), useful, len(matched), arm))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[3]))
    ucb, useful, runs, arm = ranked[0]
    return {
        "arm": arm,
        "ucb": round(ucb, 2),
        "runs": runs,
        "context": context,
        "summary": (
            f"In context {context}, the bandit prefers {arm} "
            f"(UCB {ucb:.2f} from {useful} of {runs} searches)."
        ),
    }


def _bayes_retrieval(rows: list[dict], strategy: str) -> dict:
    pool = [row for row in rows if row.get("strategy") == strategy] or list(rows)
    scored = [_posterior(pool, top_k) for top_k in _TOP_K]
    proposed = max(scored, key=lambda item: (item["ucb"], -item["top_k"]))
    observed = [item for item in scored if item["runs"] > 0]
    if proposed["runs"] == 0 and observed:
        kept = max(observed, key=lambda item: (item["mean"], -item["top_k"]))
        reason = (
            f"Acquisition prefers top_k {proposed['top_k']} (UCB {proposed['ucb']:.2f}), which has no outcomes. "
            f"Safety keeps top_k {kept['top_k']}."
        )
        decision = "withhold"
    elif proposed["runs"] == 0:
        reason = f"No retrieval outcomes. Acquisition stays on the prior for top_k {proposed['top_k']}."
        decision = "hold"
    else:
        reason = (
            f"Acquisition selects top_k {proposed['top_k']} "
            f"(UCB {proposed['ucb']:.2f} from {proposed['runs']} {strategy} searches)."
        )
        decision = "select"
    return {"component": "bayesian_retrieval", "decision": decision, "reason": reason}


def _offline_policy(rows: list[dict], arm: str) -> dict:
    logged = [row for row in rows if row.get("strategy") not in {"", "initial"}]
    if len(logged) < 2 or not arm:
        return {
            "component": "offline_policy",
            "decision": "insufficient",
            "reason": "Not enough logged decisions for offline evaluation.",
        }
    logged_value = sum(1 for row in logged if row.get("useful")) / len(logged)
    model = _reward_model(logged)
    target_draws = []
    for row in logged:
        if row.get("strategy") == arm:
            target_draws.append(1.0 if row.get("useful") else 0.0)
        else:
            target_draws.append(model.get(arm, 0.5))
    target_value = sum(target_draws) / len(target_draws)
    return {
        "component": "offline_policy",
        "decision": "doubly_robust",
        "reason": (
            f"Doubly robust estimate for {arm} is {target_value:.2f}, "
            f"against a logged value of {logged_value:.2f} on {len(logged)} searches."
        ),
        "logged_value": round(logged_value, 2),
        "target_value": round(target_value, 2),
    }


def _causal(rows: list[dict], treatment: str, context: str) -> dict:
    named = [row for row in rows if row.get("strategy") not in {"", "initial"}]
    scoped = [row for row in named if row.get("context") == context]
    pool = scoped if len(scoped) >= 2 else named
    if not treatment:
        return {
            "component": "causal",
            "decision": "unidentified",
            "reason": "No treatment arm to contrast.",
        }
    treated = [row for row in pool if row.get("strategy") == treatment]
    others: dict[str, list[dict]] = {}
    for row in pool:
        strategy = str(row.get("strategy") or "")
        if strategy == treatment:
            continue
        others.setdefault(strategy, []).append(row)
    if not others or len(treated) < 2:
        return {
            "component": "causal",
            "decision": "unidentified",
            "reason": f"The effect of {treatment} is not identified yet.",
        }
    control_name, control_rows = max(others.items(), key=lambda item: (len(item[1]), item[0]))
    if len(control_rows) < 2:
        return {
            "component": "causal",
            "decision": "unidentified",
            "reason": f"The effect of {treatment} is not identified yet.",
        }
    treated_rate = sum(1 for row in treated if row.get("useful")) / len(treated)
    control_rate = sum(1 for row in control_rows if row.get("useful")) / len(control_rows)
    ate = treated_rate - control_rate
    return {
        "component": "causal",
        "decision": "identified",
        "reason": (
            f"In context {context}, {treatment} added evidence {ate:.2f} more often than "
            f"{control_name} ({len(treated)} versus {len(control_rows)} searches)."
        ),
    }


def _consensus(report: dict) -> dict:
    metrics = report.get("metrics") or {}
    conflicts = report.get("contradictions")
    if conflicts is None:
        conflicts = (report.get("scene") or {}).get("conflicts") or []
    pending = [item for item in conflicts if item.get("status") == "pending"]
    factual = float(metrics.get("factual_support") or 0) == 1.0
    votes = [
        {
            "agent": "Lead agent",
            "vote": "yes" if factual else "no",
            "reason": "The answer cites evidence." if factual else "The answer has no factual support.",
        },
        {
            "agent": "Corroboration agent",
            "vote": "no" if pending else "yes",
            "reason": "A contradiction is still unresolved." if pending else "No contradiction is waiting.",
        },
    ]
    present = {item.get("name") for item in ((report.get("collaboration") or {}).get("agents") or [])}
    if "News agent" in present:
        votes.append({"agent": "News agent", "vote": "yes", "reason": "A news search was part of the path."})
    if "Social agent" in present:
        votes.append({"agent": "Social agent", "vote": "yes", "reason": "A social search was part of the path."})
    dissent = [vote["agent"] for vote in votes if vote["vote"] == "no"]
    if dissent:
        return {
            "component": "consensus",
            "decision": "dissent",
            "reason": f"{', '.join(dissent)} dissented. The Aggregator does not treat the path as settled.",
        }
    return {
        "component": "consensus",
        "decision": "accept",
        "reason": "The agents that voted agree the path can stand.",
    }


def _safety(bandit: dict, offline: dict) -> dict:
    arm = bandit.get("arm") or ""
    runs = int(bandit.get("runs") or 0)
    if offline.get("decision") == "insufficient" or not arm:
        return {
            "component": "safety",
            "decision": "withhold",
            "reason": "Online adaptation withheld. There are not enough logged decisions to trust a new arm.",
        }
    if runs < 2:
        return {
            "component": "safety",
            "decision": "block",
            "reason": f"Online adaptation blocked. {arm} does not have two logged searches in this context.",
        }
    target = float(offline.get("target_value") or 0)
    logged = float(offline.get("logged_value") or 0)
    if target < logged - 0.05:
        return {
            "component": "safety",
            "decision": "block",
            "reason": (
                f"Online adaptation blocked. The offline estimate for {arm} ({target:.2f}) "
                f"is below the logged policy ({logged:.2f})."
            ),
        }
    return {
        "component": "safety",
        "decision": "allow",
        "reason": (
            f"Online adaptation allowed for {arm}. The offline estimate {target:.2f} "
            f"is at least the logged value {logged:.2f}."
        ),
    }


def _governance(safety: dict, bandit: dict, retrieval: dict) -> dict:
    return {
        "component": "governance",
        "decision": safety["decision"],
        "reason": f"{safety['reason']} Retrieval: {retrieval['reason']}",
    }


def _posterior(rows: list[dict], top_k: int) -> dict:
    matched = [row for row in rows if int(row.get("top_k") or 0) == top_k]
    runs = len(matched)
    useful = sum(1 for row in matched if row.get("useful"))
    if runs == 0:
        mean, std = 0.5, 0.5
    else:
        alpha = 1 + useful
        beta = 1 + (runs - useful)
        mean = alpha / (alpha + beta)
        variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        std = math.sqrt(variance)
    return {"top_k": top_k, "runs": runs, "mean": mean, "ucb": mean + std}


def _reward_model(rows: list[dict]) -> dict[str, float]:
    grouped: dict[str, list[bool]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("strategy") or ""), []).append(bool(row.get("useful")))
    return {
        strategy: (sum(1 for useful in outcomes if useful) / len(outcomes))
        for strategy, outcomes in grouped.items()
        if strategy and outcomes
    }


def _ucb(useful: int, runs: int, total: int) -> float:
    if runs <= 0:
        return 1.0
    return useful / runs + math.sqrt(2 * math.log(max(total, 2)) / runs)


def _retrieval_rows(report: dict) -> list[dict]:
    from app.research.learning import retrieval_observations

    return retrieval_observations(report)
