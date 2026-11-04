"""Score a finished research path and predict whether a run will fall short.

A path is the ordered searches, follow-ups, and stop recorded by the research
loop. Failure means the run does not reach sufficient evidence. When at least
three earlier runs shared the same signals, their fail rate is blended with
the signals from the run being judged.
"""

from __future__ import annotations

REFUSAL = "I don't have enough evidence to answer."
WASTE = "No new claim, settled conflict, or higher confidence."

_OUTCOME_SCORE = {
    "sufficient": 1.0,
    "unresolved": 0.5,
    "unchanged": 0.45,
    "search": 0.4,
    "token": 0.4,
    "time": 0.4,
    "failed": 0.0,
    "policy": 0.42,
}
_RISK = {
    "search_failed": 0.45,
    "search_wasted": 0.35,
    "confidence_stalled": 0.2,
    "budget_tight": 0.25,
    "refusal": 0.3,
    "unresolved_conflict": 0.15,
    "repeated_question": 0.25,
}
_REASON = {
    "search_failed": "A search failed and was not recovered.",
    "search_wasted": "A search added no new evidence.",
    "confidence_stalled": "Confidence did not rise after another search.",
    "budget_tight": "Most of the search budget is already used and confidence is still low.",
    "refusal": "The answer is still missing factual support.",
    "unresolved_conflict": "Contradictions are still unresolved.",
    "repeated_question": "The loop asked a question it had already tried.",
}


_DEFAULT_TOP_K = {
    "initial": 4,
    "corroborate": 6,
    "authority": 4,
    "broaden": 8,
    "entity": 4,
    "decompose": 4,
}


class ResearchMemory:
    """Outcomes of earlier runs, grouped by failure signals, strategy, and top_k."""

    def __init__(self) -> None:
        self._runs: list[tuple[tuple[str, ...], bool]] = []
        self._strategy_rows: list[dict] = []
        self._retrieval_rows: list[dict] = []
        self._scores: list[float] = []
        self._successes = 0

    def remember(self, report: dict) -> None:
        failed = not _succeeded(report)
        self._runs.append((tuple(failure_signals(report)), failed))
        self._strategy_rows.extend(strategy_observations(report))
        self._retrieval_rows.extend(retrieval_observations(report))
        evaluation = report.get("evaluation") or {}
        if "score" in evaluation:
            self._scores.append(float(evaluation["score"]))
            if evaluation.get("outcome") == "success":
                self._successes += 1

    def scores(self) -> list[float]:
        return list(self._scores)

    def success_count(self) -> int:
        return self._successes

    def strategy_log(self) -> list[dict]:
        return [dict(row) for row in self._strategy_rows]

    def retrieval_log(self) -> list[dict]:
        return [dict(row) for row in self._retrieval_rows]

    def fail_rate(self, signature: tuple[str, ...]) -> tuple[float | None, int]:
        matches = [failed for signals, failed in self._runs if signals == signature]
        if len(matches) < 3:
            return None, len(matches)
        return sum(1 for failed in matches if failed) / len(matches), len(matches)

    def strategy_stats(self, strategy: str) -> dict:
        rows = [row for row in self._strategy_rows if row["strategy"] == strategy]
        useful = sum(1 for row in rows if row["useful"])
        runs = len(rows)
        return {
            "strategy": strategy,
            "runs": runs,
            "useful": useful,
            "useful_rate": round(useful / runs, 2) if runs else 0.0,
        }

    def strategy_table(self) -> list[dict]:
        names: list[str] = []
        for row in self._strategy_rows:
            if row["strategy"] not in names:
                names.append(row["strategy"])
        table = [self.strategy_stats(name) for name in names]
        table.sort(key=lambda row: (-row["useful_rate"], -row["runs"], row["strategy"]))
        return table

    def best_top_k(self, strategy: str) -> dict | None:
        grouped: dict[int, list[bool]] = {}
        for row in self._retrieval_rows:
            if row["strategy"] != strategy:
                continue
            grouped.setdefault(int(row["top_k"]), []).append(bool(row["useful"]))
        candidates = []
        for top_k, outcomes in grouped.items():
            if len(outcomes) < 2:
                continue
            useful = sum(1 for outcome in outcomes if outcome)
            candidates.append((useful / len(outcomes), -top_k, top_k, useful, len(outcomes)))
        if not candidates:
            return None
        _rate, _order, top_k, useful, runs = max(candidates)
        return {"top_k": top_k, "useful": useful, "runs": runs, "useful_rate": round(useful / runs, 2)}


research_memory = ResearchMemory()


def learn_strategies(memory: ResearchMemory) -> dict:
    """Rank strategies by how often their searches added evidence."""
    table = memory.strategy_table()
    named = [row for row in table if row["strategy"] != "initial" and row["runs"] >= 2]
    best = named[0]["strategy"] if named else ""
    if not table:
        summary = "No finished searches yet. The loop still uses the planner's strategy order."
    elif best:
        summary = f"{best} is the strongest strategy. {_strategy_sentence(table)}"
    else:
        summary = f"Not enough repeated searches to name a strongest strategy. {_strategy_sentence(table)}"
    return {"strategies": table, "best": best, "summary": summary}


def choose_retrieval(strategy: str, memory: ResearchMemory | None = None) -> dict:
    """Pick how many passages to retrieve for this strategy."""
    name = strategy or "initial"
    base = _DEFAULT_TOP_K.get(name, 4)
    if memory is not None:
        best = memory.best_top_k(name)
        if best is not None:
            top_k = best["top_k"]
            return {
                "top_k": top_k,
                "strategy": name,
                "basis": "history",
                "reason": (
                    f"Retrieving {top_k} passages added evidence on {best['useful']} of {best['runs']} {name} searches."
                ),
            }
    return {
        "top_k": base,
        "strategy": name,
        "basis": "default",
        "reason": f"No retrieval history for {name}. Using {base} passages.",
    }


def choose_next_step(scene: dict | None, tried: dict[str, int], memory: ResearchMemory | None = None) -> dict:
    """Pick the next follow-up, or stop when history says it will not help."""
    default = _pending_follow_ups(scene, tried)
    chosen = list(default)
    if memory is not None and chosen:
        chosen = sorted(
            chosen,
            key=lambda item: (-_adjusted_value(item, memory), -(item.get("value") or 0), str(item.get("id") or "")),
        )
    if not chosen:
        return {
            "item": None,
            "abandon": False,
            "alternatives": [],
            "policy": {"action": "stop", "strategy": "", "reason": "No follow-up left to try."},
        }
    item = chosen[0]
    first = default[0]
    alternatives = [other for other in chosen if other.get("id") != item.get("id")]
    if _abandon(item, memory):
        stats = memory.strategy_stats(str(item.get("strategy") or "")) if memory is not None else {}
        return {
            "item": item,
            "abandon": True,
            "alternatives": list(chosen),
            "policy": {
                "action": "stop",
                "strategy": str(item.get("strategy") or ""),
                "reason": (
                    f"Past {item.get('strategy')} searches added evidence on "
                    f"{stats.get('useful', 0)} of {stats.get('runs', 0)} tries, so the loop stops."
                ),
            },
        }
    if item.get("id") != first.get("id"):
        policy = {
            "action": "reorder",
            "strategy": str(item.get("strategy") or ""),
            "reason": (
                f"Past {item.get('strategy')} searches added evidence more often than {first.get('strategy')}, "
                "so the loop prefers that question."
            ),
        }
    else:
        policy = {
            "action": "follow",
            "strategy": str(item.get("strategy") or ""),
            "reason": f"Follow the highest-value {item.get('strategy')} question.",
        }
    return {"item": item, "abandon": False, "alternatives": alternatives, "policy": policy}


_AGENTS = {
    "initial": ("lead", "Lead agent", "Answer the question from the current graph."),
    "corroborate": ("corroborator", "Corroboration agent", "Settle the open contradiction with an independent source."),
    "authority": ("news", "News agent", "Find a higher-authority source."),
    "broaden": ("librarian", "Library agent", "Search documents outside the current graph."),
    "entity": ("social", "Social agent", "Find the named entity in another source."),
    "decompose": ("analyst", "Analyst agent", "Cover the part of the question the current claims miss."),
}


def assign_agent(strategy: str) -> dict:
    """The specialist that carries out one research strategy."""
    agent_id, name, task = _AGENTS.get(strategy or "initial", _AGENTS["initial"])
    return {"id": agent_id, "name": name, "task": task, "strategy": strategy or "initial"}


def open_counterfactual(decision: dict, memory: ResearchMemory | None) -> dict:
    """Record the follow-up that was chosen and the ones left unasked."""
    item = decision.get("item") or {}
    return {
        "taken_strategy": str(item.get("strategy") or ""),
        "taken_question": str(item.get("question") or ""),
        "taken_value": float(item.get("value") or 0),
        "alternatives": [_estimate(other, memory) for other in decision.get("alternatives") or []],
    }


def close_counterfactual(fork: dict, *, useful: bool) -> dict:
    """Compare the search that ran with the expected value of the questions left unasked."""
    actual = fork["taken_value"] if useful else 0.0
    return _judge_fork(fork, actual, useful, stopped=False)


def stop_counterfactual(decision: dict, memory: ResearchMemory | None) -> dict:
    """The loop stopped instead of asking a follow-up."""
    fork = {
        "taken_strategy": "stop",
        "taken_question": "",
        "taken_value": 0.0,
        "alternatives": [_estimate(other, memory) for other in decision.get("alternatives") or []],
    }
    return _judge_fork(fork, 0.0, False, stopped=True)


def summarize_counterfactuals(forks: list[dict]) -> dict:
    if not forks:
        return {
            "verdict": "none",
            "summary": "The path had no alternative follow-up to compare.",
            "forks": [],
        }
    missed = [fork for fork in forks if fork["verdict"] == "missed"]
    if missed:
        verdict = "missed"
        summary = missed[0]["summary"]
    elif any(fork["verdict"] == "matched" for fork in forks):
        verdict = "matched"
        summary = next(fork["summary"] for fork in forks if fork["verdict"] == "matched")
    else:
        verdict = "none"
        summary = forks[-1]["summary"]
    return {"verdict": verdict, "summary": summary, "forks": forks}


def collaboration_report(steps: list[dict], stop_reason: str, stop_detail: str) -> dict:
    """Group searches by the specialist that ran them and list the handoffs."""
    stats: dict[str, dict] = {}
    order: list[str] = []
    handoffs = []
    previous = ""
    for step in steps:
        if step.get("action") == "search":
            name = str(step.get("agent") or "Lead agent")
            row = stats.get(name)
            if row is None:
                known = _agent_by_name(name)
                row = {
                    "id": known["id"],
                    "name": name,
                    "task": known["task"],
                    "searches": 0,
                    "useful": 0,
                }
                stats[name] = row
                order.append(name)
            row["searches"] += 1
            if step.get("status") == "ok" and step.get("detail") != WASTE:
                row["useful"] += 1
            previous = name
        elif step.get("action") == "investigate" and step.get("status") == "ok":
            target = assign_agent(str(step.get("strategy") or "")).get("name") or "Lead agent"
            source = previous or "Lead agent"
            handoffs.append(
                {
                    "from_agent": source,
                    "to_agent": target,
                    "strategy": str(step.get("strategy") or ""),
                    "question": str(step.get("question") or ""),
                }
            )
    agents = [stats[name] for name in order]
    if handoffs:
        last = handoffs[-1]
        summary = f"{last['from_agent']} handed the {last['strategy']} question to the {last['to_agent']}"
    elif agents:
        summary = f"The {agents[0]['name']} answered without handing work to another agent"
    else:
        summary = "No agent ran a search"
    if stop_detail:
        summary = f"{summary}. The Aggregator stopped the path: {stop_detail}"
    else:
        summary = f"{summary}."
    return {
        "agents": agents,
        "handoffs": handoffs,
        "aggregator": {
            "name": "Aggregator",
            "decision": stop_reason or "stopped",
            "reason": stop_detail or "Stopped.",
        },
        "summary": summary,
    }


def benchmark_run(memory: ResearchMemory | None, report: dict, counterfactual: dict | None = None) -> dict:
    """Compare this path score with the scores of earlier runs."""
    prior = memory.scores() if memory is not None else []
    prior_successes = memory.success_count() if memory is not None else 0
    latest = float((report.get("evaluation") or {}).get("score") or 0)
    succeeded = (report.get("evaluation") or {}).get("outcome") == "success"
    runs = len(prior) + 1
    mean_prior = sum(prior) / len(prior) if prior else latest
    delta = round(latest - mean_prior, 2) if prior else 0.0
    mean_score = round((sum(prior) + latest) / runs, 2)
    success_rate = round((prior_successes + (1 if succeeded else 0)) / runs, 2)
    if not prior:
        trend = "baseline"
        summary = f"First benchmark. Path score {latest:.2f}."
    elif delta > 0.02:
        trend = "improving"
        summary = (
            f"Score {latest:.2f} is {delta:.2f} above the earlier mean of {mean_prior:.2f} "
            f"across {len(prior)} runs."
        )
    elif delta < -0.02:
        trend = "regressing"
        summary = (
            f"Score {latest:.2f} is {abs(delta):.2f} below the earlier mean of {mean_prior:.2f} "
            f"across {len(prior)} runs."
        )
    else:
        trend = "steady"
        summary = f"Score {latest:.2f} is in line with the earlier mean of {mean_prior:.2f}."
    return {
        "runs": runs,
        "latest_score": round(latest, 2),
        "mean_score": mean_score,
        "delta": delta,
        "success_rate": success_rate,
        "trend": trend,
        "summary": summary,
        "improvement": _improvement(trend, counterfactual or {}),
    }


def _estimate(item: dict, memory: ResearchMemory | None) -> dict:
    strategy = str(item.get("strategy") or "")
    value = float(item.get("value") or 0)
    question = str(item.get("question") or "")
    if memory is not None:
        stats = memory.strategy_stats(strategy)
        if stats["runs"] >= 2:
            return {
                "strategy": strategy,
                "question": question,
                "expected": round(value * stats["useful_rate"], 2),
                "basis": f"{stats['useful']} of {stats['runs']} past searches added evidence",
            }
    return {
        "strategy": strategy,
        "question": question,
        "expected": round(value * 0.5, 2),
        "basis": "half the planner value, with no matching history",
    }


def _judge_fork(fork: dict, actual: float, useful: bool, *, stopped: bool) -> dict:
    alternatives = list(fork["alternatives"])
    best = max(alternatives, key=lambda row: row["expected"]) if alternatives else None
    if best is None:
        verdict = "none"
        summary = "No other follow-up was available at this decision."
    elif best["expected"] > actual + 0.05:
        verdict = "missed"
        skipped = _skipped_label(best, fork["taken_strategy"])
        if stopped:
            summary = f"Stopping left {skipped} unasked. Its expected value was {best['expected']:.2f}."
        elif useful:
            summary = (
                f"The {fork['taken_strategy']} search added evidence, but {skipped} "
                f"had a higher expected value ({best['expected']:.2f})."
            )
        else:
            summary = (
                f"The {fork['taken_strategy']} search added nothing. {skipped[0].upper()}{skipped[1:]} "
                f"had an expected value of {best['expected']:.2f}."
            )
    elif stopped:
        verdict = "matched"
        summary = "Stopping was the better decision. The remaining questions have a low expected value."
    else:
        verdict = "matched"
        summary = f"The chosen {fork['taken_strategy']} question beat the other follow-ups that were available."
    return {
        "taken_strategy": fork["taken_strategy"],
        "taken_question": fork["taken_question"],
        "useful": useful,
        "actual": round(actual, 2),
        "verdict": verdict,
        "summary": summary,
        "alternatives": alternatives,
    }


def _skipped_label(best: dict, taken: str) -> str:
    question = str(best.get("question") or "").strip()
    strategy = str(best.get("strategy") or "")
    if strategy and strategy != taken:
        return f"the skipped {strategy} question"
    if len(question) > 72:
        question = question[:69].rstrip() + "..."
    if question:
        return f"the skipped question “{question}”"
    return "a skipped follow-up"


def _improvement(trend: str, counterfactual: dict) -> str:
    if trend == "baseline":
        return "Keep collecting runs before changing the policy."
    if trend == "improving":
        return "Keep the current policy. The latest path scored above the benchmark."
    if trend == "steady":
        return "The score is steady. Keep collecting runs before changing the policy."
    missed = next((fork for fork in counterfactual.get("forks") or [] if fork.get("verdict") == "missed"), None)
    if missed:
        alternatives = missed.get("alternatives") or []
        best = max(alternatives, key=lambda row: row.get("expected") or 0) if alternatives else None
        if best and best.get("strategy"):
            return (
                f"Prefer {best['strategy']} next time. This run scored below the benchmark "
                "and left that question unasked."
            )
    return "The latest path scored below the benchmark. Prefer the strategy with the higher historical hit rate."


def _agent_by_name(name: str) -> dict:
    for strategy, (agent_id, agent_name, task) in _AGENTS.items():
        if agent_name == name:
            return {"id": agent_id, "name": agent_name, "task": task, "strategy": strategy}
    return {"id": "lead", "name": name, "task": "Answer the question from the current graph.", "strategy": "initial"}


def observation_context(report: dict) -> str:
    """Bucket a run by whether a conflict is open and how confident the answer is."""
    confidence = float((report.get("metrics") or {}).get("confidence") or 0)
    if confidence < 0.5:
        band = "low"
    elif confidence < 0.75:
        band = "mid"
    else:
        band = "high"
    conflicts = report.get("contradictions")
    if conflicts is None:
        conflicts = (report.get("scene") or {}).get("conflicts") or []
    pending = any(item.get("status") == "pending" for item in conflicts)
    return f"{'conflict' if pending else 'clear'}:{band}"


def strategy_observations(report: dict) -> list[dict]:
    """Pair each search with the strategy that sent it."""
    pending = "initial"
    rows = []
    for step in report.get("steps") or []:
        if step.get("action") == "investigate":
            pending = str(step.get("strategy") or pending or "initial")
            continue
        if step.get("action") != "search":
            continue
        useful = step.get("status") == "ok" and step.get("detail") != WASTE
        rows.append({"strategy": pending or "initial", "useful": useful, "context": observation_context(report)})
    return rows


def retrieval_observations(report: dict) -> list[dict]:
    pending = "initial"
    rows = []
    for step in report.get("steps") or []:
        if step.get("action") == "investigate":
            pending = str(step.get("strategy") or pending or "initial")
            continue
        if step.get("action") != "search":
            continue
        top_k = int(step.get("top_k") or 0)
        if not top_k:
            continue
        useful = step.get("status") == "ok" and step.get("detail") != WASTE
        rows.append({"strategy": pending or "initial", "top_k": top_k, "useful": useful})
    return rows


def _pending_follow_ups(scene: dict | None, tried: dict[str, int]) -> list[dict]:
    pending = []
    for item in ((scene or {}).get("plan") or {}).get("items") or []:
        question = str(item.get("question") or "").strip()
        if question and question not in tried:
            pending.append(item)
    return pending


def _adjusted_value(item: dict, memory: ResearchMemory) -> float:
    value = float(item.get("value") or 0)
    stats = memory.strategy_stats(str(item.get("strategy") or ""))
    if stats["runs"] < 3:
        return value
    smoothed = (stats["useful"] + 1) / (stats["runs"] + 2)
    weight = min(1.5, max(0.5, smoothed / 0.5))
    return value * weight


def _abandon(item: dict, memory: ResearchMemory | None) -> bool:
    if memory is None:
        return False
    stats = memory.strategy_stats(str(item.get("strategy") or ""))
    return stats["runs"] >= 3 and stats["useful"] / stats["runs"] < 0.25


def _strategy_sentence(table: list[dict]) -> str:
    parts = [
        f"{row['strategy']} added evidence on {row['useful']} of {row['runs']} searches" for row in table
    ]
    return ". ".join(parts) + "."


def evaluate_trajectory(report: dict) -> dict:
    """Score the complete execution path and mark each step."""
    decisions = [_decision(index, step) for index, step in enumerate(report.get("steps") or [])]
    searches = [item for item in decisions if item["action"] == "search"]
    useful = [item for item in searches if item["verdict"] == "useful"]
    gain_rate = round(len(useful) / len(searches), 2) if searches else 0.0
    outcome = _outcome(report)
    outcome_score = _OUTCOME_SCORE.get(report.get("stop_reason") or "", 0.3)
    confidence = float((report.get("metrics") or {}).get("confidence") or 0.0)
    wasted = len(searches) - len(useful)
    budget = max(1, int(report.get("search_budget") or 1))
    efficiency = 1.0 if outcome == "success" else round(max(0.0, 1 - wasted / budget), 2)
    score = round((gain_rate + outcome_score + confidence + efficiency) / 4, 2)
    findings = _findings(report, decisions, wasted)
    summary = _summary(report, outcome, score)
    return {
        "score": score,
        "outcome": outcome,
        "summary": summary,
        "gain_rate": gain_rate,
        "efficiency": efficiency,
        "decisions": decisions,
        "findings": findings,
    }


def predict_failure(report: dict, memory: ResearchMemory | None = None) -> dict:
    """Estimate the chance this run stops before the evidence is sufficient.

    ``warned_at_search`` is the first search whose signals cross the fail line.
    The returned probability uses the finished report, blended with matching
    history when that history exists.
    """
    warned_at = None
    prefix: list[dict] = []
    seen = 0
    for step in report.get("steps") or []:
        prefix.append(step)
        if step.get("action") != "search":
            continue
        seen += 1
        assessment = _assess(_prefix_report(report, prefix, seen), memory)
        if assessment["label"] == "likely_fail" and warned_at is None:
            warned_at = seen
    final = _assess(report, memory)
    if _succeeded(report):
        final["label"] = "likely_succeed"
        final["probability"] = min(final["probability"], 0.2)
        final["reasons"] = ["The finished run reached sufficient evidence.", *final["reasons"][:3]]
    final["at_search"] = int(report.get("searches_used") or seen)
    final["warned_at_search"] = warned_at
    return final


def failure_signals(report: dict) -> list[str]:
    """Risk flags readable from the steps and metrics gathered so far."""
    steps = report.get("steps") or []
    searches = [step for step in steps if step.get("action") == "search"]
    metrics = report.get("metrics") or {}
    signals: list[str] = []
    if any(step.get("status") == "failed" for step in searches):
        signals.append("search_failed")
    if any(step.get("detail") == WASTE for step in searches) or report.get("stop_reason") == "unchanged":
        signals.append("search_wasted")
    confidences = [float(step.get("confidence") or 0) for step in searches]
    if len(confidences) >= 2 and confidences[-1] <= confidences[-2]:
        signals.append("confidence_stalled")
    budget = max(1, int(report.get("search_budget") or 1))
    used = int(report.get("searches_used") or len(searches))
    confidence = float(metrics.get("confidence") or 0)
    if used / budget >= 0.75 and not report.get("sufficient") and confidence < 0.7:
        signals.append("budget_tight")
    answer = str(report.get("answer") or "")
    if answer == REFUSAL or float(metrics.get("factual_support") or 0) == 0:
        signals.append("refusal")
    if float(metrics.get("unresolved_contradictions") or 0) >= 0.5 and not report.get("sufficient"):
        signals.append("unresolved_conflict")
    questions = [str(step.get("question") or "") for step in searches if step.get("question")]
    if len(questions) != len(set(questions)):
        signals.append("repeated_question")
    return signals


def _assess(report: dict, memory: ResearchMemory | None) -> dict:
    signals = failure_signals(report)
    probability = min(0.95, sum(_RISK[name] for name in signals))
    if not signals:
        confidence = float((report.get("metrics") or {}).get("confidence") or 0)
        probability = 0.2 if confidence < 0.5 else 0.08
    probability = round(probability, 2)
    basis = "signals"
    history_runs = 0
    reasons = [_REASON[name] for name in signals]
    if memory is not None:
        rate, history_runs = memory.fail_rate(tuple(signals))
        if rate is not None:
            probability = round(min(0.95, 0.5 * probability + 0.5 * rate), 2)
            basis = "history"
            percent = round(rate * 100)
            reasons.append(f"Past runs with these signals failed {percent}% of the time ({history_runs} runs).")
    return {
        "label": _label(probability),
        "probability": probability,
        "at_search": int(report.get("searches_used") or 0),
        "warned_at_search": None,
        "reasons": reasons,
        "basis": basis,
        "history_runs": history_runs,
    }


def _prefix_report(report: dict, steps: list[dict], seen: int) -> dict:
    searches = [step for step in steps if step.get("action") == "search"]
    useful = any(
        step.get("status") == "ok" and step.get("detail") not in {WASTE, REFUSAL} for step in searches
    )
    confidence = float(searches[-1].get("confidence") or 0) if searches else 0.0
    return {
        "steps": steps,
        "searches_used": seen,
        "search_budget": report.get("search_budget") or 1,
        "answer": "",
        "stop_reason": "",
        "sufficient": False,
        "metrics": {
            "confidence": confidence,
            "factual_support": 1.0 if useful else 0.0,
            "unresolved_contradictions": 0.0,
        },
    }


def _decision(index: int, step: dict) -> dict:
    verdict, reason = _verdict(step)
    return {
        "index": index,
        "action": str(step.get("action") or ""),
        "question": str(step.get("question") or ""),
        "strategy": str(step.get("strategy") or ""),
        "verdict": verdict,
        "reason": reason,
    }


def _verdict(step: dict) -> tuple[str, str]:
    action = step.get("action")
    status = step.get("status")
    detail = str(step.get("detail") or "")
    if action == "search" and status == "failed":
        return "failed", detail or "Search failed."
    if action == "search" and status == "recovered":
        return "recovered", detail or "Recovered from a failed search."
    if action == "search" and detail == WASTE:
        return "waste", detail
    if action == "search":
        return "useful", "The search added evidence."
    if action == "investigate" and "outside" in detail:
        return "outside", detail
    if action == "investigate" and status == "skipped":
        return "skipped", detail
    if action == "investigate":
        return "continued", detail or "Investigating the highest-value follow-up."
    if action == "stop":
        return "stopped", detail or "Stopped."
    return str(status or "recorded"), detail


def _findings(report: dict, decisions: list[dict], wasted: int) -> list[dict]:
    findings = []
    if wasted:
        noun = "search" if wasted == 1 else "searches"
        findings.append({"kind": "waste", "detail": f"{wasted} {noun} added no evidence."})
    stop = report.get("stop_reason")
    if stop in {"search", "token", "time"}:
        findings.append({"kind": "budget", "detail": f"The {stop} budget stopped the path."})
    if stop == "failed":
        findings.append({"kind": "failure", "detail": "A research step failed and the path stopped."})
    if stop == "policy":
        findings.append({"kind": "policy", "detail": "Historical outcomes stopped a strategy that rarely adds evidence."})
    if stop == "unresolved":
        findings.append({"kind": "gap", "detail": "Open gaps remained after the questions the loop could try."})
    if any(item["verdict"] == "recovered" for item in decisions):
        findings.append({"kind": "recovery", "detail": "A failed search was retried and the path continued."})
    if any(item["verdict"] == "outside" for item in decisions):
        findings.append({"kind": "outside", "detail": "A follow-up searched documents outside the current graph."})
    unresolved = float((report.get("metrics") or {}).get("unresolved_contradictions") or 0)
    if unresolved > 0:
        findings.append({"kind": "conflict", "detail": "The path ended with unresolved contradictions."})
    return findings


def _summary(report: dict, outcome: str, score: float) -> str:
    used = int(report.get("searches_used") or 0)
    noun = "search" if used == 1 else "searches"
    if outcome == "success":
        return f"The path reached sufficient evidence in {used} {noun}. Score {score:.2f}."
    if outcome == "failed":
        return f"The path failed before it gathered sufficient evidence. Score {score:.2f}."
    reason = report.get("stop_reason") or "stopped"
    confidence = float((report.get("metrics") or {}).get("confidence") or 0)
    return f"The path stopped ({reason}) with confidence {confidence:.2f}. Score {score:.2f}."


def _outcome(report: dict) -> str:
    if _succeeded(report):
        return "success"
    factual = float((report.get("metrics") or {}).get("factual_support") or 0)
    if report.get("stop_reason") == "failed" or factual == 0:
        return "failed"
    return "partial"


def _succeeded(report: dict) -> bool:
    return bool(report.get("sufficient") or report.get("stop_reason") == "sufficient")


def _label(probability: float) -> str:
    if probability >= 0.55:
        return "likely_fail"
    if probability <= 0.25:
        return "likely_succeed"
    return "uncertain"
