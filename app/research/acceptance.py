"""Acceptance checks for one finished discovery path.

Six criteria decide whether the path is good enough to keep: the conclusion
cites reliable sources, the hypothesis can be tested, agents finish their
tasks, independent agents have a shorter schedule than a series of searches,
conflicts are identified and settled, and the experiment setup can be traced.
"""

from __future__ import annotations

from app.research.learning import WASTE

_CRITERIA = (
    "evidence",
    "hypothesis",
    "reliability",
    "coordination",
    "conflicts",
    "reproducibility",
)


def measure_acceptance(report: dict, discovery: dict) -> dict:
    """Score the path against the six acceptance criteria."""
    criteria = [
        _evidence(report, discovery),
        _hypothesis(report, discovery),
        _reliability(report, discovery),
        _coordination(report),
        _conflicts(report),
        _reproducibility(discovery),
    ]
    passed = sum(1 for item in criteria if item["result"] == "pass")
    met = passed == len(_CRITERIA)
    if met:
        summary = "The path meets all 6 acceptance criteria."
    else:
        summary = f"The path meets {passed} of 6 acceptance criteria."
    return {"passed": passed, "total": len(_CRITERIA), "met": met, "summary": summary, "criteria": criteria}


def _evidence(report: dict, discovery: dict) -> dict:
    metrics = report.get("metrics") or {}
    checksums = _checksums(discovery)
    source = _mean(
        [
            float(metrics.get("factual_support") or 0),
            float(metrics.get("citation_validity") or 0),
            1.0 if checksums else 0.0,
        ]
    )
    verdict = (discovery.get("conclusion") or {}).get("verdict")
    if verdict != "concluded":
        return _criterion(
            "evidence",
            "fail",
            source,
            "The team did not accept a conclusion, so the sources do not settle the claim.",
        )
    if source >= 0.67 and checksums:
        return _criterion(
            "evidence",
            "pass",
            source,
            f"The conclusion cites checksummed sources. Evidence score {source:.2f}.",
        )
    return _criterion(
        "evidence",
        "partial",
        source,
        f"The conclusion is accepted with a source score of {source:.2f}.",
    )


def _hypothesis(report: dict, discovery: dict) -> dict:
    statement = str((discovery.get("hypothesis") or {}).get("statement") or "")
    question = str(report.get("question") or "").strip()
    status = _phase(discovery, "experiment").get("status") or "not_run"
    testable = status in {"pass", "fail", "partial"}
    relevant = bool(question) and question.casefold() in statement.casefold()
    if testable and relevant:
        return _criterion(
            "hypothesis",
            "pass",
            1.0,
            f"The hypothesis is relevant to the question and the experiment result is {status}.",
        )
    if testable:
        return _criterion(
            "hypothesis",
            "partial",
            0.5,
            f"The experiment result is {status}. The hypothesis does not name the question.",
        )
    return _criterion(
        "hypothesis",
        "fail",
        0.0,
        "The hypothesis was not turned into an experiment.",
    )


def _reliability(report: dict, discovery: dict) -> dict:
    outcomes = []
    for record in _phase(discovery, "orchestration").get("records") or []:
        state = str(record.get("detail") or "").split(" · ", 1)[0]
        outcomes.append(state == "done")
    for step in report.get("steps") or []:
        if step.get("action") != "search":
            continue
        outcomes.append(step.get("status") == "ok" and step.get("detail") != WASTE)
    if not outcomes:
        return _criterion("reliability", "fail", 0.0, "No agent task was assigned.")
    correct = sum(1 for item in outcomes if item)
    score = _mean_ratio(correct, len(outcomes))
    reason = f"{correct} of {len(outcomes)} assigned tasks completed correctly."
    if score == 1:
        result = "pass"
    elif score >= 0.5:
        result = "partial"
    else:
        result = "fail"
    return _criterion("reliability", result, score, reason)


def _coordination(report: dict) -> dict:
    agents = _search_agents(report)
    series = sum(int(agent.get("searches") or 0) for agent in agents)
    critical = max((int(agent.get("searches") or 0) for agent in agents), default=0)
    if len(agents) < 2 or series < 2 or critical >= series:
        count = series or len(agents)
        return _criterion(
            "coordination",
            "fail",
            0.0,
            f"One partition ran {_searches(count)}. The critical path is the whole series.",
        )
    score = round(1 - (critical / series), 2)
    speedup = series / critical
    elapsed = report.get("elapsed_seconds")
    clock = f" Observed run time {float(elapsed):.2f}s." if elapsed is not None else ""
    return _criterion(
        "coordination",
        "pass",
        score,
        (
            f"Critical path {_searches(critical)} across {len(agents)} agents, "
            f"against {_searches(series)} in series. The independent schedule is {speedup:.2f} times shorter.{clock}"
        ),
    )


def _conflicts(report: dict) -> dict:
    rows = list(report.get("contradictions") or [])
    pending = [row for row in rows if row.get("status") == "pending"]
    if not rows:
        return _criterion("conflicts", "pass", 1.0, "No conflicting claims were on this path.")
    if not pending:
        return _criterion(
            "conflicts",
            "pass",
            1.0,
            f"{len(rows)} conflicting claims were identified and settled.",
        )
    left = pending[0].get("left") or "one claim"
    right = pending[0].get("right") or "another claim"
    return _criterion(
        "conflicts",
        "fail",
        round(1 - (len(pending) / len(rows)), 2),
        f"{len(pending)} of {len(rows)} conflicting claims are still open: {left} versus {right}.",
    )


def _reproducibility(discovery: dict) -> dict:
    phase = _phase(discovery, "reproducibility")
    fingerprint = _record(phase, "fingerprint")
    prompt = _record(phase, "prompt")
    parameters = _record(phase, "parameters")
    result = _record(phase, "result")
    ready = all((fingerprint, prompt, parameters, result))
    if phase.get("status") == "reproducible" and ready:
        return _criterion(
            "reproducibility",
            "pass",
            1.0,
            f"Run {fingerprint} stores the dataset, prompt, tools, parameters, and result.",
        )
    if ready:
        return _criterion(
            "reproducibility",
            "partial",
            0.5,
            f"Run {fingerprint} stores the prompt, tools, parameters, and result. No passage checksum is stored.",
        )
    return _criterion(
        "reproducibility",
        "fail",
        0.0,
        "The experiment setup is missing a prompt, parameters, or result.",
    )


def _search_agents(report: dict) -> list[dict]:
    agents = list((report.get("collaboration") or {}).get("agents") or [])
    if agents:
        return agents
    stats: dict[str, dict] = {}
    order: list[str] = []
    for step in report.get("steps") or []:
        if step.get("action") != "search":
            continue
        name = str(step.get("agent") or "Lead agent")
        row = stats.get(name)
        if row is None:
            row = {"name": name, "searches": 0, "useful": 0}
            stats[name] = row
            order.append(name)
        row["searches"] += 1
        if step.get("status") == "ok" and step.get("detail") != WASTE:
            row["useful"] += 1
    return [stats[name] for name in order]


def _checksums(discovery: dict) -> bool:
    dataset = _record(_phase(discovery, "reproducibility"), "dataset")
    return "no passage checksum" not in dataset.casefold() and "(" in dataset


def _phase(discovery: dict, component: str) -> dict:
    for phase in discovery.get("phases") or []:
        if phase.get("component") == component:
            return phase
    return {}


def _record(phase: dict, label: str) -> str:
    for record in phase.get("records") or []:
        if record.get("label") == label:
            return str(record.get("detail") or "")
    return ""


def _searches(count: int) -> str:
    noun = "search" if count == 1 else "searches"
    return f"{count} {noun}"


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def _mean_ratio(correct: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(correct / total, 2)


def _criterion(criterion: str, result: str, score: float, reason: str) -> dict:
    return {"criterion": criterion, "result": result, "score": score, "reason": reason}
