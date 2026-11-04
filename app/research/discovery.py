"""A discovery team that investigates, challenges, tests, and concludes.

The Investigator reads the cited claims. The Challenger looks for open
contradictions and weak spots. The Hypothesis tester marks the claim supported,
contested, or untested. The Conclusion writes a finding only when the tester
supports it and the Challenger does not block it.

The same review also runs the discovery roadmap: register the team, rank
hypotheses, record the evidence each specialist gathered, expose assumptions,
run one experiment, take a vote, store a reproducibility manifest, and score
the result.
"""

from __future__ import annotations

import hashlib

from app.research.learning import WASTE
from app.research.society import observe_society

REFUSAL = "I don't have enough evidence to answer."
_SUPPORT_BAR = 0.5


def discover(report: dict) -> dict:
    """Review one finished research path and return the team's conclusion."""
    statement = _statement(report)
    reviews = [
        _investigator(report),
        _challenger(report),
        _tester(report, statement),
    ]
    conclusion = _conclude(statement, reviews)
    reviews.append(
        {
            "agent": "Conclusion",
            "stance": conclusion["verdict"],
            "finding": conclusion["text"],
        }
    )
    investigator, challenger, tester, _conclusion_review = reviews
    summary = (
        f"Investigator {investigator['stance']}. "
        f"Challenger {challenger['stance']}. "
        f"Hypothesis tester {tester['stance']}. "
        f"Conclusion {conclusion['verdict']}."
    )
    result = {
        "summary": summary,
        "hypothesis": {"statement": statement, "status": tester["stance"]},
        "reviews": reviews,
        "conclusion": conclusion,
        "phases": _roadmap(report, reviews, conclusion, statement),
    }
    result["society"] = observe_society(report, result)
    return result


def _investigator(report: dict) -> dict:
    shorts = _cited_shorts(report)
    if _grounded(report) and shorts:
        return {
            "agent": "Investigator",
            "stance": "found",
            "finding": f"Cited evidence: {_join(shorts)}.",
        }
    if _grounded(report):
        return {
            "agent": "Investigator",
            "stance": "found",
            "finding": "A cited answer was produced, without a named claim.",
        }
    return {
        "agent": "Investigator",
        "stance": "empty",
        "finding": "No cited answer was produced for this question.",
    }


def _challenger(report: dict) -> dict:
    blocks = []
    notes = []
    for row in _pending(report):
        left = row.get("left") or "one claim"
        right = row.get("right") or "another claim"
        blocks.append(f"Open contradiction between {left} and {right}.")
    metrics = report.get("metrics") or {}
    if not _grounded(report):
        blocks.append("The answer has no factual support.")
    elif float(metrics.get("confidence") or 0) < _SUPPORT_BAR:
        confidence = float(metrics.get("confidence") or 0)
        blocks.append(f"Confidence is {confidence:.2f}, below {_SUPPORT_BAR:.2f}.")
    if any(step.get("detail") == WASTE for step in report.get("steps") or []):
        notes.append("A later search added no new evidence.")
    if (report.get("counterfactual") or {}).get("verdict") == "missed":
        notes.append("A skipped question had a higher expected value.")
    if blocks:
        return {"agent": "Challenger", "stance": "challenge", "finding": " ".join(blocks + notes)}
    if notes:
        return {"agent": "Challenger", "stance": "caution", "finding": " ".join(notes)}
    return {
        "agent": "Challenger",
        "stance": "clear",
        "finding": "No contradiction is open, and the answer cites evidence.",
    }


def _tester(report: dict, statement: str) -> dict:
    if not _grounded(report):
        return {
            "agent": "Hypothesis tester",
            "stance": "untested",
            "finding": "The hypothesis was not tested because the answer is not grounded in cited claims.",
        }
    pending = _pending(report)
    if pending:
        return {
            "agent": "Hypothesis tester",
            "stance": "contest",
            "finding": f"The hypothesis is contested: {statement}",
        }
    confidence = float((report.get("metrics") or {}).get("confidence") or 0)
    if confidence < _SUPPORT_BAR:
        return {
            "agent": "Hypothesis tester",
            "stance": "untested",
            "finding": f"Confidence is {confidence:.2f}, so the hypothesis stays untested.",
        }
    return {
        "agent": "Hypothesis tester",
        "stance": "support",
        "finding": (
            f"The hypothesis passes. Confidence is {confidence:.2f} "
            "and no contradiction is still open."
        ),
    }


def _conclude(statement: str, reviews: list[dict]) -> dict:
    investigator, challenger, tester = reviews
    if investigator["stance"] == "empty" or tester["stance"] == "untested":
        reason = investigator["finding"] if investigator["stance"] == "empty" else tester["finding"]
        return {"verdict": "insufficient", "text": f"The team does not conclude. {reason}"}
    if challenger["stance"] == "challenge" or tester["stance"] == "contest":
        return {"verdict": "disputed", "text": f"The team does not conclude. {challenger['finding']}"}
    text = f"The team concludes: {statement}"
    if challenger["stance"] == "caution":
        text += f" The Challenger noted a limit: {challenger['finding']}"
    return {"verdict": "concluded", "text": text}


def _statement(report: dict) -> str:
    question = str(report.get("question") or "the question").strip() or "the question"
    shorts = _cited_shorts(report, limit=2)
    if _grounded(report) and shorts:
        return f"For “{question}”, the cited evidence supports {_join(shorts)}."
    answer = _answer(report)
    if answer:
        return f"For “{question}”, the cited evidence supports this answer: {answer}"
    return f"There is not enough cited evidence to state a hypothesis for “{question}”."


def _grounded(report: dict) -> bool:
    metrics = report.get("metrics") or {}
    if float(metrics.get("factual_support") or 0) < 1:
        return False
    return bool(_answer(report))


def _answer(report: dict) -> str:
    text = " ".join(str(report.get("answer") or "").split())
    if not text or text == REFUSAL:
        return ""
    if len(text) > 160:
        text = text[:157].rstrip() + "…"
    return text


def _pending(report: dict) -> list[dict]:
    return [row for row in report.get("contradictions") or [] if row.get("status") == "pending"]


def _cited_shorts(report: dict, limit: int = 3) -> list[str]:
    scene = report.get("scene") or {}
    claims = {claim.get("id"): claim for claim in scene.get("claims") or []}
    shorts: list[str] = []
    for marker in scene.get("markers") or []:
        claim = claims.get(marker.get("claim_id")) or {}
        short = " ".join(str(claim.get("short") or claim.get("text") or "").split())
        if len(short) > 80:
            short = short[:77].rstrip() + "…"
        if short and short not in shorts:
            shorts.append(short)
        if len(shorts) >= limit:
            break
    return shorts


def _join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _roadmap(report: dict, reviews: list[dict], conclusion: dict, statement: str) -> list[dict]:
    """Run the eight discovery stages and return them in roadmap order."""
    investigator, challenger, tester, _conclusion_review = reviews
    hypotheses = _hypothesis_phase(report, statement)
    experiment = _experiment_phase(report, statement, tester["stance"])
    debate = _debate_phase(report, challenger, tester["stance"])
    collaboration = _collaboration_phase(report)
    consensus = _consensus_phase(investigator, challenger, experiment, conclusion)
    reproducibility = _reproducibility_phase(report, conclusion, tester["stance"])
    evaluation = _evaluation_phase(report, experiment, consensus, reproducibility)
    orchestration = _orchestration_phase(report, investigator, experiment)
    return [
        orchestration,
        hypotheses,
        collaboration,
        debate,
        experiment,
        consensus,
        reproducibility,
        evaluation,
    ]


def _orchestration_phase(report: dict, investigator: dict, experiment: dict) -> dict:
    archivist = "done" if _dataset_rows(report) else "skipped"
    experimenter = "skipped" if experiment["status"] == "not_run" else "done"
    roster = [
        ("Investigator", "done" if investigator["stance"] == "found" else "skipped", "Read the cited claims."),
        ("Archivist", archivist, "Record the passages behind those claims."),
        ("Challenger", "done", "Challenge contradictions and weak assumptions."),
        ("Experimenter", experimenter, "Turn the top hypothesis into a test and run it."),
        ("Chair", "done", "Resolve the team's votes from the evidence."),
    ]
    records = [
        {"label": name, "detail": f"{state} · {task}"}
        for name, state, task in roster
    ]
    roster_text = ", ".join(f"{name} {state}" for name, state, _task in roster)
    status = "complete" if investigator["stance"] == "found" else "partial"
    return _phase(
        1,
        "orchestration",
        "Multi-Agent Orchestration Engine",
        status,
        f"Registered 5 agents: {roster_text}.",
        records,
    )


def _hypothesis_phase(report: dict, statement: str) -> dict:
    if not _grounded(report):
        return _phase(
            2,
            "hypotheses",
            "Hypothesis Generation Engine",
            "empty",
            "No cited evidence, so no hypothesis was ranked.",
            [{"label": "1", "detail": f"0.00 · {statement}"}],
        )
    pending_names = set()
    for row in _pending(report):
        pending_names.add(_normalize(str(row.get("left") or "")))
        pending_names.add(_normalize(str(row.get("right") or "")))
    lead_score = 0.4 if pending_names else 1.0
    ranked = [(lead_score, statement, 0)]
    seen = {_normalize(statement)}
    for short in _cited_shorts(report, limit=7):
        key = _normalize(short)
        if not key or key in seen:
            continue
        score = 0.4 if key in pending_names else 0.8
        ranked.append((score, short, 1))
        seen.add(key)
    ranked.sort(key=lambda item: (-item[0], item[2], item[1]))
    records = [
        {"label": str(index), "detail": f"{score:.2f} · {text}"}
        for index, (score, text, _order) in enumerate(ranked, start=1)
    ]
    return _phase(
        2,
        "hypotheses",
        "Hypothesis Generation Engine",
        "ranked",
        f"Ranked {len(ranked)} hypotheses. The top hypothesis scores {ranked[0][0]:.2f}.",
        records,
    )


def _collaboration_phase(report: dict) -> dict:
    agents = list((report.get("collaboration") or {}).get("agents") or [])
    if not agents:
        agents = _agents_from_steps(report)
    if not agents:
        return _phase(
            3,
            "collaboration",
            "Collaborative Research Engine",
            "empty",
            "No specialist gathered evidence.",
            [],
        )
    useful = sum(int(agent.get("useful") or 0) for agent in agents)
    searches = sum(int(agent.get("searches") or 0) for agent in agents)
    names = ", ".join(str(agent.get("name") or "Agent") for agent in agents)
    records = [
        {
            "label": str(agent.get("name") or "Agent"),
            "detail": (
                f"independent · {int(agent.get('useful') or 0)}/"
                f"{int(agent.get('searches') or 0)} searches added evidence"
            ),
        }
        for agent in agents
    ]
    status = "gathered" if useful else "partial"
    return _phase(
        3,
        "collaboration",
        "Collaborative Research Engine",
        status,
        (
            f"{len(agents)} agents gathered evidence independently: {names}. "
            f"{useful} of {searches} searches added evidence."
        ),
        records,
    )


def _debate_phase(report: dict, challenger: dict, tester_stance: str) -> dict:
    assumptions = []
    metrics = report.get("metrics") or {}
    for row in _pending(report):
        left = row.get("left") or "one claim"
        right = row.get("right") or "another claim"
        assumptions.append(f"Assumes {left} and {right} can both stand.")
    if _grounded(report) and float(metrics.get("freshness") or 0) == 0:
        assumptions.append("Assumes the cited document versions are still current.")
    if _grounded(report) and float(metrics.get("source_diversity") or 0) < 0.67:
        assumptions.append("Assumes fewer than two source types are enough.")
    if (report.get("counterfactual") or {}).get("verdict") == "missed":
        assumptions.append("Assumes the skipped question would not change the result.")
    confidence = float(metrics.get("confidence") or 0)
    if _grounded(report) and tester_stance == "support" and confidence < 0.75:
        assumptions.append(f"Assumes confidence {confidence:.2f} is enough to conclude.")
    records = [{"label": "objection" if challenger["stance"] == "challenge" else "note", "detail": challenger["finding"]}]
    records.extend({"label": "assumption", "detail": item} for item in assumptions)
    if challenger["stance"] == "challenge":
        status = "challenged"
    elif assumptions or challenger["stance"] == "caution":
        status = "noted"
    else:
        status = "clear"
    return _phase(
        4,
        "debate",
        "Scientific Debate & Criticism",
        status,
        f"The Challenger is {challenger['stance']}. {len(assumptions)} assumptions are exposed.",
        records,
    )


def _experiment_phase(report: dict, statement: str, tester_stance: str) -> dict:
    parameters = "confidence bar 0.50, minimum cited claims 2"
    if tester_stance == "untested":
        result = "not_run"
        method = "Retrieve a cited answer before testing the hypothesis."
        reason = "The hypothesis was not tested because the answer is not grounded."
    elif tester_stance == "contest":
        result = "fail"
        method = "Require an independent source that is neither side of the open contradiction."
        reason = "The experiment fails while a contradiction is still open."
    elif len(_cited_shorts(report, limit=7)) >= 2:
        result = "pass"
        method = "Hold the cited claims against the confidence bar and require two agreeing claims."
        reason = "Two cited claims agree and no contradiction is open."
    else:
        result = "partial"
        method = "Hold the cited claims against the confidence bar and require two agreeing claims."
        reason = "The hypothesis clears the confidence bar with only one cited claim."
    return _phase(
        5,
        "experiment",
        "Experiment Design & Execution",
        result,
        f"Result {result}. {reason}",
        [
            {"label": "hypothesis", "detail": statement},
            {"label": "method", "detail": method},
            {"label": "parameters", "detail": parameters},
            {"label": "result", "detail": reason},
        ],
    )


def _consensus_phase(investigator: dict, challenger: dict, experiment: dict, conclusion: dict) -> dict:
    investigator_vote = "yes" if investigator["stance"] == "found" else "no"
    if challenger["stance"] == "clear":
        challenger_vote = "yes"
    elif challenger["stance"] == "caution":
        challenger_vote = "abstain"
    else:
        challenger_vote = "no"
    experimenter_vote = {"pass": "yes", "fail": "no", "partial": "abstain", "not_run": "abstain"}[
        experiment["status"]
    ]
    verdict = conclusion["verdict"]
    if verdict == "concluded":
        status = "accept"
        summary = (
            f"Chair accepts. Investigator {investigator_vote}, "
            f"Challenger {challenger_vote}, Experimenter {experimenter_vote}."
        )
    elif verdict == "disputed":
        status = "dissent"
        summary = "Chair dissents. The Challenger blocks the conclusion."
    else:
        status = "withhold"
        summary = "Chair withholds a decision. The hypothesis is untested."
    return _phase(
        6,
        "consensus",
        "Evidence-Based Consensus",
        status,
        summary,
        [
            {"label": "Investigator", "detail": investigator_vote},
            {"label": "Challenger", "detail": challenger_vote},
            {"label": "Experimenter", "detail": experimenter_vote},
            {"label": "Chair", "detail": summary},
        ],
    )


def _reproducibility_phase(report: dict, conclusion: dict, tester_stance: str) -> dict:
    rows = _dataset_rows(report)
    checksums = [checksum for _filename, checksum in rows if checksum]
    filenames = [f"{filename} ({checksum[:12]})" if checksum else filename for filename, checksum in rows]
    asset = str(report.get("asset") or "").strip() or "unknown"
    dataset = f"{asset}: {', '.join(filenames)}" if filenames else f"{asset}: no passage checksum was stored"
    prompt = str(report.get("question") or "").strip() or "the question"
    tools = _tool_labels(report)
    parameters = (
        f"searches {int(report.get('search_budget') or 0)}, "
        f"tokens {int(report.get('token_budget') or 0)}, "
        f"seconds {int(float(report.get('time_budget') or 0))}, "
        "confidence bar 0.50"
    )
    confidence = float((report.get("metrics") or {}).get("confidence") or 0)
    result = (
        f"stop {report.get('stop_reason') or 'unknown'}, "
        f"conclusion {conclusion['verdict']}, "
        f"confidence {confidence:.2f}"
    )
    payload = "\n".join([prompt, asset, parameters, "|".join(sorted(checksums)), conclusion["verdict"], tester_stance])
    fingerprint = hashlib.sha256(payload.encode()).hexdigest()[:12]
    status = "reproducible" if checksums else "partial"
    return _phase(
        7,
        "reproducibility",
        "Reproducibility Engine",
        status,
        f"Run {fingerprint}. Dataset, prompt, tools, parameters, and result are stored.",
        [
            {"label": "fingerprint", "detail": fingerprint},
            {"label": "dataset", "detail": dataset},
            {"label": "prompt", "detail": prompt},
            {"label": "tools", "detail": ", ".join(tools) if tools else "No search tool was recorded."},
            {"label": "parameters", "detail": parameters},
            {"label": "result", "detail": result},
        ],
    )


def _evaluation_phase(report: dict, experiment: dict, consensus: dict, reproducibility: dict) -> dict:
    metrics = report.get("metrics") or {}
    quality = _mean(
        [
            float(metrics.get("confidence") or 0),
            float(metrics.get("citation_validity") or 0),
            float(metrics.get("factual_support") or 0),
        ]
    )
    steps = report.get("steps") or []
    outside = any("outside" in str(step.get("detail") or "").lower() for step in steps)
    strategies = {
        str(step.get("strategy") or "")
        for step in steps
        if step.get("action") == "investigate" and step.get("strategy")
    }
    if outside or len(strategies) >= 2:
        novelty = 1.0
    elif strategies:
        novelty = 0.5
    else:
        novelty = 0.2
    if consensus["status"] == "accept" and experiment["status"] == "pass":
        reliability = 1.0
    elif consensus["status"] == "accept":
        reliability = 0.7
    elif consensus["status"] == "dissent":
        reliability = 0.4
    else:
        reliability = 0.0
    repro = 1.0 if reproducibility["status"] == "reproducible" else 0.5
    overall = _mean([quality, novelty, reliability, repro])
    return _phase(
        8,
        "evaluation",
        "Scientific Discovery Evaluation",
        "scored",
        (
            f"Quality {quality:.2f}, novelty {novelty:.2f}, reliability {reliability:.2f}, "
            f"reproducibility {repro:.2f}. Overall {overall:.2f}."
        ),
        [
            {"label": "quality", "detail": f"{quality:.2f}"},
            {"label": "novelty", "detail": f"{novelty:.2f}"},
            {"label": "reliability", "detail": f"{reliability:.2f}"},
            {"label": "reproducibility", "detail": f"{repro:.2f}"},
            {"label": "overall", "detail": f"{overall:.2f}"},
        ],
    )


def _agents_from_steps(report: dict) -> list[dict]:
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


def _dataset_rows(report: dict) -> list[tuple[str, str]]:
    scene = report.get("scene") or {}
    claims = {claim.get("id"): claim for claim in scene.get("claims") or []}
    rows: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for marker in scene.get("markers") or []:
        claim = claims.get(marker.get("claim_id")) or {}
        provenance = claim.get("provenance") or {}
        filename = str(provenance.get("filename") or "").strip()
        checksum = str(provenance.get("checksum") or "").strip()
        if not filename and not checksum:
            continue
        key = (filename, checksum)
        if key in seen:
            continue
        seen.add(key)
        rows.append(key)
    return rows


def _tool_labels(report: dict) -> list[str]:
    labels = []
    for step in report.get("steps") or []:
        if step.get("action") != "search":
            continue
        strategy = str(step.get("strategy") or "search")
        top_k = int(step.get("top_k") or 0)
        label = f"{strategy} top_k {top_k}" if top_k else strategy
        if label not in labels:
            labels.append(label)
    return labels


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def _phase(number: int, component: str, title: str, status: str, summary: str, records: list[dict]) -> dict:
    return {
        "phase": number,
        "component": component,
        "title": title,
        "status": status,
        "summary": summary,
        "records": records,
    }
