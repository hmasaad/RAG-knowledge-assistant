"""The agent society around one discovery review.

Five roles sit in a hierarchy and share one blackboard. Search agents work
as separate partitions and post their results to that board. Messages go to
the Chair, who allocates the next task and applies a quorum before ruling.
"""

from __future__ import annotations

from app.research.learning import WASTE

_CORE = ("Chair", "Investigator", "Challenger", "Experimenter", "Archivist")


def observe_society(report: dict, discovery: dict) -> dict:
    """Describe how the agents coordinated on this path."""
    reviews = {item["agent"]: item for item in discovery.get("reviews") or []}
    investigator = reviews.get("Investigator") or {"stance": "empty", "finding": ""}
    challenger = reviews.get("Challenger") or {"stance": "clear", "finding": ""}
    conclusion = discovery.get("conclusion") or {"verdict": "insufficient", "text": ""}
    consensus = _phase(discovery, "consensus")
    experiment = _phase(discovery, "experiment")
    reproducibility = _phase(discovery, "reproducibility")
    evaluation = _phase(discovery, "evaluation")
    workers = _workers(report)
    votes = {
        "Investigator": _record(consensus, "Investigator") or "no",
        "Challenger": _record(consensus, "Challenger") or "abstain",
        "Experimenter": _record(consensus, "Experimenter") or "abstain",
    }
    notes = [
        _mas(workers),
        _hierarchy(workers),
        _blackboard(investigator, challenger, experiment, reproducibility, conclusion),
        _protocol(challenger, experiment, conclusion),
        _allocation(conclusion, workers),
        _planning(experiment, consensus),
        _debate(challenger, experiment, conclusion),
        _quorum(votes, consensus),
        _distributed(workers),
        _observability(consensus, evaluation),
    ]
    status = consensus.get("status") or "withhold"
    summary = f"Quorum {status}. { _agent_count(workers) } agents, 5 blackboard posts, 5 messages."
    return {"summary": summary, "notes": notes}


def _mas(workers: list[dict]) -> dict:
    names = list(_CORE) + [worker["name"] for worker in workers if worker["name"] not in _CORE]
    roster = _join(names)
    return _note(
        "mas",
        "formed",
        f"The society has {len(names)} agents: {roster}. The Chair coordinates the others.",
    )


def _hierarchy(workers: list[dict]) -> dict:
    level3 = ["Archivist"] + [worker["name"] for worker in workers if worker["name"] != "Archivist"]
    return _note(
        "hierarchy",
        "delegated",
        (
            "Level 1 is the Chair. Level 2 is the Investigator, the Challenger, and the Experimenter. "
            f"Level 3 is {_join(level3)}."
        ),
    )


def _blackboard(investigator: dict, challenger: dict, experiment: dict, reproducibility: dict, conclusion: dict) -> dict:
    dataset = _record(reproducibility, "dataset") or "No passage checksum was stored."
    posts = [
        f"Investigator: {investigator.get('finding') or 'No finding.'}",
        f"Archivist: {dataset}",
        f"Challenger: {challenger.get('finding') or 'No finding.'}",
        f"Experimenter: {experiment.get('summary') or 'No experiment.'}",
        f"Chair: {conclusion.get('text') or 'No ruling.'}",
    ]
    return _note(
        "blackboard",
        "posted",
        f"The shared board has {len(posts)} posts. The Chair reads the board before ruling. {posts[-1]}",
    )


def _protocol(challenger: dict, experiment: dict, conclusion: dict) -> dict:
    challenger_act = "objects to" if challenger.get("stance") == "challenge" else "informs"
    reply = {"concluded": "accept", "disputed": "reject"}.get(conclusion.get("verdict"), "withhold")
    result = experiment.get("status") or "not_run"
    return _note(
        "protocol",
        "delivered",
        (
            f"The Investigator informs the Chair. The Challenger {challenger_act} the Chair. "
            f"The Experimenter proposes {result} to the Chair. The Chair replies {reply} to the team."
        ),
    )


def _allocation(conclusion: dict, workers: list[dict]) -> dict:
    observed = _observed(workers)
    verdict = conclusion.get("verdict")
    if verdict == "concluded":
        return _note("allocation", "closed", f"The Chair closed the task list. {observed}")
    if verdict == "disputed":
        return _note(
            "allocation",
            "assigned",
            f"The next task goes to the Corroboration agent: settle the open contradiction. {observed}",
        )
    return _note(
        "allocation",
        "assigned",
        f"The next task goes to the Lead agent: retrieve a cited answer. {observed}",
    )


def _planning(experiment: dict, consensus: dict) -> dict:
    if consensus.get("status") == "withhold" or experiment.get("status") == "not_run":
        return _note(
            "planning",
            "blocked",
            "The plan stops before the experiment because the Investigator has no cited evidence.",
        )
    if consensus.get("status") == "dissent":
        return _note(
            "planning",
            "coordinated",
            "Gather, debate, experiment, and the vote finished in that order. The experiment failed, so the Chair dissented.",
        )
    return _note(
        "planning",
        "coordinated",
        "Gather, debate, experiment, and the vote finished in that order.",
    )


def _debate(challenger: dict, experiment: dict, conclusion: dict) -> dict:
    verdict = conclusion.get("verdict")
    stance = challenger.get("stance") or "clear"
    if verdict == "concluded" and stance == "caution":
        ruling = "amended"
        ruling_text = "amended the claim and kept the Challenger's limit"
    elif verdict == "concluded":
        ruling = "upheld"
        ruling_text = "upheld the claim"
    elif verdict == "disputed":
        ruling = "rejected"
        ruling_text = "rejected the claim"
    else:
        ruling = "unresolved"
        ruling_text = "left the claim unresolved"
    result = experiment.get("status") or "not_run"
    return _note(
        "debate",
        ruling,
        (
            f"Round 1 the Investigator states the hypothesis. Round 2 the Challenger is {stance}. "
            f"Round 3 the experiment is {result}. The Chair {ruling_text}."
        ),
    )


def _quorum(votes: dict[str, str], consensus: dict) -> dict:
    tally = (
        f"Investigator {votes['Investigator']}, Challenger {votes['Challenger']}, "
        f"Experimenter {votes['Experimenter']}"
    )
    status = consensus.get("status") or "withhold"
    if status == "accept":
        reason = f"Chair quorum accepts. An abstention leaves the decision to the other votes. {tally}."
    elif status == "dissent":
        reason = f"Chair quorum rejects because an agent voted no. {tally}."
    else:
        reason = f"Chair quorum withholds because the hypothesis is untested. {tally}."
    return _note("consensus", status, reason)


def _distributed(workers: list[dict]) -> dict:
    if not workers:
        return _note(
            "distributed",
            "idle",
            "No search partition ran. Review posts are the only entries on the blackboard.",
        )
    parts = [
        f"{worker['name']} {int(worker.get('useful') or 0)}/{int(worker.get('searches') or 0)}"
        for worker in workers
    ]
    if len(workers) == 1:
        return _note(
            "distributed",
            "single",
            f"One partition posts only to the blackboard: {parts[0]}.",
        )
    return _note(
        "distributed",
        "partitioned",
        f"{len(workers)} partitions post only to the blackboard: {_join(parts)}.",
    )


def _observability(consensus: dict, evaluation: dict) -> dict:
    overall = _record(evaluation, "overall") or "0.00"
    status = consensus.get("status") or "withhold"
    return _note(
        "observability",
        "observed",
        f"Observed 5 protocol messages and 5 blackboard posts. Quorum {status}. Discovery overall {overall}.",
    )


def _workers(report: dict) -> list[dict]:
    agents = list((report.get("collaboration") or {}).get("agents") or [])
    if agents:
        return [
            {
                "name": str(agent.get("name") or "Agent"),
                "searches": int(agent.get("searches") or 0),
                "useful": int(agent.get("useful") or 0),
            }
            for agent in agents
        ]
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


def _observed(workers: list[dict]) -> str:
    if not workers:
        return "No search agent ran."
    parts = []
    for worker in workers:
        count = int(worker.get("searches") or 0)
        noun = "search" if count == 1 else "searches"
        parts.append(f"{worker['name']} ran {count} {noun}")
    return _join(parts) + "."


def _agent_count(workers: list[dict]) -> int:
    return len(_CORE) + sum(1 for worker in workers if worker["name"] not in _CORE)


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


def _join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _note(component: str, decision: str, reason: str) -> dict:
    return {"component": component, "decision": decision, "reason": reason}
