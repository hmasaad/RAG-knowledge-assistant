"""Plan the next research step from gaps in a trace scene.

Each gap names the missing evidence, a follow-up question, and a search
strategy. Expected information value is the decision gain divided by the
search cost, and the list is returned in that order.
"""

from __future__ import annotations

import re

from app.retrieval.queries import decompose_query, rewrite_query

REFUSAL = "I don't have enough evidence to answer."

STRATEGIES = {
    "corroborate": "Search an independent source that can settle the contradiction.",
    "authority": "Search a higher-authority source than the claims already cited.",
    "broaden": "Search documents outside the current evidence set.",
    "entity": "Search for a named entity the question does not source.",
    "decompose": "Search the part of the question the current claims do not cover.",
}

_COST = {
    "corroborate": 2,
    "authority": 2,
    "broaden": 1,
    "entity": 1,
    "decompose": 1,
}

_STOP = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "of",
        "and",
        "or",
        "to",
        "for",
        "in",
        "on",
        "with",
        "that",
        "this",
        "about",
        "what",
        "which",
        "who",
        "how",
        "when",
        "where",
        "why",
        "does",
        "do",
        "did",
        "people",
        "saying",
        "their",
        "its",
        "from",
        "into",
        "while",
        "have",
        "has",
        "been",
        "tell",
        "me",
    }
)
_QUESTION_START = frozenset({"what", "which", "who", "how", "when", "where", "why", "tell"})


def plan_research(scene: dict) -> dict:
    """Rank follow-up research for one trace scene."""
    claims = list(scene.get("claims") or [])
    by_id = {claim["id"]: claim for claim in claims}
    question = str(scene.get("question") or "").strip()
    refused = _refused(scene)
    covered = _claim_words(claims)
    gaps: list[dict] = []
    if refused:
        gaps.append(
            _gap(
                "refusal",
                "refusal",
                "No indexed passage supports this question.",
                question or "Which document covers this question?",
                "broaden",
                0.85,
                [],
            )
        )
    support = _support_ids(scene, by_id)
    for conflict in scene.get("conflicts") or []:
        if conflict.get("status") != "pending":
            continue
        left = by_id.get(conflict.get("left_id"))
        right = by_id.get(conflict.get("right_id"))
        if left is None or right is None:
            continue
        material = conflict["left_id"] in support or conflict["right_id"] in support
        subject = _subject(scene)
        gaps.append(
            _gap(
                f"conflict:{conflict['id']}",
                "conflict",
                f"No independent source settles {left['short']} versus {right['short']}.",
                f"Which claim about {subject} holds: {left['short']} or {right['short']}?",
                "corroborate",
                0.9 if material else 0.64,
                [conflict["left_id"], conflict["right_id"]],
            )
        )
    if not refused and support and "news" not in {_role(by_id[item]) for item in support if item in by_id}:
        topic = question or scene.get("asset") or "this claim"
        gaps.append(
            _gap(
                "authority",
                "authority",
                "The answer is not backed by a news source.",
                f"What does a news source report about {topic}?",
                "authority",
                0.66,
                list(support),
            )
        )
    if not refused:
        gaps.extend(_entity_gaps(question, scene, covered))
        gaps.extend(_clause_gaps(question, covered))
    gaps.sort(key=lambda item: (-item["value"], -item["gain"], item["id"]))
    for index, item in enumerate(gaps, start=1):
        item["priority"] = index
    if gaps:
        top = gaps[0]
        summary = f"Next ({top['strategy']}, value {top['value']:.2f}): {top['question']}"
    else:
        summary = "Evidence is sufficient. No follow-up research."
    return {
        "summary": summary,
        "next_id": gaps[0]["id"] if gaps else "",
        "items": gaps,
    }


def _gap(
    gap_id: str,
    kind: str,
    missing: str,
    question: str,
    strategy: str,
    gain: float,
    claim_ids: list[str],
) -> dict:
    cost = _COST[strategy]
    return {
        "id": gap_id,
        "kind": kind,
        "missing": missing,
        "question": question,
        "strategy": strategy,
        "strategy_reason": STRATEGIES[strategy],
        "gain": gain,
        "cost": cost,
        "value": round(gain / cost, 2),
        "priority": 0,
        "claim_ids": claim_ids,
    }


def _subject(scene: dict) -> str:
    asset = str(scene.get("asset") or "").strip()
    for item in scene.get("assets") or []:
        if item.get("id") == asset and str(item.get("name") or "").strip():
            return str(item["name"]).strip()
    return asset or "this subject"


def _refused(scene: dict) -> bool:
    if str(scene.get("answer") or "").strip() == REFUSAL:
        return True
    decisions = scene.get("decisions") or []
    return bool(decisions) and decisions[-1].get("decision") == "refused"


def _support_ids(scene: dict, by_id: dict) -> list[str]:
    found: list[str] = []
    for marker in scene.get("markers") or []:
        claim_id = marker.get("claim_id")
        if claim_id not in by_id or claim_id in found:
            continue
        found.append(claim_id)
        for upstream in by_id[claim_id].get("upstream") or []:
            if upstream in by_id and upstream not in found:
                found.append(upstream)
    return found


def _role(claim: dict) -> str:
    agent = str(claim.get("agent_id") or "")
    return agent.rsplit(":", 1)[-1]


def _claim_words(claims: list[dict]) -> set[str]:
    words: set[str] = set()
    for claim in claims:
        words |= _words(claim.get("text") or "")
        words |= _words(claim.get("short") or "")
        for name in claim.get("entities") or []:
            words |= _words(name)
    return words


def _words(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in _STOP and len(word) > 2}


def _entity_gaps(question: str, scene: dict, covered: set[str]) -> list[dict]:
    ignored = {str(scene.get("asset") or "").lower()}
    for asset in scene.get("assets") or []:
        if asset.get("id") == scene.get("asset"):
            ignored |= _words(asset.get("name") or "")
            ignored.add(str(asset.get("id") or "").lower())
    gaps = []
    for name in _proper_names(question):
        tokens = _words(name)
        if not tokens or tokens <= ignored or tokens <= covered:
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        gaps.append(
            _gap(
                f"entity:{slug}",
                "entity",
                f"{name} is named in the question and missing from the evidence.",
                f"Which source mentions {name}?",
                "entity",
                0.55,
                [],
            )
        )
    return gaps


def _proper_names(question: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9'-]*", question)
    names: list[str] = []
    current: list[str] = []
    for index, word in enumerate(words):
        leading = index == 0 or words[index - 1].lower() in _QUESTION_START
        if word[0].isupper() and word.lower() not in _QUESTION_START and not leading:
            current.append(word)
            continue
        if current:
            names.append(" ".join(current))
            current = []
    if current:
        names.append(" ".join(current))
    return names


def _clause_gaps(question: str, covered: set[str]) -> list[dict]:
    clauses = decompose_query(rewrite_query(question))
    gaps = []
    for index, clause in enumerate(clauses, start=1):
        tokens = _words(clause)
        if not tokens or tokens & covered:
            continue
        gaps.append(
            _gap(
                f"clause:{index}",
                "coverage",
                f"No current claim covers: {clause}.",
                clause if clause.endswith("?") else f"What does the evidence say about {clause}?",
                "decompose",
                0.58,
                [],
            )
        )
    return gaps
