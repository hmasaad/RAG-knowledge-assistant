"""Production review for one finished research answer.

Eight areas decide whether the answer can be released. Completeness is first:
every question clause and every cited detail has to show up in the answer.
The other areas check retrieval, citation integrity, conflicts and freshness,
whether the search adapted, the quality score, access control, and the run's
latency, cost, and failures.
"""

from __future__ import annotations

import re

from app.retrieval.text import content_tokens, focus_tokens
from app.security.leakage import answer_leaks, sensitive_tokens

REFUSAL = "I don't have enough evidence to answer."
_AREAS = (
    "completeness",
    "retrieval",
    "citation",
    "conflict",
    "adaptive",
    "evaluation",
    "security",
    "observability",
)
_TITLES = {
    "completeness": "Answer quality",
    "retrieval": "Retrieval intelligence",
    "citation": "Citation integrity",
    "conflict": "Conflict and freshness",
    "adaptive": "Adaptive retrieval",
    "evaluation": "Evaluation",
    "security": "Knowledge security",
    "observability": "Observability",
}
_PASS = {
    "completeness": {"complete"},
    "retrieval": {"hybrid", "ranked"},
    "citation": {"verified"},
    "conflict": {"current", "settled"},
    "adaptive": {"adapted", "default"},
    "evaluation": {"scored"},
    "security": {"secure"},
    "observability": {"observed"},
}
_RELEASE = {"release": "Release", "revise": "Revise", "withhold": "Withhold"}
_FRAMING = frozenset(
    {
        "about",
        "based",
        "document",
        "documents",
        "file",
        "files",
        "people",
        "please",
        "saying",
        "their",
        "there",
        "these",
        "those",
        "using",
    }
)
_ADAPTIVE = frozenset({"corroborate", "authority", "broaden", "entity", "decompose"})
_FAILURES = frozenset({"failed", "recovered", "skipped"})
_METRICS = (
    "completeness",
    "factual_support",
    "citation_validity",
    "source_diversity",
    "freshness",
    "unresolved_contradictions",
)
_INJECTION = re.compile(r"(?i)ignore (?:all |any |previous |prior )+instructions|system prompt")
_SPLIT = re.compile(r"\s+\band\b\s+|\s+\bor\b\s+|,\s*", re.I)


def evaluate_production(report: dict) -> dict:
    """Score the eight production areas and decide release, revise, or withhold."""
    areas = [
        _completeness(report),
        _retrieval(report),
        _citation(report),
        _conflict(report),
        _adaptive(report),
        _evaluation(report),
        _security(report),
        _observability(report),
    ]
    decision = _decision(areas)
    passed = sum(1 for area in areas if area["status"] in _PASS[area["component"]])
    completeness = areas[0]["score"]
    summary = (
        f"{_RELEASE[decision]}. Completeness {completeness:.2f}. "
        f"{passed} of 8 production areas passed."
    )
    return {
        "summary": summary,
        "release": decision,
        "revision": _revision(report, areas),
        "areas": areas,
    }


def _completeness(report: dict) -> dict:
    answer = str(report.get("answer") or "")
    if _refused(answer):
        return _area(
            "completeness",
            "missing",
            0.0,
            "The answer refused, so no requirement was covered.",
            [{"label": "missing", "detail": "The answer has no factual coverage."}],
        )
    requirements = _requirements(report)
    covered = [item for item in requirements if item["covered"]]
    missed = [item for item in requirements if not item["covered"]]
    total = len(requirements)
    score = round(len(covered) / total, 2) if total else 0.0
    if total and not missed:
        status = "complete"
        summary = f"All {total} requirements are in the answer."
    elif covered:
        status = "partial"
        summary = f"{len(covered)} of {total} requirements are in the answer."
    else:
        status = "missing"
        summary = "No requirement from the question or the cited sources is in the answer."
    records = [
        {"label": "covered" if item["covered"] else item["kind"], "detail": item["detail"]}
        for item in requirements
    ]
    if not records:
        records = [{"label": "missing", "detail": "The question and the cited sources named nothing to check."}]
    return _area("completeness", status, score, summary, records)


def _retrieval(report: dict) -> dict:
    steps = [step for step in report.get("steps") or [] if step.get("action") == "search"]
    useful = [step for step in steps if step.get("status") == "ok"]
    strategies = _unique(str(step.get("strategy") or "initial") for step in steps)
    questions = _unique(str(step.get("question") or "") for step in steps)
    depths = _unique(str(int(step.get("top_k") or 0)) for step in steps if int(step.get("top_k") or 0) > 0)
    contextual = any(
        str(step.get("strategy") or "") in _ADAPTIVE or "outside" in str(step.get("detail") or "").casefold()
        for step in report.get("steps") or []
    )
    expanded = len(questions) > 1
    reranked = len(depths) > 1 or bool(depths)
    if not useful:
        status = "empty"
        score = 0.0
        summary = "No search returned a passage."
    elif len(strategies) > 1 or expanded or contextual or len(depths) > 1:
        status = "hybrid"
        score = 1.0
        summary = "Retrieval mixed " + ", ".join(strategies or ["initial"]) + "."
    else:
        status = "ranked"
        score = 0.75
        summary = "One ranked search answered the question."
    depth = ", ".join(depths) if depths else "not set"
    records = [
        {"label": "strategy", "detail": ", ".join(strategies) if strategies else "none"},
        {"label": "expansion", "detail": f"{len(questions)} distinct queries."},
        {"label": "rerank", "detail": f"top_k {depth}."},
        {"label": "contextual", "detail": "A follow-up left the first result set." if contextual else "The search stayed on the first result set."},
    ]
    if not reranked and useful:
        records[2]["detail"] = "top_k was not recorded."
    return _area("retrieval", status, score, summary, records)


def _citation(report: dict) -> dict:
    answer = str(report.get("answer") or "")
    cited = _cited_claims(report)
    records = []
    verified = 0
    for claim in cited:
        provenance = claim.get("provenance") or {}
        filename = str(provenance.get("filename") or "")
        checksum = str(provenance.get("checksum") or "")
        short = str(claim.get("short") or claim.get("text") or "claim")
        anchor = f"{filename} ({checksum[:12]})" if filename and checksum else "no checksum"
        if filename and checksum and _claim_covered(answer, claim):
            verified += 1
            state = "verified"
        elif filename and checksum:
            state = "unmatched"
        else:
            state = "unanchored"
        records.append({"label": "claim", "detail": f"{short} · {anchor} · {state}"})
    if cited and verified == len(cited):
        status = "verified"
        score = 1.0
        summary = f"{verified} of {len(cited)} cited claims match a checksummed source."
    elif verified:
        status = "partial"
        score = round(verified / len(cited), 2)
        summary = f"{verified} of {len(cited)} cited claims match a checksummed source."
    else:
        status = "unverified"
        score = 0.0
        summary = "No cited claim is tied to a checksummed source that the answer states."
        if not records:
            records = [{"label": "claim", "detail": "No claim was cited."}]
    return _area("citation", status, score, summary, records)


def _conflict(report: dict) -> dict:
    rows = list(report.get("contradictions") or [])
    pending = [row for row in rows if row.get("status") == "pending"]
    freshness = _freshness(report)
    superseded = [claim for claim in _cited_claims(report) if _superseded(claim)]
    records = []
    for row in rows:
        left = row.get("left") or "one claim"
        right = row.get("right") or "another claim"
        records.append({"label": "contradiction", "detail": f"{row.get('status') or 'open'} · {left} vs {right}"})
    if not records:
        records.append({"label": "contradiction", "detail": "No contradiction was detected."})
    for claim in superseded:
        provenance = claim.get("provenance") or {}
        filename = provenance.get("filename") or "a source"
        records.append(
            {
                "label": "freshness",
                "detail": f"{filename} v{int(provenance.get('version') or 0)} is older than a later version.",
            }
        )
    if not superseded:
        records.append({"label": "freshness", "detail": f"{freshness:.2f} of the freshness check is current."})
    if pending:
        status = "contested"
        score = 0.0
        summary = f"{len(pending)} contradiction{'s' if len(pending) != 1 else ''} still open."
    elif superseded:
        status = "stale"
        score = freshness if freshness < 1 else 0.5
        summary = "A cited source is older than a later version of the same document."
    elif freshness < 1:
        status = "stale"
        score = freshness
        summary = f"Freshness is {freshness:.2f}. Not every cited source is on the current version."
    elif rows:
        status = "settled"
        score = 1.0
        summary = "Every detected contradiction has a winner."
    else:
        status = "current"
        score = 1.0
        summary = "No contradiction is open, and the cited sources are current."
    return _area("conflict", status, score, summary, records)


def _adaptive(report: dict) -> dict:
    clauses = _clauses(str(report.get("question") or ""))
    compound = len(clauses) > 1
    confidence = float((report.get("metrics") or {}).get("confidence") or 0)
    changed = any(
        str(step.get("strategy") or "") not in {"", "initial"}
        for step in report.get("steps") or []
        if step.get("action") in {"search", "investigate"}
    )
    if changed:
        status = "adapted"
        score = 1.0
        summary = "The loop changed strategy after the first search."
    elif compound and confidence < 0.6 and not report.get("sufficient"):
        status = "stalled"
        score = 0.4
        summary = f"The compound question stayed on the first search at confidence {confidence:.2f}."
    else:
        status = "default"
        score = 1.0
        summary = "The first search fit the question."
    kind = "compound" if compound else "simple"
    choice = "changed strategy" if changed else "kept the initial search"
    return _area(
        "adaptive",
        status,
        score,
        summary,
        [
            {"label": "complexity", "detail": f"{kind} · {len(clauses)} clause{'s' if len(clauses) != 1 else ''}."},
            {"label": "choice", "detail": f"{choice} · confidence {confidence:.2f}."},
        ],
    )


def _evaluation(report: dict) -> dict:
    metrics = report.get("metrics") or {}
    if "completeness" not in metrics and "factual_support" not in metrics:
        return _area(
            "evaluation",
            "unscored",
            0.0,
            "The run has no quality metrics.",
            [{"label": "overall", "detail": "unscored"}],
        )
    values = []
    records = []
    for name in _METRICS:
        raw = metrics.get(name, 0.0)
        value = float(raw or 0)
        if name == "unresolved_contradictions":
            value = 1.0 - value
        values.append(value)
        records.append({"label": "metric", "detail": f"{name} {float(raw or 0):.2f}"})
    overall = round(sum(values) / len(values), 2) if values else 0.0
    records.append({"label": "overall", "detail": f"{overall:.2f}"})
    benchmark = report.get("benchmark") or {}
    if benchmark:
        trend = benchmark.get("trend") or "unrecorded"
        latest = float(benchmark.get("latest_score") or 0)
        records.append({"label": "benchmark", "detail": f"{trend} · {latest:.2f}"})
    return _area("evaluation", "scored", overall, f"Overall {overall:.2f}.", records)


def _security(report: dict) -> dict:
    answer = str(report.get("answer") or "")
    question = str(report.get("question") or "")
    caller = str(report.get("user_id") or "")
    cited = _cited_claims(report)
    filenames = [
        str((claim.get("provenance") or {}).get("filename") or "")
        for claim in cited
        if (claim.get("provenance") or {}).get("filename")
    ]
    forbidden: set[str] = set()
    for claim in (report.get("scene") or {}).get("claims") or []:
        forbidden |= sensitive_tokens(str(claim.get("text") or ""))
    leaked = bool(sensitive_tokens(answer)) or answer_leaks(answer, forbidden) or bool(_INJECTION.search(answer))
    breach = _tenant_breach(cited, caller)
    if leaked:
        status = "blocked"
        secret = "The answer contains a secret or an instruction override."
    elif breach:
        status = "blocked"
        secret = "The answer cites a private source owned by someone else."
    else:
        status = "secure" if _refused(answer) or filenames else "unscoped"
        secret = "No secret value is present." if not _refused(answer) else "The answer refuses instead of guessing."
    if status == "secure" and _refused(answer) and re.search(r"(?i)\b(password|api[_ -]?key|secret|credential)s?\b", question):
        secret = "The question asks for a secret and the answer refuses."
    access = (
        "Citations name " + ", ".join(filenames) + "."
        if filenames
        else "The answer does not name a source file."
    )
    tenant = (
        "A private source from another owner is cited."
        if breach
        else "No private source from another owner is cited."
    )
    score = 1.0 if status == "secure" else 0.0
    summary = {
        "secure": "The answer stays inside the caller's sources.",
        "blocked": "The answer fails the access check.",
        "unscoped": "The answer makes a claim without naming a source.",
    }[status]
    return _area(
        "security",
        status,
        score,
        summary,
        [
            {"label": "access", "detail": access},
            {"label": "secrets", "detail": secret},
            {"label": "tenant", "detail": tenant},
        ],
    )


def _observability(report: dict) -> dict:
    elapsed = float(report.get("elapsed_seconds") or 0)
    time_budget = report.get("time_budget")
    tokens = int(report.get("tokens_used") or 0)
    token_budget = report.get("token_budget")
    searches = int(report.get("searches_used") or 0)
    search_budget = report.get("search_budget")
    failures = [step for step in report.get("steps") or [] if step.get("status") in _FAILURES]
    stop = str(report.get("stop_reason") or "")
    feedback = str(report.get("feedback") or "").strip()
    observed = bool(stop) and "elapsed_seconds" in report and "tokens_used" in report
    if failures:
        failure_detail = f"{len(failures)} failed or recovered steps. {failures[0].get('detail') or ''}".strip()
    else:
        failure_detail = "No failed search."
    records = [
        {"label": "latency", "detail": _budget(elapsed, time_budget, "s")},
        {"label": "tokens", "detail": _budget(tokens, token_budget, "")},
        {"label": "searches", "detail": _budget(searches, search_budget, "")},
        {"label": "failures", "detail": failure_detail},
        {"label": "stop", "detail": stop or "unrecorded"},
        {"label": "feedback", "detail": feedback or "No user feedback is attached to this run."},
    ]
    if observed:
        summary = f"Stopped for {stop}. {elapsed:.3f}s, {tokens} tokens, {searches} searches, {len(failures)} failures."
        return _area("observability", "observed", 1.0, summary, records)
    return _area("observability", "blind", 0.0, "Latency, token cost, or the stop reason was not recorded.", records)


def _decision(areas: list[dict]) -> str:
    by_name = {area["component"]: area for area in areas}
    if by_name["security"]["status"] in {"blocked", "unscoped"}:
        return "withhold"
    if by_name["completeness"]["status"] == "missing" or by_name["citation"]["status"] == "unverified":
        return "withhold"
    if any(area["status"] not in _PASS[area["component"]] for area in areas):
        return "revise"
    return "release"


def _revision(report: dict, areas: list[dict]) -> str:
    answer = str(report.get("answer") or "").strip()
    if _refused(answer):
        return ""
    by_name = {area["component"]: area for area in areas}
    omitted = [
        row["detail"]
        for row in by_name["completeness"]["records"]
        if row["label"] == "omitted"
    ]
    missing = [
        row["detail"]
        for row in by_name["completeness"]["records"]
        if row["label"] in {"missing", "gap"}
    ]
    conflicts = [
        row["detail"]
        for row in by_name["conflict"]["records"]
        if row["label"] == "contradiction" and str(row["detail"]).startswith("pending")
    ]
    stale = [row["detail"] for row in by_name["conflict"]["records"] if row["label"] == "freshness" and "older than" in row["detail"]]
    if not omitted and not missing and not conflicts and not stale:
        return ""
    lines = [answer, ""]
    if omitted:
        lines.append("Omitted cited details:")
        lines.extend(f"- {item}" for item in omitted)
    if missing:
        lines.append("Still required:")
        lines.extend(f"- {item}" for item in missing)
    if conflicts:
        lines.append("Unresolved contradictions:")
        lines.extend(f"- {item}" for item in conflicts)
    if stale:
        lines.append("Outdated sources:")
        lines.extend(f"- {item}" for item in stale)
    return "\n".join(lines).strip()


def _requirements(report: dict) -> list[dict]:
    answer = str(report.get("answer") or "")
    items = []
    seen: set[tuple[str, ...]] = set()
    for clause in _clauses(str(report.get("question") or "")):
        tokens = _requirement_tokens(clause)
        key = tuple(tokens)
        if not tokens or key in seen:
            continue
        seen.add(key)
        covered = _covers(answer, tokens)
        items.append(
            {
                "covered": covered,
                "kind": "missing",
                "detail": clause,
            }
        )
    for claim in _cited_claims(report):
        text = str(claim.get("text") or claim.get("short") or "").strip()
        if not text:
            continue
        tokens = _requirement_tokens(text) or _requirement_tokens(str(claim.get("short") or ""))
        key = tuple(tokens)
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        covered = _claim_covered(answer, claim)
        filename = str((claim.get("provenance") or {}).get("filename") or "")
        detail = f"{claim.get('short') or text}: {text}"
        if filename:
            detail = f"{detail} ({filename})"
        items.append({"covered": covered, "kind": "omitted", "detail": detail})
    for gap in ((report.get("scene") or {}).get("plan") or {}).get("items") or []:
        detail = str(gap.get("missing") or gap.get("question") or "").strip()
        tokens = _requirement_tokens(detail)
        key = tuple(tokens)
        if not detail or (key and key in seen):
            continue
        if key:
            seen.add(key)
        items.append({"covered": False, "kind": "gap", "detail": detail})
    return items


def _clauses(question: str) -> list[str]:
    clauses = []
    for chunk in re.split(r"[\n;]+", question):
        for piece in _SPLIT.split(chunk):
            text = piece.strip(" ?.-*")
            if _requirement_tokens(text):
                clauses.append(text)
    return clauses


def _requirement_tokens(text: str) -> list[str]:
    tokens = []
    seen = set()
    for token in focus_tokens(text):
        if len(token) < 4 or token in _FRAMING or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def _covers(text: str, tokens: list[str]) -> bool:
    present = content_tokens(text)
    for token in tokens:
        if not any(_same_word(token, item) for item in present):
            return False
    return True


def _same_word(left: str, right: str) -> bool:
    if left == right:
        return True
    if len(left) < 5 or len(right) < 5:
        return False
    return left.startswith(right) or right.startswith(left)


def _claim_covered(answer: str, claim: dict) -> bool:
    quote = str((claim.get("citation") or {}).get("quote") or "")
    if quote and _contains(answer, quote):
        return True
    short_tokens = _requirement_tokens(str(claim.get("short") or ""))
    if short_tokens and _covers(answer, short_tokens):
        return True
    text_tokens = _requirement_tokens(str(claim.get("text") or ""))
    if not text_tokens:
        return False
    present = content_tokens(answer)
    hits = sum(1 for token in text_tokens if any(_same_word(token, item) for item in present))
    return hits / len(text_tokens) >= 0.6


def _contains(haystack: str, needle: str) -> bool:
    folded = " ".join(needle.casefold().split())
    return bool(folded) and folded in " ".join(haystack.casefold().split())


def _cited_claims(report: dict) -> list[dict]:
    scene = report.get("scene") or {}
    claims = {claim.get("id"): claim for claim in scene.get("claims") or []}
    cited = []
    seen = set()
    for marker in scene.get("markers") or []:
        claim_id = marker.get("claim_id")
        claim = claims.get(claim_id)
        if claim is None or claim_id in seen:
            continue
        seen.add(claim_id)
        cited.append(claim)
    return cited


def _superseded(claim: dict) -> bool:
    provenance = claim.get("provenance") or {}
    version = int(provenance.get("version") or 0)
    for item in provenance.get("history") or []:
        if int(item.get("version") or 0) > version:
            return True
    return False


def _freshness(report: dict) -> float:
    metrics = report.get("metrics") or {}
    if "freshness" not in metrics:
        return 1.0
    return float(metrics.get("freshness") or 0)


def _tenant_breach(cited: list[dict], caller: str) -> bool:
    if not caller:
        return False
    for claim in cited:
        provenance = claim.get("provenance") or {}
        owner = str(provenance.get("owner") or "")
        if str(provenance.get("visibility") or "") == "private" and owner and owner != caller:
            return True
    return False


def _refused(answer: str) -> bool:
    text = " ".join(answer.casefold().split())
    if not text:
        return True
    return "don't have enough evidence" in text or "do not have enough evidence" in text


def _unique(values) -> list[str]:
    found = []
    for value in values:
        cleaned = str(value).strip()
        if cleaned and cleaned not in found:
            found.append(cleaned)
    return found


def _budget(used: float, budget: object, suffix: str) -> str:
    amount = f"{used:.3f}{suffix}" if suffix == "s" else f"{used:g}{suffix}"
    if budget in (None, ""):
        return amount
    limit = f"{float(budget):.3f}{suffix}" if suffix == "s" else f"{float(budget):g}{suffix}"
    return f"{amount} of {limit}"


def _area(component: str, status: str, score: float, summary: str, records: list[dict]) -> dict:
    return {
        "priority": _AREAS.index(component) + 1,
        "component": component,
        "title": _TITLES[component],
        "status": status,
        "score": round(float(score), 2),
        "summary": summary,
        "records": records,
    }
