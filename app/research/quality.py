"""Score research reliability, then tell the adaptive loop what to do next."""

from dataclasses import dataclass

from app.generation.grounding import SUPPORT_THRESHOLD, claim_support, is_refusal
from app.research.contracts import FILL_GAPS, RESOLVE_CONTRADICTIONS, REVERIFY
from app.security.prompts import REFUSAL_TEXT

MISSING_EVIDENCE = "missing_evidence"
UNSUPPORTED_CLAIM = "unsupported_claim"
CONFLICTING_EVIDENCE = "conflicting_evidence"
STALE_SOURCE = "stale_source"
INCOMPLETE_ANSWER = "incomplete_answer"
INSUFFICIENT_REASON = "Evidence is insufficient to answer confidently."

CRITERIA = (
    "completeness",
    "factual_support",
    "citation_validity",
    "source_diversity",
    "freshness",
    "unresolved_contradictions",
)


@dataclass(frozen=True)
class ResearchQualitySpec:
    """What a reliable report must show for one question and collection."""

    expect_answer: bool
    expect_sufficient: bool
    expect_gap: bool
    expect_conflict: bool
    relevant: frozenset[str] = frozenset()
    expected_substrings: tuple[str, ...] = ()
    gap_terms: tuple[str, ...] = ()
    fresh_after: str = ""
    source_dates: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ClaimScore:
    """One cited sentence, scored before the report is averaged."""

    claim: str
    marker: int
    citation_exists: bool
    supports: bool
    support_ratio: float
    filename: str
    failures: tuple[str, ...]


@dataclass(frozen=True)
class ResearchQualityScore:
    faithfulness: float
    citation_correctness: float
    sufficiency: float
    gap_detection: float
    conflict_detection: float
    coverage: float
    source_recall: float
    completed: bool
    reliability: float
    completeness: float = 0.0
    factual_support: float = 0.0
    citation_validity: float = 0.0
    source_diversity: float = 0.0
    freshness: float = 1.0
    unresolved_contradictions: float = 0.0
    claims: tuple[ClaimScore, ...] = ()
    failures: tuple[str, ...] = ()
    recommendation: str = "stop"
    recommendation_reason: str = ""


def score_research(report, spec: ResearchQualitySpec) -> ResearchQualityScore:
    """Judge the report text, citations, gaps, and flags.

    Each cited claim is scored first. Reliability is the mean of the
    suite metrics and is not raised just because the workflow finished.
    """
    return evaluate_research(report, spec)


def evaluate_research(
    report,
    spec: ResearchQualitySpec | None = None,
) -> ResearchQualityScore:
    """Score claims, classify failures, and recommend the next loop action."""
    judged = spec if spec is not None else _observed_spec(report)
    text = getattr(report, "report", "") or ""
    citations = list(getattr(report, "citations", []) or [])
    quotes = [getattr(item, "quote", "") for item in citations]
    cited = [(getattr(item, "filename", ""), getattr(item, "quote", "")) for item in citations]
    claims = tuple(_score_claims(text, cited, set(judged.relevant), judged.expect_answer))
    faith = _faithfulness(text, quotes)
    cites = _citation_correctness(text, cited, set(judged.relevant), judged.expect_answer)
    sufficient = 1.0 if bool(getattr(report, "sufficient_evidence", False)) == judged.expect_sufficient else 0.0
    gap_score = _gap_score(getattr(report, "gaps", ()) or (), judged)
    conflicted = bool(getattr(report, "conflict", False) or getattr(report, "contradictions", ()))
    conflict_score = 1.0 if conflicted == judged.expect_conflict else 0.0
    coverage = _coverage(text, judged)
    source_recall = _source_recall(citations, judged)
    completeness = 1.0 if coverage == 1.0 and gap_score == 1.0 else 0.0
    diversity = _source_diversity(citations, judged)
    freshness = _freshness(citations, judged)
    quality = (faith, cites, sufficient, gap_score, conflict_score, coverage, source_recall)
    completed = getattr(report, "status", "completed") == "completed"
    failures = _classify_failures(
        report,
        judged,
        claims,
        coverage=coverage,
        freshness=freshness,
        conflicted=conflicted,
    )
    recommendation, reason = recommend_research(failures, judged)
    return ResearchQualityScore(
        faithfulness=faith,
        citation_correctness=cites,
        sufficiency=sufficient,
        gap_detection=gap_score,
        conflict_detection=conflict_score,
        coverage=coverage,
        source_recall=source_recall,
        completed=completed,
        reliability=sum(quality) / len(quality),
        completeness=completeness,
        factual_support=faith,
        citation_validity=cites,
        source_diversity=diversity,
        freshness=freshness,
        unresolved_contradictions=conflict_score,
        claims=claims,
        failures=failures,
        recommendation=recommendation,
        recommendation_reason=reason,
    )


def recommend_research(
    failures: tuple[str, ...] | ResearchQualityScore,
    spec: ResearchQualitySpec | None = None,
) -> tuple[str, str]:
    """Tell the adaptive loop whether to gather, reverify, resolve, or stop."""
    labels = failures.failures if isinstance(failures, ResearchQualityScore) else failures
    expect_answer = True if spec is None else spec.expect_answer
    expect_conflict = False if spec is None else spec.expect_conflict
    if not expect_answer and UNSUPPORTED_CLAIM not in labels:
        return "stop", "The report correctly stops without inventing an answer."
    if CONFLICTING_EVIDENCE in labels and not (
        expect_conflict and INCOMPLETE_ANSWER not in labels and UNSUPPORTED_CLAIM not in labels
    ):
        return RESOLVE_CONTRADICTIONS, "Unresolved contradiction."
    if expect_conflict and CONFLICTING_EVIDENCE in labels and UNSUPPORTED_CLAIM not in labels:
        return "stop", "The documented contradiction is already in the report."
    if UNSUPPORTED_CLAIM in labels:
        return REVERIFY, "A cited claim is not supported by its evidence."
    if STALE_SOURCE in labels:
        return FILL_GAPS, "A cited source is older than the freshness window."
    if MISSING_EVIDENCE in labels or INCOMPLETE_ANSWER in labels:
        return FILL_GAPS, "The answer is missing evidence."
    return "stop", "Further research is not required."


def is_confident(score: ResearchQualityScore, report) -> bool:
    """True when every published claim is grounded, or the report honestly refuses."""
    text = getattr(report, "report", "") or ""
    if is_refusal(text):
        return True
    supported = [item for item in score.claims if item.supports]
    weak = [
        item
        for item in score.claims
        if UNSUPPORTED_CLAIM in item.failures or MISSING_EVIDENCE in item.failures
    ]
    if weak:
        return False
    if not score.claims and not getattr(report, "sufficient_evidence", False):
        return False
    if not supported and score.claims:
        return False
    return True


def apply_quality_gate(report) -> ResearchQualityScore:
    """Attach claim scores and refuse when the answer is not confident."""
    score = evaluate_research(report)
    confident = is_confident(score, report)
    report.quality_action = score.recommendation
    report.quality_reason = score.recommendation_reason
    report.quality_failures = score.failures
    report.quality_claims = score.claims
    report.quality_confident = confident
    if confident:
        return score
    report.report = REFUSAL_TEXT
    report.sufficient_evidence = False
    report.action = "refuse"
    report.citations = []
    report.passages = []
    report.decision = "stop"
    report.quality_action = "stop"
    report.quality_reason = INSUFFICIENT_REASON
    return score


def _observed_spec(report) -> ResearchQualitySpec:
    gaps = tuple(getattr(report, "gaps", ()) or ())
    conflicted = bool(getattr(report, "conflict", False) or getattr(report, "contradictions", ()))
    text = getattr(report, "report", "") or ""
    return ResearchQualitySpec(
        expect_answer=not is_refusal(text),
        expect_sufficient=bool(getattr(report, "sufficient_evidence", False)),
        expect_gap=bool(gaps),
        expect_conflict=conflicted,
        relevant=frozenset(
            getattr(item, "filename", "") for item in getattr(report, "citations", []) or []
        ),
    )


def _score_claims(
    answer: str,
    passages: list[tuple[str, str]],
    relevant: set[str],
    expect_answer: bool,
) -> list[ClaimScore]:
    if is_refusal(answer):
        return []
    scored: list[ClaimScore] = []
    for claim, number in _cited_claims(answer):
        exists = 1 <= number <= len(passages)
        filename = passages[number - 1][0] if exists else ""
        quote = passages[number - 1][1] if exists else ""
        ratio = 0.0
        measured = claim_support(claim, quote) if exists else None
        ratio = 0.0 if measured is None else measured
        in_relevant = (not relevant) or filename in relevant
        supports = exists and in_relevant and measured is not None and measured >= SUPPORT_THRESHOLD
        labels: list[str] = []
        if not exists:
            labels.append(MISSING_EVIDENCE)
        elif measured is not None and not supports:
            labels.append(UNSUPPORTED_CLAIM)
        scored.append(
            ClaimScore(
                claim=claim.strip(),
                marker=number,
                citation_exists=exists,
                supports=supports,
                support_ratio=ratio if exists else 0.0,
                filename=filename,
                failures=tuple(labels),
            )
        )
    if expect_answer and not scored:
        scored.append(
            ClaimScore(
                claim=answer.strip(),
                marker=0,
                citation_exists=False,
                supports=False,
                support_ratio=0.0,
                filename="",
                failures=(MISSING_EVIDENCE,),
            )
        )
    return scored


def _classify_failures(
    report,
    spec: ResearchQualitySpec,
    claims: tuple[ClaimScore, ...],
    *,
    coverage: float,
    freshness: float,
    conflicted: bool,
) -> tuple[str, ...]:
    labels: list[str] = []
    text = getattr(report, "report", "") or ""
    gaps = tuple(getattr(report, "gaps", ()) or ())
    if any(MISSING_EVIDENCE in item.failures for item in claims) or (
        spec.expect_answer and not getattr(report, "citations", [])
    ):
        labels.append(MISSING_EVIDENCE)
    if any(UNSUPPORTED_CLAIM in item.failures for item in claims):
        labels.append(UNSUPPORTED_CLAIM)
    if conflicted:
        labels.append(CONFLICTING_EVIDENCE)
    if freshness < 1.0:
        labels.append(STALE_SOURCE)
    incomplete = spec.expect_answer and (
        is_refusal(text) or coverage < 1.0 or (spec.expect_sufficient and not getattr(report, "sufficient_evidence", False))
    )
    if incomplete or (spec.expect_answer and gaps and spec.expect_sufficient):
        labels.append(INCOMPLETE_ANSWER)
    seen: list[str] = []
    for item in labels:
        if item not in seen:
            seen.append(item)
    return tuple(seen)


def _faithfulness(answer: str, passages: list[str]) -> float:
    if is_refusal(answer):
        return 1.0
    ratios: list[float] = []
    for claim, number in _cited_claims(answer):
        if number < 1 or number > len(passages):
            ratios.append(0.0)
            continue
        ratio = claim_support(claim, passages[number - 1])
        if ratio is None:
            continue
        ratios.append(ratio)
    if not ratios:
        return 0.0
    return sum(ratios) / len(ratios)


def _citation_correctness(
    answer: str,
    passages: list[tuple[str, str]],
    relevant: set[str],
    expect_answer: bool,
) -> float:
    if is_refusal(answer):
        return 0.0 if expect_answer else 1.0
    claims = list(_cited_claims(answer))
    if not claims:
        return 0.0
    correct = 0
    for claim, number in claims:
        if number < 1 or number > len(passages):
            continue
        filename, text = passages[number - 1]
        ratio = claim_support(claim, text)
        if filename in relevant and ratio is not None and ratio >= SUPPORT_THRESHOLD:
            correct += 1
    return correct / len(claims)


def _cited_claims(answer: str) -> list[tuple[str, int]]:
    import re

    claims: list[tuple[str, int]] = []
    previous = 0
    for match in re.finditer(r"\[(\d+)\]", answer):
        claims.append((answer[previous : match.start()], int(match.group(1))))
        previous = match.end()
    return claims


def _gap_score(gaps, spec: ResearchQualitySpec) -> float:
    blob = " ".join(str(item) for item in gaps).casefold()
    if spec.expect_gap:
        if spec.gap_terms:
            return 1.0 if any(term.casefold() in blob for term in spec.gap_terms) else 0.0
        return 1.0 if blob.strip() else 0.0
    return 1.0 if not blob.strip() else 0.0


def _coverage(text: str, spec: ResearchQualitySpec) -> float:
    if not spec.expected_substrings:
        if spec.expect_answer:
            return 0.0 if text.strip() == REFUSAL_TEXT else 1.0
        return 1.0
    return 1.0 if all(item in text for item in spec.expected_substrings) else 0.0


def _source_recall(citations, spec: ResearchQualitySpec) -> float:
    if not spec.relevant:
        return 1.0
    if not spec.expect_answer:
        return 1.0
    cited = {getattr(item, "filename", "") for item in citations}
    return len(cited & spec.relevant) / len(spec.relevant)


def _source_diversity(citations, spec: ResearchQualitySpec) -> float:
    if not spec.expect_answer:
        return 1.0
    cited = {getattr(item, "filename", "") for item in citations if getattr(item, "filename", "")}
    needed = min(2, max(1, len(spec.relevant) or 1))
    if not cited:
        return 0.0
    return 1.0 if len(cited) >= needed else len(cited) / needed


def _freshness(citations, spec: ResearchQualitySpec) -> float:
    if not spec.fresh_after:
        return 1.0
    dates = dict(spec.source_dates)
    cutoff = spec.fresh_after[:10]
    cited = [getattr(item, "filename", "") for item in citations]
    if not cited:
        return 1.0 if not spec.expect_answer else 0.0
    fresh = 0
    for filename in cited:
        stamp = dates.get(filename, "")[:10]
        if stamp and stamp >= cutoff:
            fresh += 1
        elif not stamp:
            fresh += 1
    return fresh / len(cited)
