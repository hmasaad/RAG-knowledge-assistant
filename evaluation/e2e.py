"""End-to-end checks for answers, budgets, recovery, and honest refusals."""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.generation.grounding import is_refusal
from app.research.quality import ResearchQualityScore, ResearchQualitySpec, score_research
from app.research.rag import SearchFilters
from evaluation.harness import _services, resolve_document


@dataclass(frozen=True)
class EndToEndCase:
    id: str
    question: str
    documents: tuple[str, ...]
    mode: str
    spec: ResearchQualitySpec
    claim: str
    claim_verdict: str
    claim_contradiction: bool
    search_count: int | None = None
    kind: str = "answer"
    search_budget: int | None = None
    halt_after: int | None = None
    user_id: str = "local"
    owners: tuple[tuple[str, str], ...] = ()
    forbidden_substrings: tuple[str, ...] = ()


@dataclass(frozen=True)
class EndToEndResult:
    id: str
    mode: str
    kind: str
    passed: bool
    quality: ResearchQualityScore
    claim_match: bool
    contradiction_match: bool
    search_match: bool
    evidence_backed: bool
    budget_ok: bool
    recovered: bool
    refused_correctly: bool
    secure: bool
    detail: str


def load_e2e_cases(path: Path) -> list[EndToEndCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        EndToEndCase(
            id=item["id"],
            question=item["question"],
            documents=tuple(item["documents"]),
            mode=item.get("mode", "cycle"),
            spec=ResearchQualitySpec(
                expect_answer=bool(item["expect_answer"]),
                expect_sufficient=bool(item.get("expect_sufficient", item["expect_answer"])),
                expect_gap=bool(item.get("expect_gap", False)),
                expect_conflict=bool(item.get("expect_conflict", False)),
                relevant=frozenset(item.get("relevant", ())),
                expected_substrings=tuple(item.get("expected_substrings", ())),
                gap_terms=tuple(item.get("gap_terms", ())),
            ),
            claim=item["claim"],
            claim_verdict=item["claim_verdict"],
            claim_contradiction=bool(item.get("claim_contradiction", False)),
            search_count=item.get("search_count"),
            kind=item.get("kind", "answer"),
            search_budget=item.get("search_budget"),
            halt_after=item.get("halt_after"),
            user_id=item.get("user_id", "local"),
            owners=tuple((name, owner) for name, owner in (item.get("owners") or {}).items()),
            forbidden_substrings=tuple(item.get("forbidden_substrings", ())),
        )
        for item in payload
    ]


def run_end_to_end_evaluation(
    root: Path,
    cases: list[EndToEndCase] | None = None,
) -> dict[str, float]:
    """Score the integrated agent on answers, budget, recovery, and refusal."""
    return run_end_to_end_suite(root, cases)


def run_end_to_end_suite(
    root: Path,
    cases: list[EndToEndCase] | None = None,
) -> dict[str, float]:
    if cases is None:
        cases = load_e2e_cases(root / "evaluation" / "datasets" / "research_e2e.json")
    results = [_score_case(root, case) for case in cases]
    return {
        "pass_rate": _rate(item.passed for item in results),
        "reliability": _mean(
            item.quality.reliability for item in results if item.kind != "security"
        ),
        "known_answer": _mean(item.quality.coverage for item in results),
        "claim_accuracy": _rate(item.claim_match for item in results),
        "contradiction_accuracy": _rate(item.contradiction_match for item in results),
        "search_contract": _rate(item.search_match for item in results),
        "evidence_backed": _rate(item.evidence_backed for item in results),
        "budget_respect": _rate(item.budget_ok for item in results),
        "recovery": _rate(item.recovered for item in results),
        "refusal_accuracy": _rate(item.refused_correctly for item in results),
        "security_respect": _rate(item.secure for item in results),
    }


def _score_case(root: Path, case: EndToEndCase) -> EndToEndResult:
    overrides = {}
    if case.search_budget is not None:
        overrides["research_search_budget"] = case.search_budget
    with tempfile.TemporaryDirectory() as directory:
        services = _services(Path(directory), **overrides)
        owners = dict(case.owners)
        for name in case.documents:
            path = resolve_document(root, name)
            services.ingestion.ingest(
                path.name,
                path.read_bytes(),
                owner=owners.get(name, "local"),
            )
        report, recovered = _run_agent(services, case)
        judged = services.claim_judge.judge_claim(
            case.claim,
            SearchFilters(user_id=case.user_id),
            top_k=4,
        )
        searches = _search_count(report, services, case.mode)
        remaining = int(getattr(report, "remaining_budget", 0) or 0)
        side_effects = list(getattr(report, "side_effects", ()) or ())
    quality = score_research(report, case.spec)
    claim_match = judged.verdict == case.claim_verdict
    contradiction_match = judged.contradiction == case.claim_contradiction
    limit = case.search_budget if case.search_budget is not None else 16
    search_match = case.search_count is None or searches == case.search_count
    budget_ok = remaining >= 0 and searches <= limit
    text = getattr(report, "report", "") or ""
    citations = list(getattr(report, "citations", []) or [])
    refused = is_refusal(text)
    if case.spec.expect_answer:
        evidence_backed = (not refused) and quality.coverage == 1.0 and bool(citations or case.spec.expect_conflict)
    else:
        evidence_backed = refused
    refused_correctly = refused == (not case.spec.expect_answer)
    if case.kind == "recover":
        recovered = recovered and getattr(report, "status", "") == "completed"
        search_runs = [item.get("runs") for item in side_effects if item.get("action") == "search"]
        recovered = recovered and search_runs == [1]
    problems: list[str] = []
    if quality.reliability < 1.0 and case.kind != "security":
        problems.append(f"reliability={quality.reliability:.2f}")
    if not claim_match:
        problems.append(f"claim={judged.verdict}, expected {case.claim_verdict}")
    if not contradiction_match:
        problems.append(f"contradiction={judged.contradiction}, expected {case.claim_contradiction}")
    if not search_match:
        problems.append(f"search_count={searches}, expected {case.search_count}")
    if not evidence_backed:
        problems.append("answer was not evidence-backed")
    if not budget_ok:
        problems.append(f"budget remaining={remaining} searches={searches} limit={limit}")
    if not recovered:
        problems.append("session did not recover without repeating work")
    if not refused_correctly:
        problems.append(f"refused={refused}, expect_answer={case.spec.expect_answer}")
    blob = " ".join([text, *getattr(report, "gaps", ())]).casefold()
    leaked = [term for term in case.forbidden_substrings if term.casefold() in blob]
    secure = not leaked
    if not secure:
        problems.append(f"leaked {leaked!r}")
    return EndToEndResult(
        id=case.id,
        mode=case.mode,
        kind=case.kind,
        passed=not problems,
        quality=quality,
        claim_match=claim_match,
        contradiction_match=contradiction_match,
        search_match=search_match,
        evidence_backed=evidence_backed,
        budget_ok=budget_ok,
        recovered=recovered,
        refused_correctly=refused_correctly,
        secure=secure,
        detail="; ".join(problems) or "ok",
    )


def _run_agent(services, case: EndToEndCase):
    if case.kind == "recover":
        halted = services.research.cycle(
            case.question,
            top_k=4,
            user_id=case.user_id,
            halt_after=case.halt_after or 1,
        )
        if getattr(halted, "status", "") != "halted":
            return halted, False
        report = services.research.resume(halted.session_id, user_id=case.user_id)
        return report, True
    if case.mode == "research":
        return services.research.research(case.question, top_k=4, user_id=case.user_id), True
    return services.research.cycle(case.question, top_k=4, user_id=case.user_id), True


def _search_count(report, services, mode: str) -> int:
    if mode == "research":
        trace = services.traces.get(getattr(report, "trace_id", "")) or {}
        return [step["action"] for step in trace.get("steps", [])].count("search")
    return list(getattr(report, "decisions", ()) or ()).count("search")


def _rate(values) -> float:
    rows = list(values)
    if not rows:
        return 0.0
    return sum(1.0 for value in rows if value) / len(rows)


def _mean(values) -> float:
    rows = list(values)
    if not rows:
        return 0.0
    return sum(rows) / len(rows)
