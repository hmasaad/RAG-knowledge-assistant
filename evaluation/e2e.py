"""End-to-end research checks against known answers and evidence."""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class EndToEndResult:
    id: str
    mode: str
    passed: bool
    quality: ResearchQualityScore
    claim_match: bool
    contradiction_match: bool
    search_match: bool
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
        )
        for item in payload
    ]


def run_end_to_end_evaluation(
    root: Path,
    cases: list[EndToEndCase] | None = None,
) -> dict[str, float]:
    """Score planner, cycle, and claim judge on the same known collections."""
    report = run_end_to_end_suite(root, cases)
    return report


def run_end_to_end_suite(
    root: Path,
    cases: list[EndToEndCase] | None = None,
) -> dict[str, float]:
    if cases is None:
        cases = load_e2e_cases(root / "evaluation" / "datasets" / "research_e2e.json")
    results = [_score_case(root, case) for case in cases]
    return {
        "pass_rate": _rate(item.passed for item in results),
        "reliability": _mean(item.quality.reliability for item in results),
        "known_answer": _mean(item.quality.coverage for item in results),
        "claim_accuracy": _rate(item.claim_match for item in results),
        "contradiction_accuracy": _rate(item.contradiction_match for item in results),
        "search_contract": _rate(item.search_match for item in results),
    }


def _score_case(root: Path, case: EndToEndCase) -> EndToEndResult:
    with tempfile.TemporaryDirectory() as directory:
        services = _services(Path(directory))
        for name in case.documents:
            path = resolve_document(root, name)
            services.ingestion.ingest(path.name, path.read_bytes())
        if case.mode == "research":
            report = services.research.research(case.question, top_k=4)
        else:
            report = services.research.cycle(case.question, top_k=4)
        judged = services.claim_judge.judge_claim(
            case.claim,
            SearchFilters(user_id="local"),
            top_k=4,
        )
        searches = _search_count(report, services, case.mode)
    quality = score_research(report, case.spec)
    claim_match = judged.verdict == case.claim_verdict
    contradiction_match = judged.contradiction == case.claim_contradiction
    search_match = case.search_count is None or searches == case.search_count
    problems: list[str] = []
    if quality.reliability < 1.0:
        problems.append(f"reliability={quality.reliability:.2f}")
    if not claim_match:
        problems.append(f"claim={judged.verdict}, expected {case.claim_verdict}")
    if not contradiction_match:
        problems.append(f"contradiction={judged.contradiction}, expected {case.claim_contradiction}")
    if not search_match:
        problems.append(f"search_count={searches}, expected {case.search_count}")
    return EndToEndResult(
        id=case.id,
        mode=case.mode,
        passed=not problems,
        quality=quality,
        claim_match=claim_match,
        contradiction_match=contradiction_match,
        search_match=search_match,
        detail="; ".join(problems) or "ok",
    )


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
