"""Run the research agent against reviewed cases and score reliability."""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.research.quality import ResearchQualityScore, ResearchQualitySpec, score_research
from evaluation.harness import _services, resolve_document


@dataclass(frozen=True)
class ResearchQualityCase:
    id: str
    question: str
    documents: tuple[str, ...]
    mode: str
    spec: ResearchQualitySpec


def load_research_quality_cases(path: Path) -> list[ResearchQualityCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        ResearchQualityCase(
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
        )
        for item in payload
    ]


def run_research_quality_evaluation(
    root: Path,
    cases: list[ResearchQualityCase] | None = None,
) -> dict[str, float]:
    """Average reliability metrics. Completion is not one of them."""
    if cases is None:
        cases = load_research_quality_cases(
            root / "evaluation" / "datasets" / "research_quality.json"
        )
    scores = [_score_case(root, case) for case in cases]
    return {
        "faithfulness": _mean(item.faithfulness for item in scores),
        "citation_correctness": _mean(item.citation_correctness for item in scores),
        "sufficiency": _mean(item.sufficiency for item in scores),
        "gap_detection": _mean(item.gap_detection for item in scores),
        "conflict_detection": _mean(item.conflict_detection for item in scores),
        "coverage": _mean(item.coverage for item in scores),
        "source_recall": _mean(item.source_recall for item in scores),
        "reliability": _mean(item.reliability for item in scores),
    }


def _score_case(root: Path, case: ResearchQualityCase) -> ResearchQualityScore:
    with tempfile.TemporaryDirectory() as directory:
        services = _services(Path(directory))
        for name in case.documents:
            path = resolve_document(root, name)
            services.ingestion.ingest(path.name, path.read_bytes())
        if case.mode == "research":
            report = services.research.research(case.question, top_k=4)
        else:
            report = services.research.cycle(case.question, top_k=4)
    return score_research(report, case.spec)


def _mean(values) -> float:
    rows = list(values)
    if not rows:
        return 0.0
    return sum(rows) / len(rows)
