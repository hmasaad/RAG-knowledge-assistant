"""Repeatable checks for retrieval, refusal, conflicts, citations, and failures."""

import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings
from app.generation.grounding import interpret_model_answer
from app.ingestion.embeddings import LocalEmbedder
from app.ingestion.errors import KnowledgeError
from app.retrieval.search import RankedPassage, SearchService, SourceMetadata
from app.security.prompts import REFUSAL_TEXT
from app.wiring import Services, build_services

_REFUSAL_CATEGORIES = {"irrelevant", "missing"}


@dataclass(frozen=True)
class QualityCase:
    id: str
    category: str
    question: str
    documents: tuple[str, ...] = ()
    expect_evidence: bool = False
    expect_conflict: bool = False
    expect_error: bool = False
    expected_substring: str = ""
    expected_substrings: tuple[str, ...] = ()
    expected_document: str = ""
    expected_documents: tuple[str, ...] = ()
    forbidden_substring: str = ""
    model_answer: str = ""
    passage: str = ""


@dataclass
class CaseResult:
    id: str
    category: str
    passed: bool
    detail: str
    retrieval_hit: bool | None = None
    reciprocal_rank: float | None = None
    refusal_expected: bool | None = None
    refused: bool | None = None
    conflict_expected: bool | None = None
    conflict_reported: bool | None = None
    citation_rejected: bool | None = None
    failure_reported: bool | None = None
    latency_ms: float = 0.0


@dataclass
class QualityReport:
    results: list[CaseResult] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def failures(self) -> list[CaseResult]:
        return [result for result in self.results if not result.passed]


def load_quality_cases(path: Path) -> list[QualityCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases: list[QualityCase] = []
    for item in payload:
        cases.append(
            QualityCase(
                id=item["id"],
                category=item["category"],
                question=item["question"],
                documents=tuple(item.get("documents", ())),
                expect_evidence=bool(item.get("expect_evidence", False)),
                expect_conflict=bool(item.get("expect_conflict", False)),
                expect_error=bool(item.get("expect_error", False)),
                expected_substring=item.get("expected_substring", ""),
                expected_substrings=tuple(item.get("expected_substrings", ())),
                expected_document=item.get("expected_document", ""),
                expected_documents=tuple(item.get("expected_documents", ())),
                forbidden_substring=item.get("forbidden_substring", ""),
                model_answer=item.get("model_answer", ""),
                passage=item.get("passage", ""),
            )
        )
    return cases


def load_baseline(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {name: float(value) for name, value in payload.items()}


def run_quality_suite(root: Path, cases: list[QualityCase] | None = None) -> QualityReport:
    """Index each case in a fresh collection and score it against the local pipeline."""
    if cases is None:
        cases = load_quality_cases(root / "evaluation" / "datasets" / "phase4_quality.json")
    results = [_run_case(root, case) for case in cases]
    return QualityReport(results=results, metrics=summarize(results))


def summarize(results: list[CaseResult]) -> dict[str, float]:
    """Baseline rates. A missing group is reported as 1.0 so an empty slice cannot hide a gap."""
    recall_rows = [result for result in results if result.retrieval_hit is not None]
    rank_rows = [result for result in results if result.reciprocal_rank is not None]
    refusal_rows = [result for result in results if result.refusal_expected is not None]
    conflict_rows = [result for result in results if result.conflict_expected]
    citation_rows = [result for result in results if result.citation_rejected is not None]
    failure_rows = [result for result in results if result.failure_reported is not None]
    latencies = [result.latency_ms for result in results]
    return {
        "pass_rate": _rate(result.passed for result in results),
        "retrieval_recall_at_3": _rate(result.retrieval_hit for result in recall_rows),
        "mean_reciprocal_rank": _mean(result.reciprocal_rank for result in rank_rows),
        "refusal_accuracy": _rate(
            result.refused == result.refusal_expected for result in refusal_rows
        ),
        "conflict_detection_rate": _rate(result.conflict_reported for result in conflict_rows),
        "incorrect_citation_rejection_rate": _rate(
            result.citation_rejected for result in citation_rows
        ),
        "retrieval_failure_reporting_rate": _rate(
            result.failure_reported for result in failure_rows
        ),
        "mean_latency_ms": _mean(latencies),
    }


def _run_case(root: Path, case: QualityCase) -> CaseResult:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        services = _services(Path(directory))
        for name in case.documents:
            path = resolve_document(root, name)
            services.ingestion.ingest(path.name, path.read_bytes())
        if case.category == "incorrect_citation":
            result = _incorrect_citation(case)
        elif case.expect_error:
            result = _retrieval_failure(case, services)
        else:
            result = _answer_case(case, services)
    result.latency_ms = round((time.perf_counter() - started) * 1000, 2)
    return result


def resolve_document(root: Path, name: str) -> Path:
    for folder in (root / "evaluation" / "fixtures", root / "samples"):
        candidate = folder / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(name)


def _services(directory: Path, **overrides) -> Services:
    values = dict(
        embedding_provider="local",
        embedding_dimension=128,
        chroma_path=directory / "chroma",
        sqlite_path=directory / "metadata.db",
        blob_path=directory / "blobs",
        chunk_size=500,
        chunk_overlap=40,
        max_upload_bytes=100_000,
        log_level="WARNING",
        llm_provider="local",
    )
    values.update(overrides)
    return build_services(Settings(**values), embedder=LocalEmbedder(dimension=128))


def _answer_case(case: QualityCase, services: Services) -> CaseResult:
    retrieved = services.search.retrieve(case.question, top_k=3)
    answered = services.generation.answer(case.question, top_k=3)
    filenames = [passage.source.filename for passage in retrieved.passages]
    expected = _expected_documents(case)
    hit = _contains_all(filenames[:3], expected) if expected else None
    rank = _reciprocal_rank(filenames, case.expected_document) if case.expected_document else None
    refused = answered.answer == REFUSAL_TEXT and not answered.sufficient_evidence
    problems: list[str] = []
    if answered.sufficient_evidence != case.expect_evidence:
        problems.append(
            f"sufficient_evidence={answered.sufficient_evidence}, expected {case.expect_evidence}"
        )
    if case.expect_conflict and not answered.conflict:
        problems.append("conflict was not reported")
    if not case.expect_conflict and answered.conflict:
        problems.append("unexpected conflict")
    if case.expected_substring and case.expected_substring not in answered.answer:
        problems.append(f"answer missing {case.expected_substring!r}")
    for substring in case.expected_substrings:
        if substring not in answered.answer:
            problems.append(f"answer missing {substring!r}")
    if case.forbidden_substring and case.forbidden_substring in answered.answer:
        problems.append(f"answer contained {case.forbidden_substring!r}")
    cited = {citation.filename for citation in answered.citations}
    for name in case.expected_documents:
        if name not in cited:
            problems.append(f"missing citation for {name}")
    if case.expected_document and case.expect_evidence and case.expected_document not in cited:
        problems.append(f"missing citation for {case.expected_document}")
    if case.expect_evidence and not answered.citations:
        problems.append("expected a citation")
    if not case.expect_evidence and answered.citations:
        problems.append("refusal included a citation")
    if case.category == "injection" and case.expect_evidence:
        if not any(passage.instruction_detected for passage in answered.passages):
            problems.append("instruction was not detected")
    if hit is False:
        problems.append(f"expected documents not in top 3: {expected}")
    return CaseResult(
        id=case.id,
        category=case.category,
        passed=not problems,
        detail="; ".join(problems) or answered.answer,
        retrieval_hit=hit,
        reciprocal_rank=rank,
        refusal_expected=_refusal_expected(case),
        refused=refused,
        conflict_expected=case.expect_conflict,
        conflict_reported=answered.conflict and answered.sufficient_evidence,
    )


def _incorrect_citation(case: QualityCase) -> CaseResult:
    passage = RankedPassage(
        rank=1,
        text=case.passage,
        score=0.9,
        vector_score=0.8,
        lexical_score=0.7,
        answers_question=True,
        matched_terms=(),
        source=SourceMetadata(
            chunk_id="citation-check",
            document_id="citation-check",
            filename="passage.md",
            content_type="text/markdown",
            chunk_index=0,
            start_char=0,
            end_char=len(case.passage),
            page=None,
            heading=None,
        ),
        safe_text=case.passage,
    )
    answered = interpret_model_answer(case.question, case.model_answer, [passage])
    rejected = (
        answered.answer == REFUSAL_TEXT
        and not answered.sufficient_evidence
        and not answered.citations
        and (not case.forbidden_substring or case.forbidden_substring not in answered.answer)
    )
    detail = "citation rejected" if rejected else answered.answer
    return CaseResult(
        id=case.id,
        category=case.category,
        passed=rejected,
        detail=detail,
        refusal_expected=True,
        refused=answered.answer == REFUSAL_TEXT and not answered.sufficient_evidence,
        citation_rejected=rejected,
    )


def _retrieval_failure(case: QualityCase, services: Services) -> CaseResult:
    broken = SearchService(
        vectors=services.vectors,
        embedder=_FailingEmbedder(),
        metadata=services.metadata,
        default_user_id="local",
    )
    try:
        broken.retrieve(case.question, top_k=3, user_id="local")
    except KnowledgeError as exc:
        reported = exc.status_code == 503 and exc.message == "Retrieval failed."
        return CaseResult(
            id=case.id,
            category=case.category,
            passed=reported,
            detail=exc.message if reported else f"{exc.status_code} {exc.message}",
            failure_reported=reported,
        )
    return CaseResult(
        id=case.id,
        category=case.category,
        passed=False,
        detail="retrieval returned a result",
        failure_reported=False,
    )


def _expected_documents(case: QualityCase) -> tuple[str, ...]:
    if case.expected_documents:
        return case.expected_documents
    if case.expected_document:
        return (case.expected_document,)
    return ()


def _refusal_expected(case: QualityCase) -> bool:
    if case.category in _REFUSAL_CATEGORIES:
        return True
    return not case.expect_evidence


def _contains_all(filenames: list[str], expected: tuple[str, ...]) -> bool:
    present = set(filenames)
    return all(name in present for name in expected)


def _reciprocal_rank(filenames: list[str], expected: str) -> float:
    for index, name in enumerate(filenames, start=1):
        if name == expected:
            return 1.0 / index
    return 0.0


def _rate(values) -> float:
    rows = list(values)
    if not rows:
        return 1.0
    return sum(1.0 for value in rows if value) / len(rows)


def _mean(values) -> float:
    rows = [value for value in values if value is not None]
    if not rows:
        return 1.0
    return sum(rows) / len(rows)


class _FailingEmbedder:
    model = "local-hash-bow"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("vector store unavailable")

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("vector store unavailable")
