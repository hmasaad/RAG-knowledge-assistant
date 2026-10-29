"""Permanent salon MVP regression benchmark, scored after a real document upload."""

import json
import mimetypes
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.config import Settings
from app.generation.grounding import SUPPORT_THRESHOLD, claim_support, is_refusal
from app.ingestion.embeddings import LocalEmbedder
from app.ingestion.parsers import parse_document
from app.main import create_app
from evaluation.harness import load_baseline, resolve_document
from evaluation.metrics import _cited_claims

ROOT = Path(__file__).resolve().parents[1]
_MVP_MARK = re.compile(r"(?i)mvp\s+scope")
_ITEM = re.compile(r"(?:^|\n)\s*(\d+)\.\s+([^\n]+)")
_HEADING = re.compile(r"(?m)^\s*#{1,6}\s+")


@dataclass(frozen=True)
class MvpBenchmarkCase:
    id: str
    question: str
    documents: tuple[str, ...]
    required_claims: tuple[str, ...]
    forbidden_substrings: tuple[str, ...]
    mode: str = "cycle"
    retry_limit: int = 1


@dataclass(frozen=True)
class MvpBenchmarkScore:
    evidence_coverage: float
    citation_validity: float
    citation_support_rate: float
    unsupported_claim_rate: float
    repair_success_rate: float
    represented: int
    required: int
    repaired: bool


def load_mvp_benchmark_case(path: Path | None = None, *, root: Path | None = None) -> MvpBenchmarkCase:
    payload = json.loads((path or _dataset_path()).read_text(encoding="utf-8"))
    documents = tuple(payload["documents"])
    claims = tuple(payload.get("required_claims") or ())
    if not claims:
        claims = required_claims_from_files(documents, root or ROOT)
    return MvpBenchmarkCase(
        id=str(payload["id"]),
        question=str(payload["question"]),
        documents=documents,
        required_claims=claims,
        forbidden_substrings=tuple(payload.get("forbidden_substrings", ())),
        mode=str(payload.get("mode", "cycle")),
        retry_limit=int(payload.get("retry_limit", 1) or 0),
    )


def required_claims_from_files(names: tuple[str, ...], root: Path) -> tuple[str, ...]:
    """Read MVP bullets from the same files the benchmark uploads."""
    items: list[str] = []
    for name in names:
        path = resolve_document(root, name)
        items.extend(required_claims_from_bytes(path.name, path.read_bytes()))
    return tuple(dict.fromkeys(items))


def required_claims_from_bytes(filename: str, data: bytes) -> tuple[str, ...]:
    parsed = parse_document(filename, data)
    text = "\n".join(page.text for page in parsed.pages)
    return extract_mvp_scope_items(text)


def extract_mvp_scope_items(text: str) -> tuple[str, ...]:
    """Numbered bullets under an MVP Scope heading in uploaded text or PDF extract."""
    match = _MVP_MARK.search(text or "")
    if match is None:
        return ()
    rest = (text or "")[match.end() :]
    heading = _HEADING.search(rest)
    if heading is not None:
        rest = rest[: heading.start()]
    items: list[str] = []
    for found in _ITEM.finditer(rest):
        number = int(found.group(1))
        line = found.group(2).strip().rstrip(".")
        if number == 1 and items:
            break
        if number != len(items) + 1:
            continue
        items.append(line)
        if number >= 10:
            break
    return tuple(items)


def run_mvp_benchmark(
    root: Path | None = None,
    case: MvpBenchmarkCase | None = None,
) -> dict[str, float]:
    """Upload the document through the API, then score the research cycle report."""
    base = root or ROOT
    spec = case or load_mvp_benchmark_case(root=base)
    with tempfile.TemporaryDirectory() as directory:
        client = _client(Path(directory))
        with client:
            _upload(client, spec.documents, base)
            report = _ask_research(client, spec)
            first = score_mvp_report(report, spec)
            if _meets_targets(first) or spec.retry_limit < 1:
                return _metrics(first)
            attempts = 0
            successes = 0
            for _ in range(spec.retry_limit):
                attempts += 1
                repaired = score_mvp_report(_ask_research(client, spec), spec)
                if _meets_targets(repaired):
                    successes += 1
                    return _metrics(
                        MvpBenchmarkScore(
                            evidence_coverage=repaired.evidence_coverage,
                            citation_validity=repaired.citation_validity,
                            citation_support_rate=repaired.citation_support_rate,
                            unsupported_claim_rate=repaired.unsupported_claim_rate,
                            repair_success_rate=successes / attempts,
                            represented=repaired.represented,
                            required=repaired.required,
                            repaired=True,
                        )
                    )
            return _metrics(
                MvpBenchmarkScore(
                    evidence_coverage=first.evidence_coverage,
                    citation_validity=first.citation_validity,
                    citation_support_rate=first.citation_support_rate,
                    unsupported_claim_rate=first.unsupported_claim_rate,
                    repair_success_rate=0.0 if attempts else 1.0,
                    represented=first.represented,
                    required=first.required,
                    repaired=False,
                )
            )


def score_mvp_report(report, case: MvpBenchmarkCase) -> MvpBenchmarkScore:
    """Score one report. Used by the suite and by unit checks of the metric definitions."""
    payload = _as_report(report)
    text = payload.report
    quotes = _citation_texts(payload)
    required = list(case.required_claims)
    represented = sum(1 for item in required if _contains(text, item))
    coverage = represented / len(required) if required else 1.0
    claims = _cited_claims(text)
    valid = 0
    supported = 0
    evaluated = 0
    for claim, number in claims:
        if not claim.strip():
            continue
        evaluated += 1
        exists = 1 <= number <= len(quotes)
        if exists:
            valid += 1
            ratio = claim_support(claim, quotes[number - 1])
            if ratio is not None and ratio >= SUPPORT_THRESHOLD:
                supported += 1
    if is_refusal(text):
        coverage = 0.0
        represented = 0
    leaked = any(term.casefold() in text.casefold() for term in case.forbidden_substrings)
    if leaked:
        supported = 0
    citation_validity = 1.0 if not claims else valid / len(claims)
    support_rate = 1.0 if evaluated == 0 and not is_refusal(text) else (
        0.0 if evaluated == 0 else supported / evaluated
    )
    unsupported_rate = 0.0 if evaluated == 0 else (evaluated - supported) / evaluated
    if is_refusal(text):
        citation_validity = 0.0
        support_rate = 0.0
        unsupported_rate = 1.0
    return MvpBenchmarkScore(
        evidence_coverage=coverage,
        citation_validity=citation_validity,
        citation_support_rate=support_rate,
        unsupported_claim_rate=unsupported_rate,
        repair_success_rate=1.0,
        represented=represented,
        required=len(required),
        repaired=False,
    )


def mvp_benchmark_baseline(root: Path | None = None) -> dict[str, float]:
    base = root or ROOT
    return load_baseline(base / "evaluation" / "baselines" / "salon_mvp_benchmark.json")


def _client(directory: Path) -> TestClient:
    settings = Settings(
        embedding_provider="local",
        embedding_dimension=128,
        chroma_path=directory / "chroma",
        sqlite_path=directory / "metadata.db",
        blob_path=directory / "blobs",
        chunk_size=500,
        chunk_overlap=40,
        max_upload_bytes=10 * 1024 * 1024,
        log_level="WARNING",
        llm_provider="local",
    )
    return TestClient(create_app(settings, embedder=LocalEmbedder(dimension=128)))


def _upload(client: TestClient, names: tuple[str, ...], root: Path) -> None:
    for name in names:
        path = resolve_document(root, name)
        data = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix.lower() == ".md":
            content_type = "text/markdown"
        uploaded = client.post(
            "/api/documents",
            files={"file": (path.name, data, content_type)},
        )
        if uploaded.status_code not in {200, 201, 202}:
            raise RuntimeError(f"Upload failed for {path.name}: {uploaded.status_code} {uploaded.text}")


def _ask_research(client: TestClient, case: MvpBenchmarkCase) -> SimpleNamespace:
    path = "/api/research/cycle" if case.mode != "research" else "/api/research"
    response = client.post(path, json={"question": case.question, "top_k": 4})
    if response.status_code != 200:
        raise RuntimeError(f"Research failed: {response.status_code} {response.text}")
    return _as_report(response.json())


def _as_report(report) -> SimpleNamespace:
    if isinstance(report, dict):
        citations = [
            SimpleNamespace(**item) if isinstance(item, dict) else item
            for item in report.get("citations") or []
        ]
        passages = [
            SimpleNamespace(**item) if isinstance(item, dict) else item
            for item in report.get("passages") or []
        ]
        return SimpleNamespace(
            report=report.get("report") or report.get("answer") or "",
            citations=citations,
            passages=passages,
        )
    return SimpleNamespace(
        report=getattr(report, "report", "") or "",
        citations=list(getattr(report, "citations", []) or []),
        passages=list(getattr(report, "passages", []) or []),
    )


def _citation_texts(report) -> list[str]:
    citations = list(getattr(report, "citations", []) or [])
    passages = list(getattr(report, "passages", []) or [])
    by_chunk = {}
    for item in passages:
        source = getattr(item, "source", None)
        chunk_id = getattr(source, "chunk_id", None) if source is not None else None
        if chunk_id is None and isinstance(item, dict):
            chunk_id = (item.get("source") or {}).get("chunk_id") or item.get("chunk_id")
        if chunk_id:
            by_chunk[chunk_id] = item
    quotes: list[str] = []
    for item in citations:
        quote = _field(item, "quote", "")
        chunk_id = _field(item, "chunk_id", "")
        passage = by_chunk.get(chunk_id)
        if passage is not None:
            text = ""
            if hasattr(passage, "model_text"):
                text = passage.model_text()
            else:
                text = _field(passage, "text", "")
            if text:
                quote = text
        quotes.append(quote)
    if quotes:
        return quotes
    return [_field(item, "text", "") for item in passages]


def _field(item, name: str, default=""):
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _contains(answer: str, required: str) -> bool:
    return required.casefold() in (answer or "").casefold()


def _meets_targets(score: MvpBenchmarkScore) -> bool:
    return (
        score.evidence_coverage == 1.0
        and score.citation_validity == 1.0
        and score.citation_support_rate == 1.0
        and score.unsupported_claim_rate == 0.0
    )


def _metrics(score: MvpBenchmarkScore) -> dict[str, float]:
    return {
        "evidence_coverage": score.evidence_coverage,
        "citation_validity": score.citation_validity,
        "citation_support_rate": score.citation_support_rate,
        "unsupported_claim_rate": score.unsupported_claim_rate,
        "repair_success_rate": score.repair_success_rate,
    }


def _dataset_path() -> Path:
    return ROOT / "evaluation" / "datasets" / "salon_mvp_benchmark.json"
