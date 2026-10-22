"""Score the local pipeline on the standard RAG metrics."""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from evaluation.harness import _services, resolve_document
from evaluation.metrics import (
    answer_relevance,
    citation_correctness,
    faithfulness,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


@dataclass(frozen=True)
class MetricCase:
    id: str
    question: str
    documents: tuple[str, ...]
    relevant: dict[str, float]
    expect_answer: bool
    k: int


def load_metric_cases(path: Path) -> list[MetricCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        MetricCase(
            id=item["id"],
            question=item["question"],
            documents=tuple(item["documents"]),
            relevant={name: float(gain) for name, gain in item.get("relevant", {}).items()},
            expect_answer=bool(item["expect_answer"]),
            k=int(item.get("k", 3)),
        )
        for item in payload
    ]


def run_rag_evaluation(root: Path, cases: list[MetricCase] | None = None) -> dict[str, float]:
    """Index each case and average Recall@K, MRR, nDCG, and the answer metrics."""
    if cases is None:
        cases = load_metric_cases(root / "evaluation" / "datasets" / "rag_metrics.json")
    recalls: list[float] = []
    ranks: list[float] = []
    gains: list[float] = []
    faithful: list[float] = []
    relevant_answers: list[float] = []
    citations: list[float] = []
    for case in cases:
        scores = _score_case(root, case)
        if case.relevant:
            recalls.append(scores["recall_at_k"])
            ranks.append(scores["mrr"])
            gains.append(scores["ndcg_at_k"])
        faithful.append(scores["faithfulness"])
        relevant_answers.append(scores["answer_relevance"])
        citations.append(scores["citation_correctness"])
    return {
        "recall_at_k": _mean(recalls),
        "mrr": mean_reciprocal_rank(ranks),
        "ndcg_at_k": _mean(gains),
        "faithfulness": _mean(faithful),
        "answer_relevance": _mean(relevant_answers),
        "citation_correctness": _mean(citations),
    }


def _score_case(root: Path, case: MetricCase) -> dict[str, float]:
    with tempfile.TemporaryDirectory() as directory:
        services = _services(Path(directory))
        for name in case.documents:
            path = resolve_document(root, name)
            services.ingestion.ingest(path.name, path.read_bytes())
        retrieved = services.search.retrieve(case.question, top_k=case.k)
        answered = services.generation.answer(case.question, top_k=case.k)
    filenames = [passage.source.filename for passage in retrieved.passages]
    relevant = set(case.relevant)
    texts = [passage.model_text() for passage in answered.passages]
    cited = [(passage.source.filename, passage.model_text()) for passage in answered.passages]
    return {
        "recall_at_k": recall_at_k(filenames, relevant, case.k),
        "mrr": reciprocal_rank(filenames, relevant),
        "ndcg_at_k": ndcg_at_k(filenames, case.relevant, case.k),
        "faithfulness": faithfulness(answered.answer, texts),
        "answer_relevance": answer_relevance(
            case.question,
            answered.answer,
            texts,
            expect_answer=case.expect_answer,
        ),
        "citation_correctness": citation_correctness(
            answered.answer,
            cited,
            relevant,
            expect_answer=case.expect_answer,
        ),
    }


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
