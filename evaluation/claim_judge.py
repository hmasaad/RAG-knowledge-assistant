"""Precision and recall for claim verification and contradiction detection."""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from evaluation.harness import _services, resolve_document
from app.research.judge import ClaimJudgeAdapter, LocalClaimJudge
from app.research.rag import KnowledgeRagClient, SearchFilters


@dataclass(frozen=True)
class JudgeCase:
    id: str
    claim: str
    documents: tuple[str, ...]
    verdict: str
    contradiction: bool


def load_judge_cases(path: Path) -> list[JudgeCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        JudgeCase(
            id=item["id"],
            claim=item["claim"],
            documents=tuple(item["documents"]),
            verdict=item["verdict"],
            contradiction=bool(item["contradiction"]),
        )
        for item in payload
    ]


def precision_recall(gold: list[bool], predicted: list[bool]) -> tuple[float, float]:
    """Binary precision and recall. An empty denominator is 0."""
    true_positive = sum(1 for expected, actual in zip(gold, predicted, strict=True) if expected and actual)
    false_positive = sum(
        1 for expected, actual in zip(gold, predicted, strict=True) if not expected and actual
    )
    false_negative = sum(
        1 for expected, actual in zip(gold, predicted, strict=True) if expected and not actual
    )
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return precision, recall


def run_claim_judge_evaluation(root: Path, cases: list[JudgeCase] | None = None) -> dict[str, float]:
    """Score the local judge on the human-reviewed claim set."""
    if cases is None:
        cases = load_judge_cases(root / "evaluation" / "datasets" / "claim_judge.json")
    verification_gold: list[bool] = []
    verification_predicted: list[bool] = []
    contradiction_gold: list[bool] = []
    contradiction_predicted: list[bool] = []
    for case in cases:
        verdict = _judge_case(root, case)
        verification_gold.append(case.verdict == "supported")
        verification_predicted.append(verdict.verdict == "supported")
        contradiction_gold.append(case.contradiction)
        contradiction_predicted.append(verdict.contradiction)
    verification_precision, verification_recall = precision_recall(
        verification_gold, verification_predicted
    )
    contradiction_precision, contradiction_recall = precision_recall(
        contradiction_gold, contradiction_predicted
    )
    return {
        "verification_precision": verification_precision,
        "verification_recall": verification_recall,
        "contradiction_precision": contradiction_precision,
        "contradiction_recall": contradiction_recall,
    }


def _judge_case(root: Path, case: JudgeCase):
    with tempfile.TemporaryDirectory() as directory:
        services = _services(Path(directory))
        for name in case.documents:
            path = resolve_document(root, name)
            services.ingestion.ingest(path.name, path.read_bytes())
        adapter = ClaimJudgeAdapter(
            rag=KnowledgeRagClient(services.search),
            judge=LocalClaimJudge(),
        )
        return adapter.judge_claim(case.claim, SearchFilters(user_id="local"), top_k=4)
