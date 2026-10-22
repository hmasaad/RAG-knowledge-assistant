"""Recall@K, MRR, nDCG, faithfulness, answer relevance, and citation correctness."""

from pathlib import Path

from evaluation.harness import load_baseline
from evaluation.metrics import (
    answer_relevance,
    citation_correctness,
    faithfulness,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from evaluation.rag_eval import run_rag_evaluation

ROOT = Path(__file__).resolve().parents[1]


def test_ranking_metrics_score_graded_lists() -> None:
    retrieved = ["api-errors.md", "database.md", "other.md"]

    assert recall_at_k(retrieved, {"api-errors.md", "other.md"}, 2) == 0.5
    assert reciprocal_rank(retrieved, {"database.md"}) == 0.5
    assert ndcg_at_k(["database.md", "api-errors.md"], {"api-errors.md": 2, "database.md": 1}, 2) < 1
    assert ndcg_at_k(["api-errors.md", "database.md"], {"api-errors.md": 2, "database.md": 1}, 2) == 1


def test_answer_metrics_separate_a_quote_from_an_unsupported_claim() -> None:
    passage = "The file that implements retry logic is app/http/retry.py."
    quoted = f"The file that implements retry logic is app/http/retry.py. [1]"
    invented = "The office wifi password is hunter2. [1]"

    assert faithfulness(quoted, [passage]) == 1
    assert faithfulness(invented, [passage]) < 0.6
    assert faithfulness("I don't have enough evidence to answer.", [passage]) == 1
    assert (
        answer_relevance(
            "which file implements the retry logic",
            quoted,
            [passage],
            expect_answer=True,
        )
        == 1
    )
    assert (
        citation_correctness(quoted, [("api-errors.md", passage)], {"api-errors.md"}, expect_answer=True)
        == 1
    )
    assert (
        citation_correctness(invented, [("api-errors.md", passage)], {"api-errors.md"}, expect_answer=True)
        == 0
    )


def test_local_pipeline_meets_the_metric_baseline() -> None:
    report = run_rag_evaluation(ROOT)
    baseline = load_baseline(ROOT / "evaluation" / "baselines" / "rag_metrics.json")

    for name, floor in baseline.items():
        assert report[name] >= floor, f"{name}={report[name]} floor={floor}"
