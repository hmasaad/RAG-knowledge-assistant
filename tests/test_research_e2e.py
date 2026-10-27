"""The end-to-end harness scores known answers and evidence together."""

from dataclasses import replace
from pathlib import Path

from app.research.quality import ResearchQualitySpec
from evaluation.e2e import EndToEndCase, _score_case, run_end_to_end_evaluation
from evaluation.harness import load_baseline

ROOT = Path(__file__).resolve().parents[1]


def test_end_to_end_suite_meets_the_baseline() -> None:
    scores = run_end_to_end_evaluation(ROOT)
    baseline = load_baseline(ROOT / "evaluation" / "baselines" / "research_e2e.json")

    assert "completed" not in scores
    assert scores["evidence_backed"] == 1.0
    assert scores["budget_respect"] == 1.0
    assert scores["recovery"] == 1.0
    assert scores["refusal_accuracy"] == 1.0
    assert scores["security_respect"] == 1.0
    for name, floor in baseline.items():
        assert scores[name] >= floor, f"{name}={scores[name]} below {floor}"


def test_end_to_end_case_requires_the_known_answer_and_the_claim() -> None:
    case = EndToEndCase(
        id="retry-file-cycle",
        question="which file implements the retry logic",
        documents=("api-errors.md",),
        mode="cycle",
        spec=ResearchQualitySpec(
            expect_answer=True,
            expect_sufficient=True,
            expect_gap=False,
            expect_conflict=False,
            relevant=frozenset({"api-errors.md"}),
            expected_substrings=("retry.py",),
        ),
        claim="retry_request applies exponential backoff with jitter and allows three attempts.",
        claim_verdict="supported",
        claim_contradiction=False,
    )
    wrong = replace(case, claim_verdict="insufficient")
    good = _score_case(ROOT, case)
    bad = _score_case(ROOT, wrong)

    assert good.passed is True
    assert good.claim_match is True
    assert good.quality.coverage == 1.0
    assert bad.passed is False
    assert bad.claim_match is False
