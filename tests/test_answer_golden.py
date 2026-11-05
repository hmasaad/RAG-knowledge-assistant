"""Golden answers are scored with the evidence held fixed."""

from pathlib import Path

from evaluation.answer_golden import load_golden_cases, run_answer_golden, score_golden
from evaluation.harness import load_baseline

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "evaluation" / "datasets" / "answer_golden.json"


def test_every_gold_answer_is_complete_and_correctly_cited() -> None:
    for case in load_golden_cases(CASES):
        score = score_golden(case, case.gold_answer)

        assert score.completeness == 1, case.id
        assert score.citation_integrity == 1, case.id


def test_a_missing_fact_lowers_completeness_while_the_evidence_is_present() -> None:
    case = _case("both-limits")
    answer = "The retry limit is three attempts. [1]"

    score = score_golden(case, answer)

    assert score.completeness == 0.5
    assert score.citation_integrity == 0.5


def test_a_swapped_citation_keeps_the_facts_and_fails_the_citation() -> None:
    case = _case("both-limits")
    answer = "The retry limit is three attempts. [2] The retry limit is five attempts. [1]"

    score = score_golden(case, answer)

    assert score.completeness == 1
    assert score.citation_integrity == 0


def test_an_uncited_fact_does_not_count_as_citation_integrity() -> None:
    case = _case("retry-file")
    answer = "The file that implements retry logic is app/http/retry.py."

    score = score_golden(case, answer)

    assert score.completeness == 1
    assert score.citation_integrity == 0


def test_a_guess_is_not_credited_when_the_evidence_does_not_answer() -> None:
    case = _case("wifi-refusal")

    score = score_golden(case, "The office wifi password is violet. [1]")

    assert score.completeness == 0
    assert score.citation_integrity == 0


def test_the_generator_meets_the_golden_baseline_without_retrieval_metrics() -> None:
    report = run_answer_golden(ROOT)
    baseline = load_baseline(ROOT / "evaluation" / "baselines" / "answer_golden.json")

    assert set(report) == {"completeness", "citation_integrity"}
    for name, floor in baseline.items():
        assert report[name] >= floor, f"{name}={report[name]} floor={floor}"


def _case(case_id: str):
    return next(case for case in load_golden_cases(CASES) if case.id == case_id)
