"""Research quality measures the report, not whether the workflow finished."""

from pathlib import Path
from types import SimpleNamespace

from evaluation.harness import load_baseline
from evaluation.research_quality import run_research_quality_evaluation
from app.research.contracts import FILL_GAPS, RESOLVE_CONTRADICTIONS, REVERIFY
from app.research.loop import PROBE_CONFLICT, LoopState, decide
from app.research.quality import (
    CONFLICTING_EVIDENCE,
    CRITERIA,
    INCOMPLETE_ANSWER,
    MISSING_EVIDENCE,
    STALE_SOURCE,
    UNSUPPORTED_CLAIM,
    ResearchQualitySpec,
    evaluate_research,
    recommend_research,
    score_research,
)
from app.security.prompts import REFUSAL_TEXT

ROOT = Path(__file__).resolve().parents[1]


def test_completed_workflow_is_not_reliable_when_the_report_is_wrong() -> None:
    report = SimpleNamespace(
        report="The office wifi password is hunter2. [1]",
        citations=[
            SimpleNamespace(
                filename="api-errors.md",
                quote="The file that implements retry logic is app/http/retry.py.",
            )
        ],
        sufficient_evidence=True,
        gaps=(),
        conflict=False,
        contradictions=(),
        status="completed",
    )
    score = score_research(
        report,
        ResearchQualitySpec(
            expect_answer=True,
            expect_sufficient=True,
            expect_gap=False,
            expect_conflict=False,
            relevant=frozenset({"api-errors.md"}),
            expected_substrings=("retry.py",),
        ),
    )

    assert score.completed is True
    assert score.faithfulness < 1.0
    assert score.citation_correctness == 0.0
    assert score.coverage == 0.0
    assert score.completeness == 0.0
    assert score.factual_support < 1.0
    assert score.citation_validity == 0.0
    assert score.reliability < 1.0
    assert len(score.claims) == 1
    assert score.claims[0].supports is False
    assert UNSUPPORTED_CLAIM in score.claims[0].failures
    assert UNSUPPORTED_CLAIM in score.failures
    assert INCOMPLETE_ANSWER in score.failures
    assert score.recommendation == REVERIFY


def test_refusal_with_a_gap_is_reliable_when_nothing_was_answerable() -> None:
    report = SimpleNamespace(
        report=REFUSAL_TEXT,
        citations=[],
        sufficient_evidence=False,
        gaps=("what is the office wifi password",),
        conflict=False,
        contradictions=(),
        status="completed",
    )
    score = score_research(
        report,
        ResearchQualitySpec(
            expect_answer=False,
            expect_sufficient=False,
            expect_gap=True,
            expect_conflict=False,
            gap_terms=("wifi",),
        ),
    )

    assert score.completed is True
    assert score.faithfulness == 1.0
    assert score.citation_correctness == 1.0
    assert score.gap_detection == 1.0
    assert score.reliability == 1.0
    assert score.recommendation == "stop"
    assert UNSUPPORTED_CLAIM not in score.failures


def test_claim_scores_are_aggregated_after_each_claim_is_judged() -> None:
    report = SimpleNamespace(
        report="Retry lives in retry.py. [1] The wifi password is hunter2. [2]",
        citations=[
            SimpleNamespace(
                filename="api-errors.md",
                quote="The file that implements retry logic is app/http/retry.py.",
            ),
            SimpleNamespace(
                filename="api-errors.md",
                quote="The file that implements retry logic is app/http/retry.py.",
            ),
        ],
        sufficient_evidence=True,
        gaps=(),
        conflict=False,
        contradictions=(),
        status="completed",
    )
    score = evaluate_research(
        report,
        ResearchQualitySpec(
            expect_answer=True,
            expect_sufficient=True,
            expect_gap=False,
            expect_conflict=False,
            relevant=frozenset({"api-errors.md"}),
            expected_substrings=("retry.py",),
        ),
    )

    assert [item.supports for item in score.claims] == [True, False]
    assert score.citation_validity == 0.5
    assert score.factual_support < 1.0
    assert score.recommendation == REVERIFY


def test_grounding_flags_a_missing_citation_and_stale_source() -> None:
    missing = evaluate_research(
        SimpleNamespace(
            report="Retry lives in retry.py. [2]",
            citations=[
                SimpleNamespace(
                    filename="api-errors.md",
                    quote="The file that implements retry logic is app/http/retry.py.",
                )
            ],
            sufficient_evidence=True,
            gaps=(),
            conflict=False,
            contradictions=(),
            status="completed",
        ),
        ResearchQualitySpec(
            expect_answer=True,
            expect_sufficient=True,
            expect_gap=False,
            expect_conflict=False,
            relevant=frozenset({"api-errors.md"}),
            expected_substrings=("retry.py",),
        ),
    )
    stale = evaluate_research(
        SimpleNamespace(
            report="Retry lives in retry.py. [1]",
            citations=[
                SimpleNamespace(
                    filename="api-errors.md",
                    quote="The file that implements retry logic is app/http/retry.py.",
                )
            ],
            sufficient_evidence=True,
            gaps=(),
            conflict=False,
            contradictions=(),
            status="completed",
        ),
        ResearchQualitySpec(
            expect_answer=True,
            expect_sufficient=True,
            expect_gap=False,
            expect_conflict=False,
            relevant=frozenset({"api-errors.md"}),
            expected_substrings=("retry.py",),
            fresh_after="2026-01-01",
            source_dates=(("api-errors.md", "2020-01-01"),),
        ),
    )

    assert missing.claims[0].citation_exists is False
    assert MISSING_EVIDENCE in missing.failures
    assert missing.recommendation == FILL_GAPS
    assert stale.freshness == 0.0
    assert STALE_SOURCE in stale.failures
    assert stale.recommendation == FILL_GAPS
    assert set(CRITERIA) <= {
        "completeness",
        "factual_support",
        "citation_validity",
        "source_diversity",
        "freshness",
        "unresolved_contradictions",
    }


def test_conflict_is_classified_and_the_loop_is_told_to_resolve_or_stop() -> None:
    conflicted = SimpleNamespace(
        report="The retry limit is three [1] or five [2].",
        citations=[
            SimpleNamespace(filename="retry-limit-three.md", quote="The retry limit is three."),
            SimpleNamespace(filename="retry-limit-five.md", quote="The retry limit is five."),
        ],
        sufficient_evidence=True,
        gaps=(),
        conflict=True,
        contradictions=("The documents disagree on the retry limit.",),
        status="completed",
    )
    spec = ResearchQualitySpec(
        expect_answer=True,
        expect_sufficient=True,
        expect_gap=False,
        expect_conflict=True,
        relevant=frozenset({"retry-limit-three.md", "retry-limit-five.md"}),
        expected_substrings=("three", "five"),
    )
    expected = evaluate_research(conflicted, spec)
    unexpected = evaluate_research(
        conflicted,
        ResearchQualitySpec(
            expect_answer=True,
            expect_sufficient=True,
            expect_gap=False,
            expect_conflict=False,
            relevant=frozenset({"retry-limit-three.md", "retry-limit-five.md"}),
            expected_substrings=("three", "five"),
        ),
    )
    hint = decide(
        LoopState(
            complete=False,
            supported=1,
            gaps=(),
            unresolved_conflicts=('The documents disagree on "what is the retry limit".',),
            remaining_budget=4,
            quality_action=RESOLVE_CONTRADICTIONS,
            quality_reason="Unresolved contradiction.",
        )
    )

    assert CONFLICTING_EVIDENCE in expected.failures
    assert expected.source_diversity == 1.0
    assert expected.recommendation == "stop"
    assert unexpected.recommendation == RESOLVE_CONTRADICTIONS
    assert recommend_research(unexpected)[0] == RESOLVE_CONTRADICTIONS
    assert hint.action == PROBE_CONFLICT


def test_research_quality_suite_meets_the_baseline() -> None:
    scores = run_research_quality_evaluation(ROOT)
    baseline = load_baseline(ROOT / "evaluation" / "baselines" / "research_quality.json")

    assert "completed" not in scores
    for name, floor in baseline.items():
        assert scores[name] >= floor, f"{name}={scores[name]} below {floor}"
