"""The salon MVP question is a permanent regression benchmark."""

from pathlib import Path
from types import SimpleNamespace

from evaluation.harness import load_baseline
from evaluation.mvp_benchmark import (
    extract_mvp_scope_items,
    load_mvp_benchmark_case,
    mvp_benchmark_baseline,
    required_claims_from_files,
    run_mvp_benchmark,
    score_mvp_report,
)

ROOT = Path(__file__).resolve().parents[1]


def test_salon_mvp_benchmark_meets_the_baseline() -> None:
    scores = run_mvp_benchmark(ROOT)
    baseline = mvp_benchmark_baseline(ROOT)

    assert scores["evidence_coverage"] == 1.0
    assert scores["citation_validity"] == 1.0
    assert scores["citation_support_rate"] == 1.0
    assert scores["unsupported_claim_rate"] == 0.0
    assert scores["repair_success_rate"] == 1.0
    for name, floor in baseline.items():
        if name == "unsupported_claim_rate":
            assert scores[name] <= floor
        else:
            assert scores[name] >= floor, f"{name}={scores[name]} below {floor}"


def test_required_claims_come_from_the_uploaded_document() -> None:
    claims = required_claims_from_files(("salon-mvp-scope.md",), ROOT)

    assert len(claims) == 10
    assert claims[0] == "Customer registration/login and profile"
    assert claims[9] == "Basic admin management"
    assert extract_mvp_scope_items("# Other\n\n1. Not MVP.\n") == ()


def test_evidence_coverage_counts_required_mvp_items() -> None:
    case = load_mvp_benchmark_case(root=ROOT)
    perfect = "\n".join(
        f"{index}. {item}. [1]" for index, item in enumerate(case.required_claims, start=1)
    )
    quote = " ".join(f"{item}." for item in case.required_claims)
    report = SimpleNamespace(
        report=perfect,
        citations=[SimpleNamespace(filename="salon-mvp-scope.md", chunk_id="c1", quote=quote)],
        passages=[],
    )
    score = score_mvp_report(report, case)

    assert score.required == 10
    assert score.represented == 10
    assert score.evidence_coverage == 1.0
    assert score.citation_validity == 1.0
    assert score.citation_support_rate == 1.0
    assert score.unsupported_claim_rate == 0.0


def test_unsupported_addition_fails_the_support_rate() -> None:
    case = load_mvp_benchmark_case()
    report = SimpleNamespace(
        report="1. The office wifi password is hunter2. [1]",
        citations=[
            SimpleNamespace(
                filename="salon-mvp-scope.md",
                chunk_id="c1",
                quote="1. Customer registration/login and profile.",
            )
        ],
        passages=[],
    )
    score = score_mvp_report(report, case)

    assert score.evidence_coverage < 1.0
    assert score.citation_support_rate == 0.0
    assert score.unsupported_claim_rate == 1.0


def test_baseline_file_matches_the_metric_names() -> None:
    baseline = load_baseline(ROOT / "evaluation" / "baselines" / "salon_mvp_benchmark.json")

    assert set(baseline) == {
        "evidence_coverage",
        "citation_validity",
        "citation_support_rate",
        "unsupported_claim_rate",
        "repair_success_rate",
    }
