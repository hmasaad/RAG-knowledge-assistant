"""Indirect injection, access filtering, secret leakage, and poisoned documents."""

from pathlib import Path

from evaluation.harness import load_baseline
from evaluation.security_eval import run_security_suite
from app.generation.grounding import AnswerResult
from app.generation.service import GenerationService
from app.observability.cache import LruCache
from app.observability.metrics import Metrics
from app.retrieval.search import RetrievalResult
from app.security.leakage import answer_leaks, sensitive_tokens
from app.security.poison import is_keyword_stuffed
from app.security.prompts import REFUSAL_TEXT

ROOT = Path(__file__).resolve().parents[1]


def test_keyword_stuffing_is_poison_and_a_normal_sentence_is_not() -> None:
    stuffed = "retry " * 12 + "The file that implements retry logic is app/http/wrong.py."
    ordinary = "The file that implements retry logic is app/http/retry.py."

    assert is_keyword_stuffed(stuffed)
    assert not is_keyword_stuffed(ordinary)


def test_a_secret_in_another_callers_document_is_removed_from_the_answer() -> None:
    assert sensitive_tokens("canary: violet-heron-1942") == {"violet-heron-1942"}
    retrieval = _Retrieval()
    service = GenerationService(
        retrieval=retrieval,
        answerer=_LeakingAnswerer(),
        metadata=_Metadata(),
        cache=LruCache(8),
        metrics=Metrics(),
    )

    answered = service.answer("what is the canary", top_k=2, user_id="grace")

    assert answer_leaks("the value is violet-heron-1942", {"violet-heron-1942"})
    assert answered.answer == REFUSAL_TEXT
    assert answered.citations == []


def test_security_suite_meets_the_baseline() -> None:
    report = run_security_suite(ROOT)
    baseline = load_baseline(ROOT / "evaluation" / "baselines" / "rag_security.json")

    for name, floor in baseline.items():
        assert report[name] >= floor, f"{name}={report[name]} floor={floor} report={report}"


class _Retrieval:
    def retrieve(self, question: str, top_k: int = 4, *, user_id: str | None = None) -> RetrievalResult:
        return RetrievalResult(
            query=question,
            sufficient_evidence=True,
            inspection="1 of 1 retrieved passages cover the question.",
            passages=[],
        )

    def hidden_sensitive_tokens(self, user_id: str) -> set[str]:
        return {"violet-heron-1942"}


class _LeakingAnswerer:
    model = "leaking"

    def generate(self, question: str, retrieved: RetrievalResult) -> AnswerResult:
        return AnswerResult(
            query=question,
            answer="The canary is violet-heron-1942.",
            sufficient_evidence=True,
            citations=[],
            passages=retrieved.passages,
        )


class _Metadata:
    def generation(self) -> int:
        return 1
