from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.inspect import covers_question
from app.retrieval.lexical import bm25_scores
from app.retrieval.text import content_tokens, focus_tokens


def test_bm25_prefers_the_document_that_shares_query_terms() -> None:
    scores = bm25_scores(
        ["retry", "logic"],
        [
            ["schema", "migrations", "table"],
            ["file", "implements", "retry", "logic"],
        ],
    )

    assert scores[1] > scores[0]
    assert scores[0] == 0


def test_reciprocal_rank_fusion_rewards_agreement() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)

    assert fused["b"] > fused["a"] > fused["c"]


def test_coverage_inspection_rejects_unrelated_passages() -> None:
    tokens = content_tokens("which file implements the retry logic")

    assert covers_question(1.0, tuple(tokens), min_coverage=0.5)
    assert not covers_question(0.25, ("retry",), min_coverage=0.5)
    assert not covers_question(0.0, (), min_coverage=0.5)
    assert focus_tokens(
        "Can you breakdown Authentication & Profile so that i can understand it properly?"
    ) == ["authentication", "profile"]
