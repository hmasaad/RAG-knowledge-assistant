"""Score an embedder on whether related text outranks unrelated text."""

from dataclasses import dataclass

from app.ingestion.embedding_cases import EMBEDDING_PAIRS, EmbeddingPair
from app.ingestion.similarity import cosine_similarity


@dataclass(frozen=True)
class PairScore:
    id: str
    positive_similarity: float
    negative_similarity: float
    margin: float
    correct: bool


@dataclass(frozen=True)
class EmbeddingReport:
    model: str
    dimension: int
    pair_accuracy: float
    mean_positive_similarity: float
    mean_negative_similarity: float
    mean_margin: float
    pairs: tuple[PairScore, ...]


def evaluate_embedder(
    embedder: object,
    pairs: tuple[EmbeddingPair, ...] = EMBEDDING_PAIRS,
) -> EmbeddingReport:
    """Embed each query with its positive and negative passage.

    Accuracy is the share of pairs where the positive passage is closer.
    """
    scores: list[PairScore] = []
    dimension = 0
    for pair in pairs:
        vectors = embedder.embed_documents([pair.query, pair.positive, pair.negative])
        if len(vectors) != 3:
            raise ValueError("Embedding provider returned an unexpected number of vectors.")
        dimension = len(vectors[0])
        positive = cosine_similarity(vectors[0], vectors[1])
        negative = cosine_similarity(vectors[0], vectors[2])
        margin = positive - negative
        scores.append(
            PairScore(
                id=pair.id,
                positive_similarity=positive,
                negative_similarity=negative,
                margin=margin,
                correct=positive > negative,
            )
        )
    count = len(scores) or 1
    return EmbeddingReport(
        model=str(getattr(embedder, "model", "unknown")),
        dimension=dimension,
        pair_accuracy=sum(1.0 for score in scores if score.correct) / count,
        mean_positive_similarity=sum(score.positive_similarity for score in scores) / count,
        mean_negative_similarity=sum(score.negative_similarity for score in scores) / count,
        mean_margin=sum(score.margin for score in scores) / count,
        pairs=tuple(scores),
    )
