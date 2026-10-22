"""Embedding evaluation entry point. The implementation lives with the app."""

from app.ingestion.embedding_eval import EmbeddingReport, PairScore, evaluate_embedder

__all__ = ["EmbeddingReport", "PairScore", "evaluate_embedder"]
