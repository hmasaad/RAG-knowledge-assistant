"""Answerers that stay inside the retrieved evidence."""

import re
from typing import Protocol

from app.generation.agent import is_document_overview
from app.generation.grounding import AnswerResult, interpret_model_answer, refusal
from app.generation.messages import build_messages
from app.ingestion.errors import KnowledgeError
from app.retrieval.search import RankedPassage, RetrievalResult
from app.retrieval.text import content_tokens

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class Answerer(Protocol):
    model: str

    def generate(self, question: str, retrieved: RetrievalResult) -> AnswerResult: ...


class LocalAnswerer:
    """Quote the best supporting sentence. Used for tests and offline demos."""

    model = "local-extractive"

    def generate(self, question: str, retrieved: RetrievalResult) -> AnswerResult:
        if retrieved.conflict:
            return interpret_model_answer(
                question,
                _conflict_answer(retrieved),
                retrieved.passages,
                conflict=True,
            )
        supporting = _supporting_passages(question, retrieved)
        if not supporting:
            return refusal(question, retrieved.passages)
        passage = supporting[0]
        number = retrieved.passages.index(passage) + 1
        sentence = best_sentence(passage.model_text(), passage.matched_terms)
        marker = f"[{number}]"
        answer = sentence if marker in sentence else f"{sentence} {marker}"
        return interpret_model_answer(question, answer, retrieved.passages)


class OpenAIAnswerer:
    """Ask the chat model to answer from the passages and cite them."""

    def __init__(self, *, api_key: str, model: str, client: object | None = None) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
        self.model = model
        self._client = client

    def generate(self, question: str, retrieved: RetrievalResult) -> AnswerResult:
        if not retrieved.passages:
            return refusal(question, [])
        messages = build_messages(
            question,
            retrieved.passages,
            conflict=retrieved.conflict,
        )
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                max_completion_tokens=800,
            )
            content = response.choices[0].message.content or ""
        except KnowledgeError:
            raise
        except Exception as exc:
            raise KnowledgeError(
                "The language model request failed.",
                status_code=502,
            ) from exc
        return interpret_model_answer(
            question,
            content,
            retrieved.passages,
            conflict=retrieved.conflict,
        )


def _supporting_passages(question: str, retrieved: RetrievalResult) -> list[RankedPassage]:
    strong = [passage for passage in retrieved.passages if passage.answers_question]
    if strong:
        return strong
    if is_document_overview(question) and retrieved.passages:
        return [retrieved.passages[0]]
    return []


def _conflict_answer(retrieved: RetrievalResult) -> str:
    parts = ["The documents disagree."]
    seen: set[str] = set()
    for passage in retrieved.passages:
        if not passage.answers_question or passage.source.filename in seen:
            continue
        seen.add(passage.source.filename)
        number = retrieved.passages.index(passage) + 1
        sentence = best_sentence(passage.model_text(), passage.matched_terms)
        parts.append(f"{sentence} [{number}]")
    return " ".join(parts)


def best_sentence(text: str, terms: tuple[str, ...]) -> str:
    """Pick the sentence that contains the most question terms."""
    pieces = [piece.strip() for piece in _SENTENCE.split(text) if piece.strip()]
    if not pieces:
        return text.strip()[:500]
    term_set = set(terms)

    def overlap(piece: str) -> int:
        tokens = set(content_tokens(piece))
        return sum(1 for term in term_set if term in tokens)

    chosen = max(pieces, key=overlap)
    if overlap(chosen) == 0:
        chosen = pieces[0]
    return chosen[:500]


def build_answerer(*, provider: str, api_key: str, model: str) -> Answerer:
    if provider == "local":
        return LocalAnswerer()
    if provider == "openai":
        if not api_key:
            raise KnowledgeError(
                "OPENAI_API_KEY is not set. Set it, or use LLM_PROVIDER=local.",
                status_code=503,
            )
        return OpenAIAnswerer(api_key=api_key, model=model)
    raise KnowledgeError(f"Unknown language model provider: {provider}.", status_code=500)
