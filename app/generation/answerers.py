"""Answerers that stay inside the retrieved evidence."""

import re
from typing import Protocol

from app.generation.agent import is_document_overview, wants_section_summary
from app.generation.grounding import AnswerResult, interpret_model_answer, refusal
from app.generation.messages import build_messages
from app.ingestion.errors import KnowledgeError
from app.retrieval.search import RankedPassage, RetrievalResult
from app.retrieval.text import content_tokens, focus_tokens

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_NUMBERED_HEADING = re.compile(r"(?:^|\s)\d+\.$")
_PROSE = re.compile(r"\bThe\s+[A-Z][^.!?]{30,}[.!?]")
_HEADER_WORDS = frozenset(
    {
        "android",
        "date",
        "document",
        "ios",
        "page",
        "platforms",
        "status",
        "version",
        "web",
    }
)
_GENERIC_TOPIC = frozenset(
    {
        "app",
        "booking",
        "document",
        "page",
        "requirements",
        "salon",
        "specification",
    }
)
_SPEC_HEADER = re.compile(
    r"(?i)(?:^|\s)(?:[^\n.]{0,80}requirements specification\s+)?page\s+\d+\s*"
)
_BULLET = re.compile(r"(?:^|[•●]|\n\s*[-*]\s+|\n\s*\d+\.\s+)")


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
        if wants_section_summary(question) or is_document_overview(question):
            answer = section_summary(question, retrieved, supporting)
        else:
            passage = supporting[0]
            number = retrieved.passages.index(passage) + 1
            sentence = topic_excerpt(
                passage.model_text(),
                passage.matched_terms or tuple(focus_tokens(question)),
            )
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
    matched = [passage for passage in retrieved.passages if passage.matched_terms]
    if matched:
        matched.sort(key=lambda passage: (-len(passage.matched_terms), passage.rank))
        return matched[:3]
    if is_document_overview(question) and retrieved.passages:
        return list(retrieved.passages)[:3]
    return []


def section_summary(
    question: str,
    retrieved: RetrievalResult,
    supporting: list[RankedPassage] | None = None,
) -> str:
    """Quote several bullets or sentences from the matching section."""
    passages = supporting or _supporting_passages(question, retrieved)
    if not passages:
        passages = list(retrieved.passages)[:4]
    passages = sorted(passages, key=lambda item: (item.source.chunk_index, item.rank))
    items: list[tuple[str, int]] = []
    seen: set[str] = set()
    terms = tuple(focus_tokens(question))
    for passage in passages:
        require_topic = bool(terms) and not is_document_overview(question)
        number = retrieved.passages.index(passage) + 1
        for point in _section_points(passage.model_text(), terms, require_topic=require_topic):
            key = point.casefold()
            if key in seen:
                continue
            seen.add(key)
            items.append((point, number))
            if len(items) >= 8:
                break
        if len(items) >= 8:
            break
    return numbered_list(items)


def _section_points(text: str, terms: tuple[str, ...], *, require_topic: bool = True) -> list[str]:
    compact = _SPEC_HEADER.sub(" ", _clean_excerpt(text, limit=4000)).strip()
    if require_topic and terms and not (
        set(terms) & set(content_tokens(compact))
        or any(term in compact.casefold() for term in terms)
    ):
        return []
    raw = [piece.strip(" -•\t") for piece in _BULLET.split(compact) if piece.strip(" -•\t")]
    pieces: list[str] = []
    for part in raw or [compact]:
        pieces.extend(_split_sentences(part) or [part])
    points: list[str] = []
    for piece in pieces:
        cleaned = _clean_excerpt(piece, limit=280)
        if _headerish(cleaned):
            for match in _PROSE.finditer(cleaned):
                prose = _clean_excerpt(match.group(0), limit=280)
                if len(prose) >= 12:
                    points.append(prose)
            continue
        if len(cleaned) < 12:
            continue
        points.append(cleaned)
    return points[:8]


def numbered_list(items: list[tuple[str, int]]) -> str:
    """Document-order points, one numbered line each, with a citation."""
    lines = [f"{index}. {text} [{marker}]" for index, (text, marker) in enumerate(items, start=1)]
    return "\n".join(lines)


def _overview_answer(retrieved: RetrievalResult, supporting: list[RankedPassage]) -> str:
    return section_summary("", retrieved, supporting)


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


def topic_excerpt(text: str, terms: tuple[str, ...]) -> str:
    """Quote from the heading or term that names the topic, not the page header."""
    compact = _SPEC_HEADER.sub("", _clean_excerpt(text, limit=2000)).strip()
    folded = compact.casefold()
    start = _topic_start(folded, terms)
    if start is None:
        fallback = best_sentence(compact or text, terms)
        if set(terms) & set(content_tokens(fallback)):
            return fallback
        return ""
    lead = compact[max(0, start - 12) : start]
    numbered = re.search(r"(\d+(?:\.\d+)*\s+)$", lead)
    if numbered:
        start = start - len(numbered.group(1))
    return _clean_excerpt(_clip_sentence(compact[start:]))


def _topic_start(folded: str, terms: tuple[str, ...]) -> int | None:
    wanted = [term for term in terms if term]
    distinctive = [term for term in wanted if term not in _GENERIC_TOPIC and len(term) >= 4]
    phrases: list[str] = []
    if "management" in wanted and "portal" in wanted:
        phrases.extend(("management portal", "salon management portal"))
    if "admin" in wanted:
        phrases.extend(("admin portal", "platform admin", "web admin", "salon owner"))
    phrases.extend(sorted(distinctive or wanted, key=len, reverse=True))
    for phrase in phrases:
        index = 0
        while True:
            found = folded.find(phrase, index)
            if found < 0:
                break
            window = folded[max(0, found - 48) : found + 64]
            if not re.search(r"requirements specification\s+page\s+\d+", window):
                return found
            index = found + len(phrase)
    return None


def best_sentence(text: str, terms: tuple[str, ...]) -> str:
    """Pick the sentence that contains the most question terms."""
    pieces = _split_sentences(text)
    prose = [match.group(0) for match in _PROSE.finditer(" ".join((text or "").split()))]
    for item in prose:
        if item not in pieces:
            pieces.append(item)
    if not pieces:
        return _clean_excerpt(text)
    usable = [piece for piece in pieces if not _headerish(piece)]
    candidates = usable or pieces
    term_set = set(terms)
    chosen = max(candidates, key=lambda piece: _sentence_score(piece, term_set))
    return _clean_excerpt(chosen)


def _split_sentences(text: str) -> list[str]:
    raw = [piece.strip() for piece in _SENTENCE.split(text) if piece.strip()]
    merged: list[str] = []
    for piece in raw:
        if merged and _NUMBERED_HEADING.search(merged[-1]):
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return merged


def _sentence_score(piece: str, terms: set[str]) -> float:
    tokens = content_tokens(piece)
    overlap = sum(1 for term in terms if term in set(tokens))
    header = sum(1 for token in tokens if token in _HEADER_WORDS)
    length = min(len(" ".join(piece.split())), 400)
    if _headerish(piece):
        return overlap - 8
    return overlap * 10 + length / 40 - header


def _headerish(piece: str) -> bool:
    compact = " ".join(piece.split())
    tokens = content_tokens(compact)
    if len(compact) < 12:
        return True
    lead = compact[:160]
    if re.search(r"\bpage\s+\d+\b", lead, flags=re.I):
        return True
    if re.search(r"\bdocument version\b", lead, flags=re.I):
        return True
    header = sum(1 for token in tokens if token in _HEADER_WORDS)
    return bool(tokens) and header / len(tokens) >= 0.4


def _clean_excerpt(text: str, limit: int = 500) -> str:
    return " ".join((text or "").split())[:limit]


def _clip_sentence(text: str) -> str:
    stripped = (text or "").strip()
    for index, char in enumerate(stripped):
        if char in ".!?" and index >= 24 and (
            index + 1 == len(stripped) or stripped[index + 1] == " "
        ):
            return stripped[: index + 1]
    return stripped[:240]


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
