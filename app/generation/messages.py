"""Build the chat messages for a grounded answer."""

from app.retrieval.search import RankedPassage
from app.security.content import neutralize_untrusted_text
from app.security.prompts import SYSTEM_PROMPT, evidence_block, user_message


def build_messages(
    question: str,
    passages: list[RankedPassage],
    *,
    conflict: bool = False,
) -> list[dict[str, str]]:
    blocks = [
        evidence_block(number, _label(passage), _model_body(passage))
        for number, passage in enumerate(passages, start=1)
    ]
    evidence = "\n\n".join(blocks)
    if conflict:
        evidence += (
            "\n\nThe numbered passages disagree. Cite each side. "
            "Do not treat one number as the only fact."
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message(question, evidence)},
    ]


def _model_body(passage: RankedPassage) -> str:
    return neutralize_untrusted_text(passage.model_text()).text


def _label(passage: RankedPassage) -> str:
    parts = [passage.source.filename or "unknown"]
    if passage.source.heading:
        parts.append(f"heading: {passage.source.heading}")
    if passage.source.page is not None:
        parts.append(f"page: {passage.source.page}")
    return " | ".join(parts)
