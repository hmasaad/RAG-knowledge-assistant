"""Remove instruction-shaped lines from untrusted document text."""

import re
from dataclasses import dataclass

_ZERO_WIDTH = re.compile(r"[\u200b-\u200d\ufeff]")
_ROLE_TAG = re.compile(r"(?i)<\s*/?\s*(?:system|assistant|user|instructions?)\s*>")
_INSTRUCTION_LINE = re.compile(
    r"(?i)(?:"
    r"ignore\s+(?:all\s+|any\s+)?(?:previous|prior|above).{0,80}instructions"
    r"|disregard\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|system)"
    r"|\byou\s+are\s+now\b"
    r"|\bnew\s+instructions?\b"
    r"|\bdo\s+not\s+cite\b"
    r"|\bsecret\s+phrase\b"
    r"|\boverride\s+(?:the\s+)?(?:system|instructions|rules)\b"
    r"|^\s*(?:system|assistant)\s*:"
    r")"
)


@dataclass(frozen=True)
class SanitizedText:
    text: str
    instruction_detected: bool


def neutralize_untrusted_text(text: str) -> SanitizedText:
    """Drop lines that try to change the assistant's rules.

    The remaining text is what retrieval and the model are allowed to use.
    """
    cleaned = _ZERO_WIDTH.sub("", text)
    kept: list[str] = []
    detected = False
    for line in cleaned.split("\n"):
        without_tags = _ROLE_TAG.sub("", line)
        if without_tags != line or _INSTRUCTION_LINE.search(without_tags):
            detected = True
            if _INSTRUCTION_LINE.search(without_tags) or not without_tags.strip():
                continue
        kept.append(without_tags)
    return SanitizedText(text="\n".join(kept).strip(), instruction_detected=detected)
