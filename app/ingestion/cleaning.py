"""Normalize extracted text before chunking."""

import re

_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Strip noise that hurts chunk boundaries without changing wording.

    Offsets on stored chunks refer to this cleaned text.
    """
    cleaned = text.replace("\x00", "")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\u00a0", " ")
    cleaned = "\n".join(line.rstrip() for line in cleaned.split("\n"))
    cleaned = _BLANK_LINES.sub("\n\n", cleaned)
    return cleaned.strip()
