"""Split cleaned text into overlapping slices that prefer natural boundaries."""

import re
from dataclasses import dataclass

from app.ingestion.models import ChunkDraft

_HEADING = re.compile(r"^(#{1,6})[ \t]+(\S.*?)\s*$")
_SEPARATORS = ("\n\n", "\n", ". ", " ")


@dataclass(frozen=True)
class _Section:
    heading: str | None
    start: int
    end: int


@dataclass(frozen=True)
class _Span:
    text: str
    start: int
    end: int


def build_chunks(
    text: str,
    *,
    content_type: str,
    chunk_size: int,
    overlap: int,
) -> list[ChunkDraft]:
    """Chunk cleaned text. Markdown is split on headings first.

    Heading lines inside fenced code blocks stay in the surrounding section.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")
    if not text.strip():
        return []

    if content_type == "text/markdown":
        sections = _markdown_sections(text)
    else:
        sections = [_Section(heading=None, start=0, end=len(text))]

    drafts: list[ChunkDraft] = []
    for section in sections:
        body = text[section.start : section.end]
        for span in _chunk_text(body, chunk_size=chunk_size, overlap=overlap):
            drafts.append(
                ChunkDraft(
                    text=span.text,
                    start=section.start + span.start,
                    end=section.start + span.end,
                    heading=section.heading,
                )
            )
    return drafts


def _markdown_sections(text: str) -> list[_Section]:
    headings: list[tuple[int, str]] = []
    in_fence = False
    for start, end, line in _line_spans(text):
        content = line[:-1] if line.endswith("\n") else line
        if content.endswith("\r"):
            content = content[:-1]
        if content.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _HEADING.match(content)
        if match:
            headings.append((start, match.group(2).strip()))

    if not headings:
        return [_Section(heading=None, start=0, end=len(text))]

    sections: list[_Section] = []
    first_heading = headings[0][0]
    if first_heading > 0 and text[:first_heading].strip():
        sections.append(_Section(heading=None, start=0, end=first_heading))
    for index, (start, heading) in enumerate(headings):
        end = headings[index + 1][0] if index + 1 < len(headings) else len(text)
        if text[start:end].strip():
            sections.append(_Section(heading=heading, start=start, end=end))
    return sections


def _line_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start = 0
    for line in text.splitlines(keepends=True):
        spans.append((start, start + len(line), line))
        start += len(line)
    return spans


def _chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[_Span]:
    spans: list[_Span] = []
    start = 0
    length = len(text)
    while start < length:
        while start < length and text[start].isspace():
            start += 1
        if start >= length:
            break
        hard_end = min(start + chunk_size, length)
        end = hard_end if hard_end == length else _prefer_break(text, start, hard_end)
        trim_end = end
        while trim_end > start and text[trim_end - 1].isspace():
            trim_end -= 1
        if trim_end > start:
            spans.append(_Span(text=text[start:trim_end], start=start, end=trim_end))
        if end >= length:
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start
    return spans


def _prefer_break(text: str, start: int, end: int) -> int:
    """Break at the latest natural boundary that still makes progress."""
    min_end = start + max(1, (end - start) // 2)
    for separator in _SEPARATORS:
        index = text.rfind(separator, min_end, end)
        if index != -1:
            return index + len(separator)
    return end
