"""Split cleaned text into overlapping slices.

``structure`` keeps markdown sections, then splits each section on natural
boundaries. ``recursive`` uses those boundaries without headings. ``fixed``
cuts every window at ``chunk_size``. ``semantic`` groups neighboring sentences
while their embeddings stay similar.
"""

import re
from dataclasses import dataclass

from app.ingestion.models import ChunkDraft
from app.ingestion.similarity import cosine_similarity

_HEADING = re.compile(r"^(#{1,6})[ \t]+(\S.*?)\s*$")
_ADOC_HEADING = re.compile(r"^(=+)[ \t]+(\S.*?)\s*$")
_NUMBERED_SECTION = re.compile(
    r"(?:^|(?<=\n)|(?<=[.!?]\s))(\d+(?:\.\d+)*)\.\s+"
    r"([A-Z][^.!?\n]{2,70}?)(?=\s+(?:\d+\.\s+)|$|\n)"
)
_SEPARATORS = ("\n\n", "\n", ". ", " ")
STRATEGIES = ("fixed", "recursive", "semantic", "structure")
_CODE_PATTERNS = {
    "text/x-python": re.compile(r"^(?:async\s+def|def|class)\s+([A-Za-z_]\w*)"),
    "text/x-ruby": re.compile(r"^(?:def|class|module)\s+([A-Za-z_]\w*)"),
    "text/javascript": re.compile(
        r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function|class)\s+([A-Za-z_$][\w$]*)"
    ),
    "text/typescript": re.compile(
        r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function|class|interface|type)\s+([A-Za-z_$][\w$]*)"
    ),
    "text/x-go": re.compile(r"^func(?:\s+\([^)\n]*\))?\s+([A-Za-z_]\w*)"),
    "text/x-rust": re.compile(
        r"^(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:fn|struct|enum|trait)\s+([A-Za-z_]\w*)"
    ),
    "text/x-java": re.compile(
        r"^(?:(?:public|private|protected|static|final|abstract)\s+)*(?:class|interface|enum)\s+([A-Za-z_]\w*)"
    ),
    "text/x-sql": re.compile(
        r"(?i)^create\s+(?:or\s+replace\s+)?(?:table|view|function|index)\s+([A-Za-z_][\w.]*)"
    ),
    "text/x-shell": re.compile(r"^([A-Za-z_][\w]*)\s*\(\s*\)"),
}


@dataclass(frozen=True)
class _Section:
    heading: str | None
    start: int
    end: int
    number: str = ""
    parent: str = ""


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
    strategy: str = "structure",
    embedder: object | None = None,
    semantic_similarity: float = 0.5,
) -> list[ChunkDraft]:
    """Chunk cleaned text.

    ``strategy`` is ``fixed``, ``recursive``, ``semantic``, or ``structure``.
    Structure is the default: markdown and AsciiDoc are split on headings, and
    source files are split on top-level symbols. Heading lines inside fenced
    code blocks stay in the surrounding section.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown chunk strategy: {strategy}.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")
    if not text.strip():
        return []

    if strategy == "fixed":
        return _drafts(_fixed_windows(text, chunk_size=chunk_size, overlap=overlap))
    if strategy == "recursive":
        return _drafts(_chunk_text(text, chunk_size=chunk_size, overlap=overlap))
    if strategy == "semantic":
        if embedder is None:
            raise ValueError("semantic chunking requires an embedder")
        return _semantic_drafts(
            text,
            chunk_size=chunk_size,
            overlap=overlap,
            embedder=embedder,
            similarity_threshold=semantic_similarity,
        )

    if content_type == "text/markdown":
        sections = _markdown_sections(text)
    elif content_type == "text/asciidoc":
        sections = _asciidoc_sections(text)
    elif content_type in _CODE_PATTERNS:
        sections = _code_sections(text, content_type)
    elif content_type in {"application/pdf", "text/plain"}:
        sections = _numbered_sections(text)
    else:
        sections = [_Section(heading=None, start=0, end=len(text))]

    drafts: list[ChunkDraft] = []
    for section in sections:
        body = text[section.start : section.end]
        meta = _section_meta(section)
        if len(body) <= chunk_size:
            drafts.append(
                ChunkDraft(
                    text=body,
                    start=section.start,
                    end=section.end,
                    heading=meta["heading"],
                    section_id=meta["section_id"],
                    section_title=meta["section_title"],
                    section_number=meta["section_number"],
                    parent_section=meta["parent_section"],
                    chunk_type="section",
                )
            )
            continue
        for span in _chunk_text(body, chunk_size=chunk_size, overlap=overlap):
            drafts.append(
                ChunkDraft(
                    text=span.text,
                    start=section.start + span.start,
                    end=section.start + span.end,
                    heading=meta["heading"],
                    section_id=meta["section_id"],
                    section_title=meta["section_title"],
                    section_number=meta["section_number"],
                    parent_section=meta["parent_section"],
                    chunk_type="section_part",
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
    return _sections_from_marks(text, headings)


def _asciidoc_sections(text: str) -> list[_Section]:
    """Split an architecture document on AsciiDoc headings."""
    headings: list[tuple[int, str]] = []
    in_block = False
    for start, end, line in _line_spans(text):
        content = _line_content(line)
        stripped = content.strip()
        if _is_adoc_delimiter(stripped):
            in_block = not in_block
            continue
        if in_block:
            continue
        match = _ADOC_HEADING.match(content)
        if match:
            headings.append((start, match.group(2).strip()))
    if not headings:
        return [_Section(heading=None, start=0, end=len(text))]
    return _sections_from_marks(text, headings)


def _numbered_sections(text: str) -> list[_Section]:
    """Keep ``11. MVP Scope`` and its bullets together in extracted PDF text."""
    marks: list[tuple[int, str]] = []
    for match in _NUMBERED_SECTION.finditer(text):
        title = match.group(2).strip()
        if not _section_title(match.group(1), title):
            continue
        marks.append((match.start(), title, match.group(1)))
    if not marks:
        return [_Section(heading=None, start=0, end=len(text))]
    return _sections_from_marks(text, marks)


def _section_title(number: str, title: str) -> bool:
    words = title.split()
    if not words or len(words) > 8:
        return False
    if "." in number or len(number.split(".")[0]) >= 2:
        return True
    return any(
        word.casefold()
        in {"scope", "overview", "introduction", "requirements", "admin", "portal", "profile"}
        for word in words
    )


def _code_sections(text: str, content_type: str) -> list[_Section]:
    """Split source on top-level symbols. Nested definitions stay with the parent."""
    pattern = _CODE_PATTERNS[content_type]
    marks: list[tuple[int, str]] = []
    for start, end, line in _line_spans(text):
        content = _line_content(line)
        if content[:1] in {" ", "\t"}:
            continue
        match = pattern.match(content)
        if match:
            marks.append((start, match.group(1)))
    if not marks:
        return [_Section(heading=None, start=0, end=len(text))]
    return _sections_from_marks(text, marks)


def _sections_from_marks(text: str, marks: list[tuple]) -> list[_Section]:
    sections: list[_Section] = []
    first = marks[0][0]
    if first > 0 and text[:first].strip():
        sections.append(_Section(heading=None, start=0, end=first))
    for index, mark in enumerate(marks):
        start = mark[0]
        heading = mark[1]
        number = mark[2] if len(mark) > 2 else ""
        if not number:
            number, stripped = _split_numbered_title(heading)
            if number:
                heading = stripped
        end = marks[index + 1][0] if index + 1 < len(marks) else len(text)
        if text[start:end].strip():
            sections.append(_Section(heading=heading, start=start, end=end, number=number))
    return sections


def _split_numbered_title(heading: str) -> tuple[str, str]:
    match = re.match(r"^(\d+(?:\.\d+)*)\.\s+(.*)$", (heading or "").strip())
    if match:
        return match.group(1), match.group(2).strip()
    return "", (heading or "").strip()


def _section_meta(section: _Section) -> dict[str, str]:
    title = (section.heading or "").strip()
    number = section.number or _split_numbered_title(title)[0]
    if number and title and title.startswith(f"{number}."):
        title = title[len(number) + 1 :].strip()
    if number:
        section_id = "section_" + number.replace(".", "_")
        heading = f"{number}. {title}" if title else number
    elif title:
        slug = re.sub(r"[^a-z0-9]+", "_", title.casefold()).strip("_")
        section_id = f"section_{slug}" if slug else ""
        heading = title
    else:
        section_id = ""
        heading = ""
    return {
        "heading": heading or None,
        "section_id": section_id,
        "section_title": title,
        "section_number": number,
        "parent_section": section.parent,
    }


def _line_content(line: str) -> str:
    content = line[:-1] if line.endswith("\n") else line
    if content.endswith("\r"):
        content = content[:-1]
    return content


def _is_adoc_delimiter(stripped: str) -> bool:
    return len(stripped) >= 4 and stripped == stripped[0] * len(stripped) and stripped[0] in "-.*"


def _line_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start = 0
    for line in text.splitlines(keepends=True):
        spans.append((start, start + len(line), line))
        start += len(line)
    return spans


def _drafts(spans: list[_Span], *, heading: str | None = None, origin: int = 0) -> list[ChunkDraft]:
    return [
        ChunkDraft(
            text=span.text,
            start=origin + span.start,
            end=origin + span.end,
            heading=heading,
        )
        for span in spans
        if span.text.strip()
    ]


def _fixed_windows(text: str, *, chunk_size: int, overlap: int) -> list[_Span]:
    """Cut the text every ``chunk_size`` characters, with no boundary preference."""
    spans: list[_Span] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        spans.append(_Span(text=text[start:end], start=start, end=end))
        if end >= length:
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start
    return spans


def _semantic_drafts(
    text: str,
    *,
    chunk_size: int,
    overlap: int,
    embedder: object,
    similarity_threshold: float,
) -> list[ChunkDraft]:
    """Group adjacent sentences while their embeddings stay close.

    A sentence longer than ``chunk_size`` falls back to the recursive splitter.
    """
    sentences = _sentence_spans(text)
    if not sentences:
        return []
    vectors = embedder.embed_documents([sentence.text for sentence in sentences])
    if len(vectors) != len(sentences):
        raise ValueError("Embedding provider returned an unexpected number of vectors.")
    groups: list[list[int]] = [[0]]
    for index in range(1, len(sentences)):
        similarity = cosine_similarity(vectors[index - 1], vectors[index])
        start = sentences[groups[-1][0]].start
        end = sentences[index].end
        if similarity >= similarity_threshold and end - start <= chunk_size:
            groups[-1].append(index)
        else:
            groups.append([index])

    drafts: list[ChunkDraft] = []
    for group in groups:
        start = sentences[group[0]].start
        end = sentences[group[-1]].end
        piece = text[start:end]
        if not piece.strip():
            continue
        if len(piece) <= chunk_size:
            drafts.append(ChunkDraft(text=piece, start=start, end=end, heading=None))
            continue
        for span in _chunk_text(piece, chunk_size=chunk_size, overlap=overlap):
            drafts.append(
                ChunkDraft(
                    text=span.text,
                    start=start + span.start,
                    end=start + span.end,
                    heading=None,
                )
            )
    return drafts


def _sentence_spans(text: str) -> list[_Span]:
    """Split on sentence punctuation and newlines, keeping original offsets."""
    spans: list[_Span] = []
    start = 0
    index = 0
    length = len(text)
    while index < length:
        boundary = text[index] == "\n" or (
            text[index] in ".!?" and (index + 1 == length or text[index + 1].isspace())
        )
        if boundary:
            _append_sentence(spans, text, start, index + 1)
            start = index + 1
        index += 1
    if start < length:
        _append_sentence(spans, text, start, length)
    return spans


def _append_sentence(spans: list[_Span], text: str, start: int, end: int) -> None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start < end:
        spans.append(_Span(text=text[start:end], start=start, end=end))


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
