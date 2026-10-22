"""Detect when visible documents state different numeric facts."""

import re

_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_DIGITS = re.compile(r"\b\d+\b")
_WORDS = re.compile(r"[a-z]+")


def documents_conflict(passages: list[tuple[str, str, bool]]) -> bool:
    """True when two filenames answer with numbers that do not overlap.

    Each item is ``(filename, text, answers_question)``.
    """
    by_file: dict[str, set[int]] = {}
    for filename, text, answers in passages:
        if not answers:
            continue
        values = numeric_facts(text)
        if not values:
            continue
        by_file.setdefault(filename, set()).update(values)
    stated = [values for values in by_file.values() if values]
    if len(stated) < 2:
        return False
    return not set.intersection(*stated)


def numeric_facts(text: str) -> set[int]:
    values = {int(match) for match in _DIGITS.findall(text)}
    for token in _WORDS.findall(text.lower()):
        if token in _NUMBER_WORDS:
            values.add(_NUMBER_WORDS[token])
    return values
