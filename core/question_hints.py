"""Read intent signals out of a question: ordering, truncation and comparisons.

These are the cues that decide whether a query needs ORDER BY, LIMIT, or a
comparison operator other than equality. Without them the generator can express
"top 5 highest paid" in the grammar but has no reason to actually emit it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Superlatives that mean "sort descending and take the extreme".
DESCENDING_WORDS = (
    "highest", "largest", "biggest", "greatest", "maximum", "max", "most",
    "longest", "oldest", "latest", "newest", "best", "top", "heaviest",
    "richest", "tallest", "widest", "fastest",
)

ASCENDING_WORDS = (
    "lowest", "smallest", "least", "minimum", "min", "shortest", "youngest",
    "earliest", "oldest_date", "cheapest", "worst", "lightest", "slowest",
)

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

# Words that end a measure phrase; everything before them is the thing being ranked.
_PHRASE_STOP = re.compile(
    r"\b(in|of|for|from|with|by|and|or|that|who|which|where|when|among|the|a|an)\b"
)

_TOP_N = re.compile(r"\b(?:top|first|last)\s+(\d+|" + "|".join(NUMBER_WORDS) + r")\b", re.IGNORECASE)
_N_SUPERLATIVE = re.compile(
    r"\b(\d+|" + "|".join(NUMBER_WORDS) + r")\s+(?:most|least|highest|lowest|largest|smallest|biggest)\b",
    re.IGNORECASE,
)
_FREQUENCY = re.compile(r"\b(most|least)\s+(?:common|frequent|popular|numerous)\b", re.IGNORECASE)
_SORT_BY = re.compile(
    r"\b(?:sorted|ordered|order|sort|arrange[d]?|list(?:ed)?)\s+(?:them\s+)?(?:in\s+)?"
    r"(?:(ascending|descending|alphabetical|reverse)\s+)?order\s+(?:of|by)\s+(.+?)(?:[?.!,]|$)",
    re.IGNORECASE,
)
_SORT_SIMPLE = re.compile(r"\b(?:sorted|ordered)\s+by\s+(.+?)(?:[?.!,]|$)", re.IGNORECASE)
_ALPHABETICAL = re.compile(r"\balphabetical(?:ly)?\b", re.IGNORECASE)
_DESCENDING_ORDER = re.compile(r"\b(descending|reverse)\b", re.IGNORECASE)
_DIRECTION_ORDER = re.compile(
    r"\bin\s+(ascending|descending|alphabetical|reverse)\s+order\s+(?:of|by)\s+(.+?)(?:[?.!,]|$)",
    re.IGNORECASE,
)

# "at least 3" is a comparison, not a request for the minimum. These phrases are
# masked out before superlative detection so they cannot be read as rankings.
_COMPARISON_PHRASES = re.compile(
    r"\b(?:at least|at most|no less than|no more than|not less than|not more than"
    r"|minimum of|maximum of|up to|more than|less than|fewer than|greater than"
    r"|larger than|smaller than|younger than|older than)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class OrderingHint:
    """How the answer set should be ordered and truncated."""

    descending: bool
    limit: int | None = None
    measure_phrase: str = ""
    by_count: bool = False
    alphabetical: bool = False


@dataclass(frozen=True)
class ComparisonHint:
    operator: str
    value: float | int


def _to_int(token: str) -> int | None:
    token = token.strip().lower()
    if token.isdigit():
        return int(token)
    return NUMBER_WORDS.get(token)


def _measure_after(question: str, word: str) -> str:
    """Return the words naming what is being ranked, e.g. 'highest salary' -> 'salary'."""
    match = re.search(rf"\b{re.escape(word)}\b(.*)", question, re.IGNORECASE)
    if not match:
        return ""
    tail = match.group(1).strip()
    stop = _PHRASE_STOP.search(tail)
    phrase = tail[: stop.start()] if stop else tail
    return re.sub(r"[^A-Za-z0-9_ ]+", " ", phrase).strip()


def detect_ordering(question: str) -> OrderingHint | None:
    lowered = question.lower()

    frequency = _FREQUENCY.search(lowered)
    if frequency:
        # "most common X" ranks groups by how many rows they have.
        return OrderingHint(descending=frequency.group(1) == "most", limit=1, by_count=True)

    direction_order = _DIRECTION_ORDER.search(question)
    if direction_order:
        direction, measure = direction_order.groups()
        return OrderingHint(
            descending=bool(_DESCENDING_ORDER.search(direction)),
            limit=None,
            measure_phrase=measure.strip(),
            alphabetical="alphabetical" in direction.lower(),
        )

    explicit_sort = _SORT_BY.search(question) or _SORT_SIMPLE.search(question)
    if explicit_sort:
        groups = explicit_sort.groups()
        direction = groups[0] if len(groups) > 1 else None
        measure = groups[-1] or ""
        descending = bool(direction and _DESCENDING_ORDER.search(direction))
        # "ordered by age from the oldest" states the direction as a superlative
        # trailing the sort key rather than as the word "descending".
        if not descending:
            descending = any(
                re.search(rf"\b{word}\b", measure, re.IGNORECASE) for word in DESCENDING_WORDS
            )
        alphabetical = bool(direction and "alphabetical" in direction.lower())
        return OrderingHint(
            descending=descending,
            limit=None,
            measure_phrase=measure.strip(),
            alphabetical=alphabetical,
        )

    if _ALPHABETICAL.search(lowered):
        return OrderingHint(descending=False, limit=None, alphabetical=True)

    # Comparison wording must not be mistaken for a ranking request.
    ranked = _COMPARISON_PHRASES.sub(" ", lowered)

    limit: int | None = None
    top_n = _TOP_N.search(ranked)
    n_superlative = _N_SUPERLATIVE.search(ranked)
    if top_n:
        limit = _to_int(top_n.group(1))
    elif n_superlative:
        limit = _to_int(n_superlative.group(1))

    for word in DESCENDING_WORDS:
        if re.search(rf"\b{word}\b", ranked):
            return OrderingHint(descending=True, limit=limit or 1, measure_phrase=_measure_after(ranked, word))
    for word in ASCENDING_WORDS:
        if re.search(rf"\b{word}\b", ranked):
            return OrderingHint(descending=False, limit=limit or 1, measure_phrase=_measure_after(ranked, word))

    if limit is not None:
        return OrderingHint(descending=False, limit=limit)
    return None


def detect_comparison(question: str) -> ComparisonHint | None:
    """Map comparison wording to an operator and its numeric operand."""
    lowered = question.lower()
    number = re.search(r"\b(\d+(?:\.\d+)?)\b", lowered)
    if not number:
        return None
    value: float | int = float(number.group(1)) if "." in number.group(1) else int(number.group(1))

    # Ordered longest-phrase-first so "at least" is not shadowed by "least".
    patterns = (
        (r"\b(?:at least|no less than|not less than|minimum of)\b", ">="),
        (r"\b(?:at most|no more than|not more than|maximum of|up to)\b", "<="),
        (r"\b(?:more than|greater than|larger than|older than|above|over|exceeds?|exceeding)\b", ">"),
        (r"\b(?:less than|fewer than|smaller than|younger than|below|under)\b", "<"),
        (r"\b(?:exactly|equal to|equals)\b", "="),
    )
    for pattern, operator in patterns:
        if re.search(pattern, lowered):
            return ComparisonHint(operator=operator, value=value)
    return None


def asks_for_distinct(question: str) -> bool:
    return bool(re.search(r"\b(distinct|different|unique|various)\b", question, re.IGNORECASE))


# Aggregate wording, matched in the order it appears so "average, minimum and
# maximum age" projects AVG, MIN and MAX in that order.
_AGGREGATE_WORDS = (
    (re.compile(r"\b(?:average|avg|mean)\b", re.IGNORECASE), "AVG"),
    (re.compile(r"\b(?:minimum|min|smallest|lowest)\b", re.IGNORECASE), "MIN"),
    (re.compile(r"\b(?:maximum|max|largest|highest|biggest)\b", re.IGNORECASE), "MAX"),
    (re.compile(r"\b(?:sum|total)\b", re.IGNORECASE), "SUM"),
)


def detect_aggregates(question: str) -> list[str]:
    """Aggregate functions the question asks for, ordered by where they appear."""
    hits: list[tuple[int, str]] = []
    for pattern, function in _AGGREGATE_WORDS:
        match = pattern.search(question)
        if match:
            hits.append((match.start(), function))
    hits.sort()
    return [function for _, function in hits]


# Superlatives that name the quantity being ranked without saying the column.
SUPERLATIVE_COLUMN_HINTS = {
    "youngest": ("age", "birth", "born"),
    "oldest": ("age", "birth", "born", "year", "founded"),
    "cheapest": ("price", "cost", "fee"),
    "priciest": ("price", "cost", "fee"),
    "largest": ("size", "capacity", "area", "population", "amount"),
    "biggest": ("size", "capacity", "area", "population", "amount"),
    "smallest": ("size", "capacity", "area", "population", "amount"),
    "longest": ("length", "duration", "time", "distance"),
    "shortest": ("length", "duration", "time", "distance"),
    "tallest": ("height",),
    "heaviest": ("weight",),
    "richest": ("salary", "income", "revenue", "worth"),
    "newest": ("year", "date"),
    "latest": ("year", "date"),
    "earliest": ("year", "date"),
    "best": ("rating", "score", "rank"),
    "worst": ("rating", "score", "rank"),
}


def superlative_column_words(question: str) -> tuple[str, ...]:
    """Column-name hints implied by a superlative, e.g. 'youngest' implies age."""
    lowered = question.lower()
    for word, hints in SUPERLATIVE_COLUMN_HINTS.items():
        if re.search(rf"\b{word}\b", lowered):
            return hints
    return ()
