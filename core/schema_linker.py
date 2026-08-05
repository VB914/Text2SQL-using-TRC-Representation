"""Link question text to concrete schema elements and stored database values.

The generator previously recovered filter values only from quoted substrings, so
a question such as "which singers are from France" produced no WHERE clause at
all because nobody writes quotes in natural language. This module indexes the
values actually stored in the database and matches question phrases against them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from core.config import get_settings
from core.schema_utils import resolve_db_path
from models import SchemaResponse, TableInfo

LOGGER = logging.getLogger(__name__)

TEXT_TYPE_MARKERS = ("CHAR", "TEXT", "CLOB", "STRING", "VARCHAR")
NUMERIC_TYPE_MARKERS = ("INT", "REAL", "NUM", "DEC", "FLOAT", "DOUBLE")
MAX_NGRAM = 4
FUZZY_THRESHOLD = 0.86

# Words that appear in almost every question and would otherwise match stray cells.
STOPWORDS = frozenset(
    """
    a an and are as at be by do does did for from has have how in is it its of on or
    that the there these this to was were what when where which who whom whose why
    with all any each every many much more most least less few number list show give
    find name names tell me please return count total sum average maximum minimum
    """.split()
)

_WORD = re.compile(r"[A-Za-z0-9]+")

COLUMN_SYNONYMS = {
    "name": ("name", "called", "titled", "title"),
    "age": ("age", "old", "years"),
    "price": ("price", "cost", "fee", "charge"),
    "salary": ("salary", "pay", "wage", "income"),
    "year": ("year", "date", "when"),
    "count": ("count", "number", "many"),
    "rating": ("rating", "score", "rank", "grade"),
    "country": ("country", "nation", "nationality"),
    "city": ("city", "town"),
}


@dataclass(frozen=True)
class ValueMatch:
    table: str
    column: str
    value: str
    phrase: str
    exact: bool

    @property
    def score(self) -> float:
        # Longer phrases are more specific, and an exact cell hit beats a fuzzy one.
        return len(self.phrase.split()) + (1.0 if self.exact else 0.0)


def normalize(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def _is_text_column(data_type: str) -> bool:
    upper = data_type.upper()
    return any(marker in upper for marker in TEXT_TYPE_MARKERS) or not upper.strip()


def is_numeric_column(data_type: str) -> bool:
    return any(marker in data_type.upper() for marker in NUMERIC_TYPE_MARKERS)


class ValueIndex:
    """Maps normalised cell values to the columns that contain them."""

    def __init__(self, entries: dict[str, list[list[str]]] | None = None) -> None:
        self._entries: dict[str, list[tuple[str, str, str]]] = {
            key: [tuple(item) for item in value] for key, value in (entries or {}).items()
        }

    def __len__(self) -> int:
        return len(self._entries)

    def to_json(self) -> dict[str, list[list[str]]]:
        return {key: [list(item) for item in value] for key, value in self._entries.items()}

    def add(self, table: str, column: str, value: str) -> None:
        key = normalize(value)
        if not key or key in STOPWORDS:
            return
        self._entries.setdefault(key, []).append((table, column, value))

    def lookup(self, phrase: str) -> list[tuple[str, str, str]]:
        return self._entries.get(normalize(phrase), [])

    def fuzzy_lookup(self, phrase: str) -> list[tuple[str, str, str]]:
        target = normalize(phrase)
        if len(target) < 4:
            return []
        best: list[tuple[str, str, str]] = []
        best_ratio = FUZZY_THRESHOLD
        for key, hits in self._entries.items():
            if abs(len(key) - len(target)) > 4:
                continue
            ratio = SequenceMatcher(None, target, key).ratio()
            if ratio > best_ratio:
                best_ratio, best = ratio, hits
        return best


def _cache_path(db_path: Path) -> Path:
    stat = db_path.stat()
    digest = hashlib.sha1(
        f"{db_path}|{stat.st_mtime_ns}|{stat.st_size}".encode("utf-8")
    ).hexdigest()[:16]
    return get_settings().cache_dir / "value_index" / f"{digest}.json"


_MEMORY_CACHE: dict[str, ValueIndex] = {}


def build_value_index(db_path: str | None, schema: SchemaResponse | None = None) -> ValueIndex:
    """Index distinct text values, caching to disk so repeated runs stay fast."""
    settings = get_settings()
    if not settings.value_index_enabled:
        return ValueIndex()

    path = resolve_db_path(db_path)
    key = str(path)
    if key in _MEMORY_CACHE:
        return _MEMORY_CACHE[key]

    cache_file = _cache_path(path)
    if cache_file.exists():
        try:
            index = ValueIndex(json.loads(cache_file.read_text(encoding="utf-8")))
            _MEMORY_CACHE[key] = index
            return index
        except (OSError, ValueError):
            LOGGER.debug("Discarding unreadable value index cache %s", cache_file)

    index = _scan_database(path, schema)
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(index.to_json()), encoding="utf-8")
    except OSError:
        LOGGER.debug("Could not persist value index for %s", path, exc_info=True)

    _MEMORY_CACHE[key] = index
    return index


def _scan_database(path: Path, schema: SchemaResponse | None) -> ValueIndex:
    from core.schema_utils import load_schema

    schema = schema or load_schema(str(path))
    index = ValueIndex()
    max_rows = get_settings().value_index_max_rows

    with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as connection:
        cursor = connection.cursor()
        for table in schema.tables:
            for column in table.columns:
                if column.primary_key or not _is_text_column(column.data_type):
                    continue
                try:
                    rows = cursor.execute(
                        f'SELECT DISTINCT "{column.name}" FROM "{table.name}" '
                        f'WHERE "{column.name}" IS NOT NULL LIMIT {max_rows + 1}'
                    ).fetchall()
                except sqlite3.Error:
                    continue
                # A column with an unbounded number of distinct values is free text,
                # not a categorical attribute worth matching against.
                if len(rows) > max_rows:
                    continue
                for (value,) in rows:
                    if isinstance(value, str) and 0 < len(value) <= 80:
                        index.add(table.name, column.name, value)
    return index


def question_ngrams(question: str, max_n: int = MAX_NGRAM) -> list[str]:
    """Longest phrases first, so 'New York' is preferred over 'New'."""
    words = _WORD.findall(question)
    ngrams: list[str] = []
    for size in range(min(max_n, len(words)), 0, -1):
        for start in range(len(words) - size + 1):
            ngrams.append(" ".join(words[start : start + size]))
    return ngrams


def link_values(question: str, index: ValueIndex, tables: list[TableInfo]) -> list[ValueMatch]:
    """Find question phrases that name a value stored in one of ``tables``."""
    if not len(index):
        return []

    allowed = {table.name for table in tables}
    schema_words = {normalize(table.name) for table in tables}
    for table in tables:
        schema_words.update(normalize(column.name) for column in table.columns)

    matches: list[ValueMatch] = []
    consumed: set[str] = set()
    for phrase in question_ngrams(question):
        normalized = normalize(phrase)
        if not normalized or normalized in STOPWORDS or normalized in schema_words:
            continue
        if any(normalized in taken for taken in consumed):
            continue
        if all(word in STOPWORDS for word in normalized.split()):
            continue

        hits = index.lookup(phrase)
        exact = True
        if not hits and len(normalized.split()) == 1:
            hits = index.fuzzy_lookup(phrase)
            exact = False

        relevant = [hit for hit in hits if hit[0] in allowed]
        if not relevant:
            continue
        table, column, value = relevant[0]
        matches.append(ValueMatch(table=table, column=column, value=value, phrase=normalized, exact=exact))
        consumed.add(normalized)

    matches.sort(key=lambda match: match.score, reverse=True)
    return matches


def _match_position(words: list[str], column_tokens: list[str]) -> int | None:
    """Where in ``words`` the column name is mentioned, or None."""
    if not column_tokens:
        return None

    # A contiguous run of the column's words is the strongest signal.
    for start in range(len(words) - len(column_tokens) + 1):
        window = words[start : start + len(column_tokens)]
        if all(
            word == token or _singular(word) == _singular(token)
            for word, token in zip(window, column_tokens)
        ):
            return start

    if len(column_tokens) > 1:
        positions = []
        for token in column_tokens:
            found = next(
                (i for i, word in enumerate(words) if _singular(word) == _singular(token)),
                None,
            )
            if found is None:
                return None
            positions.append(found)
        return min(positions)

    token = column_tokens[0]
    return next(
        (i for i, word in enumerate(words) if _singular(word) == _singular(token)),
        None,
    )


def _singular(token: str) -> str:
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


# Columns whose names are too generic to count as an explicit mention.
_GENERIC_COLUMNS = frozenset({"id", "code", "key", "value", "type", "status", "no", "num"})


def find_mentioned_columns(question: str, tables: list[TableInfo]) -> list[tuple[TableInfo, str]]:
    """Columns the question names, in the order they are mentioned.

    Question word order is a good proxy for the order the user expects results in,
    which matters because the gold queries project columns in the order asked.
    """
    words = _WORD.findall(question.lower())
    found: list[tuple[int, str, TableInfo, str]] = []
    seen: set[tuple[str, str]] = set()

    for table in tables:
        for column in table.columns:
            key = (table.name, column.name)
            if key in seen:
                continue
            tokens = normalize(column.name).split()
            if not tokens or (len(tokens) == 1 and tokens[0] in _GENERIC_COLUMNS):
                continue
            position = _match_position(words, tokens)
            if position is not None:
                seen.add(key)
                # Longer column names are more specific, so they win ties.
                found.append((position, f"{-len(tokens)}{table.name}", table, column.name))

    found.sort(key=lambda item: (item[0], item[1]))
    return [(table, column) for _, _, table, column in found]


def link_columns(phrase: str, tables: list[TableInfo]) -> list[tuple[TableInfo, str]]:
    """Rank columns by how well their name matches ``phrase``."""
    words = set(normalize(phrase).split())
    if not words:
        return []

    scored: list[tuple[float, str, TableInfo, str]] = []
    for table in tables:
        for column in table.columns:
            column_words = set(normalize(column.name).split())
            score = float(len(words & column_words)) * 2
            for canonical, synonyms in COLUMN_SYNONYMS.items():
                if canonical in column_words and words & set(synonyms):
                    score += 1.5
            if score > 0:
                scored.append((score, f"{table.name}.{column.name}", table, column.name))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [(table, column) for _, _, table, column in scored]
