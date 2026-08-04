from __future__ import annotations

import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from core.config import get_settings
from core.schema_utils import resolve_db_path
from models import QueryResult


READ_ONLY_PREFIXES = ("SELECT", "WITH")
BLOCKED_KEYWORDS = (
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "PRAGMA",
    "ATTACH",
    "DETACH",
    "CREATE",
    # "REPLACE" is deliberately absent: it is a legitimate scalar function, and the
    # statement form (REPLACE INTO) is already rejected by the read-only prefix check.
    "VACUUM",
    "REINDEX",
    "TRUNCATE",
)

# Actions a read-only analytical query legitimately needs. Anything else is denied
# by SQLite itself, which cannot be fooled by string tricks the way a regex can.
_ALLOWED_ACTIONS = frozenset(
    action
    for action in (
        getattr(sqlite3, "SQLITE_SELECT", None),
        getattr(sqlite3, "SQLITE_READ", None),
        getattr(sqlite3, "SQLITE_FUNCTION", None),
        getattr(sqlite3, "SQLITE_RECURSIVE", None),
    )
    if action is not None
)

_STRING_LITERAL = re.compile(r"'[^']*'|\"[^\"]*\"")
_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def _strip_literals_and_comments(sql: str) -> str:
    """Blank out string literals and comments so keyword scanning sees only code.

    Without this, a perfectly safe query such as ``WHERE status = 'Update'`` is
    rejected because the blocklist matches text inside a value.
    """
    without_comments = _BLOCK_COMMENT.sub(" ", _LINE_COMMENT.sub(" ", sql))
    return _STRING_LITERAL.sub("''", without_comments)


def ensure_safe_sql(sql: str) -> str:
    cleaned = sql.strip().rstrip(";")
    if not cleaned:
        raise ValueError("SQL query is empty.")

    scannable = _strip_literals_and_comments(cleaned)
    if ";" in scannable:
        raise ValueError("Only one SQL statement is allowed.")
    if scannable.split(None, 1)[0].upper() not in READ_ONLY_PREFIXES:
        raise ValueError("Only read-only SELECT or WITH queries are allowed.")
    for keyword in BLOCKED_KEYWORDS:
        if re.search(rf"\b{keyword}\b", scannable, flags=re.IGNORECASE):
            raise ValueError(f"Blocked keyword detected in SQL: {keyword}")
    return cleaned


def _authorizer(action: int, *_args: object) -> int:
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def _read_only_uri(path: Path) -> str:
    """Build a ``file:`` URI that opens the database read-only.

    ``Path.as_uri()`` handles Windows drive letters and percent-encoding correctly,
    which hand-built URI strings routinely get wrong.
    """
    return f"{path.as_uri()}?mode=ro"


def _connect(db_path: str | None) -> sqlite3.Connection:
    path = resolve_db_path(db_path)
    return sqlite3.connect(_read_only_uri(path), uri=True)


def _apply_guards(connection: sqlite3.Connection, timeout_seconds: float) -> None:
    connection.set_authorizer(_authorizer)
    deadline = time.monotonic() + timeout_seconds
    # Returning a truthy value from the progress handler aborts the statement,
    # so a runaway cartesian product cannot hang a batch evaluation run.
    connection.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)


def execute_sql(
    db_path: str | None,
    sql: str,
    limit: int | None = None,
    timeout_seconds: float | None = None,
) -> QueryResult:
    settings = get_settings()
    safe_sql = ensure_safe_sql(sql)
    row_limit = limit or settings.max_result_rows
    budget = timeout_seconds or settings.sql_timeout_seconds

    with closing(_connect(db_path)) as connection:
        connection.row_factory = sqlite3.Row
        _apply_guards(connection, budget)
        try:
            cursor = connection.cursor()
            cursor.execute(safe_sql)
            rows = [dict(row) for row in cursor.fetchmany(row_limit)]
            columns = list(rows[0].keys()) if rows else [item[0] for item in cursor.description or []]
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc).lower():
                raise TimeoutError(f"Query exceeded the {budget:g}s time budget.") from exc
            raise
        return QueryResult(columns=columns, rows=rows, row_count=len(rows))


def sql_is_valid(db_path: str | None, sql: str) -> bool:
    """Check that SQL parses and plans against the real schema without running it."""
    safe_sql = ensure_safe_sql(sql)
    with closing(_connect(db_path)) as connection:
        _apply_guards(connection, get_settings().sql_timeout_seconds)
        connection.execute(f"EXPLAIN QUERY PLAN {safe_sql}").fetchall()
    return True
