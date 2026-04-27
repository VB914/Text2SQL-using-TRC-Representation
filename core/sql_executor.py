from __future__ import annotations

import re
import sqlite3

from core.config import get_settings
from core.schema_utils import resolve_db_path
from models import QueryResult


READ_ONLY_PREFIXES = ("SELECT", "WITH")
BLOCKED_KEYWORDS = ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "PRAGMA", "ATTACH", "DETACH", "CREATE", "REPLACE")


def ensure_safe_sql(sql: str) -> str:
    cleaned = sql.strip().rstrip(";")
    if not cleaned:
        raise ValueError("SQL query is empty.")
    if ";" in cleaned:
        raise ValueError("Only one SQL statement is allowed.")
    if cleaned.split(None, 1)[0].upper() not in READ_ONLY_PREFIXES:
        raise ValueError("Only read-only SELECT or WITH queries are allowed.")
    for keyword in BLOCKED_KEYWORDS:
        if re.search(rf"\b{keyword}\b", cleaned, flags=re.IGNORECASE):
            raise ValueError(f"Blocked keyword detected in SQL: {keyword}")
    return cleaned


def execute_sql(db_path: str | None, sql: str, limit: int | None = None) -> QueryResult:
    safe_sql = ensure_safe_sql(sql)
    row_limit = limit or get_settings().max_result_rows
    with sqlite3.connect(resolve_db_path(db_path)) as connection:
        connection.row_factory = sqlite3.Row
        cursor = connection.cursor()
        cursor.execute(safe_sql)
        rows = [dict(row) for row in cursor.fetchmany(row_limit)]
        columns = list(rows[0].keys()) if rows else [item[0] for item in cursor.description or []]
        return QueryResult(columns=columns, rows=rows, row_count=len(rows))


def sql_is_valid(db_path: str | None, sql: str) -> bool:
    safe_sql = ensure_safe_sql(sql)
    with sqlite3.connect(resolve_db_path(db_path)) as connection:
        connection.execute(f"EXPLAIN QUERY PLAN {safe_sql}").fetchall()
    return True
