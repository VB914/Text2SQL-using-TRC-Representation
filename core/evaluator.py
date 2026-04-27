from __future__ import annotations

import re
from collections import Counter

from core.sql_executor import execute_sql, sql_is_valid


def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()


def exact_match(predicted_sql: str | None, gold_sql: str | None) -> bool | None:
    if not predicted_sql or not gold_sql:
        return None
    return normalize_sql(predicted_sql) == normalize_sql(gold_sql)


def execution_accuracy(db_path: str, predicted_sql: str | None, gold_sql: str | None) -> bool | None:
    if not predicted_sql or not gold_sql:
        return None
    predicted = Counter(_freeze(row) for row in execute_sql(db_path, predicted_sql, limit=500).rows)
    gold = Counter(_freeze(row) for row in execute_sql(db_path, gold_sql, limit=500).rows)
    return predicted == gold


def _freeze(row: dict) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((key, repr(value)) for key, value in row.items()))
