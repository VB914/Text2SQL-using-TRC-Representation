from __future__ import annotations

import logging
import re
from collections import Counter
from itertools import permutations

from core.sql_executor import execute_sql
from models import QueryResult


LOGGER = logging.getLogger(__name__)

# Comparing every column ordering is factorial, so it is only attempted for the
# narrow result shapes where it is cheap and where alias drift actually happens.
MAX_PERMUTATION_ARITY = 6
COMPARISON_ROW_LIMIT = 1000
_ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)


def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()


def exact_match(predicted_sql: str | None, gold_sql: str | None) -> bool | None:
    """Compare normalised SQL text.

    This is a deliberately strict, surface-level metric: a semantically correct
    query written differently from the gold query scores zero. Execution accuracy
    is the metric that reflects whether the answer was actually right.
    """
    if not predicted_sql or not gold_sql:
        return None
    return normalize_sql(predicted_sql) == normalize_sql(gold_sql)


def gold_is_order_sensitive(gold_sql: str | None) -> bool:
    """Row order only matters when the gold query itself asked for an ordering."""
    return bool(gold_sql and _ORDER_BY.search(gold_sql))


def normalize_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        # Float noise and int/float mismatches (1 vs 1.0) are not real differences.
        return round(float(value), 6)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, str):
        return value.strip()
    return str(value)


def result_to_tuples(result: QueryResult) -> list[tuple]:
    """Convert a result to positional value tuples.

    Column *names* are discarded on purpose. A query aliased ``COUNT(*) AS n``
    and a gold query returning ``count(*)`` produce identical answers, and any
    comparison keyed on names would wrongly mark the prediction incorrect.
    """
    return [tuple(normalize_value(value) for value in row.values()) for row in result.rows]


def _matches(predicted: list[tuple], gold: list[tuple], order_sensitive: bool) -> bool:
    if order_sensitive:
        return predicted == gold
    return Counter(predicted) == Counter(gold)


def compare_results(
    predicted: QueryResult,
    gold: QueryResult,
    order_sensitive: bool = False,
) -> bool:
    predicted_rows = result_to_tuples(predicted)
    gold_rows = result_to_tuples(gold)

    if len(predicted_rows) != len(gold_rows):
        return False
    if not gold_rows:
        return True
    arity = len(gold_rows[0])
    if len(predicted_rows[0]) != arity:
        return False
    if _matches(predicted_rows, gold_rows, order_sensitive):
        return True

    # Same values, different column order still answers the question.
    if arity > MAX_PERMUTATION_ARITY:
        return False
    for perm in permutations(range(arity)):
        if perm == tuple(range(arity)):
            continue
        reordered = [tuple(row[index] for index in perm) for row in predicted_rows]
        if _matches(reordered, gold_rows, order_sensitive):
            return True
    return False


def execution_accuracy(
    db_path: str | None,
    predicted_sql: str | None,
    gold_sql: str | None,
    order_sensitive: bool | None = None,
) -> bool | None:
    """Return whether the predicted query produces the gold query's answer.

    ``None`` means the comparison could not be made (missing SQL, or gold SQL that
    does not run against this database) and should be excluded from the metric
    rather than counted as a failure.
    """
    if not predicted_sql or not gold_sql:
        return None
    if order_sensitive is None:
        order_sensitive = gold_is_order_sensitive(gold_sql)

    try:
        gold_result = execute_sql(db_path, gold_sql, limit=COMPARISON_ROW_LIMIT)
    except Exception:
        LOGGER.warning("Gold SQL could not be executed; excluding from metric.", exc_info=True)
        return None

    try:
        predicted_result = execute_sql(db_path, predicted_sql, limit=COMPARISON_ROW_LIMIT)
    except Exception:
        return False

    return compare_results(predicted_result, gold_result, order_sensitive)
