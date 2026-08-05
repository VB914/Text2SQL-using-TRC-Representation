"""Tests for result comparison.

The evaluator decides every number this project reports, so its failure modes
matter as much as the pipeline's. It previously compared rows by column name,
which silently scored correct queries as wrong whenever an alias differed.
"""

from __future__ import annotations

import pytest

from core.evaluator import (
    compare_results,
    exact_match,
    execution_accuracy,
    gold_is_order_sensitive,
    normalize_sql,
    normalize_value,
    result_to_tuples,
)
from models import QueryResult

DB = "data/sample.db"


def result(columns: list[str], rows: list[tuple]) -> QueryResult:
    dict_rows = [dict(zip(columns, row)) for row in rows]
    return QueryResult(columns=columns, rows=dict_rows, row_count=len(dict_rows))


class TestValueNormalisation:
    @pytest.mark.parametrize(
        ("left", "right"),
        [
            (1, 1.0),
            (True, 1),
            ("  Alice  ", "Alice"),
            (1.0000001, 1.0000002),
        ],
    )
    def test_values_that_should_compare_equal(self, left, right):
        assert normalize_value(left) == normalize_value(right)

    def test_none_is_preserved_and_distinct_from_zero(self):
        assert normalize_value(None) is None
        assert normalize_value(None) != normalize_value(0)

    def test_bytes_are_comparable(self):
        assert normalize_value(b"\x01\x02") == normalize_value(b"\x01\x02")


class TestColumnNameIndependence:
    def test_different_aliases_still_match(self):
        """The regression that made correct queries score as wrong."""
        predicted = result(["n"], [(5,)])
        gold = result(["count(*)"], [(5,)])

        assert compare_results(predicted, gold)

    def test_tuples_ignore_column_names(self):
        assert result_to_tuples(result(["a"], [(1,)])) == result_to_tuples(result(["b"], [(1,)]))


class TestRowAndColumnHandling:
    def test_row_order_is_ignored_by_default(self):
        predicted = result(["name"], [("Bob",), ("Alice",)])
        gold = result(["name"], [("Alice",), ("Bob",)])

        assert compare_results(predicted, gold)

    def test_row_order_matters_when_requested(self):
        predicted = result(["name"], [("Bob",), ("Alice",)])
        gold = result(["name"], [("Alice",), ("Bob",)])

        assert not compare_results(predicted, gold, order_sensitive=True)

    def test_column_order_is_tolerated(self):
        predicted = result(["b", "a"], [(2, 1)])
        gold = result(["a", "b"], [(1, 2)])

        assert compare_results(predicted, gold)

    def test_duplicate_rows_are_significant(self):
        """Comparison is a multiset, so a lost duplicate is a real difference."""
        predicted = result(["x"], [(1,), (1,)])
        gold = result(["x"], [(1,)])

        assert not compare_results(predicted, gold)

    def test_two_empty_results_match(self):
        assert compare_results(result(["x"], []), result(["y"], []))

    def test_different_arity_never_matches(self):
        assert not compare_results(result(["a", "b"], [(1, 2)]), result(["a"], [(1,)]))

    def test_genuinely_different_values_do_not_match(self):
        assert not compare_results(result(["x"], [(1,)]), result(["x"], [(2,)]))


class TestOrderSensitivityDetection:
    @pytest.mark.parametrize(
        ("sql", "sensitive"),
        [
            ("SELECT name FROM t ORDER BY name", True),
            ("SELECT name FROM t order by name desc", True),
            ("SELECT name FROM t", False),
            ("SELECT name FROM t WHERE ordering = 1", False),
        ],
    )
    def test_detects_order_by(self, sql, sensitive):
        assert gold_is_order_sensitive(sql) is sensitive

    def test_none_is_not_order_sensitive(self):
        assert gold_is_order_sensitive(None) is False


class TestExactMatch:
    def test_ignores_whitespace_case_and_trailing_semicolon(self):
        assert exact_match("SELECT  name FROM t;", "select name from t")

    def test_is_strict_about_aliases(self):
        """Documents why exact match understates quality on this pipeline."""
        assert not exact_match("SELECT s.name FROM students AS s", "SELECT name FROM students")

    def test_missing_input_returns_none(self):
        assert exact_match(None, "SELECT 1") is None
        assert exact_match("SELECT 1", None) is None

    def test_normalize_collapses_whitespace(self):
        assert normalize_sql("SELECT   a\n FROM  t ;") == "select a from t"


class TestExecutionAccuracyAgainstRealDatabase:
    def test_aliased_aggregate_matches_plain_one(self):
        assert execution_accuracy(
            DB, "SELECT COUNT(*) AS n FROM students", "SELECT COUNT(*) FROM students"
        )

    def test_wrong_answer_fails(self):
        assert not execution_accuracy(
            DB, "SELECT COUNT(*) FROM courses", "SELECT COUNT(*) FROM students"
        )

    def test_unrunnable_prediction_is_a_failure_not_an_error(self):
        assert execution_accuracy(DB, "SELECT nope FROM students", "SELECT name FROM students") is False

    def test_unrunnable_gold_is_excluded_rather_than_counted_wrong(self):
        """A broken reference query cannot judge a prediction, so it scores None."""
        assert execution_accuracy(DB, "SELECT name FROM students", "SELECT * FROM missing_table") is None

    def test_missing_sql_returns_none(self):
        assert execution_accuracy(DB, None, "SELECT 1") is None

    def test_order_is_enforced_when_gold_orders(self):
        assert not execution_accuracy(
            DB,
            "SELECT name FROM students ORDER BY name DESC",
            "SELECT name FROM students ORDER BY name ASC",
        )

    def test_order_is_ignored_when_gold_does_not_order(self):
        assert execution_accuracy(
            DB, "SELECT name FROM students ORDER BY name DESC", "SELECT name FROM students"
        )
