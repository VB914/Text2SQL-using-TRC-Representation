"""Tests for the SQL safety filter and read-only execution.

These guard the Phase 1 fixes: the connection leak, genuine read-only access,
the query time budget, and a keyword filter that no longer rejects legitimate
queries containing words like 'Update' inside a string literal.
"""

from __future__ import annotations

import os
import shutil
import sqlite3

import pytest

from core.sql_executor import _connect, ensure_safe_sql, execute_sql, sql_is_valid

DB = "data/sample.db"


@pytest.fixture
def db_copy(tmp_path):
    """A throwaway copy, so lock and write tests never touch the real database."""
    target = tmp_path / "probe.db"
    shutil.copy(DB, target)
    return str(target)


class TestRejectsDangerousSql:
    @pytest.mark.parametrize(
        "sql",
        [
            "DROP TABLE students",
            "DELETE FROM students",
            "UPDATE students SET name = 'x'",
            "INSERT INTO students VALUES (1)",
            "ALTER TABLE students ADD COLUMN x TEXT",
            "CREATE TABLE hack (x TEXT)",
            "ATTACH DATABASE 'other.db' AS other",
            "PRAGMA table_info('students')",
            "VACUUM",
        ],
    )
    def test_non_read_only_statements_are_blocked(self, sql):
        with pytest.raises(ValueError):
            ensure_safe_sql(sql)

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT 1; DROP TABLE students",
            "SELECT name FROM students; DELETE FROM students",
        ],
    )
    def test_stacked_statements_are_blocked(self, sql):
        with pytest.raises(ValueError, match="one SQL statement"):
            ensure_safe_sql(sql)

    def test_empty_sql_is_rejected(self):
        with pytest.raises(ValueError):
            ensure_safe_sql("   ")

    def test_comment_cannot_smuggle_a_second_statement(self):
        with pytest.raises(ValueError):
            ensure_safe_sql("SELECT 1 -- harmless\n; DROP TABLE students")


class TestAllowsLegitimateSql:
    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT name FROM students",
            "WITH t AS (SELECT 1 AS x) SELECT x FROM t",
            "SELECT COUNT(*) FROM students GROUP BY name HAVING COUNT(*) > 1",
        ],
    )
    def test_read_only_queries_pass(self, sql):
        assert ensure_safe_sql(sql)

    def test_literal_containing_a_blocked_word_is_allowed(self):
        """A value is not a statement: 'Update' in quotes must not trip the filter."""
        assert ensure_safe_sql("SELECT name FROM students WHERE name = 'Update'")

    def test_replace_function_is_allowed(self):
        """REPLACE is a scalar function; only REPLACE INTO is a write."""
        assert ensure_safe_sql("SELECT REPLACE(name, 'a', 'b') FROM students")

    def test_trailing_semicolon_is_stripped(self):
        assert ensure_safe_sql("SELECT 1;") == "SELECT 1"


class TestReadOnlyEnforcement:
    def test_writes_fail_against_a_read_only_connection(self, db_copy):
        with _connect(db_copy) as connection:
            with pytest.raises(sqlite3.DatabaseError, match="readonly"):
                connection.execute("CREATE TABLE hack (x TEXT)")

    def test_connection_is_closed_after_a_query(self, db_copy):
        """On Windows a leaked handle keeps the file locked, so deletion proves closure."""
        execute_sql(db_copy, "SELECT COUNT(*) FROM students")

        os.remove(db_copy)
        assert not os.path.exists(db_copy)


class TestExecution:
    def test_returns_rows_and_columns(self):
        result = execute_sql(DB, "SELECT COUNT(*) AS n FROM students")

        assert result.columns == ["n"]
        assert result.row_count == 1
        assert result.rows[0]["n"] > 0

    def test_row_limit_is_respected(self):
        result = execute_sql(DB, "SELECT name FROM students", limit=2)

        assert result.row_count <= 2

    def test_empty_result_still_reports_columns(self):
        result = execute_sql(DB, "SELECT name FROM students WHERE 1 = 0")

        assert result.row_count == 0
        assert result.columns == ["name"]

    def test_planning_accepts_valid_sql(self):
        assert sql_is_valid(DB, "SELECT name FROM students")

    def test_planning_rejects_an_unknown_column(self):
        with pytest.raises(sqlite3.DatabaseError):
            sql_is_valid(DB, "SELECT no_such_column FROM students")

    def test_runaway_query_hits_the_time_budget(self):
        """A recursive CTE would otherwise run forever and hang a batch run."""
        runaway = (
            "WITH RECURSIVE forever(x) AS "
            "(SELECT 1 UNION ALL SELECT x + 1 FROM forever) "
            "SELECT COUNT(*) FROM forever"
        )
        with pytest.raises((TimeoutError, sqlite3.DatabaseError)):
            execute_sql(DB, runaway, timeout_seconds=1)
