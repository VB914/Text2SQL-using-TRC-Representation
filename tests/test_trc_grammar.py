"""Tests for the extended TRC grammar: shaping, membership, ranges and null tests."""

from __future__ import annotations

import pytest

from core.schema_utils import load_schema
from core.sql_executor import execute_sql
from core.trc_parser import Between, TrcSyntaxError, flatten_and, parse_trc
from core.trc_to_sql import trc_to_sql
from core.trc_validator import validate_trc


@pytest.fixture(scope="module")
def schema():
    return load_schema("data/sample.db")


@pytest.mark.parametrize(
    ("trc", "expected"),
    [
        ("{ COUNT(*) | students(s) }", "SELECT COUNT(*) AS count_star FROM students AS s;"),
        ("{ DISTINCT s.name | students(s) }", "SELECT DISTINCT s.name FROM students AS s;"),
        (
            "{ s.name | students(s) } ORDER BY s.name DESC LIMIT 5",
            "SELECT s.name FROM students AS s ORDER BY s.name DESC LIMIT 5;",
        ),
        (
            "{ s.name | students(s) } ORDER BY s.name LIMIT 5 OFFSET 2",
            "SELECT s.name FROM students AS s ORDER BY s.name LIMIT 5 OFFSET 2;",
        ),
        (
            "{ s.name | students(s) AND s.name LIKE 'A%' }",
            "SELECT s.name FROM students AS s WHERE s.name LIKE 'A%';",
        ),
        (
            "{ s.name | students(s) AND s.name NOT LIKE 'A%' }",
            "SELECT s.name FROM students AS s WHERE s.name NOT LIKE 'A%';",
        ),
        (
            "{ s.name | students(s) AND s.email IS NOT NULL }",
            "SELECT s.name FROM students AS s WHERE s.email IS NOT NULL;",
        ),
        (
            "{ s.name | students(s) AND s.id IN (1, 2, 3) }",
            "SELECT s.name FROM students AS s WHERE s.id IN (1, 2, 3);",
        ),
        (
            "{ s.name | students(s) AND s.id IN { e.student_id | enrollments(e) } }",
            "SELECT s.name FROM students AS s "
            "WHERE s.id IN (SELECT e.student_id FROM enrollments AS e);",
        ),
    ],
)
def test_compiles_to_expected_sql(trc, expected):
    assert trc_to_sql(trc) == expected


def test_between_consumes_its_own_and():
    """``BETWEEN lo AND hi`` must not leak its AND into the surrounding conjunction."""
    query = parse_trc("{ s.name | students(s) AND s.id BETWEEN 1 AND 5 AND s.name = 'Alice' }")
    terms = flatten_and(query.formula)

    assert len(terms) == 3
    assert any(isinstance(term, Between) for term in terms)


def test_aggregate_restriction_becomes_having_not_a_join():
    """An aggregate comparison has two variables but can never be a join condition."""
    sql = trc_to_sql(
        "{ d.name | departments(d) AND students(s) "
        "AND s.department_id = d.id AND COUNT(s.id) > 5 }"
    )

    before_having, having = sql.split("HAVING")
    assert "ON s.department_id = d.id" in before_having
    assert "COUNT" not in before_having
    assert "COUNT(s.id) > 5" in having


def test_grouping_is_derived_from_an_aggregate_in_the_formula():
    """A HAVING clause without GROUP BY would silently collapse the result."""
    sql = trc_to_sql(
        "{ d.name | departments(d) AND students(s) "
        "AND s.department_id = d.id AND COUNT(s.id) > 1 }"
    )

    assert "GROUP BY d.name" in sql


def test_order_by_reuses_the_projected_aggregate_alias():
    sql = trc_to_sql(
        "{ d.name, COUNT(*) | departments(d) AND students(s) AND s.department_id = d.id } "
        "ORDER BY COUNT(*) DESC LIMIT 3"
    )

    assert "ORDER BY count_star DESC" in sql


def test_keyword_named_column_still_parses():
    """Spider ships a column literally named "count"."""
    assert trc_to_sql("{ y.count | yelp(y) }") == "SELECT y.count FROM yelp AS y;"


@pytest.mark.parametrize(
    "trc",
    [
        "{ * | students(s) }",
        "{ s.name | students(s) } ORDER BY s.missing_column",
        "{ s.name | students(s) } LIMIT 5 OFFSET -1",
    ],
)
def test_validator_rejects_malformed_queries(trc, schema):
    assert not validate_trc(trc, schema).valid


def test_star_outside_an_aggregate_is_rejected(schema):
    """``*`` is only meaningful inside COUNT, so the parser refuses it as a projection."""
    report = validate_trc("{ * | students(s) }", schema)

    assert not report.valid
    assert not report.parseable


def test_star_projection_built_directly_is_rejected(schema):
    """The validator still guards Star for ASTs assembled in code rather than parsed."""
    from core.trc_parser import RelationPredicate, Star, TrcQuery
    from core.trc_validator import _validate_expression

    issues = []
    _validate_expression(Star(), {"s": "students"}, {"students": {"name"}}, issues)

    assert any("aggregate" in issue.message for issue in issues)


@pytest.mark.parametrize(
    "trc",
    [
        "{ s.name | students(s) } ORDER BY",
        "{ s.name | students(s) } LIMIT abc",
        "{ s.name | students(s) AND s.name LIKE 5 }",
        "{ s.name | students(s) AND s.id IN () }",
    ],
)
def test_syntax_errors_are_reported(trc):
    with pytest.raises(TrcSyntaxError):
        parse_trc(trc)


@pytest.mark.parametrize(
    "trc",
    [
        "{ s.name | students(s) } ORDER BY s.name DESC LIMIT 3",
        "{ DISTINCT s.name | students(s) }",
        "{ s.name | students(s) AND s.name LIKE 'A%' }",
        "{ d.name, COUNT(*) | departments(d) AND students(s) AND s.department_id = d.id } "
        "ORDER BY COUNT(*) DESC LIMIT 2",
    ],
)
def test_generated_sql_executes(trc, schema):
    """Compiling is not enough; the SQL has to run against a real database."""
    report = validate_trc(trc, schema)
    assert report.valid, report.summary

    result = execute_sql("data/sample.db", trc_to_sql(trc, schema))
    assert result.row_count > 0
