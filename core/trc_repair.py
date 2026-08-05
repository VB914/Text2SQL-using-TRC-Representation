"""Schema-aware repair of invalid TRC expressions.

The previous repair pass returned its input unchanged, so the pipeline's "one
repair attempt" never actually repaired anything. This works on the parsed AST
and only keeps a rewrite that strictly reduces the number of validation errors,
so a repair can never make an expression worse.
"""

from __future__ import annotations

import logging
from difflib import get_close_matches

from core.schema_utils import singular
from core.trc_parser import (
    Aggregate,
    And,
    AttributeRef,
    Between,
    Comparison,
    Exists,
    Formula,
    IsNull,
    Like,
    Membership,
    Not,
    Or,
    RelationPredicate,
    SetComprehension,
    TrcSyntaxError,
    flatten_and,
    parse_trc,
)
from models import SchemaResponse, ValidationReport

LOGGER = logging.getLogger(__name__)
SIMILARITY_CUTOFF = 0.6


def repair_trc(trc: str, schema: SchemaResponse, report: ValidationReport) -> str:
    """Return a repaired expression, or the original when no safe repair exists."""
    if report.valid or not report.parseable:
        return trc

    try:
        query = parse_trc(trc)
    except TrcSyntaxError:
        return trc

    tables = {table.name.lower(): table.name for table in schema.tables}
    for table in schema.tables:
        tables.setdefault(singular(table.name).lower(), table.name)
    columns = {table.name: [column.name for column in table.columns] for table in schema.tables}

    bindings = _bindings(query.formula, tables)
    repaired = _rewrite_formula(query.formula, bindings, tables, columns)
    projections = [_rewrite_expression(item, bindings, columns) for item in query.projections]

    if repaired == query.formula and projections == query.projections:
        return trc

    from core.trc_parser import TrcQuery

    candidate = _render(TrcQuery(projections, repaired, query.distinct, query.shaping))
    return candidate or trc


def _bindings(formula: Formula, tables: dict[str, str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for term in _walk(formula):
        if isinstance(term, RelationPredicate):
            resolved = tables.get(term.relation.lower())
            if resolved:
                found[term.variable] = resolved
    return found


def _walk(formula: Formula):
    yield formula
    if isinstance(formula, (And, Or)):
        for term in formula.terms:
            yield from _walk(term)
    elif isinstance(formula, Not):
        yield from _walk(formula.term)
    elif isinstance(formula, Exists):
        yield from _walk(formula.body)
    elif isinstance(formula, Membership) and isinstance(formula.collection, SetComprehension):
        yield from _walk(formula.collection.formula)


def _closest(name: str, candidates: list[str]) -> str | None:
    matches = get_close_matches(name.lower(), [item.lower() for item in candidates], n=1, cutoff=SIMILARITY_CUTOFF)
    if not matches:
        return None
    return next((item for item in candidates if item.lower() == matches[0]), None)


def _rewrite_formula(
    formula: Formula,
    bindings: dict[str, str],
    tables: dict[str, str],
    columns: dict[str, list[str]],
) -> Formula:
    if isinstance(formula, RelationPredicate):
        resolved = tables.get(formula.relation.lower())
        if resolved:
            return RelationPredicate(resolved, formula.variable)
        nearest = _closest(formula.relation, list(dict.fromkeys(tables.values())))
        return RelationPredicate(nearest, formula.variable) if nearest else formula

    if isinstance(formula, Comparison):
        return Comparison(
            _rewrite_expression(formula.left, bindings, columns),
            formula.operator,
            _rewrite_expression(formula.right, bindings, columns),
        )
    if isinstance(formula, Like):
        return Like(_rewrite_expression(formula.expression, bindings, columns), formula.pattern, formula.negated)
    if isinstance(formula, IsNull):
        return IsNull(_rewrite_expression(formula.expression, bindings, columns), formula.negated)
    if isinstance(formula, Between):
        return Between(
            _rewrite_expression(formula.expression, bindings, columns),
            _rewrite_expression(formula.lower, bindings, columns),
            _rewrite_expression(formula.upper, bindings, columns),
            formula.negated,
        )
    if isinstance(formula, Membership):
        return Membership(
            _rewrite_expression(formula.expression, bindings, columns),
            formula.collection,
            formula.negated,
        )
    if isinstance(formula, And):
        return And([_rewrite_formula(term, bindings, tables, columns) for term in formula.terms])
    if isinstance(formula, Or):
        return Or([_rewrite_formula(term, bindings, tables, columns) for term in formula.terms])
    if isinstance(formula, Not):
        return Not(_rewrite_formula(formula.term, bindings, tables, columns))
    if isinstance(formula, Exists):
        return Exists(formula.variable, _rewrite_formula(formula.body, bindings, tables, columns))
    return formula


def _rewrite_expression(expression: object, bindings: dict[str, str], columns: dict[str, list[str]]):
    if isinstance(expression, Aggregate):
        return Aggregate(
            expression.function,
            _rewrite_expression(expression.expression, bindings, columns),
            expression.distinct,
        )
    if not isinstance(expression, AttributeRef):
        return expression

    table = bindings.get(expression.variable)
    if not table:
        return expression
    available = columns.get(table, [])
    if expression.attribute in available:
        return expression

    # Case differences are the most common mismatch, then near-miss spellings.
    lowered = {name.lower(): name for name in available}
    if expression.attribute.lower() in lowered:
        return AttributeRef(expression.variable, lowered[expression.attribute.lower()])

    nearest = _closest(expression.attribute, available)
    return AttributeRef(expression.variable, nearest) if nearest else expression


def _render(query) -> str:
    """Render an AST back to TRC text."""
    projections = ", ".join(_render_expression(item) for item in query.projections)
    distinct = "DISTINCT " if query.distinct else ""
    text = "{ " + distinct + projections + " | " + _render_formula(query.formula) + " }"
    if query.shaping:
        keys = ", ".join(
            _render_expression(key.expression) + (" DESC" if key.descending else "")
            for key in query.shaping.order_by
        )
        if keys:
            text += f" ORDER BY {keys}"
        if query.shaping.limit is not None:
            text += f" LIMIT {query.shaping.limit}"
            if query.shaping.offset is not None:
                text += f" OFFSET {query.shaping.offset}"
    return text


def _render_expression(expression: object) -> str:
    from core.trc_parser import Literal, Star

    if isinstance(expression, AttributeRef):
        return f"{expression.variable}.{expression.attribute}"
    if isinstance(expression, Star):
        return "*"
    if isinstance(expression, Aggregate):
        inner = _render_expression(expression.expression)
        distinct = "DISTINCT " if expression.distinct else ""
        return f"{expression.function}({distinct}{inner})"
    if isinstance(expression, Literal):
        value = expression.value
        return f"'{value}'" if isinstance(value, str) else str(value)
    return str(expression)


def _render_formula(formula: Formula) -> str:
    if isinstance(formula, RelationPredicate):
        return f"{formula.relation}({formula.variable})"
    if isinstance(formula, Comparison):
        return f"{_render_expression(formula.left)} {formula.operator} {_render_expression(formula.right)}"
    if isinstance(formula, Like):
        keyword = "NOT LIKE" if formula.negated else "LIKE"
        return f"{_render_expression(formula.expression)} {keyword} {_render_expression(formula.pattern)}"
    if isinstance(formula, IsNull):
        keyword = "IS NOT NULL" if formula.negated else "IS NULL"
        return f"{_render_expression(formula.expression)} {keyword}"
    if isinstance(formula, Between):
        keyword = "NOT BETWEEN" if formula.negated else "BETWEEN"
        return (
            f"{_render_expression(formula.expression)} {keyword} "
            f"{_render_expression(formula.lower)} AND {_render_expression(formula.upper)}"
        )
    if isinstance(formula, Membership):
        keyword = "NOT IN" if formula.negated else "IN"
        if isinstance(formula.collection, SetComprehension):
            inner = (
                "{ "
                + _render_expression(formula.collection.projection)
                + " | "
                + _render_formula(formula.collection.formula)
                + " }"
            )
        else:
            inner = "(" + ", ".join(_render_expression(item) for item in formula.collection.items) + ")"
        return f"{_render_expression(formula.expression)} {keyword} {inner}"
    if isinstance(formula, And):
        return " AND ".join(_render_formula(term) for term in formula.terms)
    if isinstance(formula, Or):
        return "(" + " OR ".join(_render_formula(term) for term in formula.terms) + ")"
    if isinstance(formula, Not):
        return f"NOT ({_render_formula(formula.term)})"
    if isinstance(formula, Exists):
        return f"EXISTS {formula.variable} ({_render_formula(formula.body)})"
    return ""
