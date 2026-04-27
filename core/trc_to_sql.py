from __future__ import annotations

import re

from core.trc_parser import (
    Aggregate,
    And,
    AttributeRef,
    Comparison,
    Exists,
    Expression,
    Formula,
    Not,
    Or,
    RelationPredicate,
    expression_variables,
    flatten_and,
    parse_trc,
)
from models import SchemaResponse


def trc_to_sql(trc_text: str, schema: SchemaResponse | None = None) -> str:
    query = parse_trc(trc_text)
    from_clause, where_clause = _compile_scope(query.formula)
    projections = ", ".join(_render_projection(item) for item in query.projections)
    parts = [f"SELECT {projections}", f"FROM {from_clause}"]
    if where_clause:
        parts.append(f"WHERE {where_clause}")
    group_by = [_render_expression(item) for item in query.projections if not isinstance(item, Aggregate)]
    if any(isinstance(item, Aggregate) for item in query.projections) and group_by:
        parts.append("GROUP BY " + ", ".join(group_by))
    return " ".join(parts) + ";"


def _render_projection(expression: Expression) -> str:
    rendered = _render_expression(expression)
    if isinstance(expression, Aggregate):
        return f"{rendered} AS {_aggregate_alias(expression)}"
    return rendered


def _compile_scope(formula: Formula) -> tuple[str, str]:
    terms = flatten_and(formula)
    relations = [term for term in terms if isinstance(term, RelationPredicate)]
    if not relations:
        raise ValueError("TRC has no relation predicates.")
    comparisons = [term for term in terms if isinstance(term, Comparison)]
    from_clause, used = _from_clause(relations, comparisons)
    where = [
        _render_formula(term)
        for term in terms
        if not isinstance(term, RelationPredicate) and id(term) not in used
    ]
    return from_clause, " AND ".join(item for item in where if item)


def _from_clause(relations: list[RelationPredicate], comparisons: list[Comparison]) -> tuple[str, set[int]]:
    parts = [f"{_quote_identifier(relations[0].relation)} AS {relations[0].variable}"]
    joined = {relations[0].variable}
    pending = relations[1:]
    used: set[int] = set()
    while pending:
        progressed = False
        for relation in list(pending):
            joins = []
            for comparison in comparisons:
                variables = expression_variables(comparison.left) | expression_variables(comparison.right)
                if relation.variable in variables and (variables - {relation.variable}) <= joined and len(variables) == 2:
                    joins.append(_render_formula(comparison))
                    used.add(id(comparison))
            if joins:
                parts.append(f"JOIN {_quote_identifier(relation.relation)} AS {relation.variable} ON {' AND '.join(joins)}")
                joined.add(relation.variable)
                pending.remove(relation)
                progressed = True
                break
        if not progressed:
            relation = pending.pop(0)
            parts.append(f"JOIN {_quote_identifier(relation.relation)} AS {relation.variable}")
            joined.add(relation.variable)
    return " ".join(parts), used


def _render_formula(formula: Formula) -> str:
    if isinstance(formula, Comparison):
        return f"{_render_expression(formula.left)} {formula.operator} {_render_expression(formula.right)}"
    if isinstance(formula, Exists):
        from_clause, where_clause = _compile_scope(formula.body)
        sql = f"SELECT 1 FROM {from_clause}"
        if where_clause:
            sql += f" WHERE {where_clause}"
        return f"EXISTS ({sql})"
    if isinstance(formula, And):
        return "(" + " AND ".join(_render_formula(term) for term in formula.terms) + ")"
    if isinstance(formula, Or):
        return "(" + " OR ".join(_render_formula(term) for term in formula.terms) + ")"
    if isinstance(formula, Not):
        return f"NOT ({_render_formula(formula.term)})"
    return ""


def _render_expression(expression: Expression) -> str:
    if isinstance(expression, AttributeRef):
        return f"{expression.variable}.{_quote_identifier(expression.attribute)}"
    if isinstance(expression, Aggregate):
        distinct = "DISTINCT " if expression.distinct else ""
        return f"{expression.function}({distinct}{_render_expression(expression.expression)})"
    value = expression.value
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return str(value)


def _aggregate_alias(expression: Aggregate) -> str:
    inner = expression.expression
    if isinstance(inner, AttributeRef):
        safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", inner.attribute).strip("_")
        return f"{expression.function.lower()}_{safe_name or 'value'}"
    return f"{expression.function.lower()}_value"


def _quote_identifier(name: str) -> str:
    if name.replace("_", "").isalnum() and not name[0].isdigit():
        return name
    return '"' + name.replace('"', '""') + '"'
