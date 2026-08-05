from __future__ import annotations

import re

from core.trc_parser import (
    Aggregate,
    And,
    AttributeRef,
    Between,
    Comparison,
    Exists,
    Expression,
    Formula,
    IsNull,
    Like,
    Membership,
    Not,
    Or,
    RelationPredicate,
    SetComprehension,
    Shaping,
    Star,
    TrcQuery,
    ValueList,
    expression_variables,
    flatten_and,
    parse_trc,
)
from models import SchemaResponse


def trc_to_sql(trc_text: str, schema: SchemaResponse | None = None) -> str:
    query = parse_trc(trc_text)
    from_clause, where_clause, having_clause = _compile_scope(query.formula)

    projections = ", ".join(_render_projection(item) for item in query.projections)
    select_keyword = "SELECT DISTINCT" if query.distinct else "SELECT"
    parts = [f"{select_keyword} {projections}", f"FROM {from_clause}"]
    if where_clause:
        parts.append(f"WHERE {where_clause}")

    group_keys = [
        _render_expression(item) for item in query.projections if not _contains_aggregate(item)
    ]
    # Grouping is required whenever an aggregate appears anywhere, including in a
    # restriction. Deriving it only from the projection list produces a HAVING
    # clause with no GROUP BY, which silently collapses the result to one row.
    # An aggregate used only for ranking still implies grouping: "the stadium with
    # the most concerts" projects stadium columns and orders by COUNT(*).
    orders_by_aggregate = bool(
        query.shaping and any(_contains_aggregate(key.expression) for key in query.shaping.order_by)
    )
    needs_group_by = (
        any(_contains_aggregate(item) for item in query.projections)
        or bool(having_clause)
        or orders_by_aggregate
    )
    if needs_group_by and group_keys:
        parts.append("GROUP BY " + ", ".join(group_keys))
    if having_clause:
        parts.append(f"HAVING {having_clause}")

    parts.extend(_render_shaping(query))
    return " ".join(parts) + ";"


def _render_shaping(query: TrcQuery) -> list[str]:
    shaping: Shaping | None = query.shaping
    if shaping is None:
        return []

    parts: list[str] = []
    if shaping.order_by:
        keys = []
        for key in shaping.order_by:
            rendered = _order_key_sql(key.expression, query)
            keys.append(f"{rendered} DESC" if key.descending else rendered)
        parts.append("ORDER BY " + ", ".join(keys))
    if shaping.limit is not None:
        parts.append(f"LIMIT {shaping.limit}")
        if shaping.offset is not None:
            parts.append(f"OFFSET {shaping.offset}")
    return parts


def _order_key_sql(expression: Expression, query: TrcQuery) -> str:
    """Prefer a projected aggregate's alias when ordering by that aggregate.

    Repeating an aggregate expression in ORDER BY is legal but fragile under
    GROUP BY, and the alias reads better in the generated SQL.
    """
    if isinstance(expression, Aggregate):
        for projection in query.projections:
            if projection == expression:
                return _aggregate_alias(expression)
    return _render_expression(expression)


def _contains_aggregate(node: object) -> bool:
    if isinstance(node, Aggregate):
        return True
    if isinstance(node, Comparison):
        return _contains_aggregate(node.left) or _contains_aggregate(node.right)
    if isinstance(node, Between):
        return any(
            _contains_aggregate(item) for item in (node.expression, node.lower, node.upper)
        )
    if isinstance(node, (Like, IsNull)):
        return _contains_aggregate(node.expression)
    if isinstance(node, Membership):
        return _contains_aggregate(node.expression)
    return False


def _render_projection(expression: Expression) -> str:
    rendered = _render_expression(expression)
    if isinstance(expression, Aggregate):
        return f"{rendered} AS {_aggregate_alias(expression)}"
    return rendered


def _compile_scope(formula: Formula) -> tuple[str, str, str]:
    terms = flatten_and(formula)
    relations = [term for term in terms if isinstance(term, RelationPredicate)]
    if not relations:
        raise ValueError("TRC has no relation predicates.")

    # An aggregate restriction is evaluated after grouping, so it can never act as
    # a join condition and must not be offered to the FROM builder.
    join_candidates = [
        term for term in terms if isinstance(term, Comparison) and not _contains_aggregate(term)
    ]
    from_clause, used = _from_clause(relations, join_candidates)

    where_terms: list[str] = []
    having_terms: list[str] = []
    for term in terms:
        if isinstance(term, RelationPredicate) or id(term) in used:
            continue
        rendered = _render_formula(term)
        if not rendered:
            continue
        if _contains_aggregate(term):
            having_terms.append(rendered)
        else:
            where_terms.append(rendered)

    return from_clause, " AND ".join(where_terms), " AND ".join(having_terms)


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
        operator = "<>" if formula.operator == "!=" else formula.operator
        return f"{_render_expression(formula.left)} {operator} {_render_expression(formula.right)}"
    if isinstance(formula, Like):
        keyword = "NOT LIKE" if formula.negated else "LIKE"
        return f"{_render_expression(formula.expression)} {keyword} {_render_expression(formula.pattern)}"
    if isinstance(formula, Between):
        keyword = "NOT BETWEEN" if formula.negated else "BETWEEN"
        return (
            f"{_render_expression(formula.expression)} {keyword} "
            f"{_render_expression(formula.lower)} AND {_render_expression(formula.upper)}"
        )
    if isinstance(formula, IsNull):
        keyword = "IS NOT NULL" if formula.negated else "IS NULL"
        return f"{_render_expression(formula.expression)} {keyword}"
    if isinstance(formula, Membership):
        keyword = "NOT IN" if formula.negated else "IN"
        return f"{_render_expression(formula.expression)} {keyword} {_render_collection(formula.collection)}"
    if isinstance(formula, Exists):
        from_clause, where_clause, having_clause = _compile_scope(formula.body)
        sql = f"SELECT 1 FROM {from_clause}"
        if where_clause:
            sql += f" WHERE {where_clause}"
        if having_clause:
            sql += f" HAVING {having_clause}"
        return f"EXISTS ({sql})"
    if isinstance(formula, And):
        return "(" + " AND ".join(_render_formula(term) for term in formula.terms) + ")"
    if isinstance(formula, Or):
        return "(" + " OR ".join(_render_formula(term) for term in formula.terms) + ")"
    if isinstance(formula, Not):
        return f"NOT ({_render_formula(formula.term)})"
    return ""


def _render_collection(collection: ValueList | SetComprehension) -> str:
    if isinstance(collection, ValueList):
        return "(" + ", ".join(_render_expression(item) for item in collection.items) + ")"

    from_clause, where_clause, having_clause = _compile_scope(collection.formula)
    sql = f"SELECT {_render_expression(collection.projection)} FROM {from_clause}"
    if where_clause:
        sql += f" WHERE {where_clause}"
    if having_clause:
        sql += f" HAVING {having_clause}"
    return f"({sql})"


def _render_expression(expression: Expression) -> str:
    if isinstance(expression, AttributeRef):
        return f"{expression.variable}.{_quote_identifier(expression.attribute)}"
    if isinstance(expression, Star):
        return "*"
    if isinstance(expression, Aggregate):
        distinct = "DISTINCT " if expression.distinct else ""
        return f"{expression.function}({distinct}{_render_expression(expression.expression)})"
    value = expression.value
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return str(value)


def _aggregate_alias(expression: Aggregate) -> str:
    inner = expression.expression
    if isinstance(inner, Star):
        return f"{expression.function.lower()}_star"
    if isinstance(inner, AttributeRef):
        safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", inner.attribute).strip("_")
        return f"{expression.function.lower()}_{safe_name or 'value'}"
    return f"{expression.function.lower()}_value"


def _quote_identifier(name: str) -> str:
    if name.replace("_", "").isalnum() and not name[0].isdigit():
        return name
    return '"' + name.replace('"', '""') + '"'
