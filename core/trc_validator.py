from __future__ import annotations

import re
from collections import defaultdict, deque

from core.schema_utils import column_lookup, table_lookup
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
    Star,
    TrcQuery,
    TrcSyntaxError,
    ValueList,
    expression_variables,
    flatten_and,
    parse_trc,
)
from models import SchemaResponse, ValidationIssue, ValidationReport


def validate_trc(trc: str, schema: SchemaResponse) -> ValidationReport:
    issues: list[ValidationIssue] = []
    text = trc.strip()
    # Bracket balance is not counted here: the parser already reports unbalanced
    # delimiters with an exact position, and naive counting misfires on brackets
    # that appear inside string literals.
    try:
        query = parse_trc(text)
    except TrcSyntaxError as exc:
        issues.append(ValidationIssue(level="error", message=str(exc), location="parser"))
        return ValidationReport(valid=False, issues=issues, parseable=False, summary=_summary(issues))

    tables = table_lookup(schema.tables)
    columns = column_lookup(schema.tables)
    bindings = _collect_bindings(query.formula, tables, issues, {})
    for projection in query.projections:
        _validate_expression(projection, bindings, columns, issues)
    _validate_formula(query.formula, bindings, tables, columns, issues)
    _check_join_consistency(query.formula, bindings, issues)
    _validate_shaping(query, bindings, columns, issues)
    _validate_grouping(query, issues)

    return ValidationReport(
        valid=not any(issue.level == "error" for issue in issues),
        issues=issues,
        parseable=True,
        summary=_summary(issues),
    )


def heuristic_repair_trc(trc: str) -> str:
    """Normalise surface noise around a TRC expression.

    This deliberately performs no structural repair. Appending parentheses to
    balance a count produces an expression that parses but means something the
    author never wrote, which is worse than a clean validation failure.
    Structural repair belongs in the schema-aware repair pass.
    """
    text = re.sub(r"^TRC\s*:\s*", "", trc.strip(), flags=re.IGNORECASE)
    text = text.replace("```", "").strip()
    if not text.startswith("{"):
        text = "{ " + text
    # A closing brace is only missing when the braces are genuinely unbalanced.
    # Testing for a trailing "}" is wrong, because a valid query may end with a
    # shaping clause such as "ORDER BY s.age" that sits outside the braces.
    if text.count("{") > text.count("}"):
        text = text + " }"
    return re.sub(r"\s+", " ", text).strip()


def _collect_bindings(
    formula: Formula,
    tables: dict[str, object],
    issues: list[ValidationIssue],
    outer: dict[str, str],
) -> dict[str, str]:
    bindings = dict(outer)
    for term in flatten_and(formula):
        if not isinstance(term, RelationPredicate):
            continue
        table = tables.get(term.relation.lower())
        if not table:
            issues.append(ValidationIssue(level="error", message=f"Invalid table '{term.relation}'.", location=term.relation))
            continue
        bindings[term.variable] = table.name
    return bindings


def _validate_formula(
    formula: Formula,
    bindings: dict[str, str],
    tables: dict[str, object],
    columns: dict[str, set[str]],
    issues: list[ValidationIssue],
) -> None:
    if isinstance(formula, Comparison):
        _validate_expression(formula.left, bindings, columns, issues)
        _validate_expression(formula.right, bindings, columns, issues)
    elif isinstance(formula, Like):
        _validate_expression(formula.expression, bindings, columns, issues)
    elif isinstance(formula, IsNull):
        _validate_expression(formula.expression, bindings, columns, issues)
    elif isinstance(formula, Between):
        for part in (formula.expression, formula.lower, formula.upper):
            _validate_expression(part, bindings, columns, issues)
    elif isinstance(formula, Membership):
        _validate_expression(formula.expression, bindings, columns, issues)
        _validate_collection(formula.collection, bindings, tables, columns, issues)
    elif isinstance(formula, Exists):
        nested = _collect_bindings(formula.body, tables, issues, bindings)
        if formula.variable not in nested:
            issues.append(
                ValidationIssue(level="error", message=f"EXISTS variable '{formula.variable}' is not bound.", location=formula.variable)
            )
        _validate_formula(formula.body, nested, tables, columns, issues)
    elif isinstance(formula, (And, Or)):
        for term in formula.terms:
            _validate_formula(term, bindings, tables, columns, issues)
    elif isinstance(formula, Not):
        _validate_formula(formula.term, bindings, tables, columns, issues)


def _validate_expression(
    expression: object,
    bindings: dict[str, str],
    columns: dict[str, set[str]],
    issues: list[ValidationIssue],
    inside_aggregate: bool = False,
) -> None:
    if isinstance(expression, Star):
        if not inside_aggregate:
            issues.append(
                ValidationIssue(
                    level="error",
                    message="'*' is only allowed as the argument of an aggregate, such as COUNT(*).",
                    location="projection",
                )
            )
    elif isinstance(expression, Aggregate):
        _validate_expression(expression.expression, bindings, columns, issues, inside_aggregate=True)
    elif isinstance(expression, AttributeRef):
        table = bindings.get(expression.variable)
        if not table:
            issues.append(ValidationIssue(level="error", message=f"Undefined tuple variable '{expression.variable}'.", location=expression.variable))
            return
        if expression.attribute not in columns.get(table, set()):
            issues.append(
                ValidationIssue(
                    level="error",
                    message=f"Invalid column '{expression.attribute}' on table '{table}'.",
                    location=f"{expression.variable}.{expression.attribute}",
                )
            )


def _validate_collection(
    collection: object,
    bindings: dict[str, str],
    tables: dict[str, object],
    columns: dict[str, set[str]],
    issues: list[ValidationIssue],
) -> None:
    if isinstance(collection, ValueList):
        if not collection.items:
            issues.append(
                ValidationIssue(level="error", message="IN requires at least one value.", location="membership")
            )
        elif len({type(item.value) is str for item in collection.items}) > 1:
            issues.append(
                ValidationIssue(
                    level="warning",
                    message="IN mixes text and numeric values, which may not compare as intended.",
                    location="membership",
                )
            )
        return

    if isinstance(collection, SetComprehension):
        # The inner set may reference outer variables, so start from the outer bindings.
        nested = _collect_bindings(collection.formula, tables, issues, bindings)
        _validate_expression(collection.projection, nested, columns, issues)
        _validate_formula(collection.formula, nested, tables, columns, issues)


def _contains_aggregate(node: object) -> bool:
    if isinstance(node, Aggregate):
        return True
    if isinstance(node, Comparison):
        return _contains_aggregate(node.left) or _contains_aggregate(node.right)
    if isinstance(node, Between):
        return any(_contains_aggregate(item) for item in (node.expression, node.lower, node.upper))
    if isinstance(node, (Like, IsNull, Membership)):
        return _contains_aggregate(node.expression)
    return False


def _validate_shaping(
    query: TrcQuery,
    bindings: dict[str, str],
    columns: dict[str, set[str]],
    issues: list[ValidationIssue],
) -> None:
    shaping = query.shaping
    if shaping is None:
        return

    for key in shaping.order_by:
        _validate_expression(key.expression, bindings, columns, issues)

    if shaping.limit is not None and shaping.limit < 0:
        issues.append(ValidationIssue(level="error", message="LIMIT must not be negative.", location="shaping"))
    if shaping.offset is not None:
        if shaping.offset < 0:
            issues.append(ValidationIssue(level="error", message="OFFSET must not be negative.", location="shaping"))
        if shaping.limit is None:
            issues.append(
                ValidationIssue(level="error", message="OFFSET requires a LIMIT.", location="shaping")
            )

    if query.distinct:
        projected = set(query.projections)
        for key in shaping.order_by:
            if key.expression not in projected:
                issues.append(
                    ValidationIssue(
                        level="warning",
                        message="Ordering a DISTINCT result by a column that is not projected is ambiguous.",
                        location="shaping",
                    )
                )


def _validate_grouping(query: TrcQuery, issues: list[ValidationIssue]) -> None:
    """Check that an aggregate query has something meaningful to group by."""
    having_terms = [term for term in flatten_and(query.formula) if _contains_aggregate(term)]
    aggregate_projections = [item for item in query.projections if _contains_aggregate(item)]
    if not having_terms and not aggregate_projections:
        return

    group_keys = [item for item in query.projections if not _contains_aggregate(item)]
    if having_terms and not aggregate_projections and not group_keys:
        issues.append(
            ValidationIssue(
                level="error",
                message="An aggregate restriction needs either a grouping column or an aggregate projection.",
                location="grouping",
            )
        )
        return

    if aggregate_projections and group_keys:
        keys = ", ".join(
            f"{item.variable}.{item.attribute}" for item in group_keys if isinstance(item, AttributeRef)
        )
        if keys:
            issues.append(
                ValidationIssue(level="info", message=f"Result will be grouped by {keys}.", location="grouping")
            )


def _check_join_consistency(formula: Formula, bindings: dict[str, str], issues: list[ValidationIssue]) -> None:
    relations = [term for term in flatten_and(formula) if isinstance(term, RelationPredicate)]
    if len(relations) < 2:
        return

    graph: dict[str, set[str]] = defaultdict(set)
    for term in flatten_and(formula):
        if not isinstance(term, Comparison):
            continue
        variables = expression_variables(term.left) | expression_variables(term.right)
        if len(variables) == 2:
            left, right = sorted(variables)
            graph[left].add(right)
            graph[right].add(left)

    visited = set()
    queue = deque([relations[0].variable])
    while queue:
        item = queue.popleft()
        if item in visited:
            continue
        visited.add(item)
        queue.extend(graph[item] - visited)

    missing = {relation.variable for relation in relations} - visited
    if missing:
        issues.append(
            ValidationIssue(
                level="warning",
                message=f"Join inconsistency: variables not connected by comparisons: {', '.join(sorted(missing))}.",
                location="joins",
            )
        )


def _summary(issues: list[ValidationIssue]) -> str:
    if not issues:
        return "TRC passed syntax, schema, and join consistency checks."
    return "; ".join(f"{issue.level.upper()}: {issue.message}" for issue in issues)
