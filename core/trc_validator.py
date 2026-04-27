from __future__ import annotations

import re
from collections import defaultdict, deque

from core.schema_utils import column_lookup, table_lookup
from core.trc_parser import (
    Aggregate,
    And,
    AttributeRef,
    Comparison,
    Exists,
    Formula,
    Not,
    Or,
    RelationPredicate,
    TrcSyntaxError,
    expression_variables,
    flatten_and,
    parse_trc,
)
from models import SchemaResponse, ValidationIssue, ValidationReport


def validate_trc(trc: str, schema: SchemaResponse) -> ValidationReport:
    issues: list[ValidationIssue] = []
    text = trc.strip()
    if text.count("{") != text.count("}"):
        issues.append(ValidationIssue(level="error", message="Unbalanced curly braces.", location="syntax"))
    if text.count("(") != text.count(")"):
        issues.append(ValidationIssue(level="error", message="Unbalanced parentheses.", location="syntax"))

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

    return ValidationReport(
        valid=not any(issue.level == "error" for issue in issues),
        issues=issues,
        parseable=True,
        summary=_summary(issues),
    )


def heuristic_repair_trc(trc: str) -> str:
    text = re.sub(r"^TRC\s*:\s*", "", trc.strip(), flags=re.IGNORECASE)
    text = text.replace("```", "").strip()
    if not text.startswith("{"):
        text = "{ " + text
    if not text.endswith("}"):
        text = text + " }"
    if text.count("(") > text.count(")"):
        text += ")" * (text.count("(") - text.count(")"))
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
) -> None:
    if isinstance(expression, Aggregate):
        _validate_expression(expression.expression, bindings, columns, issues)
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
