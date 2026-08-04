from __future__ import annotations

import re
from dataclasses import dataclass


class TrcSyntaxError(ValueError):
    pass


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    position: int
    raw: str = ""


@dataclass(frozen=True)
class AttributeRef:
    variable: str
    attribute: str


@dataclass(frozen=True)
class Literal:
    value: str | int | float


@dataclass(frozen=True)
class Star:
    """The ``*`` argument of ``COUNT(*)``; valid only inside an aggregate."""


@dataclass(frozen=True)
class Aggregate:
    function: str
    expression: "Expression"
    distinct: bool = False


Expression = AttributeRef | Literal | Aggregate | Star


@dataclass(frozen=True)
class RelationPredicate:
    relation: str
    variable: str


@dataclass(frozen=True)
class Comparison:
    left: Expression
    operator: str
    right: Expression


@dataclass(frozen=True)
class Exists:
    variable: str
    body: "Formula"


@dataclass(frozen=True)
class And:
    terms: list["Formula"]


@dataclass(frozen=True)
class Or:
    terms: list["Formula"]


@dataclass(frozen=True)
class Not:
    term: "Formula"


Formula = RelationPredicate | Comparison | Exists | And | Or | Not


@dataclass(frozen=True)
class TrcQuery:
    projections: list[Expression]
    formula: Formula


TOKEN_PATTERN = re.compile(
    r"""
    (?P<SPACE>\s+)
    |(?P<STRING>'[^']*'|"[^"]*")
    |(?P<NUMBER>\d+(?:\.\d+)?)
    |(?P<OP><=|>=|!=|=|<|>)
    |(?P<LBRACE>\{)
    |(?P<RBRACE>\})
    |(?P<LPAREN>\()
    |(?P<RPAREN>\))
    |(?P<STAR>\*)
    |(?P<COMMA>,)
    |(?P<PIPE>\|)
    |(?P<DOT>\.)
    |(?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)
    """,
    re.VERBOSE,
)
KEYWORDS = {"AND", "OR", "NOT", "EXISTS", "DISTINCT", "COUNT", "SUM", "AVG", "MIN", "MAX"}
AGGREGATES = {"COUNT", "SUM", "AVG", "MIN", "MAX"}


def tokenize(text: str) -> list[Token]:
    tokens = []
    position = 0
    while position < len(text):
        match = TOKEN_PATTERN.match(text, position)
        if not match:
            raise TrcSyntaxError(f"Unexpected token near position {position}: {text[position:position + 20]!r}")
        kind = match.lastgroup or ""
        value = match.group()
        if kind != "SPACE":
            if kind == "IDENT" and value.upper() in KEYWORDS:
                # ``value`` is upper-cased so keyword matching stays case-insensitive,
                # while ``raw`` preserves the original spelling for identifier positions
                # (a column may legitimately be named "count" or "order").
                tokens.append(Token("KEYWORD", value.upper(), position, value))
            else:
                tokens.append(Token(kind, value, position, value))
        position = match.end()
    tokens.append(Token("EOF", "", position, ""))
    return tokens


class TrcParser:
    def __init__(self, text: str):
        self.tokens = tokenize(text)
        self.index = 0

    def peek(self, offset: int = 0) -> Token:
        return self.tokens[self.index + offset]

    def match(self, kind: str, value: str | None = None) -> Token | None:
        token = self.peek()
        if token.kind != kind or (value is not None and token.value != value):
            return None
        self.index += 1
        return token

    def expect(self, kind: str, value: str | None = None) -> Token:
        token = self.peek()
        if token.kind != kind or (value is not None and token.value != value):
            expected = value or kind
            raise TrcSyntaxError(f"Expected {expected!r} at position {token.position}, found {token.value!r}")
        self.index += 1
        return token

    def parse(self) -> TrcQuery:
        self.expect("LBRACE")
        projections = self.parse_projection_list()
        self.expect("PIPE")
        formula = self.parse_formula()
        self.expect("RBRACE")
        self.expect("EOF")
        return TrcQuery(projections, formula)

    def parse_projection_list(self) -> list[Expression]:
        projections = [self.parse_expression()]
        while self.match("COMMA"):
            projections.append(self.parse_expression())
        return projections

    def parse_formula(self) -> Formula:
        return self.parse_or()

    def parse_or(self) -> Formula:
        terms = [self.parse_and()]
        while self.match("KEYWORD", "OR"):
            terms.append(self.parse_and())
        return terms[0] if len(terms) == 1 else Or(terms)

    def parse_and(self) -> Formula:
        terms = [self.parse_unary()]
        while self.match("KEYWORD", "AND"):
            terms.append(self.parse_unary())
        return terms[0] if len(terms) == 1 else And(terms)

    def parse_unary(self) -> Formula:
        if self.match("KEYWORD", "NOT"):
            return Not(self.parse_unary())
        if self.match("KEYWORD", "EXISTS"):
            variable = self.expect("IDENT").value
            self.expect("LPAREN")
            body = self.parse_formula()
            self.expect("RPAREN")
            return Exists(variable, body)
        if self.match("LPAREN"):
            formula = self.parse_formula()
            self.expect("RPAREN")
            return formula
        return self.parse_predicate()

    def parse_predicate(self) -> Formula:
        if self._looks_like_relation():
            relation = self.parse_name()
            self.expect("LPAREN")
            variable = self.expect("IDENT").value
            self.expect("RPAREN")
            return RelationPredicate(relation, variable)
        left = self.parse_expression()
        operator = self.expect("OP").value
        right = self.parse_expression()
        return Comparison(left, operator, right)

    def parse_expression(self) -> Expression:
        token = self.peek()
        if token.kind == "KEYWORD" and token.value in AGGREGATES:
            function = self.expect("KEYWORD").value
            self.expect("LPAREN")
            distinct = bool(self.match("KEYWORD", "DISTINCT"))
            if self.match("STAR"):
                expression: Expression = Star()
            else:
                expression = self.parse_expression()
            self.expect("RPAREN")
            return Aggregate(function, expression, distinct)
        if token.kind == "STRING":
            return Literal(self.expect("STRING").value[1:-1])
        if token.kind == "NUMBER":
            raw = self.expect("NUMBER").value
            return Literal(float(raw) if "." in raw else int(raw))
        if token.kind == "IDENT":
            variable = self.expect("IDENT").value
            self.expect("DOT")
            attribute = self.parse_name()
            return AttributeRef(variable, attribute)
        raise TrcSyntaxError(f"Expected expression at position {token.position}, found {token.value!r}")

    def parse_name(self) -> str:
        token = self.peek()
        if token.kind == "IDENT":
            return self.expect("IDENT").value
        if token.kind == "STRING":
            return self.expect("STRING").value[1:-1]
        if token.kind == "KEYWORD":
            # Identifier positions are unambiguous, so a table or column named after a
            # keyword (Spider ships a "count" column) is accepted with its original spelling.
            self.index += 1
            return token.raw or token.value
        raise TrcSyntaxError(f"Expected identifier at position {token.position}, found {token.value!r}")

    def _looks_like_relation(self) -> bool:
        return (
            self.peek().kind in {"IDENT", "STRING"}
            and self.peek(1).kind == "LPAREN"
            and self.peek(2).kind == "IDENT"
            and self.peek(3).kind == "RPAREN"
        )


def parse_trc(text: str) -> TrcQuery:
    return TrcParser(text).parse()


def flatten_and(formula: Formula) -> list[Formula]:
    if isinstance(formula, And):
        terms = []
        for term in formula.terms:
            terms.extend(flatten_and(term))
        return terms
    return [formula]


def expression_variables(expression: Expression) -> set[str]:
    if isinstance(expression, AttributeRef):
        return {expression.variable}
    if isinstance(expression, Aggregate):
        return expression_variables(expression.expression)
    return set()
