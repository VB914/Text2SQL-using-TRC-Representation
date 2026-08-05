from __future__ import annotations

import re
from dataclasses import dataclass, field


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


@dataclass(frozen=True)
class Like:
    expression: Expression
    pattern: Literal
    negated: bool = False


@dataclass(frozen=True)
class ValueList:
    items: list[Literal]


@dataclass(frozen=True)
class SetComprehension:
    """A set-builder used as the right side of IN, e.g. ``{ y.name | authors(y) }``.

    This is the most calculus-native way to express membership and compiles to a
    correlated ``IN (SELECT ...)`` subquery.
    """

    projection: Expression
    formula: "Formula"


@dataclass(frozen=True)
class Membership:
    expression: Expression
    collection: "ValueList | SetComprehension"
    negated: bool = False


@dataclass(frozen=True)
class Between:
    expression: Expression
    lower: Expression
    upper: Expression
    negated: bool = False


@dataclass(frozen=True)
class IsNull:
    expression: Expression
    negated: bool = False


Formula = (
    RelationPredicate | Comparison | Exists | And | Or | Not | Like | Membership | Between | IsNull
)


@dataclass(frozen=True)
class OrderKey:
    expression: Expression
    descending: bool = False


@dataclass(frozen=True)
class Shaping:
    """How the answer set is presented.

    Ordering and truncation are not part of classical tuple relational calculus,
    because ``{ t | phi(t) }`` denotes an unordered set. They are modelled here as
    operators applied to that set from the outside, which keeps the calculus itself
    unchanged while still supporting execution-oriented queries.
    """

    order_by: list[OrderKey] = field(default_factory=list)
    limit: int | None = None
    offset: int | None = None


@dataclass(frozen=True)
class TrcQuery:
    projections: list[Expression]
    formula: Formula
    distinct: bool = False
    shaping: Shaping | None = None


TOKEN_PATTERN = re.compile(
    r"""
    (?P<SPACE>\s+)
    |(?P<STRING>'[^']*'|"[^"]*")
    |(?P<NUMBER>\d+(?:\.\d+)?)
    |(?P<OP><=|>=|!=|<>|=|<|>)
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
KEYWORDS = {
    "AND",
    "OR",
    "NOT",
    "EXISTS",
    "DISTINCT",
    "COUNT",
    "SUM",
    "AVG",
    "MIN",
    "MAX",
    "LIKE",
    "IN",
    "BETWEEN",
    "IS",
    "NULL",
    "ORDER",
    "BY",
    "ASC",
    "DESC",
    "LIMIT",
    "OFFSET",
}
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
        distinct = bool(self.match("KEYWORD", "DISTINCT"))
        projections = self.parse_projection_list()
        self.expect("PIPE")
        formula = self.parse_formula()
        self.expect("RBRACE")
        # Ordering and truncation are applied to the completed set, so they are
        # parsed after the closing brace rather than inside the calculus.
        shaping = self.parse_shaping()
        self.expect("EOF")
        return TrcQuery(projections, formula, distinct=distinct, shaping=shaping)

    def parse_projection_list(self) -> list[Expression]:
        projections = [self.parse_expression()]
        while self.match("COMMA"):
            projections.append(self.parse_expression())
        return projections

    def parse_shaping(self) -> Shaping | None:
        order_by: list[OrderKey] = []
        limit: int | None = None
        offset: int | None = None

        if self.match("KEYWORD", "ORDER"):
            self.expect("KEYWORD", "BY")
            order_by.append(self.parse_order_key())
            while self.match("COMMA"):
                order_by.append(self.parse_order_key())

        if self.match("KEYWORD", "LIMIT"):
            limit = self._parse_non_negative_integer("LIMIT")
            if self.match("KEYWORD", "OFFSET"):
                offset = self._parse_non_negative_integer("OFFSET")

        if not order_by and limit is None:
            return None
        return Shaping(order_by=order_by, limit=limit, offset=offset)

    def parse_order_key(self) -> OrderKey:
        expression = self.parse_expression()
        descending = False
        if self.match("KEYWORD", "DESC"):
            descending = True
        else:
            self.match("KEYWORD", "ASC")
        return OrderKey(expression, descending)

    def _parse_non_negative_integer(self, clause: str) -> int:
        token = self.expect("NUMBER")
        if "." in token.value:
            raise TrcSyntaxError(f"{clause} requires a whole number, found {token.value!r}.")
        return int(token.value)

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
        negated = bool(self.match("KEYWORD", "NOT"))

        if self.match("KEYWORD", "LIKE"):
            return Like(left, self._parse_string_literal("LIKE"), negated)
        if self.match("KEYWORD", "IN"):
            return Membership(left, self.parse_collection(), negated)
        if self.match("KEYWORD", "BETWEEN"):
            lower = self.parse_expression()
            # The AND separating the bounds belongs to BETWEEN and must be consumed
            # here; otherwise parse_and would treat the upper bound as a new conjunct.
            self.expect("KEYWORD", "AND")
            upper = self.parse_expression()
            return Between(left, lower, upper, negated)
        if self.match("KEYWORD", "IS"):
            is_negated = bool(self.match("KEYWORD", "NOT"))
            self.expect("KEYWORD", "NULL")
            return IsNull(left, is_negated)

        if negated:
            token = self.peek()
            raise TrcSyntaxError(
                f"Expected LIKE, IN or BETWEEN after NOT at position {token.position}, found {token.value!r}"
            )
        operator = self.expect("OP").value
        right = self.parse_expression()
        return Comparison(left, operator, right)

    def parse_collection(self) -> ValueList | SetComprehension:
        if self.match("LBRACE"):
            projection = self.parse_expression()
            self.expect("PIPE")
            formula = self.parse_formula()
            self.expect("RBRACE")
            return SetComprehension(projection, formula)

        self.expect("LPAREN")
        items = [self._parse_literal("IN")]
        while self.match("COMMA"):
            items.append(self._parse_literal("IN"))
        self.expect("RPAREN")
        return ValueList(items)

    def _parse_literal(self, clause: str) -> Literal:
        token = self.peek()
        if token.kind == "STRING":
            return Literal(self.expect("STRING").value[1:-1])
        if token.kind == "NUMBER":
            raw = self.expect("NUMBER").value
            return Literal(float(raw) if "." in raw else int(raw))
        raise TrcSyntaxError(f"{clause} expects literal values, found {token.value!r} at {token.position}.")

    def _parse_string_literal(self, clause: str) -> Literal:
        token = self.peek()
        if token.kind != "STRING":
            raise TrcSyntaxError(f"{clause} expects a string pattern, found {token.value!r} at {token.position}.")
        return Literal(self.expect("STRING").value[1:-1])

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
