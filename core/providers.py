from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache

from core.config import get_settings
from core.prompt_builder import load_few_shot_examples
from core.question_hints import (
    OrderingHint,
    asks_for_distinct,
    detect_aggregates,
    detect_comparison,
    detect_ordering,
    superlative_column_words,
)
from core.schema_linker import (
    build_value_index,
    find_mentioned_columns,
    is_numeric_column,
    link_columns,
    link_values,
)
from core.schema_utils import singular
from core.trc_parser import And, Exists, Formula, Not, Or, RelationPredicate, parse_trc
from core.trc_to_sql import trc_to_sql
from models import SchemaResponse, TableInfo


LOGGER = logging.getLogger(__name__)


@dataclass
class ProviderResult:
    provider: str
    trc: str
    sql: str | None = None
    entities: list[str] = field(default_factory=list)
    schema_mappings: list[str] = field(default_factory=list)
    raw_output: str | None = None
    reasoning: str = ""


class RuleBasedProvider:
    name = "rule-based-fallback"

    def generate_trc(self, question: str, schema: SchemaResponse, prompt: str) -> ProviderResult:
        matched = self._match_few_shot(question, schema)
        if matched:
            return matched
        return self._heuristic_generation(question, schema)

    def repair_trc(self, question: str, schema: SchemaResponse, trc: str, errors: list[str], prompt: str) -> str:
        return trc.strip()

    def _match_few_shot(self, question: str, schema: SchemaResponse) -> ProviderResult | None:
        schema_tables = {table.name.lower() for table in schema.tables}
        normalized_question = _normalize(question)
        best: tuple[float, dict[str, object]] | None = None
        for example in load_few_shot_examples():
            trc = str(example.get("trc", "")).strip()
            if not trc:
                continue
            relation_names = {name.lower() for name in _trc_relations(trc)}
            if relation_names and not relation_names <= schema_tables:
                continue
            score = SequenceMatcher(None, normalized_question, _normalize(str(example.get("question", "")))).ratio()
            if best is None or score > best[0]:
                best = (score, example)
        if not best or best[0] < 0.74:
            return None

        example = best[1]
        trc = str(example["trc"])
        sql = str(example.get("sql") or "")
        return ProviderResult(
            provider=self.name,
            trc=trc,
            sql=sql,
            entities=[str(item) for item in example.get("entities", [])],
            schema_mappings=[str(item) for item in example.get("schema_mapping", [])],
            raw_output=f"Matched curated few-shot example with similarity {best[0]:.2f}.",
            reasoning="The question closely matches a curated few-shot pattern, so the fallback reused its TRC form.",
        )

    def _heuristic_generation(self, question: str, schema: SchemaResponse) -> ProviderResult:
        tables = _rank_tables(question, schema.tables)
        if not tables:
            raise ValueError("The selected database has no user tables.")

        asks_count = bool(re.search(r"\b(how many|count|number of|total)\b", question, re.IGNORECASE))
        group_hint = bool(re.search(r"\b(each|per|by|for every)\b", question, re.IGNORECASE))

        value_matches = _linked_values(question, schema)
        selected = _select_tables(question, schema, tables, value_matches)
        count_table, group_table = _count_group_tables(question, schema.tables, selected) if asks_count else (None, None)

        if asks_count and group_hint and count_table and group_table and count_table.name != group_table.name:
            selected = [count_table, group_table]
            relation_tables = _expand_join_tables([group_table, count_table], schema)
        else:
            relation_tables = _expand_join_tables(selected, schema)

        # The counted and grouped tables are chosen from the whole schema, so they are
        # not guaranteed to be among the joined relations. Without this the alias
        # lookup below raises KeyError and the generator dies on the question.
        relation_names = {table.name for table in relation_tables}
        if count_table and count_table.name not in relation_names:
            count_table = next((table for table in relation_tables if table.name == count_table.name), None) or (
                relation_tables[0] if relation_tables else None
            )
        if group_table and group_table.name not in relation_names:
            group_table = None
            group_hint = False

        aliases = _aliases(relation_tables)
        predicates = [f"{table.name}({aliases[table.name]})" for table in relation_tables]
        predicates.extend(_join_predicates(relation_tables, aliases))

        # Columns consumed by a filter are not also projected, so "countries where
        # singers are above age 20" selects the country and not the age.
        filtered_columns: set[tuple[str, str]] = set()
        predicates.extend(
            _filter_predicates(question, relation_tables, aliases, value_matches, filtered_columns)
        )

        mentioned = [
            (table, column)
            for table, column in find_mentioned_columns(question, relation_tables)
            if table.name in aliases and (table.name, column) not in filtered_columns
        ]
        aggregates = detect_aggregates(question)

        ordering = detect_ordering(question)
        # "the stadium with the most concerts" asks for the stadium, ranked by a
        # count. The count belongs in ORDER BY, not in the projection list.
        ranks_by_count = bool(
            asks_count and mentioned and ordering is not None and ordering.limit is not None
        )

        projections: list[str]
        mappings: list[str] = []
        if ranks_by_count:
            projections = [
                _attribute(aliases[table.name], column) for table, column in mentioned[:2]
            ]
            mappings.append("ranked by COUNT(*)")
        elif aggregates and mentioned:
            # "the average, minimum and maximum age" is three aggregates over one column.
            table, column = mentioned[0]
            attribute = _attribute(aliases[table.name], column)
            projections = [f"{function}({attribute})" for function in aggregates]
            mappings.append(f"aggregate {'/'.join(aggregates)} -> {table.name}.{column}")
        elif asks_count:
            counted = count_table or _first_in(relation_tables, selected)
            grouped = _grouping_target(mentioned, group_table, relation_tables, selected, aliases, group_hint)
            if grouped is not None:
                grouped_table, group_col = grouped
                projections = [
                    _attribute(aliases[grouped_table.name], group_col),
                    "COUNT(*)",
                ]
                mappings.append(f"grouping -> {grouped_table.name}.{group_col}")
            else:
                # Counting rows of a single relation is COUNT(*) in the gold queries.
                projections = ["COUNT(*)"]
                mappings.append(f"count target -> rows of {counted.name}")
        else:
            projections = []
            for table, column in mentioned[:4]:
                rendered = _attribute(aliases[table.name], column)
                if rendered not in projections:
                    projections.append(rendered)
                    mappings.append(f"projection -> {table.name}.{column}")
            if not projections:
                primary = _first_in(relation_tables, selected)
                column = _display_column(primary) or _primary_key(primary) or primary.columns[0].name
                projections.append(_attribute(aliases[primary.name], column))
                mappings.append(f"projection -> {primary.name}.{column}")

        distinct = "DISTINCT " if asks_for_distinct(question) and not asks_count and not aggregates else ""
        trc = "{ " + distinct + ", ".join(projections) + " | " + " AND ".join(predicates) + " }"

        if ranks_by_count:
            direction = " DESC" if ordering.descending else ""
            shaping = f" ORDER BY COUNT(*){direction} LIMIT {ordering.limit}"
        elif aggregates:
            # "the maximum age" is an aggregate, not a request to sort and take one row.
            shaping = ""
        else:
            shaping = _shaping_clause(question, relation_tables, aliases, projections, asks_count)
        if shaping:
            trc += shaping
            mappings.append(f"shaping -> {shaping.strip()}")

        sql = None
        try:
            sql = trc_to_sql(trc, schema)
        except Exception:
            LOGGER.debug("Could not compile heuristic TRC during provider stage.", exc_info=True)

        entities = [table.name for table in selected]
        return ProviderResult(
            provider=self.name,
            trc=trc,
            sql=sql,
            entities=entities,
            schema_mappings=mappings,
            raw_output="Generated by deterministic schema-aware fallback.",
            reasoning="The fallback ranked schema tables and columns from question tokens, added foreign-key joins, then emitted parseable TRC.",
        )


class TransformersProvider:
    name = "transformers"

    def __init__(self) -> None:
        settings = get_settings()
        try:
            from transformers import pipeline

            self.generator = pipeline("text2text-generation", model=settings.default_model_name)
        except Exception as exc:
            raise RuntimeError(f"Could not load model {settings.default_model_name}: {exc}") from exc

    def generate_trc(self, question: str, schema: SchemaResponse, prompt: str) -> ProviderResult:
        output = self.generator(prompt, max_new_tokens=256, do_sample=False)[0]["generated_text"]
        trc = _section(output, "TRC") or output.strip()
        return ProviderResult(
            provider=self.name,
            trc=trc,
            sql=_section(output, "SQL"),
            entities=_lines(_section(output, "ENTITIES")),
            schema_mappings=_lines(_section(output, "SCHEMA_MAPPING")),
            raw_output=output,
            reasoning="A local Hugging Face text-to-text model generated the structured sections from the prompt.",
        )

    def repair_trc(self, question: str, schema: SchemaResponse, trc: str, errors: list[str], prompt: str) -> str:
        output = self.generator(prompt, max_new_tokens=160, do_sample=False)[0]["generated_text"]
        return (_section(output, "TRC") or output).strip()


class HybridProvider:
    def __init__(self) -> None:
        self.rule_based = RuleBasedProvider()
        self.model = self._load_model_if_requested()

    def generate_trc(self, question: str, schema: SchemaResponse, prompt: str) -> ProviderResult:
        if self.model:
            try:
                return self.model.generate_trc(question, schema, prompt)
            except Exception as exc:
                LOGGER.warning("Model generation failed; falling back to rules: %s", exc)
        return self.rule_based.generate_trc(question, schema, prompt)

    def repair_trc(self, question: str, schema: SchemaResponse, trc: str, errors: list[str], prompt: str) -> str:
        if self.model:
            try:
                return self.model.repair_trc(question, schema, trc, errors, prompt)
            except Exception as exc:
                LOGGER.warning("Model repair failed; using heuristic TRC: %s", exc)
        return self.rule_based.repair_trc(question, schema, trc, errors, prompt)

    def _load_model_if_requested(self) -> TransformersProvider | None:
        provider = get_settings().default_provider.lower()
        if provider not in {"transformers", "hf", "huggingface"}:
            return None
        try:
            return TransformersProvider()
        except Exception as exc:
            LOGGER.warning("Local model unavailable; using rule-based fallback: %s", exc)
            return None


@lru_cache(maxsize=1)
def get_provider() -> HybridProvider:
    return HybridProvider()


def _linked_values(question: str, schema: SchemaResponse) -> list:
    """Values from the question that exist in the database, or an empty list."""
    try:
        index = build_value_index(schema.db_path)
    except Exception:
        LOGGER.debug("Value index unavailable; continuing without value linking.", exc_info=True)
        return []
    return link_values(question, index, schema.tables)


_NAME_GLUE = frozenset({"in", "of", "to", "and", "the", "by", "for"})


def _mentions_table(question: str, table: TableInfo) -> bool:
    """True only when the question names the whole table, not just part of it.

    Partial overlap is not enough: "how many singers" shares the token "singer"
    with ``singer_in_concert``, and admitting that bridge table adds a join that
    multiplies rows and corrupts the count.
    """
    if _is_bridge_table(table):
        return False
    tokens = _token_set(question)
    name_tokens = {token for token in _tokens(table.name) if token not in _NAME_GLUE}
    if not name_tokens:
        return False
    return all(
        token in tokens or _singular_token(token) in tokens for token in name_tokens
    )


def _select_tables(
    question: str,
    schema: SchemaResponse,
    ranked: list[TableInfo],
    value_matches: list,
) -> list[TableInfo]:
    """Choose the relations the query needs, based on evidence rather than rank alone.

    Previously the top two ranked tables were always taken, which forced a join into
    single-table questions. An unnecessary join changes the row multiset and quietly
    corrupts counts, so a second relation is only admitted when the question actually
    points at it.
    """
    by_name = {table.name: table for table in schema.tables}
    primary = ranked[0]
    selected = [primary]

    # A value matched in another table means that table has to be present.
    for match in value_matches:
        table = by_name.get(match.table)
        if table and table.name not in {item.name for item in selected}:
            selected.append(table)

    if len(selected) == 1:
        for candidate in ranked[1:]:
            if _mentions_table(question, candidate) and candidate.name != primary.name:
                selected.append(candidate)
                break

    return selected[:3]


def _grouping_target(
    mentioned: list[tuple[TableInfo, str]],
    group_table: TableInfo | None,
    relation_tables: list[TableInfo],
    selected: list[TableInfo],
    aliases: dict[str, str],
    group_hint: bool,
) -> tuple[TableInfo, str] | None:
    """The column to group by, preferring one the question actually names."""
    if not group_hint:
        return None
    if mentioned:
        return mentioned[0]
    if group_table and group_table.name in aliases:
        column = _display_column(group_table) or group_table.columns[0].name
        return group_table, column
    fallback = _first_in(relation_tables, selected[1:]) if len(selected) > 1 else None
    if fallback is not None:
        column = _display_column(fallback) or fallback.columns[0].name
        return fallback, column
    return None


def _first_in(available: list[TableInfo], preferred: list[TableInfo]) -> TableInfo:
    """Pick the first preferred table that is actually available, else the first available."""
    names = {table.name for table in available}
    for table in preferred:
        if table.name in names:
            return table
    return available[0]


def _rank_tables(question: str, tables: list[TableInfo]) -> list[TableInfo]:
    tokens = _token_set(question)
    scored = []
    for table in tables:
        table_tokens = _token_set(table.name) | {_normalize(singular(table.name))}
        column_tokens = {token for column in table.columns for token in _token_set(column.name)}
        score = len(tokens & table_tokens) * 3 + len(tokens & column_tokens)
        if _normalize(singular(table.name)) in tokens:
            score += 2
        if _is_bridge_table(table):
            score -= 2
        scored.append((score, table.name, table))
    scored.sort(key=lambda item: (-item[0], item[1]))
    positive = [table for score, _, table in scored if score > 0]
    return positive or [item[2] for item in scored[:1]]


def _count_group_tables(
    question: str,
    tables: list[TableInfo],
    ranked: list[TableInfo],
) -> tuple[TableInfo | None, TableInfo | None]:
    count_table = _table_from_phrase(_count_phrase(question), tables)
    group_table = _table_from_phrase(_group_phrase(question), tables)

    if not count_table and ranked:
        count_table = ranked[0]
    if not group_table:
        group_table = next((table for table in ranked if count_table and table.name != count_table.name), None)
    return count_table, group_table


def _count_phrase(question: str) -> str:
    patterns = [
        r"\bnumber of\s+(?:the\s+)?(.+?)(?:\s+(?:for|in|by|per|with|that|who|where)\b|[?.!,]|$)",
        r"\bcount of\s+(?:the\s+)?(.+?)(?:\s+(?:for|in|by|per|with|that|who|where)\b|[?.!,]|$)",
        r"\bhow many\s+(?:the\s+)?(.+?)(?:\s+(?:are|is|do|does|did|have|has|were|was)\b|[?.!,]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return question


def _group_phrase(question: str) -> str:
    patterns = [
        r"\bnames? of\s+(?:the\s+)?(.+?)(?:\s+and\b|\s+with\b|[?.!,]|$)",
        r"\bfor each\s+(.+?)(?:[?.!,]|$)",
        r"\bin each\s+(.+?)(?:[?.!,]|$)",
        r"\bper\s+(.+?)(?:[?.!,]|$)",
        r"\bby\s+(.+?)(?:[?.!,]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def _table_from_phrase(phrase: str, tables: list[TableInfo]) -> TableInfo | None:
    tokens = _token_set(phrase)
    if not tokens:
        return None

    best: tuple[int, str, TableInfo] | None = None
    normalized_phrase = _normalize(phrase)
    for table in tables:
        table_name = _normalize(table.name)
        table_singular = _normalize(singular(table.name))
        table_tokens = _token_set(table.name)
        score = len(tokens & table_tokens) * 4
        if table_name in normalized_phrase or table_singular in normalized_phrase:
            score += 8
        if _is_bridge_table(table):
            score -= 3
        candidate = (score, table.name, table)
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best[2] if best and best[0] > 0 else None


def _trc_relations(trc: str) -> set[str]:
    try:
        query = parse_trc(trc)
    except Exception:
        return set()
    names: set[str] = set()
    _collect_relation_names(query.formula, names)
    return names


def _collect_relation_names(formula: Formula, names: set[str]) -> None:
    if isinstance(formula, RelationPredicate):
        names.add(formula.relation)
    elif isinstance(formula, Exists):
        _collect_relation_names(formula.body, names)
    elif isinstance(formula, (And, Or)):
        for term in formula.terms:
            _collect_relation_names(term, names)
    elif isinstance(formula, Not):
        _collect_relation_names(formula.term, names)


def _expand_join_tables(selected: list[TableInfo], schema: SchemaResponse) -> list[TableInfo]:
    by_name = {table.name: table for table in schema.tables}
    if not selected:
        return []
    names = [selected[0].name]
    for table in selected[1:]:
        path = _shortest_path(names[0], table.name, schema)
        for name in path or [table.name]:
            if name not in names:
                names.append(name)
    return [by_name[name] for name in names if name in by_name]


def _shortest_path(start: str, end: str, schema: SchemaResponse) -> list[str] | None:
    if start == end:
        return [start]
    graph: dict[str, set[str]] = {table.name: set() for table in schema.tables}
    for table in schema.tables:
        for fk in table.foreign_keys:
            graph.setdefault(fk.source_table, set()).add(fk.target_table)
            graph.setdefault(fk.target_table, set()).add(fk.source_table)
    queue: list[list[str]] = [[start]]
    seen = {start}
    while queue:
        path = queue.pop(0)
        for next_name in graph.get(path[-1], set()):
            if next_name == end:
                return path + [next_name]
            if next_name not in seen:
                seen.add(next_name)
                queue.append(path + [next_name])
    return None


def _join_predicates(tables: list[TableInfo], aliases: dict[str, str]) -> list[str]:
    names = {table.name for table in tables}
    predicates = []
    seen = set()
    for table in tables:
        for fk in table.foreign_keys:
            if fk.target_table not in names:
                continue
            key = tuple(sorted([(fk.source_table, fk.source_column), (fk.target_table, fk.target_column)]))
            if key in seen:
                continue
            seen.add(key)
            predicates.append(
                f"{_attribute(aliases[fk.source_table], fk.source_column)} = "
                f"{_attribute(aliases[fk.target_table], fk.target_column)}"
            )
    return predicates


def _shaping_clause(
    question: str,
    tables: list[TableInfo],
    aliases: dict[str, str],
    projections: list[str],
    asks_count: bool,
) -> str:
    """Translate ordering intent into the TRC shaping clause, if any."""
    hint = detect_ordering(question)
    if hint is None:
        return ""

    key = _ordering_key(hint, tables, aliases, projections, asks_count, question)
    if not key:
        # A bare LIMIT without an ordering is rarely what the question meant.
        return f" LIMIT {hint.limit}" if hint.limit else ""

    clause = f" ORDER BY {key}" + (" DESC" if hint.descending else "")
    if hint.limit:
        clause += f" LIMIT {hint.limit}"
    return clause


def _ordering_key(
    hint: OrderingHint,
    tables: list[TableInfo],
    aliases: dict[str, str],
    projections: list[str],
    asks_count: bool,
    question: str = "",
) -> str:
    if hint.by_count:
        # "most common X" ranks groups by their size.
        return "COUNT(*)" if not asks_count else next(
            (item for item in projections if item.startswith("COUNT(")), "COUNT(*)"
        )

    if asks_count:
        aggregate = next((item for item in projections if item.startswith("COUNT(")), None)
        if aggregate:
            return aggregate

    if hint.measure_phrase:
        for table, column_name in find_mentioned_columns(hint.measure_phrase, tables):
            if table.name in aliases:
                return _attribute(aliases[table.name], column_name)

    # "youngest" names no column but implies one. This is checked before the loose
    # name-overlap fallback, which would happily rank "youngest singer" by Singer_ID.
    for word in superlative_column_words(question):
        for table in tables:
            if table.name not in aliases:
                continue
            for column in table.columns:
                if word in _normalize(column.name).split() and is_numeric_column(column.data_type):
                    return _attribute(aliases[table.name], column.name)

    if hint.measure_phrase:
        for table, column_name in link_columns(hint.measure_phrase, tables):
            if table.name in aliases:
                return _attribute(aliases[table.name], column_name)

    if hint.alphabetical:
        for table in tables:
            column = _display_column(table)
            if column and table.name in aliases:
                return _attribute(aliases[table.name], column)

    # Fall back to ordering by whatever the query already projects.
    return projections[0] if projections else ""


def _filter_predicates(
    question: str,
    tables: list[TableInfo],
    aliases: dict[str, str],
    value_matches: list,
    used_columns: set[tuple[str, str]] | None = None,
) -> list[str]:
    filters: list[str] = []
    used_columns = used_columns if used_columns is not None else set()

    # Values quoted in the question are unambiguous, so they are trusted first.
    for value in re.findall(r"['\"]([^'\"]+)['\"]", question):
        target = _best_text_column(question, tables)
        if target:
            table, column = target
            filters.append(f"{_attribute(aliases[table.name], column.name)} = '{_escape(value)}'")
            used_columns.add((table.name, column.name))

    for match in value_matches:
        if match.table not in aliases or (match.table, match.column) in used_columns:
            continue
        attribute = _attribute(aliases[match.table], match.column)
        if match.exact:
            filters.append(f"{attribute} = '{_escape(match.value)}'")
        else:
            filters.append(f"{attribute} LIKE '%{_escape(match.value)}%'")
        used_columns.add((match.table, match.column))

    filters.extend(_numeric_filters(question, tables, aliases, used_columns))
    return filters


def _numeric_filters(
    question: str,
    tables: list[TableInfo],
    aliases: dict[str, str],
    used_columns: set[tuple[str, str]],
) -> list[str]:
    hint = detect_comparison(question)
    if hint is None:
        return []

    # Prefer a numeric column the question actually names; falling back to the first
    # numeric column produced filters such as "Singer_ID > 20" for "age above 20".
    for table, column_name in find_mentioned_columns(question, tables):
        column = next((item for item in table.columns if item.name == column_name), None)
        if column and is_numeric_column(column.data_type) and (table.name, column.name) not in used_columns:
            used_columns.add((table.name, column.name))
            return [f"{_attribute(aliases[table.name], column.name)} {hint.operator} {hint.value}"]

    target = _best_numeric_column(question, tables)
    if not target:
        return []
    table, column = target
    if not is_numeric_column(column.data_type) or (table.name, column.name) in used_columns:
        return []
    used_columns.add((table.name, column.name))
    return [f"{_attribute(aliases[table.name], column.name)} {hint.operator} {hint.value}"]


def _escape(value: str) -> str:
    return value.replace("'", "''")


def _best_text_column(question: str, tables: list[TableInfo]):
    tokens = _token_set(question)
    for table in tables:
        for column in table.columns:
            if column.name.lower() in {"name", "title"} or tokens & _token_set(column.name):
                return table, column
    return None


def _best_numeric_column(question: str, tables: list[TableInfo]):
    tokens = _token_set(question)
    numeric_markers = ("INT", "REAL", "NUM", "DEC", "FLOAT", "DOUBLE")
    for table in tables:
        for column in table.columns:
            column_tokens = _token_set(column.name)
            if tokens & column_tokens and any(marker in column.data_type.upper() for marker in numeric_markers):
                return table, column
    for table in tables:
        for column in table.columns:
            if not column.primary_key and any(marker in column.data_type.upper() for marker in numeric_markers):
                return table, column
    return None


def _display_column(table: TableInfo) -> str | None:
    priority = ["name", "title", "department_name", "course_name"]
    columns = {column.name.lower(): column.name for column in table.columns}
    for name in priority:
        if name in columns:
            return columns[name]
    for column in table.columns:
        if "TEXT" in column.data_type.upper() or "CHAR" in column.data_type.upper():
            return column.name
    return None


def _primary_key(table: TableInfo) -> str | None:
    for column in table.columns:
        if column.primary_key:
            return column.name
    for column in table.columns:
        if column.name.lower() == "id":
            return column.name
    return None


def _aliases(tables: list[TableInfo]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    used: set[str] = set()
    for table in tables:
        base = re.sub(r"[^A-Za-z]", "", table.name)[:1].lower() or "t"
        alias = base
        counter = 2
        while alias in used:
            alias = f"{base}{counter}"
            counter += 1
        aliases[table.name] = alias
        used.add(alias)
    return aliases


def _attribute(alias: str, column: str) -> str:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", column):
        return f"{alias}.{column}"
    return f'{alias}."{column}"'


def _section(text: str, name: str) -> str | None:
    pattern = re.compile(rf"{name}\s*:\s*(.*?)(?=\n[A-Z_]+\s*:|\Z)", re.IGNORECASE | re.DOTALL)
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def _lines(value: str | None) -> list[str]:
    if not value:
        return []
    return [line.strip("-* \t") for line in value.splitlines() if line.strip()]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))


def _token_set(text: str) -> set[str]:
    tokens = set(_tokens(text))
    expanded = set(tokens)
    for token in tokens:
        expanded.add(_singular_token(token))
    return expanded


def _singular_token(token: str) -> str:
    if token == "people":
        return "person"
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def _is_bridge_table(table: TableInfo) -> bool:
    if len(table.foreign_keys) >= 2:
        return True
    column_names = [column.name.lower() for column in table.columns]
    id_like = sum(1 for name in column_names if name.endswith("_id") or name == "id")
    return len(column_names) > 1 and id_like == len(column_names)


def _normalize(text: str) -> str:
    return " ".join(_tokens(text))
