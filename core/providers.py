from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache

from core.config import get_settings
from core.prompt_builder import load_few_shot_examples
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
        selected = tables[:2] if len(tables) > 1 else tables[:1]
        count_table, group_table = _count_group_tables(question, schema.tables, selected) if asks_count else (None, None)

        if asks_count and group_hint and count_table and group_table and count_table.name != group_table.name:
            selected = [count_table, group_table]
            relation_tables = _expand_join_tables([group_table, count_table], schema)
        else:
            relation_tables = _expand_join_tables(selected, schema)

        aliases = _aliases(relation_tables)
        predicates = [f"{table.name}({aliases[table.name]})" for table in relation_tables]
        predicates.extend(_join_predicates(relation_tables, aliases))
        predicates.extend(_filter_predicates(question, relation_tables, aliases))

        projections: list[str]
        mappings: list[str] = []
        if asks_count:
            counted = count_table or selected[0]
            count_col = _primary_key(counted) or counted.columns[0].name
            if group_hint and (group_table or len(selected) > 1):
                grouped = group_table or selected[1]
                group_col = _display_column(grouped) or grouped.columns[0].name
                projections = [
                    _attribute(aliases[grouped.name], group_col),
                    f"COUNT({_attribute(aliases[counted.name], count_col)})",
                ]
                mappings.extend([f"count target -> {counted.name}.{count_col}", f"grouping -> {grouped.name}.{group_col}"])
            else:
                projections = [f"COUNT({_attribute(aliases[counted.name], count_col)})"]
                mappings.append(f"count target -> {counted.name}.{count_col}")
        else:
            projections = []
            for table in selected[:2]:
                column = _display_column(table) or _primary_key(table) or table.columns[0].name
                projections.append(_attribute(aliases[table.name], column))
                mappings.append(f"projection -> {table.name}.{column}")
            if not projections:
                first = selected[0]
                projections.append(_attribute(aliases[first.name], first.columns[0].name))

        trc = "{ " + ", ".join(projections) + " | " + " AND ".join(predicates) + " }"
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


def _filter_predicates(question: str, tables: list[TableInfo], aliases: dict[str, str]) -> list[str]:
    filters = []
    for value in re.findall(r"['\"]([^'\"]+)['\"]", question):
        target = _best_text_column(question, tables)
        if target:
            table, column = target
            filters.append(f"{_attribute(aliases[table.name], column.name)} = '{value}'")

    number_match = re.search(r"\b(\d+(?:\.\d+)?)\b", question)
    if number_match:
        value = number_match.group(1)
        operator = ">"
        if re.search(r"\b(less|under|below|younger|fewer)\b", question, re.IGNORECASE):
            operator = "<"
        elif re.search(r"\b(equal|exactly|is|are)\b", question, re.IGNORECASE):
            operator = "="
        target = _best_numeric_column(question, tables)
        if target:
            table, column = target
            filters.append(f"{_attribute(aliases[table.name], column.name)} {operator} {value}")
    return filters


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
