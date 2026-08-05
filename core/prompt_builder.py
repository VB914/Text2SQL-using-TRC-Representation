from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.config import get_settings
from core.datasets import extract_spider_few_shots
from models import SchemaResponse


PIPELINE_STEPS = [
    "Step 1: Identify entities in the question.",
    "Step 2: Map entities to tables, columns, and foreign keys in the schema.",
    "Step 3: Generate a tuple relational calculus expression.",
    "Step 4: Validate TRC syntax and schema references.",
    "Step 5: Convert the validated TRC to SQLite SQL.",
]


def build_trc_prompt(question: str, schema: SchemaResponse, evidence: str | None = None) -> str:
    system_prompt = _read_text(get_settings().system_prompt_path)
    examples = _render_examples(load_few_shot_examples())
    evidence_text = f"\nEvidence:\n{evidence}\n" if evidence else ""
    return f"""{system_prompt}

You are generating an intermediate TRC representation before SQL.

Required reasoning order:
{chr(10).join(PIPELINE_STEPS)}

TRC grammar used in this project:
- Query: {{ projection_list | formula }} with an optional shaping clause after the closing brace
- Relation predicate: table_alias_name(tuple_variable), for example students(s)
- Attribute reference: variable.attribute, for example s.name
- Quoted attributes are allowed for complex schema names: f."County Name"
- Connectives: AND, OR, NOT, EXISTS x (...)
- Comparisons: =, !=, <, <=, >, >=
- Aggregates: COUNT(*), COUNT(x.attribute), SUM(...), AVG(...), MIN(...), MAX(...)
- Duplicate removal: {{ DISTINCT s.name | students(s) }}
- Pattern match: s.name LIKE 'A%' and s.name NOT LIKE 'A%'
- Membership against a value list: s.dept IN ('CS', 'EE')
- Membership against a set: s.id IN {{ e.student_id | enrollments(e) }}
- Range: s.age BETWEEN 18 AND 25
- Null tests: s.email IS NULL, s.email IS NOT NULL
- Ordering and truncation are written AFTER the closing brace, because the braces
  denote an unordered set:
  {{ s.name | students(s) }} ORDER BY s.gpa DESC LIMIT 5
- Do not write HAVING. A condition that contains an aggregate is automatically
  applied after grouping, so write it as an ordinary conjunct:
  {{ d.name | departments(d) AND students(s) AND s.department_id = d.id AND COUNT(s.id) > 5 }}

Few-shot examples:
{examples}

Current schema:
{schema.formatted_schema}
{evidence_text}
Question:
{question}

Return exactly these sections:
ENTITIES:
SCHEMA_MAPPING:
TRC:
SQL:
"""


def build_repair_prompt(question: str, schema: SchemaResponse, trc: str, errors: list[str]) -> str:
    return f"""Repair the TRC expression so it validates against the schema.

Question:
{question}

Schema:
{schema.formatted_schema}

Invalid TRC:
{trc}

Validation errors:
{chr(10).join(errors)}

Return only the corrected TRC expression.
"""


def prompt_summary() -> list[str]:
    return PIPELINE_STEPS.copy()


@lru_cache(maxsize=1)
def load_few_shot_examples() -> list[dict[str, Any]]:
    settings = get_settings()
    examples = _load_json(settings.few_shot_examples_path)
    if settings.auto_spider_few_shots:
        examples.extend(extract_spider_few_shots(settings.spider_few_shot_count))
    return examples


def _render_examples(examples: list[dict[str, Any]]) -> str:
    blocks = []
    for index, example in enumerate(examples, start=1):
        block = [
            f"Example {index}",
            f"Question: {example.get('question', '')}",
            "Schema:",
            example.get("schema", ""),
        ]
        if example.get("entities"):
            block.append("Entities: " + ", ".join(example["entities"]))
        if example.get("schema_mapping"):
            block.append("Schema mapping: " + "; ".join(example["schema_mapping"]))
        if example.get("trc"):
            block.append("TRC: " + example["trc"])
        if example.get("sql"):
            block.append("SQL: " + example["sql"])
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def _load_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, list) else []


def _read_text(path: Path) -> str:
    if not path.exists():
        return "You are a careful Text-to-SQL research assistant."
    return path.read_text(encoding="utf-8").strip()
