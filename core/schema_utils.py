from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict
from pathlib import Path

from core.config import get_settings
from models import ColumnInfo, ForeignKeyInfo, SchemaResponse, TableInfo

LOGGER = logging.getLogger(__name__)
SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}


def resolve_db_path(db_path: str | None) -> Path:
    settings = get_settings()
    path = Path(db_path) if db_path else settings.sample_db_path
    if not path.is_absolute():
        path = settings.project_root / path
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {path}")
    if path.suffix.lower() not in SQLITE_SUFFIXES:
        raise ValueError("Only SQLite database files are supported.")
    return path.resolve()


def list_database_options() -> list[str]:
    settings = get_settings()
    candidates = [settings.sample_db_path]
    if settings.uploads_dir.exists():
        candidates.extend(path for path in settings.uploads_dir.rglob("*") if path.is_file())
    return [
        str(path.resolve())
        for path in candidates
        if path.exists() and path.suffix.lower() in SQLITE_SUFFIXES
    ]


def load_schema(db_path: str | None) -> SchemaResponse:
    path = resolve_db_path(db_path)
    LOGGER.info("Loading schema for %s", path)
    with sqlite3.connect(path) as connection:
        cursor = connection.cursor()
        tables = []
        for table_name in _table_names(cursor):
            tables.append(
                TableInfo(
                    name=table_name,
                    columns=_columns(cursor, table_name),
                    foreign_keys=_foreign_keys(cursor, table_name),
                )
            )

    inferred = _infer_foreign_keys(tables)
    by_table: dict[str, list[ForeignKeyInfo]] = defaultdict(list)
    for table in tables:
        for foreign_key in table.foreign_keys:
            by_table[foreign_key.source_table].append(foreign_key)
    existing = {
        (fk.source_table, fk.source_column, fk.target_table, fk.target_column)
        for fks in by_table.values()
        for fk in fks
    }
    for foreign_key in inferred:
        key = (
            foreign_key.source_table,
            foreign_key.source_column,
            foreign_key.target_table,
            foreign_key.target_column,
        )
        if key not in existing:
            by_table[foreign_key.source_table].append(foreign_key)

    enriched = [
        TableInfo(name=table.name, columns=table.columns, foreign_keys=by_table.get(table.name, []))
        for table in tables
    ]
    return SchemaResponse(
        db_path=str(path),
        tables=enriched,
        formatted_schema=format_schema_for_prompt(enriched),
        database_options=list_database_options(),
    )


def format_schema_for_prompt(tables: list[TableInfo]) -> str:
    lines = []
    for table in tables:
        columns = ", ".join(
            f"{column.name} {column.data_type}{' PRIMARY KEY' if column.primary_key else ''}"
            for column in table.columns
        )
        lines.append(f"Table {table.name}: {columns}")
        for foreign_key in table.foreign_keys:
            marker = "inferred" if foreign_key.inferred else "declared"
            lines.append(
                f"  ForeignKey {foreign_key.source_table}.{foreign_key.source_column} -> "
                f"{foreign_key.target_table}.{foreign_key.target_column} ({marker})"
            )
    return "\n".join(lines)


def singular(name: str) -> str:
    lower = name.lower()
    if lower.endswith("ies"):
        return name[:-3] + "y"
    if lower.endswith("s") and len(name) > 1:
        return name[:-1]
    return name


def table_lookup(tables: list[TableInfo]) -> dict[str, TableInfo]:
    lookup = {}
    for table in tables:
        lookup[table.name.lower()] = table
        lookup[singular(table.name).lower()] = table
    return lookup


def column_lookup(tables: list[TableInfo]) -> dict[str, set[str]]:
    return {table.name: {column.name for column in table.columns} for table in tables}


def _table_names(cursor: sqlite3.Cursor) -> list[str]:
    rows = cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [row[0] for row in rows]


def _columns(cursor: sqlite3.Cursor, table_name: str) -> list[ColumnInfo]:
    rows = cursor.execute(f"PRAGMA table_info('{table_name}')").fetchall()
    return [
        ColumnInfo(name=row[1], data_type=row[2] or "TEXT", nullable=not bool(row[3]), primary_key=bool(row[5]))
        for row in rows
    ]


def _foreign_keys(cursor: sqlite3.Cursor, table_name: str) -> list[ForeignKeyInfo]:
    rows = cursor.execute(f"PRAGMA foreign_key_list('{table_name}')").fetchall()
    return [
        ForeignKeyInfo(
            source_table=table_name,
            source_column=row[3],
            target_table=row[2],
            target_column=row[4],
        )
        for row in rows
    ]


def _infer_foreign_keys(tables: list[TableInfo]) -> list[ForeignKeyInfo]:
    inferred = []
    table_names = {table.name.lower(): table.name for table in tables}
    singular_names = {singular(table.name).lower(): table.name for table in tables}
    name_map = {**table_names, **singular_names}
    for table in tables:
        for column in table.columns:
            if column.primary_key or not column.name.lower().endswith("_id"):
                continue
            target_table = name_map.get(column.name[:-3].lower())
            if target_table and target_table != table.name:
                inferred.append(
                    ForeignKeyInfo(
                        source_table=table.name,
                        source_column=column.name,
                        target_table=target_table,
                        target_column="id",
                        inferred=True,
                    )
                )
    return inferred
