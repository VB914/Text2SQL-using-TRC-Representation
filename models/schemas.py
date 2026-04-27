from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ColumnInfo(BaseModel):
    name: str
    data_type: str
    nullable: bool = True
    primary_key: bool = False


class ForeignKeyInfo(BaseModel):
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    inferred: bool = False


class TableInfo(BaseModel):
    name: str
    columns: list[ColumnInfo]
    foreign_keys: list[ForeignKeyInfo] = Field(default_factory=list)


class SchemaResponse(BaseModel):
    success: bool = True
    db_path: str
    tables: list[TableInfo]
    formatted_schema: str
    database_options: list[str] = Field(default_factory=list)
    error: str | None = None


class ValidationIssue(BaseModel):
    level: str
    message: str
    location: str | None = None


class ValidationReport(BaseModel):
    valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    repaired_trc: str | None = None
    parseable: bool = False
    summary: str = ""


class QueryResult(BaseModel):
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0


class PipelineRequest(BaseModel):
    question: str
    db_path: str | None = None
    gold_sql: str | None = None
    evidence: str | None = None
    dataset: str | None = None
    split: str | None = None
    db_id: str | None = None
    difficulty: str | None = None
    question_id: str | int | None = None


class PipelineResponse(BaseModel):
    success: bool
    provider: str
    question: str
    db_path: str
    prompt_summary: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    schema_mappings: list[str] = Field(default_factory=list)
    trc: str | None = None
    sql: str | None = None
    validation: ValidationReport | None = None
    execution_result: QueryResult | None = None
    raw_output: str | None = None
    reasoning: dict[str, str] = Field(default_factory=dict)
    error: str | None = None


class ExecuteRequest(BaseModel):
    sql: str
    db_path: str | None = None
    limit: int = 100


class ValidateTRCRequest(BaseModel):
    trc: str
    db_path: str | None = None
    question: str | None = None


class DatasetInfo(BaseModel):
    id: str
    name: str
    available: bool
    splits: list[str] = Field(default_factory=list)
    database_count: int = 0
    example_count: int = 0
    notes: str | None = None


class DatasetExample(BaseModel):
    dataset: str
    split: str
    index: int
    db_id: str
    question: str
    db_path: str | None = None
    gold_sql: str | None = None
    evidence: str | None = None
    difficulty: str | None = None
    question_id: str | int | None = None


class DatasetExamplesResponse(BaseModel):
    success: bool
    dataset: str
    split: str
    examples: list[DatasetExample] = Field(default_factory=list)
    total: int = 0
    error: str | None = None


class JsonlExportRequest(BaseModel):
    dataset: str = "spider"
    split: str = "train"
    output_path: str | None = None
    limit: int | None = None
    include_evidence: bool = True


class JsonlExportResponse(BaseModel):
    success: bool
    output_path: str
    rows_written: int
    error: str | None = None
