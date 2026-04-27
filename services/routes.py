from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from core.config import get_settings
from core.datasets import dataset_catalog, export_finetuning_jsonl, list_dataset_databases, list_dataset_examples
from core.pipeline import run_trc_pipeline
from core.schema_utils import list_database_options, load_schema
from core.sql_executor import execute_sql
from core.trc_validator import validate_trc
from models import (
    DatasetExamplesResponse,
    ExecuteRequest,
    JsonlExportRequest,
    PipelineRequest,
    ValidateTRCRequest,
)


router = APIRouter()


@router.get("/health")
def health() -> dict[str, object]:
    return {"success": True, "status": "ok"}


@router.get("/databases")
def databases() -> dict[str, object]:
    return {"success": True, "databases": list_database_options()}


@router.post("/databases/upload")
def upload_database(file: UploadFile = File(...)) -> dict[str, object]:
    settings = get_settings()
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".db", ".sqlite", ".sqlite3"}:
        return {"success": False, "error": "Only SQLite database files are supported."}

    target = settings.uploads_dir / Path(file.filename or "uploaded.sqlite").name
    with target.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    return {"success": True, "db_path": str(target.resolve())}


@router.get("/schemas")
def schema(db_path: str | None = None):
    try:
        return load_schema(db_path)
    except Exception as exc:
        return {"success": False, "error": str(exc), "tables": [], "formatted_schema": "", "db_path": db_path or ""}


@router.get("/datasets")
def datasets() -> dict[str, object]:
    return {"success": True, "datasets": [item.model_dump() for item in dataset_catalog()]}


@router.get("/datasets/databases")
def dataset_databases(dataset: str = "sample", split: str = "demo") -> dict[str, object]:
    try:
        return {"success": True, "databases": list_dataset_databases(dataset, split)}
    except Exception as exc:
        return {"success": False, "error": str(exc), "databases": []}


@router.get("/datasets/examples")
def dataset_examples(
    dataset: str = "sample",
    split: str = "demo",
    limit: int = 30,
    offset: int = 0,
    db_id: str | None = None,
    search: str | None = None,
) -> DatasetExamplesResponse:
    try:
        examples, total = list_dataset_examples(dataset, split, limit, offset, db_id, search)
        return DatasetExamplesResponse(success=True, dataset=dataset, split=split, examples=examples, total=total)
    except Exception as exc:
        return DatasetExamplesResponse(success=False, dataset=dataset, split=split, error=str(exc))


@router.post("/datasets/export-jsonl")
def export_jsonl(request: JsonlExportRequest):
    return export_finetuning_jsonl(request)


@router.post("/generate")
@router.post("/generate/trc")
def generate(request: PipelineRequest):
    return run_trc_pipeline(request)


@router.post("/validate/trc")
def validate(request: ValidateTRCRequest):
    schema_info = load_schema(request.db_path)
    return validate_trc(request.trc, schema_info)


@router.post("/execute")
def execute(request: ExecuteRequest):
    try:
        return {"success": True, "result": execute_sql(request.db_path, request.sql, request.limit)}
    except Exception as exc:
        return {"success": False, "error": str(exc)}
