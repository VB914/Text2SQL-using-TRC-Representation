from __future__ import annotations

import logging

from core.datasets import resolve_dataset_db_path
from core.prompt_builder import build_repair_prompt, build_trc_prompt, prompt_summary
from core.providers import get_provider
from core.schema_utils import load_schema
from core.sql_executor import execute_sql, sql_is_valid
from core.trc_repair import repair_trc
from core.trc_to_sql import trc_to_sql
from core.trc_validator import heuristic_repair_trc, validate_trc
from models import PipelineRequest, PipelineResponse, ValidationReport


LOGGER = logging.getLogger(__name__)


def run_trc_pipeline(request: PipelineRequest) -> PipelineResponse:
    db_path = _request_db_path(request)
    question = request.question.strip()
    if not question:
        return PipelineResponse(
            success=False,
            provider="none",
            question=request.question,
            db_path=str(db_path or ""),
            error="Question cannot be empty.",
        )

    try:
        schema = load_schema(str(db_path) if db_path else request.db_path)
        prompt = build_trc_prompt(question, schema, request.evidence)
        provider = get_provider()
        generated = provider.generate_trc(question, schema, prompt)
        trc = heuristic_repair_trc(generated.trc)
        validation = validate_trc(trc, schema)

        if not validation.valid:
            trc, validation = _repair_once(provider, question, schema, trc, validation)

        if not validation.valid:
            return PipelineResponse(
                success=False,
                provider=generated.provider,
                question=question,
                db_path=schema.db_path,
                prompt_summary=prompt_summary(),
                entities=generated.entities,
                schema_mappings=generated.schema_mappings,
                trc=trc,
                validation=validation,
                raw_output=generated.raw_output,
                reasoning={
                    "trc_generation": generated.reasoning,
                    "validation": "TRC validation failed after one repair pass.",
                },
                error=validation.summary,
            )

        sql = trc_to_sql(trc, schema)
        # Plan the statement before running it. This catches SQL that compiles from
        # valid TRC but cannot execute against this database, and reports it as a
        # generation failure rather than an unhandled error mid-execution.
        try:
            sql_valid = sql_is_valid(schema.db_path, sql)
        except Exception as exc:
            LOGGER.info("Generated SQL failed planning: %s", exc)
            return PipelineResponse(
                success=False,
                provider=generated.provider,
                question=question,
                db_path=schema.db_path,
                prompt_summary=prompt_summary(),
                entities=generated.entities,
                schema_mappings=generated.schema_mappings,
                trc=trc,
                sql=sql,
                sql_valid=False,
                validation=validation,
                raw_output=generated.raw_output,
                reasoning={
                    "trc_generation": generated.reasoning,
                    "validation": validation.summary,
                    "sql_generation": "The TRC compiled, but the resulting SQL could not be planned.",
                },
                error=f"Generated SQL is not executable: {exc}",
            )

        result = execute_sql(schema.db_path, sql)
        return PipelineResponse(
            success=True,
            sql_valid=sql_valid,
            provider=generated.provider,
            question=question,
            db_path=schema.db_path,
            prompt_summary=prompt_summary(),
            entities=generated.entities,
            schema_mappings=generated.schema_mappings,
            trc=trc,
            sql=sql,
            validation=validation,
            execution_result=result,
            raw_output=generated.raw_output,
            reasoning={
                "trc_generation": generated.reasoning,
                "validation": validation.summary,
                "sql_generation": "The validated TRC was deterministically compiled into SQLite SQL.",
                "execution": f"Query executed safely with a read-only SQLite cursor and returned {result.row_count} row(s).",
            },
        )
    except Exception as exc:
        LOGGER.exception("TRC pipeline failed.")
        return PipelineResponse(
            success=False,
            provider="pipeline",
            question=request.question,
            db_path=str(db_path or request.db_path or ""),
            error=str(exc),
        )


def _repair_once(provider, question, schema, trc: str, validation: ValidationReport) -> tuple[str, ValidationReport]:
    errors = [issue.message for issue in validation.issues if issue.level == "error"]

    # Schema-aware repair first: it fixes the common cases (wrong case, near-miss
    # table and column names) deterministically.
    schema_repaired = repair_trc(trc, schema, validation)
    if schema_repaired != trc:
        report = validate_trc(schema_repaired, schema)
        if _error_count(report) < _error_count(validation):
            report.repaired_trc = schema_repaired
            return schema_repaired, report

    repair_prompt = build_repair_prompt(question, schema, trc, errors)
    repaired = heuristic_repair_trc(provider.repair_trc(question, schema, trc, errors, repair_prompt))
    report = validate_trc(repaired, schema)
    # Never accept a rewrite that leaves the expression in a worse state.
    if _error_count(report) > _error_count(validation):
        return trc, validation
    report.repaired_trc = repaired if repaired != trc else None
    return repaired, report


def _error_count(report: ValidationReport) -> int:
    return sum(1 for issue in report.issues if issue.level == "error")


def _request_db_path(request: PipelineRequest) -> str | None:
    if request.db_path:
        return request.db_path
    if request.dataset and request.db_id:
        path = resolve_dataset_db_path(request.dataset, request.db_id, request.split or "dev")
        return str(path) if path else None
    return None
