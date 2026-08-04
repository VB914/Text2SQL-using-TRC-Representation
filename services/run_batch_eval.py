from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from core.evaluator import exact_match, execution_accuracy, gold_is_order_sensitive
from core.pipeline import run_trc_pipeline
from models import PipelineRequest


def _percent(count: int, total: int) -> float:
    return round(100.0 * count / total, 1) if total else 0.0


def evaluate_questions(
    questions: list[dict[str, Any]],
    db_path: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in questions:
        # Per-example database first: a single global path is wrong for Spider-style
        # sets where every question targets a different database.
        example_db = item.get("db_path") or db_path
        response = run_trc_pipeline(
            PipelineRequest(
                question=item["question"],
                db_path=example_db,
                gold_sql=item.get("gold_sql"),
            )
        )

        gold_sql = item.get("gold_sql")
        # The pipeline resolves the real path, so metrics never run against None.
        resolved_db = response.db_path or example_db
        executed = response.execution_result is not None

        accuracy: bool | None = None
        if gold_sql and response.sql and resolved_db:
            accuracy = execution_accuracy(
                resolved_db,
                response.sql,
                gold_sql,
                order_sensitive=gold_is_order_sensitive(gold_sql),
            )

        rows.append(
            {
                "question": item["question"],
                "db_path": resolved_db,
                "predicted_sql": response.sql,
                "gold_sql": gold_sql,
                "exact_match": exact_match(response.sql, gold_sql) if gold_sql else None,
                "execution_accuracy": accuracy,
                "trc": response.trc,
                "trc_valid": response.validation.valid if response.validation else None,
                "executed": executed,
                "success": response.success,
                "error": response.error,
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    trc_valid = sum(1 for row in rows if row["trc_valid"])
    sql_generated = sum(1 for row in rows if row["predicted_sql"])
    executed = sum(1 for row in rows if row["executed"])

    scored_em = [row for row in rows if row["exact_match"] is not None]
    scored_ex = [row for row in rows if row["execution_accuracy"] is not None]

    return {
        "total": total,
        "trc_valid": trc_valid,
        "trc_valid_pct": _percent(trc_valid, total),
        "sql_generated": sql_generated,
        "sql_generated_pct": _percent(sql_generated, total),
        "executed": executed,
        "execution_success_pct": _percent(executed, total),
        "exact_match_scored": len(scored_em),
        "exact_match_pct": _percent(sum(1 for row in scored_em if row["exact_match"]), len(scored_em)),
        "execution_accuracy_scored": len(scored_ex),
        "execution_accuracy_pct": _percent(
            sum(1 for row in scored_ex if row["execution_accuracy"]), len(scored_ex)
        ),
    }


def _write_outputs(out_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "per_example.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TRC pipeline over a question set and report metrics.")
    parser.add_argument("input", nargs="?", default="data/sample_questions.json")
    parser.add_argument("--db-path", default=None, help="Fallback database when an example does not name one.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", default=None, help="Directory for summary.json and per_example.csv.")
    parser.add_argument("--json", action="store_true", help="Print per-example rows instead of the summary table.")
    args = parser.parse_args()

    questions = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]

    rows = evaluate_questions(questions, args.db_path)
    summary = summarize(rows)

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        width = max(len(key) for key in summary)
        for key, value in summary.items():
            print(f"{key.ljust(width)} : {value}")

    if args.out_dir and rows:
        _write_outputs(Path(args.out_dir), rows, summary)
        print(f"\nWrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
