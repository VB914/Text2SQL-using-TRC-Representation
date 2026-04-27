from __future__ import annotations

import json
import sys
from pathlib import Path

from core.evaluator import exact_match, execution_accuracy
from core.pipeline import run_trc_pipeline
from models import PipelineRequest


def main() -> None:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/sample_questions.json")
    db_path = sys.argv[2] if len(sys.argv) > 2 else None

    questions = json.loads(input_path.read_text(encoding="utf-8"))
    rows = []
    for item in questions:
        response = run_trc_pipeline(
            PipelineRequest(
                question=item["question"],
                db_path=db_path,
                gold_sql=item.get("gold_sql"),
            )
        )
        gold_sql = item.get("gold_sql")
        rows.append(
            {
                "question": item["question"],
                "predicted_sql": response.sql,
                "gold_sql": gold_sql,
                "exact_match": exact_match(response.sql or "", gold_sql or "") if gold_sql else None,
                "execution_accuracy": execution_accuracy(db_path, response.sql or "", gold_sql or "") if gold_sql and response.sql else None,
                "trc_valid": response.validation.valid if response.validation else None,
                "error": response.error,
            }
        )

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
