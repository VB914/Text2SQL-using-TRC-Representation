from __future__ import annotations

import argparse
import json

from core.datasets import export_finetuning_jsonl
from models import JsonlExportRequest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Spider/BIRD examples as fine-tuning-ready JSONL.")
    parser.add_argument("--dataset", choices=["sample", "spider", "bird"], default="spider")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-evidence", action="store_true")
    args = parser.parse_args()

    response = export_finetuning_jsonl(
        JsonlExportRequest(
            dataset=args.dataset,
            split=args.split,
            output_path=args.output,
            limit=args.limit,
            include_evidence=not args.no_evidence,
        )
    )
    print(json.dumps(response.model_dump(), indent=2))


if __name__ == "__main__":
    main()
