from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from core.config import get_settings
from core.schema_utils import load_schema
from models import DatasetExample, DatasetInfo, JsonlExportRequest, JsonlExportResponse


def dataset_catalog() -> list[DatasetInfo]:
    settings = get_settings()
    sample_examples = _load_json_list(settings.sample_questions_path)
    spider_available = (settings.spider_root / "train_spider.json").exists()
    bird_available = (settings.bird_root / "dev.json").exists()

    spider_splits = []
    if spider_available:
        spider_splits = [split for split in ["train", "dev", "test"] if _spider_file(split).exists()]

    return [
        DatasetInfo(
            id="sample",
            name="Bundled SQLite Demo",
            available=settings.sample_db_path.exists(),
            splits=["demo"],
            database_count=1 if settings.sample_db_path.exists() else 0,
            example_count=len(sample_examples),
            notes="Small course-enrollment database used for quick demonstrations.",
        ),
        DatasetInfo(
            id="spider",
            name="Spider",
            available=spider_available,
            splits=spider_splits,
            database_count=_count_sqlite_files(settings.spider_root / "database"),
            example_count=sum(_safe_len(path) for path in [_spider_file("train"), _spider_file("dev"), _spider_file("test")]),
            notes="Cross-domain academic Text-to-SQL benchmark.",
        ),
        DatasetInfo(
            id="bird",
            name="BIRD Dev",
            available=bird_available,
            splits=["dev"] if bird_available else [],
            database_count=_count_sqlite_files(settings.bird_root / "dev_databases"),
            example_count=_safe_len(settings.bird_root / "dev.json"),
            notes="BIRD development split with evidence text and real-world schemas.",
        ),
    ]


def list_dataset_examples(
    dataset: str,
    split: str,
    limit: int = 50,
    offset: int = 0,
    db_id: str | None = None,
    search: str | None = None,
) -> tuple[list[DatasetExample], int]:
    examples = _all_examples(dataset.lower(), split.lower())
    if db_id:
        examples = [item for item in examples if item.db_id == db_id]
    if search:
        needle = search.lower()
        examples = [
            item
            for item in examples
            if needle in item.question.lower() or needle in item.db_id.lower()
        ]
    total = len(examples)
    return examples[offset : offset + limit], total


def list_dataset_databases(dataset: str, split: str = "dev") -> list[dict[str, Any]]:
    examples = _all_examples(dataset.lower(), split.lower())
    counts = Counter(item.db_id for item in examples)
    rows = []
    for db_id, count in sorted(counts.items()):
        db_path = _database_path(dataset.lower(), db_id, split.lower())
        rows.append(
            {
                "db_id": db_id,
                "db_path": str(db_path) if db_path and db_path.exists() else None,
                "example_count": count,
            }
        )
    return rows


def resolve_dataset_db_path(dataset: str, db_id: str, split: str = "dev") -> Path | None:
    path = _database_path(dataset.lower(), db_id, split.lower())
    return path if path and path.exists() else None


def export_finetuning_jsonl(request: JsonlExportRequest) -> JsonlExportResponse:
    try:
        examples, _ = list_dataset_examples(
            dataset=request.dataset,
            split=request.split,
            limit=request.limit or 100_000,
        )
        settings = get_settings()
        output = Path(request.output_path) if request.output_path else settings.exports_dir / f"{request.dataset}_{request.split}_text2sql.jsonl"
        if not output.is_absolute():
            output = settings.project_root / output
        output.parent.mkdir(parents=True, exist_ok=True)

        schema_cache: dict[str, str] = {}
        with output.open("w", encoding="utf-8") as handle:
            for example in examples:
                schema_text = _schema_text(example, schema_cache)
                user_prompt = _training_prompt(example, schema_text, request.include_evidence)
                completion = _training_completion(example)
                handle.write(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "system", "content": "Generate valid SQLite SQL from natural language and schema context."},
                                {"role": "user", "content": user_prompt},
                                {"role": "assistant", "content": completion},
                            ],
                            "dataset": example.dataset,
                            "split": example.split,
                            "db_id": example.db_id,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        return JsonlExportResponse(success=True, output_path=str(output), rows_written=len(examples))
    except Exception as exc:
        fallback = request.output_path or ""
        return JsonlExportResponse(success=False, output_path=fallback, rows_written=0, error=str(exc))


def extract_spider_few_shots(limit: int = 6) -> list[dict[str, str]]:
    examples, _ = list_dataset_examples("spider", "train", limit=200)
    chosen: list[DatasetExample] = []
    buckets = [
        lambda sql: " join " not in sql.lower() and "count" not in sql.lower(),
        lambda sql: " join " in sql.lower(),
        lambda sql: "count" in sql.lower(),
        lambda sql: " group by " in sql.lower(),
        lambda sql: " in (" in sql.lower() or "exists" in sql.lower(),
    ]
    for predicate in buckets:
        for example in examples:
            if example not in chosen and example.gold_sql and predicate(example.gold_sql):
                chosen.append(example)
                break
            if len(chosen) >= limit:
                break
    for example in examples:
        if len(chosen) >= limit:
            break
        if example not in chosen:
            chosen.append(example)

    rows = []
    cache: dict[str, str] = {}
    for example in chosen[:limit]:
        rows.append(
            {
                "question": example.question,
                "schema": _schema_text(example, cache),
                "sql": example.gold_sql or "",
                "dataset": "spider",
                "db_id": example.db_id,
            }
        )
    return rows


def _all_examples(dataset: str, split: str) -> list[DatasetExample]:
    if dataset == "sample":
        return _sample_examples()
    if dataset == "spider":
        return _spider_examples(split)
    if dataset == "bird":
        return _bird_examples(split)
    raise ValueError(f"Unknown dataset: {dataset}")


def _sample_examples() -> list[DatasetExample]:
    settings = get_settings()
    rows = _load_json_list(settings.sample_questions_path)
    return [
        DatasetExample(
            dataset="sample",
            split="demo",
            index=index,
            db_id="sample",
            question=row.get("question", ""),
            db_path=str(settings.sample_db_path),
            gold_sql=row.get("gold_sql"),
        )
        for index, row in enumerate(rows)
    ]


def _spider_examples(split: str) -> list[DatasetExample]:
    rows = []
    if split == "train":
        rows.extend(_load_json_list(_spider_file("train")))
        rows.extend(_load_json_list(get_settings().spider_root / "train_others.json"))
    else:
        rows.extend(_load_json_list(_spider_file(split)))
    examples = []
    for index, row in enumerate(rows):
        db_id = row.get("db_id", "")
        db_path = _database_path("spider", db_id, split)
        examples.append(
            DatasetExample(
                dataset="spider",
                split=split,
                index=index,
                db_id=db_id,
                question=row.get("question", ""),
                db_path=str(db_path) if db_path and db_path.exists() else None,
                gold_sql=row.get("query"),
            )
        )
    return examples


def _bird_examples(split: str) -> list[DatasetExample]:
    if split != "dev":
        return []
    rows = _load_json_list(get_settings().bird_root / "dev.json")
    examples = []
    for index, row in enumerate(rows):
        db_id = row.get("db_id", "")
        db_path = _database_path("bird", db_id, split)
        examples.append(
            DatasetExample(
                dataset="bird",
                split="dev",
                index=index,
                db_id=db_id,
                question=row.get("question", ""),
                db_path=str(db_path) if db_path and db_path.exists() else None,
                gold_sql=row.get("SQL"),
                evidence=row.get("evidence"),
                difficulty=row.get("difficulty"),
                question_id=row.get("question_id"),
            )
        )
    return examples


def _spider_file(split: str) -> Path:
    settings = get_settings()
    if split == "train":
        return settings.spider_root / "train_spider.json"
    return settings.spider_root / f"{split}.json"


def _database_path(dataset: str, db_id: str, split: str) -> Path | None:
    settings = get_settings()
    suffixes = (".sqlite", ".db")
    if dataset == "sample":
        return settings.sample_db_path
    if dataset == "spider":
        roots = [settings.spider_root / "database", settings.spider_root / "test_database"]
        for root in roots:
            for suffix in suffixes:
                candidate = root / db_id / f"{db_id}{suffix}"
                if candidate.exists():
                    return candidate
    if dataset == "bird":
        for suffix in suffixes:
            candidate = settings.bird_root / "dev_databases" / db_id / f"{db_id}{suffix}"
            if candidate.exists():
                return candidate
    return None


def _schema_text(example: DatasetExample, cache: dict[str, str]) -> str:
    if not example.db_path:
        return f"Database id: {example.db_id}"
    if example.db_path not in cache:
        try:
            cache[example.db_path] = load_schema(example.db_path).formatted_schema
        except Exception:
            cache[example.db_path] = f"Database id: {example.db_id}"
    return cache[example.db_path]


def _training_prompt(example: DatasetExample, schema_text: str, include_evidence: bool) -> str:
    parts = [f"Question: {example.question}", "Schema:", schema_text]
    if include_evidence and example.evidence:
        parts.append(f"Evidence: {example.evidence}")
    return "\n".join(parts)


def _training_completion(example: DatasetExample) -> str:
    return f"SQL:\n{example.gold_sql or ''}".strip()


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, list) else []


def _safe_len(path: Path) -> int:
    return len(_load_json_list(path))


def _count_sqlite_files(root: Path) -> int:
    if not root.exists():
        return 0
    return len({path.stem for path in root.rglob("*") if path.suffix.lower() in {".sqlite", ".db", ".sqlite3"}})
