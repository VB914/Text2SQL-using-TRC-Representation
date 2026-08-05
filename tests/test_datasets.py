from pathlib import Path

import pytest

from core.datasets import (
    dataset_catalog,
    export_finetuning_jsonl,
    extract_spider_few_shots,
    list_dataset_examples,
)
from models import JsonlExportRequest
from tests.conftest import missing_dataset


def test_dataset_catalog_includes_sample() -> None:
    catalog = {item.id: item for item in dataset_catalog()}

    assert "sample" in catalog
    assert catalog["sample"].available is True


@pytest.mark.requires_spider
def test_spider_examples_load_when_dataset_present(spider_root) -> None:
    catalog = {item.id: item for item in dataset_catalog()}
    if not catalog.get("spider") or not catalog["spider"].available:
        missing_dataset(spider_root, "Spider dev data")

    examples, total = list_dataset_examples("spider", "dev", limit=3)

    assert total > 0
    assert len(examples) == 3
    assert examples[0].question
    assert examples[0].db_path and Path(examples[0].db_path).exists()


@pytest.mark.requires_spider
def test_auto_mined_spider_few_shots_have_sql(spider_root) -> None:
    examples = extract_spider_few_shots(limit=2)
    if not examples:
        missing_dataset(spider_root, "Spider train data")

    assert len(examples) <= 2
    assert all(example.get("sql") for example in examples)


def test_export_finetuning_jsonl_for_sample(tmp_path: Path) -> None:
    output_path = tmp_path / "sample.jsonl"

    response = export_finetuning_jsonl(
        JsonlExportRequest(
            dataset="sample",
            split="demo",
            output_path=str(output_path),
            limit=2,
        )
    )

    assert response.success is True
    assert response.rows_written == 2
    assert output_path.exists()
