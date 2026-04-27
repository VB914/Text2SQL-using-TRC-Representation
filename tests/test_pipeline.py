from pathlib import Path

import pytest

from core.pipeline import run_trc_pipeline
from models import PipelineRequest


def test_grouped_count_query_uses_entity_name_and_count() -> None:
    db_path = Path("data/spider_data/spider_data/database/concert_singer/concert_singer.sqlite")
    if not db_path.exists():
        pytest.skip("Spider concert_singer database is not present")

    response = run_trc_pipeline(
        PipelineRequest(
            question="What are the names of the singers and number of concerts for each person?",
            db_path=str(db_path),
        )
    )

    assert response.success is True
    assert response.sql is not None
    assert "GROUP BY s.Name" in response.sql
    assert response.execution_result is not None
    assert {"Name", "count_concert_ID"} <= set(response.execution_result.columns)
    assert response.execution_result.row_count == 5
