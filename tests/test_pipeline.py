import pytest

from core.pipeline import run_trc_pipeline
from models import PipelineRequest


@pytest.mark.requires_spider
def test_grouped_count_query_uses_entity_name_and_count(spider_db) -> None:
    response = run_trc_pipeline(
        PipelineRequest(
            question="What are the names of the singers and number of concerts for each person?",
            db_path=spider_db,
        )
    )

    assert response.success is True
    assert response.sql is not None
    assert "GROUP BY s.Name" in response.sql
    assert response.execution_result is not None

    # The count column is aliased from whatever is being counted, so assert on the
    # shape of the answer rather than on the generated alias.
    columns = response.execution_result.columns
    assert "Name" in columns
    assert any(column.startswith("count_") for column in columns)
    assert response.execution_result.row_count == 5
    assert all(row["Name"] and row[columns[1]] > 0 for row in response.execution_result.rows)
