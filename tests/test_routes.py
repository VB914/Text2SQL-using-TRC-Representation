"""Tests for the FastAPI layer.

The API was previously untested end to end, so a broken route or a changed
response shape would only surface in the UI.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from services.main import app

DB = "data/sample.db"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


class TestHealthAndDiscovery:
    def test_health_reports_ok(self, client):
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_databases_are_listed(self, client):
        payload = client.get("/databases").json()

        assert payload["success"] is True
        assert any("sample.db" in item for item in payload["databases"])

    def test_dataset_catalog_is_available(self, client):
        payload = client.get("/datasets").json()

        assert payload["success"] is True
        assert isinstance(payload["datasets"], list)


class TestSchema:
    def test_returns_tables_for_the_sample_database(self, client):
        payload = client.get("/schemas", params={"db_path": DB}).json()

        names = [table["name"] for table in payload["tables"]]
        assert "students" in names

    def test_unknown_database_is_reported_not_crashed(self, client):
        response = client.get("/schemas", params={"db_path": "does/not/exist.db"})

        assert response.status_code == 200
        assert response.json()["success"] is False


class TestGeneration:
    def test_generates_sql_for_a_simple_question(self, client):
        payload = client.post(
            "/generate", json={"question": "How many students are there?", "db_path": DB}
        ).json()

        assert payload["success"] is True
        assert "COUNT(*)" in payload["sql"]
        assert payload["execution_result"]["row_count"] == 1

    def test_empty_question_is_rejected_gracefully(self, client):
        payload = client.post("/generate", json={"question": "   ", "db_path": DB}).json()

        assert payload["success"] is False
        assert payload["error"]

    def test_trc_alias_route_behaves_the_same(self, client):
        payload = client.post(
            "/generate/trc", json={"question": "List student names", "db_path": DB}
        ).json()

        assert payload["success"] is True
        assert payload["trc"]


class TestValidation:
    def test_valid_trc_passes(self, client):
        payload = client.post(
            "/validate/trc", json={"trc": "{ s.name | students(s) }", "db_path": DB}
        ).json()

        assert payload["valid"] is True

    def test_unknown_column_is_reported(self, client):
        payload = client.post(
            "/validate/trc", json={"trc": "{ s.nope | students(s) }", "db_path": DB}
        ).json()

        assert payload["valid"] is False
        assert any(issue["level"] == "error" for issue in payload["issues"])


class TestExecuteEndpointSafety:
    def test_read_only_query_runs(self, client):
        response = client.post("/execute", json={"sql": "SELECT COUNT(*) FROM students", "db_path": DB})

        assert response.status_code == 200
        assert response.json()["success"] is True

    @pytest.mark.parametrize(
        "sql",
        ["DROP TABLE students", "DELETE FROM students", "SELECT 1; DROP TABLE students"],
    )
    def test_destructive_sql_is_refused_through_the_api(self, client, sql):
        """The safety filter must hold at the HTTP boundary, not just in library code."""
        payload = client.post("/execute", json={"sql": sql, "db_path": DB}).json()

        assert payload["success"] is False
