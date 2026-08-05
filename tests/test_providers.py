"""Tests for TRC generation.

The strongest guarantee this generator can offer is that whatever it emits
parses, validates against the schema, compiles, and runs. These tests assert
that property across a spread of question shapes rather than pinning exact SQL,
which would break on every heuristic change.
"""

from __future__ import annotations

import pytest

from core.providers import HybridProvider, RuleBasedProvider, get_provider
from core.schema_utils import load_schema
from core.sql_executor import execute_sql
from core.trc_to_sql import trc_to_sql
from core.trc_validator import validate_trc

DB = "data/sample.db"

QUESTIONS = [
    "List all student names",
    "How many students are there?",
    "Show the names of all courses",
    "How many students are in each department?",
    "What are the distinct department names?",
    "Show student names sorted by name",
    "List the top 3 students by name",
    "How many courses are there in total?",
    "Show the names of instructors",
    "What is the name of every department?",
]


@pytest.fixture(scope="module")
def schema():
    return load_schema(DB)


@pytest.fixture(scope="module")
def provider():
    return RuleBasedProvider()


class TestGeneratedTrcIsAlwaysUsable:
    @pytest.mark.parametrize("question", QUESTIONS)
    def test_output_validates_compiles_and_executes(self, question, provider, schema):
        result = provider.generate_trc(question, schema, "")

        report = validate_trc(result.trc, schema)
        assert report.valid, f"{question!r} produced invalid TRC {result.trc!r}: {report.summary}"

        sql = trc_to_sql(result.trc, schema)
        execute_sql(DB, sql)

    @pytest.mark.parametrize("question", QUESTIONS)
    def test_reports_what_it_did(self, question, provider, schema):
        result = provider.generate_trc(question, schema, "")

        assert result.provider
        assert result.reasoning
        assert result.trc.startswith("{")


class TestGenerationChoices:
    def test_counting_one_relation_uses_count_star(self, provider, schema):
        sql = provider.generate_trc("How many students are there?", schema, "").sql

        assert "COUNT(*)" in sql
        assert "JOIN" not in sql, "a count over one table must not join another"

    def test_grouping_question_produces_group_by(self, provider, schema):
        sql = provider.generate_trc("How many students are in each department?", schema, "").sql

        assert "GROUP BY" in sql

    def test_sorted_question_produces_order_by(self, provider, schema):
        sql = provider.generate_trc("Show student names sorted by name", schema, "").sql

        assert "ORDER BY" in sql

    def test_top_n_question_produces_limit(self, provider, schema):
        sql = provider.generate_trc("List the top 3 students by name", schema, "").sql

        assert "LIMIT 3" in sql

    def test_distinct_question_produces_distinct(self, provider, schema):
        sql = provider.generate_trc("What are the distinct department names?", schema, "").sql

        assert "DISTINCT" in sql

    def test_empty_schema_is_reported_clearly(self, provider):
        from models import SchemaResponse

        empty = SchemaResponse(db_path=DB, tables=[], formatted_schema="", database_options=[])
        with pytest.raises(ValueError, match="no user tables"):
            provider.generate_trc("anything", empty, "")


class TestHybridProvider:
    def test_falls_back_to_rules_when_no_model_is_configured(self, schema):
        hybrid = HybridProvider()

        assert hybrid.model is None
        assert hybrid.generate_trc("List all student names", schema, "").provider

    def test_model_failure_does_not_break_generation(self, schema, monkeypatch):
        """A model that raises must not take the pipeline down with it."""

        class Exploding:
            def generate_trc(self, *_args, **_kwargs):
                raise RuntimeError("model unavailable")

        hybrid = HybridProvider()
        monkeypatch.setattr(hybrid, "model", Exploding())

        result = hybrid.generate_trc("List all student names", schema, "")

        assert result.trc
        assert validate_trc(result.trc, schema).valid

    def test_get_provider_is_cached(self):
        assert get_provider() is get_provider()
