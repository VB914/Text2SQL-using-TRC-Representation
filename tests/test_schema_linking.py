"""Tests for value linking, column linking, question hints and TRC repair."""

from __future__ import annotations

import pytest

from core.question_hints import (
    asks_for_distinct,
    detect_aggregates,
    detect_comparison,
    detect_ordering,
    superlative_column_words,
)
from core.schema_linker import (
    ValueIndex,
    build_value_index,
    find_mentioned_columns,
    normalize,
    question_ngrams,
)
from core.schema_utils import load_schema
from core.trc_repair import repair_trc
from core.trc_validator import validate_trc

SAMPLE_DB = "data/sample.db"


@pytest.fixture(scope="module")
def schema():
    return load_schema(SAMPLE_DB)


class TestValueIndex:
    def test_matches_a_value_that_is_not_quoted(self):
        index = ValueIndex()
        index.add("singer", "Country", "France")

        assert index.lookup("france") == [("singer", "Country", "France")]

    def test_lookup_is_case_and_punctuation_insensitive(self):
        index = ValueIndex()
        index.add("stadium", "Location", "Raith Rovers")

        assert index.lookup("raith rovers") == [("stadium", "Location", "Raith Rovers")]

    def test_common_words_are_not_indexed(self):
        index = ValueIndex()
        index.add("t", "c", "the")

        assert index.lookup("the") == []

    def test_survives_a_round_trip_through_json(self):
        index = ValueIndex()
        index.add("singer", "Country", "France")

        assert ValueIndex(index.to_json()).lookup("France")

    def test_builds_and_caches_for_a_real_database(self, schema):
        first = build_value_index(SAMPLE_DB, schema)
        second = build_value_index(SAMPLE_DB, schema)

        assert len(first) > 0
        assert first is second


def test_longer_phrases_are_offered_before_shorter_ones():
    ngrams = question_ngrams("New York City", max_n=3)

    assert ngrams[0] == "New York City"
    assert ngrams.index("New York") < ngrams.index("New")


def test_normalize_strips_punctuation_and_case():
    assert normalize("  Raith-Rovers!  ") == "raith rovers"


class TestColumnMentions:
    def test_returns_columns_in_the_order_the_question_asks(self, schema):
        found = find_mentioned_columns("Show the name and department_id of students", schema.tables)
        names = [column for _, column in found]

        assert names.index("name") < names.index("department_id")

    def test_ignores_generic_column_names(self, schema):
        found = find_mentioned_columns("show me the id", schema.tables)

        assert all(column.lower() != "id" for _, column in found)


class TestOrderingHints:
    @pytest.mark.parametrize(
        ("question", "descending", "limit"),
        [
            ("What is the name of the singer with the highest rating?", True, 1),
            ("Show the top 5 highest paid employees", True, 5),
            ("List the 3 largest stadiums", True, 3),
            ("What is the youngest age of any student?", False, 1),
            ("List all students sorted by name", False, None),
            ("List concerts in descending order of year", True, None),
        ],
    )
    def test_detects_direction_and_limit(self, question, descending, limit):
        hint = detect_ordering(question)

        assert hint is not None
        assert hint.descending is descending
        assert hint.limit == limit

    def test_trailing_superlative_sets_descending(self):
        """'ordered by age from the oldest' states direction after the sort key."""
        hint = detect_ordering("Show singers ordered by age from the oldest")

        assert hint is not None
        assert hint.descending is True

    def test_comparison_wording_is_not_read_as_a_ranking(self):
        """'at least 3' is a filter; 'least' must not be treated as a superlative."""
        assert detect_ordering("Show courses with at least 3 credits") is None

    def test_superlative_implies_a_column(self):
        assert "age" in superlative_column_words("Who is the youngest singer?")


class TestComparisonHints:
    @pytest.mark.parametrize(
        ("question", "operator", "value"),
        [
            ("students older than 20", ">", 20),
            ("courses with at least 3 credits", ">=", 3),
            ("employees with salary less than 50000", "<", 50000),
            ("items with at most 5 units", "<=", 5),
            ("exactly 4 members", "=", 4),
        ],
    )
    def test_maps_wording_to_operators(self, question, operator, value):
        hint = detect_comparison(question)

        assert hint is not None
        assert hint.operator == operator
        assert hint.value == value

    def test_returns_nothing_without_a_number(self):
        assert detect_comparison("show all students") is None


class TestAggregateAndDistinct:
    def test_detects_aggregates_in_the_order_mentioned(self):
        assert detect_aggregates("the average, minimum, and maximum age") == ["AVG", "MIN", "MAX"]

    def test_no_aggregate_words_means_no_aggregates(self):
        assert detect_aggregates("show all student names") == []

    @pytest.mark.parametrize("question", ["distinct countries", "different names", "unique values"])
    def test_detects_distinct_requests(self, question):
        assert asks_for_distinct(question)


class TestRepair:
    def test_fixes_a_column_whose_case_is_wrong(self, schema):
        trc = "{ s.NAME | students(s) }"
        report = validate_trc(trc, schema)
        assert not report.valid

        repaired = repair_trc(trc, schema, report)

        assert validate_trc(repaired, schema).valid

    def test_fixes_a_misspelled_column(self, schema):
        trc = "{ s.nme | students(s) }"
        report = validate_trc(trc, schema)
        assert not report.valid

        repaired = repair_trc(trc, schema, report)

        assert validate_trc(repaired, schema).valid

    def test_leaves_a_valid_expression_untouched(self, schema):
        trc = "{ s.name | students(s) }"

        assert repair_trc(trc, schema, validate_trc(trc, schema)) == trc

    def test_gives_up_rather_than_guessing_wildly(self, schema):
        """An unrelated name has no close match, so the input is returned unchanged."""
        trc = "{ s.zzzzzzzz | students(s) }"

        repaired = repair_trc(trc, schema, validate_trc(trc, schema))

        assert not validate_trc(repaired, schema).valid
