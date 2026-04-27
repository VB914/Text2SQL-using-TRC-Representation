from core.schema_utils import load_schema
from core.trc_validator import heuristic_repair_trc, validate_trc


def test_validator_accepts_valid_trc() -> None:
    schema = load_schema(None)
    trc = "{ s.name | students(s) AND departments(d) AND s.department_id = d.id }"

    report = validate_trc(trc, schema)

    assert report.valid is True
    assert report.parseable is True


def test_validator_rejects_undefined_column() -> None:
    schema = load_schema(None)
    trc = "{ s.nickname | students(s) }"

    report = validate_trc(trc, schema)

    assert report.valid is False
    assert any("Invalid column" in issue.message for issue in report.issues)


def test_heuristic_repair_wraps_missing_braces() -> None:
    repaired = heuristic_repair_trc("s.name | students(s)")

    assert repaired.startswith("{")
    assert repaired.endswith("}")
