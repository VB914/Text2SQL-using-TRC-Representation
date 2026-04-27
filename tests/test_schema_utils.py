from core.schema_utils import load_schema


def test_load_schema_reads_sample_database() -> None:
    schema = load_schema(None)
    table_names = {table.name for table in schema.tables}

    assert schema.success is True
    assert "students" in table_names
    assert "courses" in table_names
    assert "ForeignKey students.department_id -> departments.id" in schema.formatted_schema


def test_schema_contains_columns() -> None:
    schema = load_schema(None)
    students = next(table for table in schema.tables if table.name == "students")
    column_names = {column.name for column in students.columns}

    assert {"id", "name", "age", "department_id"} <= column_names
