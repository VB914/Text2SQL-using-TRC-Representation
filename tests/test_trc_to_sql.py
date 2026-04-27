from core.schema_utils import load_schema
from core.sql_executor import execute_sql
from core.trc_to_sql import trc_to_sql


def test_trc_to_sql_translates_join_query() -> None:
    schema = load_schema(None)
    trc = "{ s.name, c.title | students(s) AND enrollments(e) AND courses(c) AND e.student_id = s.id AND e.course_id = c.id }"

    sql = trc_to_sql(trc, schema)

    assert "JOIN enrollments AS e ON e.student_id = s.id" in sql
    assert "JOIN courses AS c ON e.course_id = c.id" in sql


def test_translated_sql_executes_against_sample_db() -> None:
    schema = load_schema(None)
    trc = "{ i.name | instructors(i) AND course_offerings(o) AND courses(c) AND o.instructor_id = i.id AND o.course_id = c.id AND c.title = 'Databases' }"

    sql = trc_to_sql(trc, schema)
    result = execute_sql(schema.db_path, sql)

    assert result.row_count == 1
    assert result.rows[0]["name"] == "Dr. Ada"
