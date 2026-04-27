from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv("TEXT2SQL_API_URL", "http://localhost:8000").rstrip("/")


st.set_page_config(
    page_title="TRC Text2SQL Research Demo",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def api_get(path: str, **params: Any) -> dict[str, Any]:
    response = requests.get(f"{API_URL}{path}", params={k: v for k, v in params.items() if v is not None}, timeout=30)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=90)
    response.raise_for_status()
    return response.json()


def upload_database(file) -> str | None:
    response = requests.post(
        f"{API_URL}/databases/upload",
        files={"file": (file.name, file.getvalue(), "application/octet-stream")},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("success"):
        return payload.get("db_path")
    st.error(payload.get("error", "Upload failed."))
    return None


def inject_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(38, 166, 154, 0.18), transparent 34rem),
                linear-gradient(135deg, #0e141b 0%, #111827 52%, #172033 100%);
            color: #eef3f8;
        }
        [data-testid="stSidebar"] {
            background: rgba(9, 14, 21, 0.92);
            border-right: 1px solid rgba(255,255,255,0.08);
        }
        .hero {
            padding: 1.3rem 1.5rem;
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(255,255,255,0.09), rgba(255,255,255,0.035));
            box-shadow: 0 22px 70px rgba(0,0,0,0.22);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2.1rem;
            letter-spacing: -0.04em;
        }
        .hero p {
            margin: 0.45rem 0 0;
            color: #b7c4d2;
            max-width: 820px;
        }
        .metric-card {
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 18px;
            padding: 0.85rem 1rem;
            background: rgba(255,255,255,0.055);
        }
        .section-title {
            font-weight: 700;
            font-size: 1.05rem;
            margin-bottom: 0.35rem;
            color: #f4f7fb;
        }
        .muted {
            color: #a7b3c2;
            font-size: 0.92rem;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: rgba(255,255,255,0.11);
            background: rgba(255,255,255,0.045);
        }
        .stButton > button {
            width: 100%;
            border-radius: 999px;
            border: 0;
            background: linear-gradient(90deg, #26a69a, #84d6c5);
            color: #061018;
            font-weight: 800;
            padding: 0.7rem 1.1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_schema(schema_payload: dict[str, Any] | None) -> None:
    st.sidebar.markdown("### Database Schema")
    if not schema_payload or not schema_payload.get("success", True):
        st.sidebar.info("Select a database to inspect its schema.")
        return

    for table in schema_payload.get("tables", []):
        columns = table.get("columns", [])
        with st.sidebar.expander(f"{table['name']}  ·  {len(columns)} columns", expanded=False):
            for column in columns:
                pk = " primary key" if column.get("primary_key") else ""
                st.markdown(f"`{column['name']}`  <span class='muted'>{column['data_type']}{pk}</span>", unsafe_allow_html=True)
            foreign_keys = table.get("foreign_keys", [])
            if foreign_keys:
                st.caption("Foreign keys")
                for fk in foreign_keys:
                    inferred = " inferred" if fk.get("inferred") else ""
                    st.caption(f"{fk['source_column']} -> {fk['target_table']}.{fk['target_column']}{inferred}")


def render_validation(validation: dict[str, Any] | None) -> None:
    if not validation:
        st.warning("No validation report returned.")
        return
    valid = validation.get("valid", False)
    st.markdown("✔ Valid" if valid else "❌ Error")
    st.caption(validation.get("summary", ""))
    issues = validation.get("issues", [])
    for issue in issues:
        level = issue.get("level", "info").upper()
        st.write(f"{level}: {issue.get('message', '')}")


def render_results(result: dict[str, Any] | None) -> None:
    if not result:
        st.info("No rows to display yet.")
        return
    rows = result.get("rows", [])
    columns = result.get("columns", [])
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.dataframe(pd.DataFrame(columns=columns), use_container_width=True, hide_index=True)
    st.caption(f"{result.get('row_count', 0)} row(s) returned")


def main() -> None:
    inject_style()

    try:
        datasets_payload = api_get("/datasets")
    except Exception as exc:
        st.error(f"API service is not reachable at {API_URL}. Start the project with `python run.py`.")
        st.caption(str(exc))
        return

    datasets = [item for item in datasets_payload.get("datasets", []) if item.get("available")]
    if not datasets:
        st.error("No datasets or SQLite databases were found in the data folder.")
        return

    st.markdown(
        """
        <div class="hero">
            <h1>TRC-Guided Text-to-SQL</h1>
            <p>Structured prompting generates Tuple Relational Calculus first, validates it against the selected schema, compiles it to SQL, and executes it safely on SQLite.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Dataset")
        dataset_labels = {f"{item['name']} ({item['id']})": item for item in datasets}
        dataset_label = st.selectbox("Select benchmark", list(dataset_labels.keys()), label_visibility="collapsed")
        dataset_info = dataset_labels[dataset_label]
        split = st.selectbox("Split", dataset_info.get("splits") or ["demo"])

        uploaded_file = st.file_uploader("Upload SQLite database", type=["db", "sqlite", "sqlite3"])
        if uploaded_file and st.button("Use uploaded database"):
            uploaded_path = upload_database(uploaded_file)
            if uploaded_path:
                st.session_state["uploaded_db_path"] = uploaded_path
                st.success("Database uploaded.")

    uploaded_db_path = st.session_state.get("uploaded_db_path")
    db_rows = []
    if not uploaded_db_path:
        db_payload = api_get("/datasets/databases", dataset=dataset_info["id"], split=split)
        db_rows = db_payload.get("databases", [])

    col_left, col_right = st.columns([0.38, 0.62], gap="large")
    with col_left:
        with st.container(border=True):
            st.markdown("<div class='section-title'>Database</div>", unsafe_allow_html=True)
            if uploaded_db_path:
                selected_db_id = "uploaded"
                selected_db_path = uploaded_db_path
                st.info(Path(uploaded_db_path).name)
            else:
                db_labels = {
                    f"{row['db_id']} · {row.get('example_count', 0)} questions": row
                    for row in db_rows
                    if row.get("db_path")
                }
                if not db_labels:
                    st.warning("No SQLite database files found for this split.")
                    return
                db_label = st.selectbox("Select database", list(db_labels.keys()))
                selected_db = db_labels[db_label]
                selected_db_id = selected_db["db_id"]
                selected_db_path = selected_db["db_path"]
            st.caption(selected_db_path)

    examples = []
    selected_example: dict[str, Any] | None = None
    if not uploaded_db_path:
        examples_payload = api_get(
            "/datasets/examples",
            dataset=dataset_info["id"],
            split=split,
            db_id=selected_db_id,
            limit=40,
        )
        examples = examples_payload.get("examples", [])

    with col_right:
        with st.container(border=True):
            st.markdown("<div class='section-title'>Question</div>", unsafe_allow_html=True)
            if examples:
                labels = ["Write my own question"] + [f"{item['index']}: {item['question'][:96]}" for item in examples]
                chosen = st.selectbox("Example question", labels)
                if chosen != labels[0]:
                    selected_example = examples[labels.index(chosen) - 1]
            default_question = selected_example["question"] if selected_example else ""
            question = st.text_area(
                "Natural language input",
                value=default_question,
                height=120,
                placeholder="Example: Which instructor teaches the Databases course?",
            )
            st.markdown("<div class='muted'>Visible pipeline: NL + schema -> structured prompt -> TRC -> validation -> SQL -> execution.</div>", unsafe_allow_html=True)

    schema_payload = api_get("/schemas", db_path=selected_db_path)
    render_schema(schema_payload)

    generate_col = st.columns([0.32, 0.36, 0.32])[1]
    with generate_col:
        generate = st.button("Generate SQL", type="primary")

    if generate:
        payload = {
            "question": question,
            "db_path": selected_db_path,
            "dataset": dataset_info["id"] if not uploaded_db_path else "uploaded",
            "split": split,
            "db_id": selected_db_id,
        }
        if selected_example:
            payload.update(
                {
                    "gold_sql": selected_example.get("gold_sql"),
                    "evidence": selected_example.get("evidence"),
                    "difficulty": selected_example.get("difficulty"),
                    "question_id": selected_example.get("question_id"),
                }
            )
        with st.spinner("Generating TRC, validating, compiling SQL, and executing..."):
            st.session_state["pipeline_response"] = api_post("/generate", payload)

    response = st.session_state.get("pipeline_response")
    if not response:
        return

    status_cols = st.columns(3)
    status_cols[0].markdown(f"<div class='metric-card'><b>Provider</b><br>{response.get('provider', '-')}</div>", unsafe_allow_html=True)
    status_cols[1].markdown(f"<div class='metric-card'><b>Status</b><br>{'Success' if response.get('success') else 'Needs review'}</div>", unsafe_allow_html=True)
    status_cols[2].markdown(f"<div class='metric-card'><b>Database</b><br>{Path(response.get('db_path', '')).name}</div>", unsafe_allow_html=True)

    if response.get("error"):
        st.error(response["error"])

    with st.container(border=True):
        st.markdown("<div class='section-title'>1. TRC Generation</div>", unsafe_allow_html=True)
        st.caption(response.get("reasoning", {}).get("trc_generation", ""))
        if response.get("entities"):
            st.write("Entities: " + ", ".join(response["entities"]))
        if response.get("schema_mappings"):
            st.write("Schema mapping: " + "; ".join(response["schema_mappings"]))
        st.code(response.get("trc") or "", language="text")

    with st.container(border=True):
        st.markdown("<div class='section-title'>2. TRC Validation</div>", unsafe_allow_html=True)
        render_validation(response.get("validation"))

    with st.container(border=True):
        st.markdown("<div class='section-title'>3. SQL Generation</div>", unsafe_allow_html=True)
        st.caption(response.get("reasoning", {}).get("sql_generation", ""))
        st.code(response.get("sql") or "-- SQL was not generated because validation failed.", language="sql")

    with st.container(border=True):
        st.markdown("<div class='section-title'>4. Execution Result</div>", unsafe_allow_html=True)
        st.caption(response.get("reasoning", {}).get("execution", ""))
        render_results(response.get("execution_result"))


if __name__ == "__main__":
    main()
