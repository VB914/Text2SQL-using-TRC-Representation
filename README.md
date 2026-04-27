# TRC-Guided Text2SQL

This project is a local research demo for Text-to-SQL generation using an explicit Tuple Relational Calculus (TRC) intermediate representation.

The visible pipeline is:

```text
Natural Language + Schema
        ->
Structured Prompt Builder
        ->
LLM Provider or Rule-Based Fallback
        ->
TRC Generation
        ->
TRC Validation
        ->
TRC to SQL Conversion
        ->
SQLite Execution
        ->
Result Table
```

The app is dataset-first: it can use the bundled sample database, Spider data under `data/spider_data`, BIRD dev data under `data/dev`, or an uploaded SQLite database.

## Project Structure

```text
app/          Streamlit UI
core/         schema loading, prompting, providers, TRC parser, validator, compiler, execution
services/     FastAPI app and API routes
models/       Pydantic request/response models
data/         sample DB, uploaded DBs, Spider/BIRD datasets, JSONL exports
prompts/      system prompt and few-shot examples
tests/        unit tests
run.py        one-command local launcher
```

## Run the Project

Step 1: create and activate a virtual environment.

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Step 2: install dependencies.

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Step 3: start the full app from one terminal.

```powershell
python run.py
```

The launcher starts:

```text
FastAPI:   http://localhost:8000
Streamlit: http://localhost:8501
```

The browser should open automatically at `http://localhost:8501`.

## Manual Run Commands

If you want to run the API and UI separately:

```powershell
python -m uvicorn services.main:app --reload --port 8000
```

In another terminal:

```powershell
.venv\Scripts\activate
streamlit run app/app.py
```

## How to Use

1. Select a dataset in the left panel.
2. Select a split and database.
3. Inspect the schema in the sidebar.
4. Choose an example question or type your own question.
5. Click `Generate SQL`.
6. Review the TRC expression, validation report, generated SQL, and execution result.

The UI intentionally shows one research pipeline only: TRC generation, validation, SQL conversion, and execution.

## Structured Prompting

Prompts are built in `core/prompt_builder.py` and use:

```text
Step 1: Identify entities
Step 2: Map schema
Step 3: Generate TRC
Step 4: Validate TRC
Step 5: Convert to SQL
```

Few-shot examples are loaded from:

```text
prompts/few_shot_examples.json
```

If Spider train data is available, the project can also auto-mine a small number of Spider examples for extra prompt grounding.

## TRC Validation

The validator in `core/trc_validator.py` checks:

- balanced braces and parentheses
- malformed TRC syntax
- invalid table names
- invalid column names
- undefined tuple variables
- disconnected joins

If validation fails, the pipeline attempts one repair pass before SQL conversion.

## Model Provider

By default, the project uses a deterministic schema-aware fallback so the demo runs without downloading a model.

To try a local Hugging Face text-to-text model:

```powershell
$env:TEXT2SQL_PROVIDER="transformers"
$env:TEXT2SQL_MODEL_NAME="google/flan-t5-base"
python run.py
```

If the model cannot load, the system falls back to the rule-based provider.

## Dataset and Fine-Tuning Utilities

Export fine-tuning-ready JSONL:

```powershell
python -m services.export_finetuning_data --dataset spider --split train --output data/exports/spider_train_text2sql.jsonl
```

For BIRD dev:

```powershell
python -m services.export_finetuning_data --dataset bird --split dev --output data/exports/bird_dev_text2sql.jsonl
```

Optional LoRA support is scaffolded in `services/train_lora.py`. Install the optional dependencies only if you plan to train:

```powershell
pip install -r requirements-lora.txt
python -m services.train_lora --train-file data/exports/spider_train_text2sql.jsonl --output-dir models/text2sql-lora
```

## Tests

Run the test suite:

```powershell
python -m pytest tests
```

## Troubleshooting

If `fastapi` or `streamlit` is missing:

```powershell
pip install -r requirements.txt
```

If `uvicorn` is not found:

```powershell
python -m uvicorn services.main:app --reload --port 8000
```

If port `8000` or `8501` is already in use, stop the old process or run the services manually on different ports.

If a SQLite database does not load, make sure the file extension is `.db`, `.sqlite`, or `.sqlite3`.
