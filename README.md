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

## TRC Grammar

The intermediate representation is a tuple relational calculus expression with an
optional shaping clause:

```text
{ [DISTINCT] projection_list | formula } [ORDER BY key [ASC|DESC], ...] [LIMIT n [OFFSET m]]
```

Supported inside the formula: relation predicates `students(s)`, attribute
references `s.name`, `AND`/`OR`/`NOT`, `EXISTS x (...)`, comparisons, aggregates
including `COUNT(*)`, `LIKE`, `IN` over a value list or a set comprehension,
`BETWEEN`, and `IS [NOT] NULL`.

Two design points are worth calling out:

- **Ordering and truncation sit outside the braces.** `{ t | phi(t) }` denotes a
  set, and sets are unordered, so `ORDER BY` and `LIMIT` are applied to the
  resulting set rather than written inside the calculus.
- **There is no `HAVING` keyword.** A conjunct containing an aggregate cannot be
  evaluated per tuple, so the compiler routes it to `HAVING` and everything else
  to `WHERE`, deriving `GROUP BY` from the remaining projections.

Out of scope by design: set operations (`UNION`/`INTERSECT`/`EXCEPT`), subqueries
in `FROM`, and arithmetic expressions. Together these account for roughly 15% of
the Spider dev set.

## TRC Validation

The validator in `core/trc_validator.py` checks:

- malformed TRC syntax, with the position of the offending token
- invalid table names
- invalid column names, checked against the specific table a variable is bound to
- undefined tuple variables
- disconnected joins (reported as a warning)
- ordering keys, `LIMIT`/`OFFSET` sanity, and grouping consistency

If validation fails, `core/trc_repair.py` attempts a schema-aware repair on the
parsed tree, correcting case differences and near-miss table and column names. A
repair is kept only when it strictly reduces the number of errors, so a repair
can never leave the expression worse than it was.

Once the TRC compiles, the generated SQL is planned with `EXPLAIN QUERY PLAN`
before it is executed, so SQL that cannot run is reported as a generation
failure rather than an error part-way through execution.

## Schema Linking

Filter values are recovered by indexing the distinct values actually stored in
the database (`core/schema_linker.py`), so a question such as "singers from
France" resolves to `country = 'France'` without the value being quoted. The
index is cached under `data/cache/` and keyed on the database's modification
time and size.

`core/question_hints.py` reads intent from the question: superlatives and sort
direction, `LIMIT`, aggregate functions, comparison wording such as "at least"
or "older than", and requests for distinct results.

## Safety

Generated SQL is never trusted:

- databases are opened read-only through a `file:` URI, enforced by SQLite
- only single `SELECT`/`WITH` statements are allowed, checked after string
  literals and comments are stripped so a value like `'Update'` is not mistaken
  for a statement
- a SQLite authorizer denies anything that is not a read
- queries are abandoned once they exceed a time budget
  (`TEXT2SQL_SQL_TIMEOUT_SECONDS`, default 10 seconds)

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

> **Note:** `services/train_lora.py` is scaffolding and is currently not usable.
> It fine-tunes a causal model while the inference path loads a sequence-to-sequence
> pipeline, and nothing loads the trained adapter back at inference time. It needs
> reworking before it will do anything useful.

## Results

Measured on the full Spider dev set (1034 questions, unseen databases) using the
default deterministic generator:

| Metric | Value |
|---|---|
| Execution accuracy | 21.4% |
| TRC validation failures | 0 |
| Generator crashes | 0 |

Reproduce with:

```powershell
python -m services.run_batch_eval data/sample_questions.json --out-dir data/exports/eval
```

Exact match currently reports 0% because it compares normalised SQL text, and
this pipeline emits explicit aliases (`SELECT s.name FROM students AS s`) where
the Spider gold queries do not (`SELECT name FROM students`). The queries are
semantically identical; the metric is measuring formatting. Spider's official
exact-set-match metric is the appropriate comparison and is not yet implemented.

### Known limitations

- Accuracy is bounded by the deterministic generator, which decides joins from
  keyword overlap. Questions needing nested subqueries or set operations are not
  handled.
- The Hugging Face provider is experimental. `google/flan-t5-base` does not
  reliably emit well-formed TRC, so the rule-based generator is the default.

## Tests

```powershell
python -m pytest tests
```

Tests that need Spider or BIRD skip when the data is absent. To make those skips
fail instead, which is what you want on a machine where the data should be
present:

```powershell
$env:TEXT2SQL_REQUIRE_DATASETS="1"
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
