# ragops-lab

Evaluation-first RAG and LLMOps platform for production-grade document question answering.

This project is built to demonstrate practical AI engineering, not just prompt wiring. It focuses on ingestion, retrieval quality, grounded generation, measurable evaluation, and traceability through a reusable Python package, CLI, API, tests, and sample assets.

## What it does

- Ingests `.txt`, `.md`, `.csv`, and optionally `.pdf` documents into reusable chunks.
- Supports lexical, vector, and hybrid retrieval through package APIs.
- Generates evidence-grounded answers with citation validation.
- Evaluates context precision, context recall, faithfulness, citation support, unsupported claims, and refusal correctness.
- Stores RAG traces with latency and token estimates.
- Exposes the workflow through both a CLI and a FastAPI service.
- Includes prompt regression coverage and a retrieval baseline notebook.

## Quickstart

### 1. Install

```bash
poetry install --with dev
```

### 2. Ingest sample documents

```bash
poetry run python -m ragops_lab.cli ingest data/sample_documents --out data/processed/chunks.jsonl
```

### 3. Ask a question

```bash
poetry run python -m ragops_lab.cli ask "Which Apollo mission first landed on the Moon?" --chunks data/processed/chunks.jsonl
```

### 4. Run the API

```bash
poetry run uvicorn ragops_lab.api.app:app --host 0.0.0.0 --port 8000
```

Then use:

- `POST /ingest`
- `POST /search`
- `POST /ask`
- `POST /evaluate`
- `GET /traces/{id}`
- `GET /dashboard`

## Example API calls

### Ingest

```bash
curl -X POST http://localhost:8000/ingest ^
  -H "Content-Type: application/json" ^
  -d "{\"input_dir\":\"data/sample_documents\",\"out_path\":\"data/processed/chunks.jsonl\"}"
```

### Ask

```bash
curl -X POST http://localhost:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"Which Apollo mission landed on the Moon?\",\"chunks_path\":\"data/processed/chunks.jsonl\",\"top_k\":3,\"mode\":\"hybrid\"}"
```

## Architecture

The system is organized as reusable package modules instead of notebook-only logic:

```text
Raw documents
  -> ingestion
  -> chunk store
  -> retrieval
  -> generation
  -> evaluation
  -> trace store
  -> CLI / API / dashboard
```

The design principle is evaluation-first development: every generated answer should be linked to retrieved evidence, validated for citations, and measurable through explicit metrics.

## Project structure

```text
src/ragops_lab/domain       Core typed models
src/ragops_lab/ingestion    Document loading, chunking, JSONL persistence
src/ragops_lab/retrieval    Tokenizer, BM25, vector, hybrid, retrieval metrics
src/ragops_lab/generation   LLM abstraction and grounded answer generation
src/ragops_lab/evaluation   RAG evaluation metrics and report export
src/ragops_lab/tracing      Trace persistence
src/ragops_lab/api          FastAPI service and dashboard
src/ragops_lab/cli.py       Command line interface
tests/                      Unit and integration coverage
data/sample_documents/      Local demo corpus
data/golden/                Prompt and retrieval regression fixtures
notebooks/                  Notebook demos built on package code
```

## Quality checks

```bash
poetry run ruff check .
poetry run mypy src
poetry run pytest -q
```

Or via `make`:

```bash
make install
make lint
make typecheck
make test
```

## Notes

- The default CLI and API generation path uses a deterministic local heuristic client so the repo works without external model credentials.
- The vector retrieval layer includes a fake embedding client for tests and an optional `sentence-transformers` adapter for local experimentation.
- PDF ingestion is intentionally optional and exposed through an interface boundary rather than a hard dependency.

## Portfolio signal

This repo shows operational depth around RAG systems: retrieval baselines, grounded generation, regression testing, evaluation exports, and traceable service behavior rather than a thin demo wrapper around an LLM call.
