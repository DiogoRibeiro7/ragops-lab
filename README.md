# ragops-lab

[![CI](https://github.com/DiogoRibeiro7/ragops-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/DiogoRibeiro7/ragops-lab/actions/workflows/ci.yml)
[![GitHub release](https://img.shields.io/github/v/release/DiogoRibeiro7/ragops-lab)](https://github.com/DiogoRibeiro7/ragops-lab/releases)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21805398.svg)](https://doi.org/10.5281/zenodo.21805398)

RAGOps Lab is an open-source software project for building, testing, and observing retrieval-augmented generation systems.

This project is built to demonstrate practical AI engineering, not just prompt wiring. It focuses on ingestion, retrieval quality, grounded generation, measurable evaluation, and traceability through a reusable Python package, CLI, API, tests, and sample assets.

## Repository status

- Python package under `src/` with typed domain models and service boundaries.
- CLI and FastAPI entrypoints.
- Deterministic offline defaults; no model API keys are required for the main demo path.
- CI runs linting, type checking, and tests on Python 3.11 and 3.12.
- Licensed under MIT.
- All versions DOI: [`10.5281/zenodo.21805398`](https://doi.org/10.5281/zenodo.21805398).
- Release history is tracked in [`CHANGELOG.md`](CHANGELOG.md).

## What it does

- Ingests `.txt`, `.md`, `.csv`, and optionally `.pdf` documents into reusable chunks.
- Supports lexical, vector, and hybrid retrieval through package APIs.
- Generates evidence-grounded answers with citation validation.
- Evaluates context precision, context recall, faithfulness, citation support, unsupported claims, and refusal correctness.
- Stores RAG traces with latency and token estimates.
- Exposes the workflow through both a CLI and a FastAPI service.
- Includes prompt regression coverage and an analytical notebook suite (see below).

## Quickstart

### 1. Install

Requires Python 3.11 or 3.12 and Poetry.

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

You can also use the installed console script:

```bash
poetry run ragops-lab ask "Which Apollo mission first landed on the Moon?" --chunks data/processed/chunks.jsonl
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

### 5. Run with Docker

```bash
docker compose up --build
```

The API is served at `http://localhost:8000`.

## Notebooks

The [`notebooks/`](notebooks/) suite is an analytical walkthrough of the system,
built entirely on the package code (no notebook-only logic) and committed with
executed outputs and figures so it renders on GitHub without being re-run.

| Notebook | Focus |
| --- | --- |
| [`01_retrieval_baseline`](notebooks/01_retrieval_baseline.ipynb) | BM25 baseline, per-query score anatomy, and a `k1`×`b` parameter sweep |
| [`02_retrieval_strategies`](notebooks/02_retrieval_strategies.ipynb) | Lexical vs vector vs hybrid: recall/MRR curves, a fusion-weight sweep, and per-question win/loss analysis |
| [`03_grounded_generation`](notebooks/03_grounded_generation.ipynb) | Grounded prompting, citation validation, refusal guard-rails, and latency profiling |
| [`04_rag_evaluation`](notebooks/04_rag_evaluation.ipynb) | End-to-end scoring, operational budgets, trace persistence, and a CI-style regression gate |

They run against the bundled corpus and golden set with deterministic offline
clients, so `poetry install --with dev` is the only prerequisite. To re-execute:

```bash
poetry run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

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

The design principle is evaluation-first development: every generated answer should be linked to retrieved evidence, validated for citations, and measurable through explicit metrics. See [`docs/architecture.md`](docs/architecture.md) for more detail.

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
make check
```

Or run individual gates:

```bash
make lint
make typecheck
make test
```

Pre-commit hooks are available:

```bash
poetry run pre-commit install
poetry run pre-commit run --all-files
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for contribution guidelines and [`SECURITY.md`](SECURITY.md) for vulnerability reporting.

## Notes

- The default CLI and API generation path uses a deterministic local heuristic client so the repo works without external model credentials.
- The vector retrieval layer includes a fake embedding client for tests and an optional `sentence-transformers` adapter for local experimentation.
- PDF ingestion is intentionally optional and exposed through an interface boundary rather than a hard dependency.

## Portfolio signal

This repo shows operational depth around RAG systems: retrieval baselines, grounded generation, regression testing, evaluation exports, and traceable service behavior rather than a thin demo wrapper around an LLM call.
