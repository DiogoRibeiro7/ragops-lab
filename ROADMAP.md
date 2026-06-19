# Roadmap

This roadmap tracks the product and engineering direction for `ragops-lab` as an evaluation-first RAG and LLMOps platform.

## Current state

The repository already includes a working local MVP with:

- typed domain models for documents, chunks, retrieval results, generated answers, evaluations, and traces
- reusable ingestion for text, markdown, csv, and optional pdf adapters
- lexical, vector, and hybrid retrieval flows
- grounded answer generation with citation validation
- evaluation metrics and report export
- CLI commands for ingestion and question answering
- FastAPI endpoints for ingestion, retrieval, answering, evaluation, and trace lookup
- local trace persistence and a minimal dashboard
- automated linting, type-checking, and test coverage

## Milestone 1 — Complete the evaluation pipeline

- Add a dedicated CI job for regression-style RAG evaluation.
- Enforce failure thresholds for faithfulness and citation support.
- Persist evaluation outputs to `artifacts/evaluation`.
- Upload evaluation artifacts from CI for inspection.
- Add clearer reporting for retrieval metrics and answer-quality metrics.

## Milestone 2 — Strengthen runtime and API behavior

- Add request-size and payload-size guards to the API surface.
- Make runtime paths and storage locations configurable through environment settings.
- Improve error responses and validation messages across API and CLI entrypoints.
- Add better trace summaries, filtering, and inspection views in the dashboard.

## Milestone 3 — Move beyond in-memory retrieval

- Add a persistent local vector store for embeddings.
- Add indexing and reload flows for processed documents and chunks.
- Separate retrieval configuration from runtime execution paths.
- Support configurable retrieval profiles for lexical, vector, and hybrid search.

## Milestone 4 — Improve generation and evaluation depth

- Add provider-backed LLM and embedding integrations behind the existing abstractions.
- Add stronger claim extraction and evidence matching for faithfulness checks.
- Add dataset-oriented evaluation commands for repeated benchmark runs.
- Expand refusal evaluation for unanswerable and weak-context cases.

## Milestone 5 — Product and deployment polish

- Improve the dashboard into a more useful inspection surface for traces and metrics.
- Add richer API usage documentation and example payloads.
- Add release notes and a first tagged release from `main`.
- Add deployment guidance for local demos and portfolio presentation.

## Near-term priorities

1. Finish CI-backed evaluation regression checks and artifact publishing.
2. Harden API safety limits and runtime configuration.
3. Add persistent retrieval storage instead of relying on in-memory indexing.
4. Improve documentation and release readiness for external users.
