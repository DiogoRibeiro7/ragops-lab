# Roadmap

## Phase 1 — Foundation

- Define data models for documents, chunks, retrieval results, generated answers, and evaluations.
- Create CLI for ingestion and question answering.

## Phase 2 — Ingestion

- Support Markdown, text, PDF placeholder adapter, and CSV metadata.
- Implement chunking strategies.
- Store normalized documents and chunks.

## Phase 3 — Retrieval

- Implement BM25 baseline.
- Add vector retrieval adapter.
- Add hybrid reranking.
- Add retrieval evaluation datasets.

## Phase 4 — Generation

- Add provider-agnostic LLM client.
- Add answer generation with evidence constraints.
- Add citation validation.

## Phase 5 — Evaluation

- Implement automatic metrics.
- Add golden dataset support.
- Add prompt regression test suite.

## Phase 6 — API and observability

- Add FastAPI service.
- Store traces, latency, and cost estimates.
- Add dashboard.

## Phase 7 — Deployment

- Add Docker Compose.
- Add local vector store.
- Add CI checks for prompt regressions.
