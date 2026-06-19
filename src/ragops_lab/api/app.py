"""FastAPI application for the RAGOps lab."""

from __future__ import annotations

import uuid
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ragops_lab.domain import RagTrace, RetrievalResult
from ragops_lab.evaluation import evaluate_answer
from ragops_lab.generation import GenerationService, HeuristicLLMClient
from ragops_lab.ingestion import ChunkingConfig, ingest_directory, load_chunks_jsonl
from ragops_lab.retrieval import (
    BM25Retriever,
    FakeEmbeddingClient,
    HybridRetriever,
    VectorRetriever,
)
from ragops_lab.tracing import JsonlTraceStore

app = FastAPI(title="RAGOps Lab API", version="0.1.0")


class IngestRequest(BaseModel):
    """Request body for ingestion."""

    input_dir: str = Field(default="data/raw")
    out_path: str = Field(default="data/processed/chunks.jsonl")
    chunk_size: int = Field(default=500, gt=0)
    overlap: int = Field(default=50, ge=0)


class SearchRequest(BaseModel):
    """Search request."""

    query: str = Field(min_length=1)
    chunks_path: str = Field(default="data/processed/chunks.jsonl")
    top_k: int = Field(default=5, ge=1)
    mode: str = Field(default="lexical")


class AskRequest(SearchRequest):
    """Ask request."""


class EvaluateRequest(BaseModel):
    """Evaluation request."""

    question: str = Field(min_length=1)
    answer_text: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)
    chunks_path: str = Field(default="data/processed/chunks.jsonl")
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    reference_chunk_ids: list[str] = Field(default_factory=list)
    expected_answer: str = Field(default="")
    expected_unanswerable: bool | None = Field(default=None)


TRACE_STORE = JsonlTraceStore(Path("artifacts/traces/traces.jsonl"))


def _load_retriever(
    chunks_path: str, mode: str
) -> tuple[list[RetrievalResult], BM25Retriever | VectorRetriever | HybridRetriever]:
    chunks = load_chunks_jsonl(Path(chunks_path))
    lexical = BM25Retriever(chunks)
    vector = VectorRetriever(chunks, FakeEmbeddingClient())
    if mode == "lexical":
        return [], lexical
    if mode == "vector":
        return [], vector
    if mode == "hybrid":
        return [], HybridRetriever(lexical, vector)
    raise HTTPException(status_code=400, detail=f"Unsupported retrieval mode: {mode}")


def _search(request: SearchRequest) -> list[RetrievalResult]:
    chunks = load_chunks_jsonl(Path(request.chunks_path))
    lexical = BM25Retriever(chunks)
    vector = VectorRetriever(chunks, FakeEmbeddingClient())
    if request.mode == "lexical":
        return lexical.search(request.query, top_k=request.top_k)
    if request.mode == "vector":
        return vector.search(request.query, top_k=request.top_k)
    if request.mode == "hybrid":
        return HybridRetriever(lexical, vector).search(request.query, top_k=request.top_k)
    raise HTTPException(status_code=400, detail=f"Unsupported retrieval mode: {request.mode}")


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict[str, int | str]:
    """Ingest local files into chunk JSONL."""
    chunks = ingest_directory(
        Path(request.input_dir),
        Path(request.out_path),
        config=ChunkingConfig(chunk_size=request.chunk_size, overlap=request.overlap),
    )
    return {"chunks_written": len(chunks), "out_path": request.out_path}


@app.post("/search")
def search(request: SearchRequest) -> list[dict[str, object]]:
    """Search indexed chunks."""
    return [result.model_dump(mode="json") for result in _search(request)]


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, object]:
    """Answer a question using retrieved evidence."""
    started = perf_counter()
    results = _search(request)
    service = GenerationService(HeuristicLLMClient())
    answer = service.answer(request.query, results, model_name="heuristic-grounded")
    evaluation = evaluate_answer(request.query, answer, results)
    trace = RagTrace(
        trace_id=f"trace-{uuid.uuid4().hex[:12]}",
        question=request.query,
        retrieved_chunks=results,
        answer=answer,
        evaluation=evaluation,
        model_name=answer.model_name,
        latency_ms=(perf_counter() - started) * 1000,
        token_estimate=sum(result.chunk.token_count for result in results),
    )
    TRACE_STORE.save(trace)
    return {
        "trace_id": trace.trace_id,
        "answer": answer.model_dump(mode="json"),
        "evaluation": evaluation.model_dump(mode="json"),
    }


@app.post("/evaluate")
def evaluate(request: EvaluateRequest) -> dict[str, object]:
    """Evaluate an answer against retrieved evidence."""
    chunks = {chunk.chunk_id: chunk for chunk in load_chunks_jsonl(Path(request.chunks_path))}
    results = [
        RetrievalResult(chunk=chunks[chunk_id], score=1.0, rank=index, retrieval_method="manual")
        for index, chunk_id in enumerate(request.retrieved_chunk_ids, start=1)
        if chunk_id in chunks
    ]
    from ragops_lab.domain import GeneratedAnswer

    answer = GeneratedAnswer(
        question=request.question,
        answer_text=request.answer_text,
        citations=request.citations,
        model_name="manual-eval",
        refusal=False,
        grounded=bool(request.citations),
    )
    report = evaluate_answer(
        request.question,
        answer,
        results,
        reference_chunk_ids=request.reference_chunk_ids,
        expected_answer=request.expected_answer,
        expected_unanswerable=request.expected_unanswerable,
    )
    return report.model_dump(mode="json")


@app.get("/traces/{trace_id}")
def get_trace(trace_id: str) -> dict[str, object]:
    """Fetch a single trace."""
    trace = TRACE_STORE.get(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found.")
    return trace.model_dump(mode="json")


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    """Render a minimal trace dashboard."""
    traces = TRACE_STORE.list()
    items = "\n".join(
        f"<li><strong>{trace.trace_id}</strong>: {trace.question} "
        f"(faithfulness={trace.evaluation.faithfulness if trace.evaluation else 'n/a':.2f})</li>"
        if trace.evaluation is not None
        else f"<li><strong>{trace.trace_id}</strong>: {trace.question}</li>"
        for trace in traces
    )
    return (
        "<html><body><h1>RAGOps Traces</h1><ul>"
        f"{items or '<li>No traces yet.</li>'}"
        "</ul></body></html>"
    )
