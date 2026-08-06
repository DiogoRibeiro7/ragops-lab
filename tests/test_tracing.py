from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from ragops_lab.domain import (
    DocumentChunk,
    EvaluationResult,
    GeneratedAnswer,
    RagTrace,
    RagTraceSummary,
    RetrievalResult,
)
from ragops_lab.tracing import JsonlTraceStore


def _trace(trace_id: str, question: str, faithfulness: float, created_at: datetime) -> RagTrace:
    chunk = DocumentChunk(
        chunk_id=f"{trace_id}:0",
        document_id=trace_id,
        text="Apollo 11 landed on the Moon.",
        start_offset=0,
        end_offset=30,
        token_count=6,
    )
    result = RetrievalResult(chunk=chunk, score=1.0, rank=1, retrieval_method="lexical")
    answer = GeneratedAnswer(
        question=question,
        answer_text="Apollo 11 landed on the Moon.",
        citations=[chunk.chunk_id],
        model_name="heuristic-grounded",
        grounded=True,
    )
    evaluation = EvaluationResult(
        context_precision=1.0,
        context_recall=1.0,
        answer_relevance=1.0,
        faithfulness=faithfulness,
        citation_support=1.0,
        unsupported_claim_count=0,
    )
    return RagTrace(
        trace_id=trace_id,
        question=question,
        retrieved_chunks=[result],
        answer=answer,
        evaluation=evaluation,
        model_name=answer.model_name,
        latency_ms=12.5,
        token_estimate=chunk.token_count,
        created_at=created_at,
    )


def test_trace_summary_compacts_full_trace() -> None:
    trace = _trace("trace-a", "Which mission landed on the Moon?", 1.0, datetime.now(UTC))

    summary = RagTraceSummary.from_trace(trace)

    assert summary.trace_id == trace.trace_id
    assert summary.retrieved_chunk_count == 1
    assert summary.faithfulness == 1.0
    assert summary.citation_support == 1.0
    assert summary.grounded is True


def test_trace_store_lists_filtered_summaries(tmp_path: Path) -> None:
    store = JsonlTraceStore(tmp_path / "traces.jsonl")
    now = datetime.now(UTC)
    store.save(_trace("trace-old", "What is lexical retrieval?", 0.5, now - timedelta(days=1)))
    store.save(_trace("trace-new", "Which Apollo mission landed?", 1.0, now))

    summaries = store.list_summaries(query="apollo", min_faithfulness=0.9, limit=10)

    assert [summary.trace_id for summary in summaries] == ["trace-new"]
