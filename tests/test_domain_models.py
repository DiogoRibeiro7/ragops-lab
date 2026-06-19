from __future__ import annotations

from datetime import UTC, datetime

import pytest
from ragops_lab.domain import (
    Document,
    DocumentChunk,
    EvaluationResult,
    GeneratedAnswer,
    RagTrace,
    RetrievalResult,
)


def test_domain_models_accept_valid_values() -> None:
    document = Document(document_id="apollo-doc", title="Apollo", text="Moon landing facts.")
    chunk = DocumentChunk(
        chunk_id="apollo-doc:0",
        document_id=document.document_id,
        text="Apollo 11 landed on the Moon.",
        start_offset=0,
        end_offset=30,
        token_count=6,
    )
    result = RetrievalResult(chunk=chunk, score=0.9, rank=1, retrieval_method="lexical")
    answer = GeneratedAnswer(
        question="Which mission landed on the Moon?",
        answer_text="Apollo 11 landed on the Moon.",
        citations=[chunk.chunk_id],
        model_name="test-model",
        grounded=True,
    )
    evaluation = EvaluationResult(
        context_precision=1.0,
        context_recall=1.0,
        answer_relevance=1.0,
        faithfulness=1.0,
        citation_support=1.0,
        unsupported_claim_count=0,
    )
    trace = RagTrace(
        trace_id="trace-123",
        question=answer.question,
        retrieved_chunks=[result],
        answer=answer,
        evaluation=evaluation,
        model_name="test-model",
        latency_ms=12.5,
        token_estimate=42,
        created_at=datetime.now(UTC),
    )

    assert trace.answer.citations == ["apollo-doc:0"]


@pytest.mark.parametrize(
    ("factory", "expected_message"),
    [
        (lambda: Document(document_id="!", title="Bad", text="text"), "document_id"),
        (
            lambda: DocumentChunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                text="text",
                start_offset=10,
                end_offset=5,
                token_count=1,
            ),
            "end_offset",
        ),
        (
            lambda: GeneratedAnswer(
                question="question",
                answer_text="answer",
                citations=["bad citation"],
                model_name="model",
            ),
            "citation",
        ),
        (
            lambda: EvaluationResult(
                context_precision=1.2,
                context_recall=0.0,
                answer_relevance=0.0,
                faithfulness=0.0,
                citation_support=0.0,
                unsupported_claim_count=0,
            ),
            "less than or equal to 1",
        ),
    ],
)
def test_domain_models_reject_invalid_values(factory: object, expected_message: str) -> None:
    with pytest.raises(Exception) as exc_info:
        factory()  # type: ignore[misc]

    assert expected_message in str(exc_info.value)
