from __future__ import annotations

from pathlib import Path

from ragops_lab.domain import DocumentChunk, GeneratedAnswer, RetrievalResult
from ragops_lab.evaluation import (
    evaluate_answer,
    export_evaluation_report_csv,
    export_evaluation_report_markdown,
)


def test_evaluation_metrics_and_exports(tmp_path: Path) -> None:
    chunk = DocumentChunk(
        chunk_id="apollo:0",
        document_id="apollo",
        text="Apollo 11 was the first mission to land humans on the Moon.",
        start_offset=0,
        end_offset=60,
        token_count=12,
    )
    result = RetrievalResult(chunk=chunk, score=1.0, rank=1, retrieval_method="lexical")
    answer = GeneratedAnswer(
        question="Which mission first landed humans on the Moon?",
        answer_text="Apollo 11 was the first mission to land humans on the Moon.",
        citations=["apollo:0"],
        model_name="heuristic",
        grounded=True,
    )

    report = evaluate_answer(
        answer.question,
        answer,
        [result],
        reference_chunk_ids=["apollo:0"],
        expected_answer=chunk.text,
        expected_unanswerable=False,
    )
    csv_path = tmp_path / "report.csv"
    md_path = tmp_path / "report.md"
    export_evaluation_report_csv(report, csv_path)
    export_evaluation_report_markdown(report, md_path)

    assert report.faithfulness == 1.0
    assert report.citation_support == 1.0
    assert report.claim_count == 1
    assert report.supported_claim_count == 1
    assert csv_path.exists()
    assert md_path.exists()


def test_claim_support_tolerates_light_paraphrase() -> None:
    chunk = DocumentChunk(
        chunk_id="metrics:0",
        document_id="metrics",
        text="Context precision measures the share of retrieved chunks that are relevant.",
        start_offset=0,
        end_offset=72,
        token_count=10,
    )
    answer = GeneratedAnswer(
        question="What does context precision measure?",
        answer_text="Context precision is the proportion of relevant retrieved chunks.",
        citations=["metrics:0"],
        model_name="heuristic",
        grounded=True,
    )

    report = evaluate_answer(
        answer.question,
        answer,
        [RetrievalResult(chunk=chunk, score=1.0, rank=1, retrieval_method="lexical")],
    )

    assert report.faithfulness == 1.0
    assert report.unsupported_claim_count == 0
    assert report.claim_support[0].supported is True


def test_claim_support_flags_hallucinated_claims() -> None:
    chunk = DocumentChunk(
        chunk_id="apollo:0",
        document_id="apollo",
        text="Apollo 11 landed humans on the Moon in 1969.",
        start_offset=0,
        end_offset=46,
        token_count=9,
    )
    answer = GeneratedAnswer(
        question="What happened on Apollo 11?",
        answer_text=(
            "Apollo 11 landed humans on the Moon in 1969. "
            "Buzz Aldrin commanded the mission."
        ),
        citations=["apollo:0"],
        model_name="heuristic",
        grounded=True,
    )

    report = evaluate_answer(
        answer.question,
        answer,
        [RetrievalResult(chunk=chunk, score=1.0, rank=1, retrieval_method="lexical")],
    )

    assert report.faithfulness == 0.5
    assert report.unsupported_claim_count == 1
    assert report.unsupported_claims == ["Buzz Aldrin commanded the mission."]


def test_claim_support_requires_matching_numbers() -> None:
    chunk = DocumentChunk(
        chunk_id="apollo:0",
        document_id="apollo",
        text="Apollo 11 was the first mission to land humans on the Moon.",
        start_offset=0,
        end_offset=60,
        token_count=12,
    )
    answer = GeneratedAnswer(
        question="Which mission landed humans on the Moon?",
        answer_text="Apollo 12 was the first mission to land humans on the Moon.",
        citations=["apollo:0"],
        model_name="heuristic",
        grounded=True,
    )

    report = evaluate_answer(
        answer.question,
        answer,
        [RetrievalResult(chunk=chunk, score=1.0, rank=1, retrieval_method="lexical")],
    )

    assert report.faithfulness == 0.0
    assert report.claim_support[0].missing_terms == ["12"]


def test_claim_support_prefers_cited_evidence() -> None:
    cited_chunk = DocumentChunk(
        chunk_id="apollo:0",
        document_id="apollo",
        text="Apollo 11 landed humans on the Moon.",
        start_offset=0,
        end_offset=36,
        token_count=7,
    )
    uncited_chunk = DocumentChunk(
        chunk_id="apollo:1",
        document_id="apollo",
        text="Neil Armstrong commanded Gemini 8.",
        start_offset=37,
        end_offset=70,
        token_count=5,
    )
    answer = GeneratedAnswer(
        question="What did Neil Armstrong command?",
        answer_text="Neil Armstrong commanded Gemini 8.",
        citations=["apollo:0"],
        model_name="heuristic",
        grounded=True,
    )

    report = evaluate_answer(
        answer.question,
        answer,
        [
            RetrievalResult(chunk=cited_chunk, score=1.0, rank=1, retrieval_method="lexical"),
            RetrievalResult(chunk=uncited_chunk, score=0.8, rank=2, retrieval_method="lexical"),
        ],
    )

    assert report.faithfulness == 0.0
    assert report.claim_support[0].evidence_chunk_id == "apollo:0"
