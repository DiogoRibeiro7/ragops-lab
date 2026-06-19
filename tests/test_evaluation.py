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
    assert csv_path.exists()
    assert md_path.exists()
