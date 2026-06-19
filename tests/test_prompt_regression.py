from __future__ import annotations

import json
from pathlib import Path

from ragops_lab.evaluation import (
    evaluate_answer,
    export_evaluation_report_csv,
    export_evaluation_report_markdown,
)
from ragops_lab.generation import GenerationService, HeuristicLLMClient
from ragops_lab.ingestion import ChunkingConfig, ingest_directory, load_chunks_jsonl
from ragops_lab.retrieval import BM25Retriever


def test_prompt_regression_thresholds(tmp_path: Path) -> None:
    source_dir = Path("data/sample_documents")
    chunks_path = tmp_path / "chunks.jsonl"
    chunks = ingest_directory(source_dir, chunks_path, ChunkingConfig(chunk_size=220, overlap=20))
    golden = json.loads(Path("data/golden/qa.json").read_text(encoding="utf-8"))
    chunk_ids = {chunk.chunk_id for chunk in chunks}

    reports = []
    for example in golden:
        results = BM25Retriever(load_chunks_jsonl(chunks_path)).search(example["query"], top_k=2)
        answer = GenerationService(HeuristicLLMClient()).answer(
            example["query"],
            results,
            model_name="heuristic-grounded",
        )
        report = evaluate_answer(
            example["query"],
            answer,
            results,
            reference_chunk_ids=example["relevant_chunk_ids"],
            expected_unanswerable=False,
        )
        reports.append(report)
        assert set(example["relevant_chunk_ids"]) <= chunk_ids

    average_faithfulness = sum(report.faithfulness for report in reports) / len(reports)
    average_citation_support = sum(report.citation_support for report in reports) / len(reports)
    artifact_dir = tmp_path / "artifacts" / "evaluation"
    export_evaluation_report_csv(reports[0], artifact_dir / "report.csv")
    export_evaluation_report_markdown(reports[0], artifact_dir / "report.md")

    assert average_faithfulness >= 0.8
    assert average_citation_support >= 1.0
