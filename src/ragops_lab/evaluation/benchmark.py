"""Dataset-oriented RAG benchmark runner."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from pydantic import BaseModel, Field

from ragops_lab.domain import EvaluationResult, GeneratedAnswer
from ragops_lab.evaluation.service import evaluate_answer
from ragops_lab.generation import GenerationService, HeuristicLLMClient
from ragops_lab.ingestion import ChunkingConfig, ingest_directory
from ragops_lab.retrieval import BM25Retriever
from ragops_lab.retrieval.evaluation import (
    RetrievalGoldenExample,
    recall_at_k,
    reciprocal_rank,
)


class EvaluationCase(BaseModel):
    """Serializable result for one golden-set example."""

    query: str
    retrieved_chunk_ids: list[str]
    relevant_chunk_ids: list[str]
    recall_at_k: float = Field(ge=0.0, le=1.0)
    reciprocal_rank: float = Field(ge=0.0, le=1.0)
    answer: GeneratedAnswer
    evaluation: EvaluationResult


class EvaluationSummary(BaseModel):
    """Aggregate regression metrics and threshold status for one run."""

    case_count: int = Field(ge=0)
    top_k: int = Field(ge=1)
    average_recall_at_k: float = Field(ge=0.0, le=1.0)
    mean_reciprocal_rank: float = Field(ge=0.0, le=1.0)
    average_faithfulness: float = Field(ge=0.0, le=1.0)
    average_citation_support: float = Field(ge=0.0, le=1.0)
    min_faithfulness: float = Field(ge=0.0, le=1.0)
    min_citation_support: float = Field(ge=0.0, le=1.0)
    passed: bool


class BenchmarkRun(BaseModel):
    """One benchmark repeat with its case-level results."""

    run_id: int = Field(ge=1)
    summary: EvaluationSummary
    cases: list[EvaluationCase]


class BenchmarkSummary(BaseModel):
    """Aggregate metrics across repeated benchmark runs."""

    run_count: int = Field(ge=1)
    case_count: int = Field(ge=0)
    top_k: int = Field(ge=1)
    average_recall_at_k: float = Field(ge=0.0, le=1.0)
    mean_reciprocal_rank: float = Field(ge=0.0, le=1.0)
    average_faithfulness: float = Field(ge=0.0, le=1.0)
    lowest_run_faithfulness: float = Field(ge=0.0, le=1.0)
    average_citation_support: float = Field(ge=0.0, le=1.0)
    lowest_run_citation_support: float = Field(ge=0.0, le=1.0)
    min_faithfulness: float = Field(ge=0.0, le=1.0)
    min_citation_support: float = Field(ge=0.0, le=1.0)
    passed: bool


def load_golden_examples(path: Path) -> list[RetrievalGoldenExample]:
    """Load golden retrieval examples from JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [RetrievalGoldenExample.model_validate(example) for example in payload]


def run_evaluation(
    *,
    source_dir: Path,
    golden_path: Path,
    chunks_path: Path,
    top_k: int,
    chunk_size: int,
    overlap: int,
    min_faithfulness: float,
    min_citation_support: float,
) -> tuple[EvaluationSummary, list[EvaluationCase]]:
    """Run one deterministic RAG evaluation pass over a golden dataset."""
    _validate_benchmark_inputs(
        source_dir=source_dir,
        golden_path=golden_path,
        top_k=top_k,
        chunk_size=chunk_size,
        overlap=overlap,
        min_faithfulness=min_faithfulness,
        min_citation_support=min_citation_support,
    )
    chunks = ingest_directory(
        source_dir,
        chunks_path,
        ChunkingConfig(chunk_size=chunk_size, overlap=overlap),
    )
    retriever = BM25Retriever(chunks)
    generation = GenerationService(HeuristicLLMClient())
    golden_examples = load_golden_examples(golden_path)
    available_chunk_ids = {chunk.chunk_id for chunk in chunks}

    cases: list[EvaluationCase] = []
    for example in golden_examples:
        missing_ids = set(example.relevant_chunk_ids) - available_chunk_ids
        if missing_ids:
            raise ValueError(f"Golden example references missing chunk ids: {sorted(missing_ids)}")

        results = retriever.search(example.query, top_k=top_k)
        retrieved_ids = [result.chunk.chunk_id for result in results]
        relevant_ids = set(example.relevant_chunk_ids)
        answer = generation.answer(
            example.query,
            results,
            model_name="heuristic-grounded",
        )
        report = evaluate_answer(
            example.query,
            answer,
            results,
            reference_chunk_ids=example.relevant_chunk_ids,
            expected_unanswerable=False,
        )
        cases.append(
            EvaluationCase(
                query=example.query,
                retrieved_chunk_ids=retrieved_ids,
                relevant_chunk_ids=example.relevant_chunk_ids,
                recall_at_k=recall_at_k(retrieved_ids, relevant_ids),
                reciprocal_rank=reciprocal_rank(retrieved_ids, relevant_ids),
                answer=answer,
                evaluation=report,
            )
        )

    case_count = len(cases)
    divisor = max(case_count, 1)
    average_faithfulness = sum(case.evaluation.faithfulness for case in cases) / divisor
    average_citation_support = sum(case.evaluation.citation_support for case in cases) / divisor
    summary = EvaluationSummary(
        case_count=case_count,
        top_k=top_k,
        average_recall_at_k=sum(case.recall_at_k for case in cases) / divisor,
        mean_reciprocal_rank=sum(case.reciprocal_rank for case in cases) / divisor,
        average_faithfulness=average_faithfulness,
        average_citation_support=average_citation_support,
        min_faithfulness=min_faithfulness,
        min_citation_support=min_citation_support,
        passed=(
            average_faithfulness >= min_faithfulness
            and average_citation_support >= min_citation_support
        ),
    )
    return summary, cases


def run_benchmark(
    *,
    source_dir: Path,
    golden_path: Path,
    output_dir: Path,
    runs: int,
    top_k: int,
    chunk_size: int,
    overlap: int,
    min_faithfulness: float,
    min_citation_support: float,
) -> tuple[BenchmarkSummary, list[BenchmarkRun]]:
    """Run repeated benchmark passes and aggregate their metrics."""
    if runs < 1:
        raise ValueError("runs must be greater than or equal to 1.")
    _validate_benchmark_inputs(
        source_dir=source_dir,
        golden_path=golden_path,
        top_k=top_k,
        chunk_size=chunk_size,
        overlap=overlap,
        min_faithfulness=min_faithfulness,
        min_citation_support=min_citation_support,
    )

    benchmark_runs: list[BenchmarkRun] = []
    for run_id in range(1, runs + 1):
        run_dir = output_dir / f"run-{run_id:03d}"
        summary, cases = run_evaluation(
            source_dir=source_dir,
            golden_path=golden_path,
            chunks_path=run_dir / "chunks.jsonl",
            top_k=top_k,
            chunk_size=chunk_size,
            overlap=overlap,
            min_faithfulness=min_faithfulness,
            min_citation_support=min_citation_support,
        )
        benchmark_runs.append(BenchmarkRun(run_id=run_id, summary=summary, cases=cases))

    divisor = len(benchmark_runs)
    benchmark_summary = BenchmarkSummary(
        run_count=divisor,
        case_count=benchmark_runs[0].summary.case_count if benchmark_runs else 0,
        top_k=top_k,
        average_recall_at_k=sum(
            run.summary.average_recall_at_k for run in benchmark_runs
        )
        / divisor,
        mean_reciprocal_rank=sum(
            run.summary.mean_reciprocal_rank for run in benchmark_runs
        )
        / divisor,
        average_faithfulness=sum(
            run.summary.average_faithfulness for run in benchmark_runs
        )
        / divisor,
        lowest_run_faithfulness=min(run.summary.average_faithfulness for run in benchmark_runs),
        average_citation_support=sum(
            run.summary.average_citation_support for run in benchmark_runs
        )
        / divisor,
        lowest_run_citation_support=min(
            run.summary.average_citation_support for run in benchmark_runs
        ),
        min_faithfulness=min_faithfulness,
        min_citation_support=min_citation_support,
        passed=all(run.summary.passed for run in benchmark_runs),
    )
    return benchmark_summary, benchmark_runs


def write_artifacts(
    summary: EvaluationSummary,
    cases: list[EvaluationCase],
    output_dir: Path,
) -> None:
    """Write one evaluation run's JSON, CSV, and Markdown artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        summary.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (output_dir / "cases.json").write_text(
        json.dumps([case.model_dump(mode="json") for case in cases], indent=2),
        encoding="utf-8",
    )

    with (output_dir / "cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query",
                "retrieved_chunk_ids",
                "relevant_chunk_ids",
                "recall_at_k",
                "reciprocal_rank",
                "faithfulness",
                "citation_support",
                "unsupported_claim_count",
                "refusal_correct",
            ],
        )
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "query": case.query,
                    "retrieved_chunk_ids": " ".join(case.retrieved_chunk_ids),
                    "relevant_chunk_ids": " ".join(case.relevant_chunk_ids),
                    "recall_at_k": f"{case.recall_at_k:.4f}",
                    "reciprocal_rank": f"{case.reciprocal_rank:.4f}",
                    "faithfulness": f"{case.evaluation.faithfulness:.4f}",
                    "citation_support": f"{case.evaluation.citation_support:.4f}",
                    "unsupported_claim_count": case.evaluation.unsupported_claim_count,
                    "refusal_correct": case.evaluation.refusal_correct,
                }
            )

    lines = [
        "# RAG Evaluation Regression Report",
        "",
        f"- Cases: {summary.case_count}",
        f"- Top k: {summary.top_k}",
        f"- Recall@k: {summary.average_recall_at_k:.2f}",
        f"- MRR: {summary.mean_reciprocal_rank:.2f}",
        f"- Faithfulness: {summary.average_faithfulness:.2f}",
        f"- Citation support: {summary.average_citation_support:.2f}",
        f"- Required faithfulness: {summary.min_faithfulness:.2f}",
        f"- Required citation support: {summary.min_citation_support:.2f}",
        f"- Status: {'passed' if summary.passed else 'failed'}",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_benchmark_artifacts(
    summary: BenchmarkSummary,
    runs: list[BenchmarkRun],
    output_dir: Path,
) -> None:
    """Write aggregate and per-run benchmark artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        write_artifacts(run.summary, run.cases, output_dir / f"run-{run.run_id:03d}")
    if len(runs) == 1:
        write_artifacts(runs[0].summary, runs[0].cases, output_dir)

    (output_dir / "benchmark-summary.json").write_text(
        summary.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (output_dir / "benchmark-runs.json").write_text(
        json.dumps(
            [
                {"run_id": run.run_id, "summary": run.summary.model_dump(mode="json")}
                for run in runs
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    with (output_dir / "benchmark-runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "case_count",
                "recall_at_k",
                "mrr",
                "faithfulness",
                "citation_support",
                "passed",
            ],
        )
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "run_id": run.run_id,
                    "case_count": run.summary.case_count,
                    "recall_at_k": f"{run.summary.average_recall_at_k:.4f}",
                    "mrr": f"{run.summary.mean_reciprocal_rank:.4f}",
                    "faithfulness": f"{run.summary.average_faithfulness:.4f}",
                    "citation_support": f"{run.summary.average_citation_support:.4f}",
                    "passed": run.summary.passed,
                }
            )

    lines = [
        "# RAG Benchmark Summary",
        "",
        f"- Runs: {summary.run_count}",
        f"- Cases per run: {summary.case_count}",
        f"- Top k: {summary.top_k}",
        f"- Average recall@k: {summary.average_recall_at_k:.2f}",
        f"- Mean reciprocal rank: {summary.mean_reciprocal_rank:.2f}",
        f"- Average faithfulness: {summary.average_faithfulness:.2f}",
        f"- Lowest run faithfulness: {summary.lowest_run_faithfulness:.2f}",
        f"- Average citation support: {summary.average_citation_support:.2f}",
        f"- Lowest run citation support: {summary.lowest_run_citation_support:.2f}",
        f"- Required faithfulness: {summary.min_faithfulness:.2f}",
        f"- Required citation support: {summary.min_citation_support:.2f}",
        f"- Status: {'passed' if summary.passed else 'failed'}",
    ]
    (output_dir / "benchmark-summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _validate_benchmark_inputs(
    *,
    source_dir: Path,
    golden_path: Path,
    top_k: int,
    chunk_size: int,
    overlap: int,
    min_faithfulness: float,
    min_citation_support: float,
) -> None:
    if not source_dir.exists():
        raise ValueError(f"Source directory not found: {source_dir}")
    if not source_dir.is_dir():
        raise ValueError(f"Source path is not a directory: {source_dir}")
    if not golden_path.exists():
        raise ValueError(f"Golden dataset not found: {golden_path}")
    if top_k < 1:
        raise ValueError("top_k must be greater than or equal to 1.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")
    if not 0.0 <= min_faithfulness <= 1.0:
        raise ValueError("min_faithfulness must be between 0 and 1.")
    if not 0.0 <= min_citation_support <= 1.0:
        raise ValueError("min_citation_support must be between 0 and 1.")
