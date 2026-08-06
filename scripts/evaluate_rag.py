"""Run the deterministic RAG evaluation regression gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pydantic import BaseModel, Field

from ragops_lab.domain import EvaluationResult, GeneratedAnswer
from ragops_lab.evaluation import evaluate_answer
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
    """Aggregate regression metrics and threshold status."""

    case_count: int = Field(ge=0)
    top_k: int = Field(ge=1)
    average_recall_at_k: float = Field(ge=0.0, le=1.0)
    mean_reciprocal_rank: float = Field(ge=0.0, le=1.0)
    average_faithfulness: float = Field(ge=0.0, le=1.0)
    average_citation_support: float = Field(ge=0.0, le=1.0)
    min_faithfulness: float = Field(ge=0.0, le=1.0)
    min_citation_support: float = Field(ge=0.0, le=1.0)
    passed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/sample_documents"))
    parser.add_argument("--golden-path", type=Path, default=Path("data/golden/qa.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--overlap", type=int, default=60)
    parser.add_argument("--min-faithfulness", type=float, default=0.80)
    parser.add_argument("--min-citation-support", type=float, default=1.00)
    return parser.parse_args()


def load_golden_examples(path: Path) -> list[RetrievalGoldenExample]:
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


def write_artifacts(
    summary: EvaluationSummary,
    cases: list[EvaluationCase],
    output_dir: Path,
) -> None:
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


def main() -> int:
    args = parse_args()
    summary, cases = run_evaluation(
        source_dir=args.source_dir,
        golden_path=args.golden_path,
        chunks_path=args.output_dir / "chunks.jsonl",
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        min_faithfulness=args.min_faithfulness,
        min_citation_support=args.min_citation_support,
    )
    write_artifacts(summary, cases, args.output_dir)
    print(summary.model_dump_json(indent=2))
    return 0 if summary.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
