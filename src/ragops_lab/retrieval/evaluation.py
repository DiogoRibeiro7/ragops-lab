"""Retrieval evaluation metrics."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .lexical import BM25Retriever


class RetrievalGoldenExample(BaseModel):
    """Golden retrieval evaluation example."""

    query: str = Field(min_length=1)
    relevant_chunk_ids: list[str] = Field(min_length=1)


class RetrievalEvaluationReport(BaseModel):
    """Aggregate retrieval metrics."""

    recall_at_k: float = Field(ge=0.0, le=1.0)
    mean_reciprocal_rank: float = Field(ge=0.0, le=1.0)


def recall_at_k(result_ids: list[str], relevant_ids: set[str]) -> float:
    """Compute recall over known relevant chunks."""
    if not relevant_ids:
        return 0.0
    return len(set(result_ids) & relevant_ids) / len(relevant_ids)


def reciprocal_rank(result_ids: list[str], relevant_ids: set[str]) -> float:
    """Compute reciprocal rank."""
    for index, chunk_id in enumerate(result_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / index
    return 0.0


def evaluate_retrieval(
    retriever: BM25Retriever,
    golden_dataset: list[RetrievalGoldenExample],
    *,
    top_k: int = 5,
) -> RetrievalEvaluationReport:
    """Evaluate retrieval quality over a small golden set."""
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    for example in golden_dataset:
        results = retriever.search(example.query, top_k=top_k)
        result_ids = [result.chunk.chunk_id for result in results]
        relevant_ids = set(example.relevant_chunk_ids)
        recalls.append(recall_at_k(result_ids, relevant_ids))
        reciprocal_ranks.append(reciprocal_rank(result_ids, relevant_ids))
    divisor = max(len(golden_dataset), 1)
    return RetrievalEvaluationReport(
        recall_at_k=sum(recalls) / divisor,
        mean_reciprocal_rank=sum(reciprocal_ranks) / divisor,
    )
