"""Hybrid retrieval."""

from __future__ import annotations

from ragops_lab.domain import RetrievalResult

from .lexical import BM25Retriever
from .vector import VectorRetriever


class HybridRetriever:
    """Combine lexical and vector retrieval scores."""

    def __init__(
        self,
        lexical_retriever: BM25Retriever,
        vector_retriever: VectorRetriever,
        *,
        lexical_weight: float = 0.5,
        vector_weight: float = 0.5,
    ) -> None:
        self.lexical_retriever = lexical_retriever
        self.vector_retriever = vector_retriever
        self.lexical_weight = lexical_weight
        self.vector_weight = vector_weight

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]:
        lexical_results = self.lexical_retriever.search(query, top_k=top_k * 2)
        vector_results = self.vector_retriever.search(query, top_k=top_k * 2)
        lexical_scores = {result.chunk.chunk_id: result.score for result in lexical_results}
        vector_scores = {result.chunk.chunk_id: result.score for result in vector_results}
        lexical_max = max(lexical_scores.values(), default=1.0)
        vector_max = max(vector_scores.values(), default=1.0)
        by_chunk_id = {
            result.chunk.chunk_id: result.chunk for result in [*lexical_results, *vector_results]
        }
        combined: list[tuple[str, float]] = []
        for chunk_id in by_chunk_id:
            lexical_score = lexical_scores.get(chunk_id, 0.0) / lexical_max
            vector_score = vector_scores.get(chunk_id, 0.0) / vector_max
            final_score = lexical_score * self.lexical_weight + vector_score * self.vector_weight
            if final_score > 0.0:
                combined.append((chunk_id, final_score))
        combined.sort(key=lambda item: item[1], reverse=True)
        return [
            RetrievalResult(
                chunk=by_chunk_id[chunk_id],
                score=score,
                rank=rank,
                retrieval_method="hybrid",
                matched_terms=[],
            )
            for rank, (chunk_id, score) in enumerate(combined[:top_k], start=1)
        ]
