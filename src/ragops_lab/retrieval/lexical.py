"""Lexical BM25-style retrieval."""

from __future__ import annotations

import math
from collections import Counter

from ragops_lab.domain import DocumentChunk, RetrievalResult

from .tokenizer import tokenize


class BM25Retriever:
    """In-memory lexical retriever."""

    def __init__(self, chunks: list[DocumentChunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.corpus_tokens = [tokenize(chunk.text) for chunk in chunks]
        self.document_frequencies = self._build_document_frequencies()
        self.avg_doc_length = sum(len(tokens) for tokens in self.corpus_tokens) / max(
            len(self.corpus_tokens), 1
        )

    def _build_document_frequencies(self) -> Counter[str]:
        frequencies: Counter[str] = Counter()
        for tokens in self.corpus_tokens:
            frequencies.update(set(tokens))
        return frequencies

    def _inverse_document_frequency(self, token: str) -> float:
        frequency = self.document_frequencies.get(token, 0)
        numerator = len(self.chunks) - frequency + 0.5
        denominator = frequency + 0.5
        return math.log1p(max(numerator, 0.0) / denominator)

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]:
        """Search the indexed chunks."""
        query_tokens = tokenize(query)
        query_terms = set(query_tokens)
        scored: list[tuple[DocumentChunk, float, list[str]]] = []
        for chunk, tokens in zip(self.chunks, self.corpus_tokens, strict=True):
            term_counts = Counter(tokens)
            document_length = len(tokens) or 1
            score = 0.0
            for term in query_terms:
                term_frequency = term_counts.get(term, 0)
                if term_frequency == 0:
                    continue
                idf = self._inverse_document_frequency(term)
                numerator = term_frequency * (self.k1 + 1)
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * document_length / max(self.avg_doc_length, 1.0)
                )
                score += idf * numerator / denominator
            if score > 0.0:
                matched_terms = sorted(term for term in query_terms if term in term_counts)
                scored.append((chunk, score, matched_terms))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            RetrievalResult(
                chunk=chunk,
                score=score,
                rank=rank,
                retrieval_method="lexical",
                matched_terms=matched_terms,
            )
            for rank, (chunk, score, matched_terms) in enumerate(scored[:top_k], start=1)
        ]
