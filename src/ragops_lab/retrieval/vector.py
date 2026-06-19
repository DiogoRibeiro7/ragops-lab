"""Vector retrieval adapters."""

from __future__ import annotations

import math
from collections import Counter
from typing import Protocol

from ragops_lab.domain import DocumentChunk, RetrievalResult

from .tokenizer import tokenize


class EmbeddingClient(Protocol):
    """Embedding client abstraction."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""

    def embed_query(self, query: str) -> list[float]:
        """Embed a query."""


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity."""
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(
        left_value * right_value for left_value, right_value in zip(left, right, strict=True)
    )
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(0.0, numerator / (left_norm * right_norm))


class FakeEmbeddingClient:
    """Deterministic embedding client used in tests and local demos."""

    def __init__(self, vocabulary: list[str] | None = None) -> None:
        self.vocabulary = vocabulary or []

    def _build_vocabulary(self, texts: list[str]) -> list[str]:
        if self.vocabulary:
            return self.vocabulary
        terms = sorted({term for text in texts for term in tokenize(text)})
        return terms[:256]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vocabulary = self._build_vocabulary(texts)
        return [self._embed_with_vocabulary(text, vocabulary) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        vocabulary = self.vocabulary or self._build_vocabulary([query])
        return self._embed_with_vocabulary(query, vocabulary)

    def _embed_with_vocabulary(self, text: str, vocabulary: list[str]) -> list[float]:
        counts = Counter(tokenize(text))
        return [float(counts.get(term, 0)) for term in vocabulary]


class SentenceTransformerEmbeddingClient:
    """Optional sentence-transformers adapter."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install it to use vector retrieval."
            ) from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, vector)) for vector in vectors]

    def embed_query(self, query: str) -> list[float]:
        vector = self._model.encode([query], normalize_embeddings=True)[0]
        return list(map(float, vector))


class VectorRetriever:
    """In-memory vector retriever."""

    def __init__(self, chunks: list[DocumentChunk], embedding_client: EmbeddingClient) -> None:
        self.chunks = chunks
        self.embedding_client = embedding_client
        self.chunk_vectors = embedding_client.embed_texts([chunk.text for chunk in chunks])

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]:
        query_vector = self.embedding_client.embed_query(query)
        scored: list[tuple[DocumentChunk, float]] = []
        for chunk, vector in zip(self.chunks, self.chunk_vectors, strict=True):
            score = cosine_similarity(query_vector, vector)
            if score > 0.0:
                scored.append((chunk, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            RetrievalResult(
                chunk=chunk,
                score=score,
                rank=rank,
                retrieval_method="vector",
                matched_terms=[],
            )
            for rank, (chunk, score) in enumerate(scored[:top_k], start=1)
        ]
