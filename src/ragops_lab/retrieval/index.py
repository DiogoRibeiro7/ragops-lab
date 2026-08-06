"""Persistent local retrieval indexes."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from ragops_lab.domain import DocumentChunk

from .vector import FakeEmbeddingClient, VectorRetriever


class LocalVectorIndex(BaseModel):
    """Serializable local vector index for deterministic offline retrieval."""

    index_version: int = Field(default=1)
    embedding_model: str = Field(default="fake-bow")
    vocabulary: list[str] = Field(default_factory=list)
    chunks: list[DocumentChunk] = Field(default_factory=list)
    vectors: list[list[float]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_vector_shape(self) -> LocalVectorIndex:
        """Ensure persisted vectors match chunk and vocabulary dimensions."""
        if len(self.chunks) != len(self.vectors):
            raise ValueError("Local vector index must contain one vector per chunk.")
        if self.vocabulary:
            expected_width = len(self.vocabulary)
            invalid_widths = [
                len(vector) for vector in self.vectors if len(vector) != expected_width
            ]
            if invalid_widths:
                raise ValueError("Local vector index vectors must match vocabulary size.")
        return self

    @classmethod
    def build(cls, chunks: list[DocumentChunk]) -> LocalVectorIndex:
        """Build a deterministic local vector index from chunks."""
        embedding_client = FakeEmbeddingClient()
        vectors = embedding_client.embed_texts([chunk.text for chunk in chunks])
        return cls(
            vocabulary=embedding_client.vocabulary,
            chunks=chunks,
            vectors=vectors,
        )

    def save(self, path: Path) -> None:
        """Persist the index as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> LocalVectorIndex:
        """Load a persisted local vector index."""
        if not path.exists():
            raise FileNotFoundError(f"Vector index not found: {path}")
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def as_retriever(self) -> VectorRetriever:
        """Rehydrate the index as a vector retriever."""
        return VectorRetriever(
            self.chunks,
            FakeEmbeddingClient(self.vocabulary),
            chunk_vectors=self.vectors,
        )
