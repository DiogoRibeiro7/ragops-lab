"""Persistent local retrieval indexes."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from ragops_lab.domain import DocumentChunk

from .vector import EmbeddingClient, FakeEmbeddingClient, VectorRetriever, build_embedding_client


class LocalVectorIndex(BaseModel):
    """Serializable local vector index for deterministic offline retrieval."""

    index_version: int = Field(default=1)
    embedding_provider: str = Field(default="fake")
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
    def build(
        cls,
        chunks: list[DocumentChunk],
        embedding_client: EmbeddingClient | None = None,
        *,
        embedding_provider: str = "fake",
        embedding_model: str = "fake-bow",
    ) -> LocalVectorIndex:
        """Build a deterministic local vector index from chunks."""
        client = embedding_client or FakeEmbeddingClient()
        vectors = client.embed_texts([chunk.text for chunk in chunks])
        vocabulary = client.vocabulary if isinstance(client, FakeEmbeddingClient) else []
        return cls(
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            vocabulary=vocabulary,
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

    def as_retriever(self, embedding_client: EmbeddingClient | None = None) -> VectorRetriever:
        """Rehydrate the index as a vector retriever."""
        client = embedding_client
        if client is None and self.embedding_provider == "fake":
            client = FakeEmbeddingClient(self.vocabulary)
        if client is None:
            from ragops_lab.config import EmbeddingSettings

            client = build_embedding_client(
                EmbeddingSettings(provider=self.embedding_provider, model=self.embedding_model)
            )
        return VectorRetriever(
            self.chunks,
            client,
            chunk_vectors=self.vectors,
        )
