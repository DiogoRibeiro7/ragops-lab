"""Retrieval domain models."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from .chunk import DocumentChunk


class RetrievalResult(BaseModel):
    """Result returned by a retriever."""

    chunk: DocumentChunk
    score: float = Field(ge=0.0, description="Ranking score.")
    rank: int = Field(ge=1, description="1-based rank.")
    retrieval_method: str = Field(min_length=1, description="lexical, vector, or hybrid.")
    matched_terms: list[str] = Field(default_factory=list)

    @field_validator("retrieval_method")
    @classmethod
    def normalize_method(cls, value: str) -> str:
        stripped = value.strip().lower()
        if not stripped:
            raise ValueError("retrieval_method must not be empty.")
        return stripped
