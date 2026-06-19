"""Chunk domain model."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from .document import validate_identifier


class DocumentChunk(BaseModel):
    """Single retrievable chunk derived from a document."""

    chunk_id: str = Field(description="Stable chunk identifier.")
    document_id: str = Field(description="Parent document identifier.")
    text: str = Field(min_length=1, description="Chunk text.")
    start_offset: int = Field(ge=0, description="Inclusive character start offset.")
    end_offset: int = Field(gt=0, description="Exclusive character end offset.")
    token_count: int = Field(ge=1, description="Approximate token count.")
    source_path: str | None = Field(default=None, description="Source document path.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Chunk metadata.")

    @model_validator(mode="after")
    def validate_chunk(self) -> DocumentChunk:
        self.chunk_id = validate_identifier(self.chunk_id, "chunk_id")
        self.document_id = validate_identifier(self.document_id, "document_id")
        if self.end_offset <= self.start_offset:
            raise ValueError("end_offset must be greater than start_offset.")
        if not self.text.strip():
            raise ValueError("text must not be blank.")
        return self
