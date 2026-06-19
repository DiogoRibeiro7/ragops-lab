"""Trace domain model."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field, field_validator

from .answer import GeneratedAnswer
from .document import ensure_aware_timestamp, validate_identifier
from .evaluation import EvaluationResult
from .retrieval import RetrievalResult


class RagTrace(BaseModel):
    """Persisted trace for a retrieval and answer generation request."""

    trace_id: str = Field(description="Stable trace identifier.")
    question: str = Field(min_length=1)
    retrieved_chunks: list[RetrievalResult] = Field(default_factory=list)
    answer: GeneratedAnswer
    evaluation: EvaluationResult | None = Field(default=None)
    model_name: str = Field(min_length=1)
    latency_ms: float = Field(ge=0.0)
    token_estimate: int = Field(ge=0, default=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("trace_id")
    @classmethod
    def validate_trace_id(cls, value: str) -> str:
        return validate_identifier(value, "trace_id")

    @field_validator("question", "model_name")
    @classmethod
    def normalize_text_fields(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be empty.")
        return stripped

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return ensure_aware_timestamp(value)
