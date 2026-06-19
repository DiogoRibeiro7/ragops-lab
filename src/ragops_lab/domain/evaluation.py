"""Evaluation domain model."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field, field_validator

from .document import ensure_aware_timestamp


class EvaluationResult(BaseModel):
    """Evaluation output for a generated answer."""

    context_precision: float = Field(ge=0.0, le=1.0)
    context_recall: float = Field(ge=0.0, le=1.0)
    answer_relevance: float = Field(ge=0.0, le=1.0)
    faithfulness: float = Field(ge=0.0, le=1.0)
    citation_support: float = Field(ge=0.0, le=1.0)
    unsupported_claim_count: int = Field(ge=0)
    refusal_correct: bool | None = Field(default=None)
    notes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return ensure_aware_timestamp(value)
