"""Evaluation domain model."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field, field_validator

from .document import ensure_aware_timestamp


class ClaimSupportResult(BaseModel):
    """Claim-level evidence matching result."""

    claim: str = Field(min_length=1)
    supported: bool
    score: float = Field(ge=0.0, le=1.0)
    evidence_chunk_id: str | None = Field(default=None)
    matched_terms: list[str] = Field(default_factory=list)
    missing_terms: list[str] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    """Evaluation output for a generated answer."""

    context_precision: float = Field(ge=0.0, le=1.0)
    context_recall: float = Field(ge=0.0, le=1.0)
    answer_relevance: float = Field(ge=0.0, le=1.0)
    faithfulness: float = Field(ge=0.0, le=1.0)
    citation_support: float = Field(ge=0.0, le=1.0)
    unsupported_claim_count: int = Field(ge=0)
    claim_count: int = Field(default=0, ge=0)
    supported_claim_count: int = Field(default=0, ge=0)
    unsupported_claims: list[str] = Field(default_factory=list)
    claim_support: list[ClaimSupportResult] = Field(default_factory=list)
    refusal_correct: bool | None = Field(default=None)
    notes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return ensure_aware_timestamp(value)
