"""Generated answer domain model."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from .document import validate_identifier


class GeneratedAnswer(BaseModel):
    """Answer returned by the generation layer."""

    question: str = Field(min_length=1)
    answer_text: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)
    model_name: str = Field(min_length=1)
    refusal: bool = Field(default=False)
    grounded: bool = Field(default=False)
    prompt: str | None = Field(default=None)

    @field_validator("question", "answer_text", "model_name")
    @classmethod
    def normalize_text_fields(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be empty.")
        return stripped

    @field_validator("citations")
    @classmethod
    def validate_citations(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        for citation in value:
            identifier = validate_identifier(citation, "citation")
            if identifier not in seen:
                seen.add(identifier)
                normalized.append(identifier)
        return normalized
