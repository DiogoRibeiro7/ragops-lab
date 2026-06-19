"""Document domain model."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

ID_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-")


def validate_identifier(value: str, field_name: str) -> str:
    """Validate stable identifiers used across traces, chunks, and documents."""
    if len(value) < 3:
        raise ValueError(f"{field_name} must be at least 3 characters long.")
    if not value[0].isalnum():
        raise ValueError(f"{field_name} must start with an alphanumeric character.")
    if any(character not in ID_CHARS for character in value):
        raise ValueError(f"{field_name} contains unsupported characters.")
    return value


def ensure_aware_timestamp(value: datetime) -> datetime:
    """Normalize timestamps to timezone-aware UTC datetimes."""
    if value.tzinfo is None:
        raise ValueError("timestamp must include timezone information.")
    return value.astimezone(UTC)


class Document(BaseModel):
    """Normalized source document."""

    document_id: str = Field(description="Stable identifier for the source document.")
    title: str = Field(min_length=1, description="Human-readable title.")
    text: str = Field(min_length=1, description="Normalized document text.")
    source_path: str | None = Field(default=None, description="Original filesystem path.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Source metadata.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        return validate_identifier(value, "document_id")

    @field_validator("title", "text")
    @classmethod
    def strip_non_empty_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be empty.")
        return stripped

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return ensure_aware_timestamp(value)
