"""Trace storage."""

from __future__ import annotations

import json
from pathlib import Path

from ragops_lab.domain import RagTrace


class JsonlTraceStore:
    """Simple append-only trace store."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def save(self, trace: RagTrace) -> None:
        """Append a trace to storage."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(trace.model_dump_json())
            handle.write("\n")

    def list(self) -> list[RagTrace]:
        """List all traces."""
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            return [RagTrace.model_validate(json.loads(line)) for line in handle if line.strip()]

    def get(self, trace_id: str) -> RagTrace | None:
        """Retrieve a trace by id."""
        for trace in self.list():
            if trace.trace_id == trace_id:
                return trace
        return None
