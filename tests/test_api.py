from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from ragops_lab.api.app import TRACE_STORE, app
from ragops_lab.ingestion import ChunkingConfig, ingest_directory


def test_api_end_to_end(tmp_path: Path) -> None:
    client = TestClient(app)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "apollo.txt").write_text(
        "Apollo 11 was the first mission to land on the Moon in July 1969.",
        encoding="utf-8",
    )
    chunks_path = tmp_path / "chunks.jsonl"
    ingest_directory(raw_dir, chunks_path, ChunkingConfig(chunk_size=120, overlap=10))
    TRACE_STORE.path = tmp_path / "traces.jsonl"

    search_response = client.post(
        "/search", json={"query": "moon landing mission", "chunks_path": str(chunks_path)}
    )
    ask_response = client.post(
        "/ask", json={"query": "Which mission landed on the Moon?", "chunks_path": str(chunks_path)}
    )

    assert search_response.status_code == 200
    assert ask_response.status_code == 200
    trace_id = ask_response.json()["trace_id"]
    trace_response = client.get(f"/traces/{trace_id}")

    assert trace_response.status_code == 200
    assert trace_response.json()["trace_id"] == trace_id
