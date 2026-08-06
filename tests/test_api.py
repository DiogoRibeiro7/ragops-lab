from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from ragops_lab import __version__
from ragops_lab.api.app import SETTINGS, TRACE_STORE, app
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


def test_api_ingest_and_evaluate_endpoints(tmp_path: Path) -> None:
    client = TestClient(app)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "apollo.txt").write_text(
        "Apollo 11 landed humans on the Moon.",
        encoding="utf-8",
    )
    chunks_path = tmp_path / "chunks.jsonl"

    ingest_response = client.post(
        "/ingest",
        json={
            "input_dir": str(raw_dir),
            "out_path": str(chunks_path),
            "chunk_size": 120,
            "overlap": 10,
        },
    )
    evaluate_response = client.post(
        "/evaluate",
        json={
            "question": "Which mission landed humans on the Moon?",
            "answer_text": "Apollo 11 landed humans on the Moon.",
            "citations": ["apollo:0"],
            "chunks_path": str(chunks_path),
            "retrieved_chunk_ids": ["apollo:0"],
            "reference_chunk_ids": ["apollo:0"],
        },
    )

    assert ingest_response.status_code == 200
    assert ingest_response.json()["chunks_written"] == 1
    assert evaluate_response.status_code == 200
    assert evaluate_response.json()["citation_support"] == 1.0


def test_api_uses_configured_default_chunk_path() -> None:
    assert app.version == __version__
    assert SETTINGS.paths.chunk_path.as_posix() == "data/processed/chunks.jsonl"


def test_api_rejects_top_k_above_runtime_limit(tmp_path: Path) -> None:
    client = TestClient(app)
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text("", encoding="utf-8")

    response = client.post(
        "/search",
        json={
            "query": "moon landing",
            "chunks_path": str(chunks_path),
            "top_k": SETTINGS.api_max_top_k + 1,
        },
    )

    assert response.status_code == 422
    assert "top_k" in response.text


def test_api_rejects_overlong_query() -> None:
    client = TestClient(app)
    response = client.post(
        "/search",
        json={
            "query": "x" * (SETTINGS.api_max_query_chars + 1),
            "chunks_path": "missing.jsonl",
        },
    )

    assert response.status_code == 422
    assert "query" in response.text


def test_api_rejects_unsupported_retrieval_mode(tmp_path: Path) -> None:
    client = TestClient(app)
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text("", encoding="utf-8")

    response = client.post(
        "/search",
        json={"query": "moon landing", "chunks_path": str(chunks_path), "mode": "unknown"},
    )

    assert response.status_code == 400
    assert "Unsupported retrieval mode" in response.text


def test_api_reports_missing_chunks_file() -> None:
    client = TestClient(app)
    response = client.post(
        "/search",
        json={"query": "moon landing", "chunks_path": "missing/chunks.jsonl"},
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "resource_not_found"
    assert "missing" in response.json()["error"]["message"]


def test_api_reports_missing_ingest_directory(tmp_path: Path) -> None:
    client = TestClient(app)
    missing_dir = tmp_path / "missing"
    response = client.post(
        "/ingest",
        json={"input_dir": str(missing_dir), "out_path": str(tmp_path / "chunks.jsonl")},
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "resource_not_found"
    assert "Input directory not found" in response.json()["error"]["message"]


def test_api_reports_missing_manual_evaluation_chunk_ids(tmp_path: Path) -> None:
    client = TestClient(app)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "apollo.txt").write_text("Apollo 11 landed on the Moon.", encoding="utf-8")
    chunks_path = tmp_path / "chunks.jsonl"
    ingest_directory(raw_dir, chunks_path, ChunkingConfig(chunk_size=120, overlap=10))

    response = client.post(
        "/evaluate",
        json={
            "question": "Which mission landed on the Moon?",
            "answer_text": "Apollo 11 landed on the Moon.",
            "chunks_path": str(chunks_path),
            "retrieved_chunk_ids": ["missing:0"],
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
    assert "Retrieved chunk ids not found" in response.json()["error"]["message"]


def test_api_rejects_oversized_request_body() -> None:
    client = TestClient(app)
    response = client.post(
        "/search",
        content=b"x" * (SETTINGS.api_max_request_bytes + 1),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413
    assert "Request body is too large" in response.text
