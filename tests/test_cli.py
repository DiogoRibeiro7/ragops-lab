from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch
from typer.testing import CliRunner

from ragops_lab import cli
from ragops_lab.cli import app
from ragops_lab.ingestion import ChunkingConfig, ingest_directory


def test_cli_ask_reports_missing_chunks_file(tmp_path: Path) -> None:
    runner = CliRunner()
    missing_path = tmp_path / "missing.jsonl"

    result = runner.invoke(app, ["ask", "What happened?", "--chunks", str(missing_path)])

    assert result.exit_code == 1
    assert "Chunks file not found" in result.output


def test_cli_ingest_reports_missing_input_directory(tmp_path: Path) -> None:
    runner = CliRunner()
    missing_dir = tmp_path / "missing"
    out_path = tmp_path / "chunks.jsonl"

    result = runner.invoke(app, ["ingest", str(missing_dir), "--out", str(out_path)])

    assert result.exit_code == 1
    assert "Input directory not found" in result.output


def test_cli_builds_index_and_asks_with_vector_mode(tmp_path: Path) -> None:
    runner = CliRunner()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "metrics.txt").write_text(
        "Faithfulness and citation support are critical RAG metrics.",
        encoding="utf-8",
    )
    chunks_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "vector_index.json"
    ingest_directory(raw_dir, chunks_path, ChunkingConfig(chunk_size=120, overlap=10))

    index_result = runner.invoke(
        app,
        ["index", "--chunks", str(chunks_path), "--out", str(index_path)],
    )
    ask_result = runner.invoke(
        app,
        [
            "ask",
            "What metrics matter?",
            "--chunks",
            str(chunks_path),
            "--index-path",
            str(index_path),
            "--profile",
            "vector",
            "--mode",
            "vector",
        ],
    )

    assert index_result.exit_code == 0
    assert index_path.exists()
    assert ask_result.exit_code == 0
    assert "Faithfulness" in ask_result.output


def test_cli_reports_unknown_retrieval_profile(tmp_path: Path) -> None:
    runner = CliRunner()
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text("", encoding="utf-8")

    result = runner.invoke(
        app,
        ["ask", "What happened?", "--chunks", str(chunks_path), "--profile", "missing"],
    )

    assert result.exit_code == 1
    assert "Unknown retrieval profile" in result.output


def test_cli_reports_missing_llm_provider_configuration(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runner = CliRunner()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "apollo.txt").write_text("Apollo 11 landed on the Moon.", encoding="utf-8")
    chunks_path = tmp_path / "chunks.jsonl"
    ingest_directory(raw_dir, chunks_path, ChunkingConfig(chunk_size=120, overlap=10))
    monkeypatch.setenv("RAGOPS_LLM_PROVIDER", "openai-compatible")
    monkeypatch.delenv("RAGOPS_LLM_ENDPOINT", raising=False)

    result = runner.invoke(app, ["ask", "Which mission landed?", "--chunks", str(chunks_path)])

    assert result.exit_code == 1
    assert "RAGOPS_LLM_ENDPOINT" in result.output


def test_cli_index_reports_embedding_provider_errors(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runner = CliRunner()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "metrics.txt").write_text(
        "Faithfulness and citation support are critical RAG metrics.",
        encoding="utf-8",
    )
    chunks_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "vector_index.json"
    ingest_directory(raw_dir, chunks_path, ChunkingConfig(chunk_size=120, overlap=10))

    def fail_embedding_provider(_: object) -> object:
        raise RuntimeError("embedding provider is unavailable")

    monkeypatch.setattr(cli, "build_embedding_client", fail_embedding_provider)

    result = runner.invoke(
        app,
        ["index", "--chunks", str(chunks_path), "--out", str(index_path)],
    )

    assert result.exit_code == 1
    assert "embedding provider is unavailable" in result.output
