from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from ragops_lab.cli import app


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
