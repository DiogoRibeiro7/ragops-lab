from __future__ import annotations

import csv
from pathlib import Path

from ragops_lab.cli import app
from ragops_lab.ingestion import (
    ChunkingConfig,
    discover_documents,
    ingest_directory,
    load_chunks_jsonl,
)
from typer.testing import CliRunner


def test_ingestion_supports_text_markdown_and_csv(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "note.txt").write_text("hello world", encoding="utf-8")
    (raw_dir / "guide.md").write_text("# Title\nrag metrics", encoding="utf-8")
    with (raw_dir / "facts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["topic", "value"])
        writer.writeheader()
        writer.writerow({"topic": "mission", "value": "Apollo 11"})

    documents = discover_documents(raw_dir)
    output_path = tmp_path / "chunks.jsonl"
    chunks = ingest_directory(raw_dir, output_path, ChunkingConfig(chunk_size=30, overlap=5))

    assert len(documents) == 3
    assert output_path.exists()
    assert len(load_chunks_jsonl(output_path)) == len(chunks)


def test_cli_ingest_writes_chunks(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "doc.txt").write_text("Apollo 11 landed on the Moon.", encoding="utf-8")
    output_path = tmp_path / "processed" / "chunks.jsonl"

    result = CliRunner().invoke(app, ["ingest", str(raw_dir), "--out", str(output_path)])

    assert result.exit_code == 0
    assert output_path.exists()
