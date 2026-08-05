from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragops_lab.cli import app
from ragops_lab.domain import DocumentChunk, RetrievalResult
from ragops_lab.generation import FakeLLMClient, GenerationService
from ragops_lab.ingestion import save_chunks_jsonl


def _results() -> list[RetrievalResult]:
    chunk = DocumentChunk(
        chunk_id="apollo:0",
        document_id="apollo",
        text="Apollo 11 was the first mission to land on the Moon.",
        start_offset=0,
        end_offset=55,
        token_count=11,
    )
    return [RetrievalResult(chunk=chunk, score=1.0, rank=1, retrieval_method="lexical")]


def test_generation_accepts_valid_citations() -> None:
    response = json.dumps(
        {
            "answer_text": "Apollo 11 was the first mission to land on the Moon.",
            "citations": ["apollo:0"],
            "refusal": False,
        }
    )
    answer = GenerationService(FakeLLMClient(response)).answer(
        "Which mission first landed on the Moon?",
        _results(),
        model_name="fake",
    )

    assert answer.citations == ["apollo:0"]
    assert answer.grounded is True


def test_generation_rejects_unknown_citations() -> None:
    response = json.dumps(
        {"answer_text": "Apollo 11.", "citations": ["unknown:0"], "refusal": False}
    )

    with pytest.raises(ValueError):
        GenerationService(FakeLLMClient(response)).answer(
            "Which mission first landed on the Moon?",
            _results(),
            model_name="fake",
        )


def test_cli_ask_uses_saved_chunks(tmp_path: Path) -> None:
    output_path = tmp_path / "chunks.jsonl"
    save_chunks_jsonl([result.chunk for result in _results()], output_path)

    result = CliRunner().invoke(
        app,
        ["ask", "Which mission first landed on the Moon?", "--chunks", str(output_path)],
    )

    assert result.exit_code == 0
    assert "Apollo 11" in result.stdout
