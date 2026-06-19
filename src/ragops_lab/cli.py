"""Command line interface for the project."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import RuntimeSettings
from .evaluation import evaluate_answer
from .generation import GenerationService, HeuristicLLMClient
from .ingestion import ChunkingConfig, ingest_directory, load_chunks_jsonl
from .retrieval import BM25Retriever

app = typer.Typer(help="Portfolio project command line interface.")
console = Console()


@app.command()
def info() -> None:
    """Print validated runtime settings."""
    settings = RuntimeSettings()
    console.print(settings.model_dump())


@app.command()
def ingest(
    input_dir: Path,
    out: Path = Path("data/processed/chunks.jsonl"),
    chunk_size: int = 500,
    overlap: int = 50,
) -> None:
    """Ingest documents into chunk JSONL."""
    chunks = ingest_directory(
        input_dir,
        out,
        config=ChunkingConfig(chunk_size=chunk_size, overlap=overlap),
    )
    console.print({"chunks_written": len(chunks), "out": str(out)})


@app.command()
def ask(
    question: str,
    chunks: Path = Path("data/processed/chunks.jsonl"),
    top_k: int = 3,
) -> None:
    """Ask a grounded question over ingested chunks."""
    chunk_list = load_chunks_jsonl(chunks)
    results = BM25Retriever(chunk_list).search(question, top_k=top_k)
    answer = GenerationService(HeuristicLLMClient()).answer(
        question, results, model_name="heuristic-grounded"
    )
    evaluation = evaluate_answer(question, answer, results)
    table = Table(title="RAG Answer")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Question", question)
    table.add_row("Answer", answer.answer_text)
    table.add_row("Citations", ", ".join(answer.citations) or "none")
    table.add_row("Faithfulness", f"{evaluation.faithfulness:.2f}")
    table.add_row("Citation support", f"{evaluation.citation_support:.2f}")
    console.print(table)


if __name__ == "__main__":
    app()
