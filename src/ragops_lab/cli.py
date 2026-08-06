"""Command line interface for the project."""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import typer
from rich.console import Console
from rich.table import Table

from .config import RuntimeSettings
from .evaluation import evaluate_answer
from .generation import GenerationService, HeuristicLLMClient
from .ingestion import ChunkingConfig, ingest_directory, load_chunks_jsonl
from .retrieval import BM25Retriever, HybridRetriever, LocalVectorIndex

app = typer.Typer(help="Portfolio project command line interface.")
console = Console()


@app.command()
def info() -> None:
    """Print validated runtime settings."""
    settings = RuntimeSettings.from_env()
    console.print(settings.model_dump())


def _fail(message: str) -> NoReturn:
    console.print(f"Error: {message}", style="red")
    raise typer.Exit(1)


@app.command()
def ingest(
    input_dir: Path,
    out: Path | None = None,
    chunk_size: int = 500,
    overlap: int = 50,
) -> None:
    """Ingest documents into chunk JSONL."""
    settings = RuntimeSettings.from_env()
    output_path = out or settings.paths.chunk_path
    if not input_dir.exists():
        _fail(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        _fail(f"Input path is not a directory: {input_dir}")
    try:
        chunks = ingest_directory(
            input_dir,
            output_path,
            config=ChunkingConfig(chunk_size=chunk_size, overlap=overlap),
        )
    except ValueError as exc:
        _fail(str(exc))
    console.print({"chunks_written": len(chunks), "out": str(output_path)})


@app.command()
def index(
    chunks: Path | None = None,
    out: Path | None = None,
) -> None:
    """Build a persistent local vector index from chunk JSONL."""
    settings = RuntimeSettings.from_env()
    chunks_path = chunks or settings.paths.chunk_path
    output_path = out or settings.paths.vector_index_path
    if not chunks_path.exists():
        _fail(f"Chunks file not found: {chunks_path}")
    chunk_list = load_chunks_jsonl(chunks_path)
    LocalVectorIndex.build(chunk_list).save(output_path)
    console.print({"chunks_indexed": len(chunk_list), "out": str(output_path)})


@app.command()
def ask(
    question: str,
    chunks: Path | None = None,
    index_path: Path | None = None,
    profile: str = "lexical",
    mode: str | None = None,
    top_k: int | None = None,
    lexical_weight: float | None = None,
    vector_weight: float | None = None,
) -> None:
    """Ask a grounded question over ingested chunks."""
    settings = RuntimeSettings.from_env()
    try:
        retrieval = settings.resolve_retrieval_profile(
            profile,
            mode=mode,
            top_k=top_k,
            lexical_weight=lexical_weight,
            vector_weight=vector_weight,
        )
    except ValueError as exc:
        _fail(str(exc))
    chunks_path = chunks or settings.paths.chunk_path
    vector_index_path = index_path or settings.paths.vector_index_path
    if retrieval.mode in {"lexical", "hybrid"} and not chunks_path.exists():
        _fail(f"Chunks file not found: {chunks_path}")
    if retrieval.mode in {"vector", "hybrid"} and not vector_index_path.exists():
        _fail(f"Vector index not found: {vector_index_path}")
    if retrieval.mode == "lexical":
        chunk_list = load_chunks_jsonl(chunks_path)
        results = BM25Retriever(chunk_list).search(question, top_k=retrieval.top_k)
    elif retrieval.mode == "vector":
        results = LocalVectorIndex.load(vector_index_path).as_retriever().search(
            question,
            top_k=retrieval.top_k,
        )
    else:
        chunk_list = load_chunks_jsonl(chunks_path)
        lexical = BM25Retriever(chunk_list)
        vector = LocalVectorIndex.load(vector_index_path).as_retriever()
        results = HybridRetriever(
            lexical,
            vector,
            lexical_weight=retrieval.lexical_weight,
            vector_weight=retrieval.vector_weight,
        ).search(question, top_k=retrieval.top_k)
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
