"""Reusable ingestion pipeline."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ragops_lab.domain import Document, DocumentChunk

from ..retrieval.tokenizer import tokenize


class PdfExtractor(Protocol):
    """Optional PDF extraction dependency boundary."""

    def extract_text(self, path: Path) -> str:
        """Extract text from a PDF file."""


@dataclass(frozen=True)
class ChunkingConfig:
    """Chunking controls."""

    chunk_size: int = 500
    overlap: int = 50
    strategy: str = "chars"


SUPPORTED_CHUNKING_STRATEGIES = frozenset({"chars"})


def slugify(value: str) -> str:
    """Create stable ASCII-ish identifiers from filenames."""
    normalized = "".join(character.lower() if character.isalnum() else "-" for character in value)
    collapsed = "-".join(part for part in normalized.split("-") if part)
    return collapsed or "document"


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_csv_file(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[str] = []
        for index, row in enumerate(reader, start=1):
            parts = [f"{key}: {value or ''}".strip() for key, value in row.items()]
            rows.append(f"Row {index}: " + " | ".join(parts))
    return "\n".join(rows)


def load_document(path: Path, pdf_extractor: PdfExtractor | None = None) -> Document:
    """Load a single supported document."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        text = _read_text_file(path)
    elif suffix == ".csv":
        text = _read_csv_file(path)
    elif suffix == ".pdf":
        if pdf_extractor is None:
            raise ValueError("PDF ingestion requires a PdfExtractor implementation.")
        text = pdf_extractor.extract_text(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    title = path.stem.replace("_", " ").replace("-", " ").strip().title()
    return Document(
        document_id=slugify(path.stem),
        title=title or path.stem,
        text=text,
        source_path=str(path),
        metadata={"suffix": suffix},
    )


def discover_documents(
    input_dir: Path, pdf_extractor: PdfExtractor | None = None
) -> list[Document]:
    """Load all supported documents from a directory tree."""
    documents: list[Document] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".txt", ".md", ".csv", ".pdf"}:
            continue
        documents.append(load_document(path, pdf_extractor=pdf_extractor))
    return documents


def chunk_document(document: Document, config: ChunkingConfig | None = None) -> list[DocumentChunk]:
    """Split a document into reusable chunks."""
    chunking = config or ChunkingConfig()
    if chunking.chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunking.overlap < 0 or chunking.overlap >= chunking.chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")
    if chunking.strategy not in SUPPORTED_CHUNKING_STRATEGIES:
        supported = ", ".join(sorted(SUPPORTED_CHUNKING_STRATEGIES))
        raise ValueError(
            f"Unsupported chunking strategy: {chunking.strategy}. Supported: {supported}."
        )

    text = document.text
    step = chunking.chunk_size - chunking.overlap
    chunks: list[DocumentChunk] = []
    index = 0
    for start in range(0, len(text), step):
        end = min(len(text), start + chunking.chunk_size)
        content = text[start:end].strip()
        if not content:
            continue
        chunks.append(
            DocumentChunk(
                chunk_id=f"{document.document_id}:{index}",
                document_id=document.document_id,
                text=content,
                start_offset=start,
                end_offset=end,
                token_count=max(1, len(tokenize(content))),
                source_path=document.source_path,
                metadata=document.metadata | {"title": document.title, "chunk_index": index},
            )
        )
        index += 1
        if end >= len(text):
            break
    return chunks


def ingest_directory(
    input_dir: Path,
    output_path: Path,
    config: ChunkingConfig | None = None,
    pdf_extractor: PdfExtractor | None = None,
) -> list[DocumentChunk]:
    """Ingest a directory and persist chunks as JSONL."""
    chunks: list[DocumentChunk] = []
    for document in discover_documents(input_dir, pdf_extractor=pdf_extractor):
        chunks.extend(chunk_document(document, config=config))
    save_chunks_jsonl(chunks, output_path)
    return chunks


def save_chunks_jsonl(chunks: list[DocumentChunk], output_path: Path) -> None:
    """Persist chunks to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json())
            handle.write("\n")


def load_chunks_jsonl(path: Path) -> list[DocumentChunk]:
    """Load chunks from JSONL."""
    with path.open("r", encoding="utf-8") as handle:
        return [DocumentChunk.model_validate(json.loads(line)) for line in handle if line.strip()]
