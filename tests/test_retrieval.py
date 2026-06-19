from __future__ import annotations

from ragops_lab.domain import DocumentChunk
from ragops_lab.retrieval import (
    BM25Retriever,
    FakeEmbeddingClient,
    HybridRetriever,
    RetrievalGoldenExample,
    VectorRetriever,
    evaluate_retrieval,
    tokenize,
)


def _chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="apollo:0",
            document_id="apollo",
            text="Apollo 11 landed on the Moon in 1969.",
            start_offset=0,
            end_offset=38,
            token_count=8,
        ),
        DocumentChunk(
            chunk_id="metrics:0",
            document_id="metrics",
            text="Faithfulness and citation support are critical RAG metrics.",
            start_offset=0,
            end_offset=60,
            token_count=9,
        ),
    ]


def test_tokenizer_normalizes_terms() -> None:
    assert tokenize("Apollo-11, MOON!") == ["apollo", "11", "moon"]


def test_lexical_vector_and_hybrid_retrieval_rank_expected_chunk() -> None:
    chunks = _chunks()
    lexical = BM25Retriever(chunks)
    vector = VectorRetriever(
        chunks, FakeEmbeddingClient(["apollo", "moon", "faithfulness", "citation"])
    )
    hybrid = HybridRetriever(lexical, vector, lexical_weight=0.7, vector_weight=0.3)

    assert lexical.search("moon mission", top_k=1)[0].chunk.chunk_id == "apollo:0"
    assert vector.search("citation support", top_k=1)[0].chunk.chunk_id == "metrics:0"
    assert hybrid.search("moon mission", top_k=1)[0].chunk.chunk_id == "apollo:0"


def test_retrieval_evaluation_computes_recall_and_mrr() -> None:
    report = evaluate_retrieval(
        BM25Retriever(_chunks()),
        [RetrievalGoldenExample(query="moon landing", relevant_chunk_ids=["apollo:0"])],
        top_k=2,
    )

    assert report.recall_at_k == 1.0
    assert report.mean_reciprocal_rank == 1.0
