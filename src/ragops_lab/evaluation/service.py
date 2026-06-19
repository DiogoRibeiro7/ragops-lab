"""Evaluation services and report exporters."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Protocol

from ragops_lab.domain import EvaluationResult, GeneratedAnswer, RetrievalResult

from ..retrieval.tokenizer import tokenize


class RelevanceJudge(Protocol):
    """Judge abstraction used for answer relevance scoring."""

    def score(self, question: str, answer_text: str, references: str) -> float:
        """Return a normalized score."""


class OverlapJudge:
    """Simple lexical overlap judge for local evaluation."""

    def score(self, question: str, answer_text: str, references: str) -> float:
        question_terms = set(tokenize(question))
        answer_terms = set(tokenize(answer_text))
        reference_terms = set(tokenize(references))
        if not question_terms or not answer_terms:
            return 0.0
        aligned = len((question_terms & answer_terms) & reference_terms)
        return min(1.0, aligned / len(question_terms))


def context_precision(results: list[RetrievalResult], reference_chunk_ids: set[str]) -> float:
    """Measure the proportion of retrieved chunks that are relevant."""
    if not results:
        return 0.0
    relevant = sum(1 for result in results if result.chunk.chunk_id in reference_chunk_ids)
    return relevant / len(results)


def context_recall(results: list[RetrievalResult], reference_chunk_ids: set[str]) -> float:
    """Measure how much of the needed context was retrieved."""
    if not reference_chunk_ids:
        return 0.0
    retrieved = {result.chunk.chunk_id for result in results}
    return len(retrieved & reference_chunk_ids) / len(reference_chunk_ids)


def extract_claims(answer_text: str) -> list[str]:
    """Break an answer into crude sentence-level claims."""
    claims = [part.strip() for part in answer_text.replace("!", ".").replace("?", ".").split(".")]
    return [claim for claim in claims if claim]


def unsupported_claim_count(answer_text: str, evidence_text: str) -> int:
    """Count claims not supported by lexical overlap with the evidence."""
    evidence_terms = set(tokenize(evidence_text))
    unsupported = 0
    for claim in extract_claims(answer_text):
        claim_terms = set(tokenize(claim))
        if claim_terms and not claim_terms <= evidence_terms:
            unsupported += 1
    return unsupported


def citation_support(answer: GeneratedAnswer, results: list[RetrievalResult]) -> float:
    """Measure whether cited chunks exist in retrieved context."""
    if answer.refusal:
        return 1.0
    if not answer.citations:
        return 0.0
    retrieved_chunk_ids = {result.chunk.chunk_id for result in results}
    supported = sum(1 for citation in answer.citations if citation in retrieved_chunk_ids)
    return supported / len(answer.citations)


def refusal_correctness(answer: GeneratedAnswer, expected_unanswerable: bool) -> bool:
    """Check whether a refusal matched the scenario."""
    return answer.refusal is expected_unanswerable


def evaluate_answer(
    question: str,
    answer: GeneratedAnswer,
    results: list[RetrievalResult],
    *,
    reference_chunk_ids: list[str] | None = None,
    expected_answer: str = "",
    expected_unanswerable: bool | None = None,
    judge: RelevanceJudge | None = None,
) -> EvaluationResult:
    """Evaluate a generated answer."""
    reference_ids = set(reference_chunk_ids or [])
    evidence_text = "\n".join(result.chunk.text for result in results)
    overlap_judge = judge or OverlapJudge()
    unsupported = unsupported_claim_count(answer.answer_text, evidence_text)
    claim_count = max(len(extract_claims(answer.answer_text)), 1)
    faithfulness = max(0.0, 1.0 - unsupported / claim_count)
    notes: list[str] = []
    citation_score = citation_support(answer, results)
    if citation_score < 1.0:
        notes.append("One or more citations were unsupported by retrieved context.")
    if unsupported > 0:
        notes.append("Unsupported claims detected in answer text.")
    return EvaluationResult(
        context_precision=context_precision(results, reference_ids),
        context_recall=context_recall(results, reference_ids),
        answer_relevance=overlap_judge.score(
            question, answer.answer_text, expected_answer or evidence_text
        ),
        faithfulness=faithfulness,
        citation_support=citation_score,
        unsupported_claim_count=unsupported,
        refusal_correct=(
            refusal_correctness(answer, expected_unanswerable)
            if expected_unanswerable is not None
            else None
        ),
        notes=notes,
    )


def export_evaluation_report_csv(report: EvaluationResult, output_path: Path) -> None:
    """Export a single-row CSV evaluation report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report.model_dump().keys()))
        writer.writeheader()
        writer.writerow(report.model_dump(mode="json"))


def export_evaluation_report_markdown(report: EvaluationResult, output_path: Path) -> None:
    """Export an evaluation report as Markdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Evaluation Report",
        "",
        f"- Context precision: {report.context_precision:.2f}",
        f"- Context recall: {report.context_recall:.2f}",
        f"- Answer relevance: {report.answer_relevance:.2f}",
        f"- Faithfulness: {report.faithfulness:.2f}",
        f"- Citation support: {report.citation_support:.2f}",
        f"- Unsupported claims: {report.unsupported_claim_count}",
        f"- Refusal correct: {report.refusal_correct}",
    ]
    if report.notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in report.notes)
    output_path.write_text("\n".join(lines), encoding="utf-8")
