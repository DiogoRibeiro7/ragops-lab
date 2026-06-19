"""Generation service with citation constraints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, Field

from ragops_lab.domain import GeneratedAnswer, RetrievalResult


class LLMClient(Protocol):
    """LLM generation interface."""

    def generate(self, prompt: str) -> str:
        """Generate a response for a prompt."""


class AnswerPayload(BaseModel):
    """Structured model output."""

    answer_text: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)
    refusal: bool = Field(default=False)


@dataclass(frozen=True)
class PromptTemplate:
    """Prompt template enforcing evidence-grounded answers."""

    system_instruction: str = (
        "Answer only using the provided contexts. Return valid JSON with "
        "answer_text, citations, and refusal. Cite chunk_ids exactly."
    )

    def render(self, question: str, contexts: list[RetrievalResult]) -> str:
        context_blocks = [f"[{result.chunk.chunk_id}] {result.chunk.text}" for result in contexts]
        return f"{self.system_instruction}\nQuestion: {question}\nContexts:\n" + "\n".join(
            context_blocks
        )


class FakeLLMClient:
    """Test double for generation."""

    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, prompt: str) -> str:
        return self.response


class HeuristicLLMClient:
    """Local deterministic generator for demos and CLI usage."""

    def generate(self, prompt: str) -> str:
        question_line = next(
            (line for line in prompt.splitlines() if line.startswith("Question:")), ""
        )
        question = question_line.replace("Question:", "", 1).strip().lower()
        context_lines = [line for line in prompt.splitlines() if line.startswith("[")]
        if not context_lines:
            return json.dumps(
                {
                    "answer_text": "I do not have enough evidence to answer.",
                    "citations": [],
                    "refusal": True,
                }
            )
        ranked = sorted(
            context_lines,
            key=lambda line: sum(1 for word in question.split() if word in line.lower()),
            reverse=True,
        )
        best = ranked[0]
        citation = best.split("]", 1)[0][1:]
        content = best.split("]", 1)[1].strip()
        if sum(1 for word in question.split() if word in content.lower()) == 0:
            return json.dumps(
                {
                    "answer_text": "I do not have enough evidence to answer.",
                    "citations": [],
                    "refusal": True,
                }
            )
        answer_text = content.split(".")[0].strip()
        return json.dumps({"answer_text": answer_text, "citations": [citation], "refusal": False})


class GenerationService:
    """Coordinates prompt creation and citation validation."""

    def __init__(
        self, llm_client: LLMClient, prompt_template: PromptTemplate | None = None
    ) -> None:
        self.llm_client = llm_client
        self.prompt_template = prompt_template or PromptTemplate()

    def answer(
        self, question: str, contexts: list[RetrievalResult], *, model_name: str
    ) -> GeneratedAnswer:
        prompt = self.prompt_template.render(question, contexts)
        raw_response = self.llm_client.generate(prompt)
        payload = AnswerPayload.model_validate(json.loads(raw_response))
        available_citations = {result.chunk.chunk_id for result in contexts}
        invalid_citations = [
            citation for citation in payload.citations if citation not in available_citations
        ]
        if invalid_citations:
            raise ValueError(f"Unknown citations returned by model: {invalid_citations}")
        grounded = bool(payload.citations) and not payload.refusal
        return GeneratedAnswer(
            question=question,
            answer_text=payload.answer_text,
            citations=payload.citations,
            model_name=model_name,
            refusal=payload.refusal,
            grounded=grounded,
            prompt=prompt,
        )
