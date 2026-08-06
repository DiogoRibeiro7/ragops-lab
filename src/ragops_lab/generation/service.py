"""Generation service with citation constraints."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, Field

from ragops_lab.config import LLMSettings
from ragops_lab.domain import GeneratedAnswer, RetrievalResult
from ragops_lab.retrieval import tokenize

GENERATION_GROUNDING_THRESHOLD = 0.50
GENERATION_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)


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
        contexts = _parse_context_blocks(prompt)
        question_terms = _content_terms(question)
        if not contexts:
            return json.dumps(
                {
                    "answer_text": "I do not have enough evidence to answer.",
                    "citations": [],
                    "refusal": True,
                }
            )
        ranked = sorted(
            contexts,
            key=lambda context: _grounding_score(question_terms, context[1]),
            reverse=True,
        )
        citation, content = ranked[0]
        if _grounding_score(question_terms, content) < GENERATION_GROUNDING_THRESHOLD:
            return json.dumps(
                {
                    "answer_text": "I do not have enough evidence to answer.",
                    "citations": [],
                    "refusal": True,
                }
            )
        answer_text = content.split(".")[0].strip()
        return json.dumps({"answer_text": answer_text, "citations": [citation], "refusal": False})


def _content_terms(text: str) -> set[str]:
    return {term for term in tokenize(text) if term not in GENERATION_STOPWORDS}


def _grounding_score(question_terms: set[str], evidence_text: str) -> float:
    if not question_terms:
        return 0.0
    return len(question_terms & _content_terms(evidence_text)) / len(question_terms)


def _parse_context_blocks(prompt: str) -> list[tuple[str, str]]:
    context_section = prompt.split("Contexts:", 1)[-1]
    matches = re.finditer(
        r"(?ms)^\[(?P<citation>[^\]]+)\]\s*(?P<content>.*?)(?=^\[[^\]]+\]|\Z)",
        context_section,
    )
    return [
        (match.group("citation"), match.group("content").strip())
        for match in matches
        if match.group("content").strip()
    ]


class OpenAICompatibleLLMClient:
    """HTTP client for OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        model: str,
        timeout_seconds: float,
    ) -> None:
        parsed_endpoint = urllib.parse.urlparse(endpoint)
        if parsed_endpoint.scheme not in {"http", "https"}:
            raise ValueError("LLM endpoint must use http or https.")
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")
        request = urllib.request.Request(  # noqa: S310
            self.endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(  # noqa: S310
                request,
                timeout=self.timeout_seconds,
            ) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM provider request failed: {exc}") from exc
        choices = response_payload.get("choices", [])
        if not choices:
            raise RuntimeError("LLM provider response did not contain choices.")
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM provider response did not contain message content.")
        return content


def build_llm_client(settings: LLMSettings) -> tuple[LLMClient, str]:
    """Build an LLM client from runtime settings."""
    if settings.provider == "heuristic":
        return HeuristicLLMClient(), settings.model
    if settings.provider == "openai-compatible":
        if not settings.endpoint:
            raise ValueError("RAGOPS_LLM_ENDPOINT is required for openai-compatible LLMs.")
        api_key = os.getenv(settings.api_key_env)
        if not api_key:
            raise ValueError(f"{settings.api_key_env} is required for openai-compatible LLMs.")
        return (
            OpenAICompatibleLLMClient(
                endpoint=settings.endpoint,
                api_key=api_key,
                model=settings.model,
                timeout_seconds=settings.timeout_seconds,
            ),
            settings.model,
        )
    raise ValueError(f"Unsupported LLM provider: {settings.provider}")


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
