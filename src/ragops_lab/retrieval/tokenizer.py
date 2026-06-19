"""Tokenizer used by retrieval and evaluation."""

from __future__ import annotations

import re

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text into normalized lexical terms."""
    return TOKEN_PATTERN.findall(text.lower())
