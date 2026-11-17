"""Small helpers for parsing LLM output."""

from __future__ import annotations

import re

_FENCE_RE = re.compile(r"```[ \t]*([\w+-]*)[ \t]*\n(.*?)```", re.DOTALL)


def extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """Return ``(language, code)`` for each fenced block in ``text``."""
    return [(lang, body.strip()) for lang, body in _FENCE_RE.findall(text)]


def first_code_block(text: str, languages: tuple[str, ...]) -> str | None:
    """Return the first fenced block whose language is in ``languages``.

    If no fences are present, ``""`` in ``languages`` allows the raw text.
    """
    blocks = extract_code_blocks(text)
    for lang, body in blocks:
        if lang.lower() in languages:
            return body
    if not blocks and "" in languages and text.strip():
        return text.strip()
    return None


def to_probability(log_odds: float) -> float:
    import numpy as np

    return float(1 / (1 + np.exp(-log_odds)))
