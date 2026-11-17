"""Provider-agnostic LLM interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Sequence


@dataclass(frozen=True)
class Attachment:
    """A binary input (image or PDF) passed alongside the text prompt."""

    kind: Literal["image", "pdf"]
    media_type: str
    data: str  # base64-encoded


class LLMClient(ABC):
    """Minimal interface an InFact node needs from any LLM backend."""

    #: Whether the backend can ingest images / PDFs natively.
    supports_images: bool = False
    supports_pdf: bool = False

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        attachments: Sequence[Attachment] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Return the model's text response for ``prompt``."""
