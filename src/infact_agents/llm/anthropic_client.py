"""Anthropic backend (requires the ``anthropic`` extra)."""

from __future__ import annotations

from typing import Sequence

from .base import Attachment, LLMClient


class AnthropicClient(LLMClient):
    supports_images = True
    supports_pdf = True

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.1,
        max_tokens: int = 8192,
    ):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Anthropic support requires the extra: `uv sync --extra anthropic`"
            ) from e

        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for the anthropic provider")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key)

    def generate(
        self,
        prompt: str,
        *,
        attachments: Sequence[Attachment] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        content: list[dict] = []
        for a in attachments:
            block_type = "image" if a.kind == "image" else "document"
            content.append(
                {
                    "type": block_type,
                    "source": {"type": "base64", "media_type": a.media_type, "data": a.data},
                }
            )
        content.append({"type": "text", "text": prompt})

        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            temperature=self.temperature if temperature is None else temperature,
            messages=[{"role": "user", "content": content}],
        )
        return "".join(b.text for b in message.content if getattr(b, "type", "") == "text")
