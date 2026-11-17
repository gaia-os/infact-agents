"""Ollama backend (default)."""

from __future__ import annotations

from typing import Sequence

from .base import Attachment, LLMClient


class OllamaClient(LLMClient):
    supports_images = True  # requires a vision-capable model (e.g. llava)
    supports_pdf = False  # PDFs are converted to text upstream

    def __init__(
        self,
        model: str = "llama3.1",
        host: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 8192,
    ):
        import ollama

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = ollama.Client(host=host)

    def generate(
        self,
        prompt: str,
        *,
        attachments: Sequence[Attachment] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        message: dict = {"role": "user", "content": prompt}
        images = [a.data for a in attachments if a.kind == "image"]
        if images:
            message["images"] = images

        response = self._client.chat(
            model=self.model,
            messages=[message],
            options={
                "temperature": self.temperature if temperature is None else temperature,
                "num_predict": self.max_tokens if max_tokens is None else max_tokens,
            },
        )
        return response.message.content or ""
