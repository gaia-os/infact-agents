"""LLM backends and factory."""

from __future__ import annotations

from ..settings import Settings
from .base import Attachment, LLMClient


def create_llm_client(settings: Settings | None = None) -> LLMClient:
    """Build an :class:`LLMClient` from settings (default: Ollama)."""
    settings = settings or Settings()
    provider = settings.llm_provider.lower()

    if provider == "ollama":
        from .ollama_client import OllamaClient

        return OllamaClient(
            model=settings.llm_model,
            host=settings.ollama_host,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    if provider == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(
            api_key=settings.anthropic_api_key or "",
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider!r}")


__all__ = ["Attachment", "LLMClient", "create_llm_client"]
