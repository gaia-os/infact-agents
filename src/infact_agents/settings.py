"""Central configuration for InFact nodes.

All values can be overridden with ``INFACT_*`` environment variables or by
passing explicit keyword arguments to :class:`Settings`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(f"INFACT_{name}", default)


@dataclass
class Settings:
    # --- LLM backend -------------------------------------------------------
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "ollama"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "gpt-oss:20b"))
    llm_temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE", "0.1")))
    llm_max_tokens: int = field(default_factory=lambda: int(_env("LLM_MAX_TOKENS", "8192")))

    # Provider specific
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434"))
    anthropic_api_key: str | None = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY")
    )

    # --- Analysis ----------------------------------------------------------
    max_log_likelihood_ratio: float = field(
        default_factory=lambda: float(_env("MAX_LOG_LIKELIHOOD_RATIO", "5.0"))
    )
    max_debug_attempts: int = field(default_factory=lambda: int(_env("MAX_DEBUG_ATTEMPTS", "5")))
    # Guard against overflowing small local-model context windows.
    max_input_chars: int = field(default_factory=lambda: int(_env("MAX_INPUT_CHARS", "60000")))

    # --- API ---------------------------------------------------------------
    # If set, clients must send it in the X-API-Key header.
    api_key: str | None = field(default_factory=lambda: os.environ.get("INFACT_API_KEY"))
    api_host: str = field(default_factory=lambda: _env("API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: int(_env("API_PORT", "8000")))
    max_upload_bytes: int = field(
        default_factory=lambda: int(_env("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
    )

    # --- Logging -----------------------------------------------------------
    log_dir: Path = field(default_factory=lambda: Path(_env("LOG_DIR", "logs")))
    log_level: int = field(
        default_factory=lambda: getattr(logging, _env("LOG_LEVEL", "INFO").upper(), logging.INFO)
    )
