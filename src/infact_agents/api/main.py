"""Standalone entrypoint: ``uvicorn infact_agents.api.main:app`` or ``infact-api``."""

from __future__ import annotations

from ..settings import Settings
from .app import create_app

settings = Settings()
app = create_app(settings)


def run() -> None:
    import uvicorn

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
