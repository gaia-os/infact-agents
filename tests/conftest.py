"""Shared fixtures: PDF generation, a fake LLM, and API clients."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import infact_agents.node as node_module
from infact_agents.api import create_app
from infact_agents.llm import LLMClient
from infact_agents.settings import Settings

HYPOTHESIS = "Global average surface temperature has increased since 1900."

# (filename, text) - short evidence documents.
PDF_DOCS = {
    "nasa_report.pdf": (
        "NASA GISS surface temperature analysis: global mean temperature has risen "
        "about 1.1 degrees Celsius since 1880, with most warming after 1975."
    ),
    "noaa_summary.pdf": (
        "NOAA reports that the ten warmest years on record have all occurred since 2010. "
        "The long-term trend is about 0.08 degrees Celsius per decade since 1880."
    ),
    "blog_post.pdf": (
        "An unsourced blog post claims temperatures have not changed in decades. "
        "No data, methodology or references are provided."
    ),
}


def make_pdf(text: str) -> bytes:
    """Build a minimal single-page PDF containing ``text`` (extractable by pypdf)."""
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 10 Tf 40 750 Td ({safe}) Tj ET".encode("latin-1")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


@pytest.fixture
def pdfs() -> dict[str, bytes]:
    return {name: make_pdf(text) for name, text in PDF_DOCS.items()}


class FakeLLM(LLMClient):
    """Deterministic stand-in for an LLM. Records every prompt it receives."""

    supports_pdf = False  # like Ollama: PDFs must arrive as extracted text

    def __init__(self):
        self.prompts: list[str] = []
        self.redundant = False
        self.log_likelihoods = (-1.0, -3.0)

    def generate(self, prompt, *, attachments=(), temperature=None, max_tokens=None):
        self.prompts.append(prompt)
        if "Generate Python code" in prompt:
            lp, lm = self.log_likelihoods
            return f"```python\ndef calculate_log_likelihoods(d):\n    return {lp}, {lm}\n```"
        if "is the new data redundant" in prompt.lower():
            return "true" if self.redundant else "false"
        return '```json\n{"numerical_values": [1.1], "confidence_assessment": {"confidence_score": 0.9}}\n```'


@pytest.fixture
def fake_llm(monkeypatch) -> FakeLLM:
    llm = FakeLLM()
    monkeypatch.setattr(node_module, "create_llm_client", lambda settings: llm)
    return llm


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(api_key=None, log_dir=tmp_path / "logs", max_upload_bytes=1024 * 1024)


@pytest.fixture
def client(fake_llm, settings) -> TestClient:
    return TestClient(create_app(settings))
