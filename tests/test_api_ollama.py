"""End-to-end API tests against a real Ollama server.

Run with:  uv run pytest -m ollama
Choose the model with INFACT_LLM_MODEL (default: gpt-oss:20b).
Skipped automatically if Ollama or the model is unavailable.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from infact_agents.api import create_app
from infact_agents.settings import Settings

from .conftest import HYPOTHESIS

pytestmark = pytest.mark.ollama


@pytest.fixture(scope="module")
def ollama_settings(tmp_path_factory) -> Settings:
    settings = Settings(
        api_key=None,
        llm_provider="ollama",
        log_dir=tmp_path_factory.mktemp("logs"),
    )
    try:
        import ollama

        installed = {m.model for m in ollama.Client(host=settings.ollama_host).list().models}
    except Exception as e:
        pytest.skip(f"Ollama not reachable at {settings.ollama_host}: {e}")
    if settings.llm_model not in installed:
        pytest.skip(f"Model {settings.llm_model!r} not installed in Ollama")
    return settings


def test_ollama_with_pdfs(ollama_settings, pdfs):
    client = TestClient(create_app(ollama_settings))

    r = client.post("/nodes", json={"hypothesis": HYPOTHESIS})
    assert r.status_code == 201, r.text
    node_id = r.json()["id"]

    last = None
    for name in ("nasa_report.pdf", "noaa_summary.pdf", "blog_post.pdf"):
        r = client.post(
            f"/nodes/{node_id}/data",
            files={"file": (name, pdfs[name], "application/pdf")},
        )
        assert r.status_code == 200, f"{name}: {r.text}"
        last = r.json()
        assert 0.0 <= last["probability"] <= 1.0

    detail = client.get(f"/nodes/{node_id}").json()
    assert detail["num_data_points"] >= 1
    for dp in detail["data_points"]:
        assert dp["metadata"]["file_type"] == ".pdf"
        assert dp["analysis_rationale"]  # generated analysis code was kept
        # likelihood difference is capped by settings
        assert abs(dp["l_plus"] - dp["l_minus"]) <= ollama_settings.max_log_likelihood_ratio + 1e-9

    # Two supporting sources and one weak contradicting blog: belief should lean true.
    assert last["probability"] > 0.5
