"""API tests using a fake LLM (fast, deterministic, no Ollama needed)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader

from infact_agents.api import create_app
from infact_agents.settings import Settings

from .conftest import HYPOTHESIS, PDF_DOCS, make_pdf


def _create(client, **extra) -> str:
    r = client.post("/nodes", json={"hypothesis": HYPOTHESIS, **extra})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _upload(client, node_id: str, name: str, data: bytes):
    return client.post(
        f"/nodes/{node_id}/data", files={"file": (name, data, "application/pdf")}
    )


def test_default_provider_is_ollama():
    assert Settings().llm_provider == "ollama"


def test_generated_pdfs_are_readable(pdfs, tmp_path):
    for name, data in pdfs.items():
        path = tmp_path / name
        path.write_bytes(data)
        assert PDF_DOCS[name][:40] in PdfReader(str(path)).pages[0].extract_text()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_create_and_get_node(client):
    node_id = _create(client, prior_log_odds=0.5)
    body = client.get(f"/nodes/{node_id}").json()
    assert body["hypothesis"] == HYPOTHESIS
    assert body["num_data_points"] == 0
    assert body["data_points"] == []
    assert body["probability"] == pytest.approx(0.6224593, abs=1e-6)
    assert [n["id"] for n in client.get("/nodes").json()] == [node_id]


def test_unknown_node_404(client):
    assert client.get("/nodes/nope").status_code == 404
    assert _upload(client, "nope", "a.pdf", make_pdf("x")).status_code == 404
    assert client.delete("/nodes/nope").status_code == 404


def test_single_pdf(client, fake_llm, pdfs):
    node_id = _create(client)
    r = _upload(client, node_id, "nasa_report.pdf", pdfs["nasa_report.pdf"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["redundant"] is False
    assert body["raw_data"] == {
        "numerical_values": [1.1],
        "confidence_assessment": {"confidence_score": 0.9},
    }
    assert body["num_data_points"] == 1
    # l_plus - l_minus = 2 from the fake LLM, prior 0
    assert body["current_posterior"] == pytest.approx(2.0)
    assert body["probability"] > 0.5

    # PDF text reached the LLM (Ollama path: extracted text, not attachment)
    assert any("NASA GISS" in p for p in fake_llm.prompts)

    detail = client.get(f"/nodes/{node_id}").json()
    assert detail["data_points"][0]["metadata"]["filename"] == "nasa_report.pdf"
    assert detail["data_points"][0]["metadata"]["page_count"] == 1


def test_multiple_pdfs_accumulate_evidence(client, fake_llm, pdfs):
    node_id = _create(client)
    posteriors = []
    for name, data in pdfs.items():
        r = _upload(client, node_id, name, data)
        assert r.status_code == 200, r.text
        posteriors.append(r.json()["current_posterior"])

    assert posteriors == pytest.approx([2.0, 4.0, 6.0])
    detail = client.get(f"/nodes/{node_id}").json()
    assert detail["num_data_points"] == len(pdfs)
    assert {dp["metadata"]["filename"] for dp in detail["data_points"]} == set(pdfs)


def test_contradicting_pdf_lowers_belief(client, fake_llm, pdfs):
    node_id = _create(client)
    _upload(client, node_id, "nasa_report.pdf", pdfs["nasa_report.pdf"])
    fake_llm.log_likelihoods = (-3.0, -1.0)
    r = _upload(client, node_id, "blog_post.pdf", pdfs["blog_post.pdf"])
    assert r.json()["current_posterior"] == pytest.approx(0.0)


def test_redundant_pdf_is_skipped(client, fake_llm, pdfs):
    node_id = _create(client)
    _upload(client, node_id, "nasa_report.pdf", pdfs["nasa_report.pdf"])
    fake_llm.redundant = True
    r = _upload(client, node_id, "nasa_copy.pdf", pdfs["nasa_report.pdf"])
    body = r.json()
    assert body["redundant"] is True
    assert body["raw_data"] is None
    assert body["num_data_points"] == 1
    assert body["current_posterior"] == pytest.approx(2.0)


def test_unsafe_filename_is_reduced_to_basename(client, pdfs):
    node_id = _create(client)
    r = _upload(client, node_id, "../../etc/evil.pdf", pdfs["nasa_report.pdf"])
    assert r.status_code == 200
    detail = client.get(f"/nodes/{node_id}").json()
    assert detail["data_points"][0]["metadata"]["filename"] == "evil.pdf"


def test_upload_too_large(client):
    node_id = _create(client)
    r = _upload(client, node_id, "big.pdf", b"0" * (1024 * 1024 + 1))
    assert r.status_code == 413


def test_delete_node(client):
    node_id = _create(client)
    assert client.delete(f"/nodes/{node_id}").status_code == 204
    assert client.get(f"/nodes/{node_id}").status_code == 404


def test_processing_failure_returns_502(client, fake_llm, pdfs):
    node_id = _create(client)
    fake_llm.generate = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    r = _upload(client, node_id, "nasa_report.pdf", pdfs["nasa_report.pdf"])
    assert r.status_code == 502
    assert "RuntimeError" in r.json()["detail"]
    assert "boom" not in r.json()["detail"]


def test_api_key_required(fake_llm, settings, pdfs):
    settings.api_key = "secret"
    c = TestClient(create_app(settings))
    assert c.get("/nodes").status_code == 401
    assert c.get("/nodes", headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.get("/nodes", headers={"X-API-Key": "secret"}).status_code == 200
    assert c.get("/health").status_code == 200  # health stays open


def test_stateless_analyze(client, fake_llm, pdfs):
    r = client.post(
        "/analyze",
        data={"hypothesis": HYPOTHESIS, "prior_log_odds": "0"},
        files={"file": ("nasa_report.pdf", pdfs["nasa_report.pdf"], "application/pdf")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["hypothesis"] == HYPOTHESIS
    assert body["prior_log_odds"] == 0
    assert body["current_posterior"] == pytest.approx(2.0)
    assert body["probability"] > 0.5
    assert body["redundant"] is False
    assert body["raw_data"]["numerical_values"] == [1.1]
    assert body["l_plus"] == -1.0
    assert body["l_minus"] == -3.0
    assert body["metadata"]["filename"] == "nasa_report.pdf"
    assert client.get("/nodes").json() == []
