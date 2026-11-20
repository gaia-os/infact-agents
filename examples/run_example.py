"""Evaluate a hypothesis against PDF documents using the InFact HTTP API.

Start the API first (in another terminal)::

    uv sync --extra api
    uv run infact-api

Edit the settings below, then run::

    uv run python examples/run_example.py
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx

# ----------------------------------------------------------------------------
# Settings: edit these
# ----------------------------------------------------------------------------
HYPOTHESIS = "The Earth's average temperature has risen since 1900"

# PDFs to analyse. Either list files explicitly, or leave empty to use every
# *.pdf in DOCS_DIR.
PDF_FILES: list[str] = []
DOCS_DIR = Path(__file__).parent / "docs"

API_URL = "http://127.0.0.1:8000"
API_KEY = os.getenv("INFACT_API_KEY")  # only needed if the server sets it

PRIOR_LOG_ODDS = 0.0  # 0.0 = 50% prior
LLM_PROVIDER: str | None = None  # e.g. "ollama" or "anthropic"; None = server default
LLM_MODEL: str | None = None  # e.g. "gpt-oss:20b"; None = server default
KEEP_NODE = False  # True = don't delete the node on the server afterwards
TIMEOUT_SECONDS = 600.0  # LLM processing can take minutes per document
# ----------------------------------------------------------------------------


def print_belief(label: str, s: dict) -> None:
    lo, hi = s["interval"]
    print(
        f"{label}: log-odds={s['current_posterior']:.2f}  P={s['probability']:.6f}  "
        f"95% interval=({lo:.6f}, {hi:.6f})  data points={s['num_data_points']}"
    )


def main() -> None:
    files = [Path(f) for f in PDF_FILES] or sorted(DOCS_DIR.glob("*.pdf"))
    if not files:
        raise SystemExit(f"No PDFs found. Put some in {DOCS_DIR} or set PDF_FILES.")
    missing = [f for f in files if not f.is_file()]
    if missing:
        raise SystemExit(f"File(s) not found: {', '.join(map(str, missing))}")

    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    with httpx.Client(
        base_url=API_URL, headers=headers, timeout=httpx.Timeout(TIMEOUT_SECONDS)
    ) as client:
        try:
            client.get("/health").raise_for_status()
        except httpx.HTTPError as e:
            raise SystemExit(f"Cannot reach API at {API_URL}: {e}\nStart it with: uv run infact-api")

        body = {"hypothesis": HYPOTHESIS, "prior_log_odds": PRIOR_LOG_ODDS}
        if LLM_PROVIDER:
            body["llm_provider"] = LLM_PROVIDER
        if LLM_MODEL:
            body["llm_model"] = LLM_MODEL
        r = client.post("/nodes", json=body)
        r.raise_for_status()
        node = r.json()
        node_id = node["id"]
        print(f"Hypothesis: {node['hypothesis']}")
        print_belief("Prior    ", node)

        try:
            for f in files:
                print(f"\nProcessing {f.name} ...")
                with f.open("rb") as fh:
                    r = client.post(
                        f"/nodes/{node_id}/data",
                        files={"file": (f.name, fh, "application/pdf")},
                    )
                if r.status_code != 200:
                    print(f"  failed ({r.status_code}): {r.text}")
                    continue
                res = r.json()
                if res["redundant"]:
                    print("  skipped: redundant with existing evidence")
                print_belief("  Posterior", res)

            detail = client.get(f"/nodes/{node_id}").json()
            print("\n=== Final belief ===")
            print_belief("Result   ", detail)
            for i, dp in enumerate(detail["data_points"], 1):
                print(
                    f"\n[{i}] L+={dp['l_plus']:.3f} L-={dp['l_minus']:.3f} "
                    f"posterior={dp['posterior']:.3f}"
                )
                print(f"    {dp['analysis_rationale']}")
        finally:
            if not KEEP_NODE:
                client.delete(f"/nodes/{node_id}")


if __name__ == "__main__":
    main()
