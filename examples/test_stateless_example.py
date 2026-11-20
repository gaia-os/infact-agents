"""Exercise the stateless /analyze API without creating server-side nodes."""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx

HYPOTHESIS = "The Earth's average temperature has risen since 1900"
PDF_FILES: list[str] = []
DOCS_DIR = Path(__file__).parent / "docs"
API_URL = "http://127.0.0.1:8000"
API_KEY = os.getenv("INFACT_API_KEY")
PRIOR_LOG_ODDS = 0.0
LLM_PROVIDER: str | None = None
LLM_MODEL: str | None = None
TIMEOUT_SECONDS = 600.0


def main() -> None:
    files = [Path(value) for value in PDF_FILES] or sorted(DOCS_DIR.glob("*.pdf"))
    if not files:
        raise SystemExit(f"No PDFs found. Put some in {DOCS_DIR} or set PDF_FILES.")

    missing = [path for path in files if not path.is_file()]
    if missing:
        raise SystemExit(f"File(s) not found: {', '.join(map(str, missing))}")

    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    current_posterior = PRIOR_LOG_ODDS
    existing_data_points: list[dict] = []

    with httpx.Client(
        base_url=API_URL,
        headers=headers,
        timeout=httpx.Timeout(TIMEOUT_SECONDS),
    ) as client:
        try:
            client.get("/health").raise_for_status()
        except httpx.HTTPError as exc:
            raise SystemExit(
                f"Cannot reach API at {API_URL}: {exc}\n"
                "Start it with: uv run infact-api"
            ) from exc

        print(f"Hypothesis: {HYPOTHESIS}")
        prior_probability = 1 / (1 + pow(2.718281828, -current_posterior))
        print(f"Prior: log-odds={current_posterior:.2f}  P={prior_probability:.6f}")

        for path in files:
            print(f"\nProcessing {path.name} ...")
            data = {
                "hypothesis": HYPOTHESIS,
                "prior_log_odds": str(PRIOR_LOG_ODDS),
                "current_posterior": str(current_posterior),
                "existing_data_points": json.dumps(existing_data_points),
            }
            if LLM_PROVIDER:
                data["llm_provider"] = LLM_PROVIDER
            if LLM_MODEL:
                data["llm_model"] = LLM_MODEL

            with path.open("rb") as stream:
                response = client.post(
                    "/analyze",
                    data=data,
                    files={"file": (path.name, stream, "application/pdf")},
                )

            response.raise_for_status()
            result = response.json()

            if result["redundant"]:
                print("  skipped: redundant with existing evidence")
            else:
                existing_data_points.append(
                    {
                        "raw_data": result["raw_data"],
                        "l_plus": result["l_plus"],
                        "l_minus": result["l_minus"],
                        "posterior": result["posterior"],
                    }
                )

                print("  raw_data keys and value lengths:")
                for key, value in result["raw_data"].items():
                    length = len(value) if isinstance(value, (str, list, dict)) else "n/a"
                    print(f"    {key}: {length}")
                print(f"  L+={result['l_plus']:.3f}  L-={result['l_minus']:.3f}")

            print(f"  analysis_rationale returned: {bool(result.get('analysis_rationale'))}")
            print(result.get("analysis_rationale"))
            current_posterior = result["current_posterior"]
            low, high = result["interval"]
            print(
                f"  Posterior: log-odds={current_posterior:.2f}  "
                f"P={result['probability']:.6f}  "
                f"95% interval=({low:.6f}, {high:.6f})"
            )

        print("\nNo server-side node was created or retained.")


if __name__ == "__main__":
    main()
