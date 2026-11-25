# infact-agents
agentic fact-checking framework combining autonomous evidence retrieval, multi-step reasoning, and probabilistic belief estimation for reliable claim verification.

## Setup (uv)

```bash
uv venv
uv sync                      # core + Ollama
uv sync --extra anthropic    # optionally add Anthropic
```

## Usage

```python
from infact_agents import InFactNode, Settings

# Default: Ollama (INFACT_LLM_MODEL=gpt-oss:20b at http://localhost:11434)
node = InFactNode("The Earth's average temperature has risen since 1900")
posterior, (lo, hi) = node.process_data("evidence.pdf")

# Anthropic via settings
node = InFactNode("...", settings=Settings(llm_provider="anthropic", llm_model="claude-3-5-sonnet-20241022"))

# Or bring your own backend by subclassing `infact_agents.LLMClient`
```

## API (FastAPI)

```bash
uv sync --extra api
uv run infact-api          # or: uv run uvicorn infact_agents.api.main:app
```

| Method | Path | Purpose |
|---|---|---|
| POST | `/nodes` | create node `{hypothesis, prior_log_odds?, llm_provider?, llm_model?}` |
| GET | `/nodes`, `/nodes/{id}` | list / full state incl. data points |
| POST | `/nodes/{id}/data` | multipart upload (`file`), returns updated belief |
| DELETE | `/nodes/{id}` | remove node |
| GET | `/health` | liveness |

Env: `INFACT_API_KEY` (if set, send `X-API-Key`), `INFACT_API_HOST`, `INFACT_API_PORT`, `INFACT_MAX_UPLOAD_BYTES`.
Node state is in memory only.

### API Consumer Guide

Use this section when sharing the API with other teams.

#### 1) Reachability

Start the service:

```bash
uv sync --extra api
INFACT_API_HOST=0.0.0.0 INFACT_API_PORT=8000 uv run infact-api
```

Base URL examples:

- Local caller on same machine: `http://127.0.0.1:8000`
- Remote caller on network: `http://<server-ip>:8000`

OpenAPI docs:

- Swagger UI: `/docs`
- OpenAPI JSON: `/openapi.json`

#### 2) Authentication

If `INFACT_API_KEY` is set on the server, every protected request must include:

```http
X-API-Key: <your-key>
```

`/health` stays open without a key.

#### 3) Quick curl examples

Set these once:

```bash
export API_BASE="http://127.0.0.1:8000"
export API_KEY=""  # set to real key if server requires auth
```

Health:

```bash
curl -s "$API_BASE/health"
```

Create node:

```bash
curl -s -X POST "$API_BASE/nodes" \
	-H "Content-Type: application/json" \
	-H "X-API-Key: $API_KEY" \
	-d '{
		"hypothesis": "Global average temperature has risen since 1900",
		"prior_log_odds": 0.0,
		"llm_provider": "ollama",
		"llm_model": "gpt-oss:20b"
	}'
```

List nodes:

```bash
curl -s "$API_BASE/nodes" -H "X-API-Key: $API_KEY"
```

Get node detail:

```bash
curl -s "$API_BASE/nodes/<node_id>" -H "X-API-Key: $API_KEY"
```

Upload evidence to a node:

```bash
curl -s -X POST "$API_BASE/nodes/<node_id>/data" \
	-H "X-API-Key: $API_KEY" \
	-F "file=@examples/docs/nasa_report.pdf;type=application/pdf"
```

Delete node:

```bash
curl -s -X DELETE "$API_BASE/nodes/<node_id>" -H "X-API-Key: $API_KEY" -i
```

Stateless analyze (no server-side node retention):

```bash
curl -s -X POST "$API_BASE/analyze" \
	-H "X-API-Key: $API_KEY" \
	-F "hypothesis=The Earth's average temperature has risen since 1900" \
	-F "prior_log_odds=0.0" \
	-F "current_posterior=0.0" \
	-F "existing_data_points=[]" \
	-F "llm_provider=ollama" \
	-F "llm_model=gpt-oss:20b" \
	-F "file=@examples/docs/nasa_report.pdf;type=application/pdf"
```

#### 4) Python consumer example (httpx)

```python
from __future__ import annotations

import httpx

API_BASE = "http://127.0.0.1:8000"
API_KEY = ""  # set if required
HYPOTHESIS = "The Earth's average temperature has risen since 1900"

headers = {"X-API-Key": API_KEY} if API_KEY else {}

with httpx.Client(base_url=API_BASE, headers=headers, timeout=600.0) as client:
		client.get("/health").raise_for_status()

		create = client.post(
				"/nodes",
				json={"hypothesis": HYPOTHESIS, "prior_log_odds": 0.0},
		)
		create.raise_for_status()
		node_id = create.json()["id"]

		with open("examples/docs/nasa_report.pdf", "rb") as f:
				process = client.post(
						f"/nodes/{node_id}/data",
						files={"file": ("nasa_report.pdf", f, "application/pdf")},
				)
		process.raise_for_status()
		print("Posterior:", process.json()["current_posterior"])

		detail = client.get(f"/nodes/{node_id}")
		detail.raise_for_status()
		print("Data points:", detail.json()["num_data_points"])

		client.delete(f"/nodes/{node_id}")
```

#### 5) Endpoint expectations

- `GET /health`: no request body, returns liveness.
- `POST /nodes`: JSON body with `hypothesis` (required), optional `prior_log_odds`, `llm_provider`, `llm_model`.
- `GET /nodes`: no body, returns node summaries.
- `GET /nodes/{id}`: no body, returns full node state and data points.
- `POST /nodes/{id}/data`: `multipart/form-data` with required `file` field.
- `DELETE /nodes/{id}`: no body, removes node.
- `POST /analyze`: `multipart/form-data` with required `hypothesis` and `file`; optional `prior_log_odds`, `current_posterior`, `existing_data_points` (JSON string), `llm_provider`, `llm_model`.

### Using this repo as a submodule

```bash
git submodule add <url> libs/infact-agents
uv add --editable "./libs/infact-agents[api]"
```

Then either run it as its own service (`uv run infact-api`) and call it over HTTP, or mount it in your FastAPI app:

```python
from infact_agents.api import create_router
app.include_router(create_router(), prefix="/infact")
```

Or skip HTTP entirely: `from infact_agents import InFactNode`.

## Tests

```bash
uv run pytest -m "not ollama"   # fast, fake LLM
uv run pytest -m ollama         # real Ollama (INFACT_LLM_MODEL, default gpt-oss:20b); skipped if unavailable
```

## Layout

- `src/infact_agents/settings.py` - config (`INFACT_*` env vars)
- `src/infact_agents/llm/` - model-agnostic LLM clients (Ollama, Anthropic)
- `src/infact_agents/process_data/process.py` - parsing, metadata, redundancy check
- `src/infact_agents/analyse_data.py` - LLM-generated likelihood code + debug loop
- `src/infact_agents/node.py` - `InFactNode`

Note: generated analysis code is executed in-process without a sandbox.
