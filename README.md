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
