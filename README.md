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

# Default: Ollama (INFACT_LLM_MODEL=llama3.1 at http://localhost:11434)
node = InFactNode("The Earth's average temperature has risen since 1900")
posterior, (lo, hi) = node.process_data("evidence.pdf")

# Anthropic via settings
node = InFactNode("...", settings=Settings(llm_provider="anthropic", llm_model="claude-3-5-sonnet-20241022"))

# Or bring your own backend by subclassing `infact_agents.LLMClient`
```

## Layout

- `src/infact_agents/settings.py` - config (`INFACT_*` env vars)
- `src/infact_agents/llm/` - model-agnostic LLM clients (Ollama, Anthropic)
- `src/infact_agents/process_data/process.py` - parsing, metadata, redundancy check
- `src/infact_agents/analyse_data.py` - LLM-generated likelihood code + debug loop
- `src/infact_agents/node.py` - `InFactNode`

Note: generated analysis code is executed in-process without a sandbox.
