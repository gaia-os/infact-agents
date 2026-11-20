# Example: evaluate a hypothesis with PDFs via the API

1. Put your PDFs in `examples/docs/`.
2. Start the API (needs a running LLM backend, e.g. Ollama):

   ```bash
   uv sync --extra api
   uv run infact-api
   ```

3. Edit the settings block at the top of `run_example.py` (`HYPOTHESIS`, `PDF_FILES`/`DOCS_DIR`,
   `API_URL`, `PRIOR_LOG_ODDS`, `LLM_PROVIDER`, `LLM_MODEL`, `KEEP_NODE`), then in another terminal:

   ```bash
   uv run python examples/run_example.py
   ```

If the server uses `INFACT_API_KEY`, it is picked up from the environment (or set `API_KEY` in the script).

The script creates a node, uploads each PDF in turn (belief is updated after each),
then prints the final probability, interval and per-document rationale.

## Stateless example

To demonstrate `/analyze` without any server-side node storage:

```bash
uv run python examples/test_stateless_example.py
```

The script keeps the current posterior and accepted evidence in memory in the
client process, and sends both back with each new PDF. The API retains no node.
