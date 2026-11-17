"""FastAPI router/app exposing InFact nodes.

Node state is held in memory and is lost on restart; use ``GET /nodes/{id}``
to export state if you need to persist it on the caller side.
"""

from __future__ import annotations

import dataclasses
import hmac
import tempfile
import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, File, Header, HTTPException, UploadFile, status

from ..node import InFactNode
from ..settings import Settings
from .schemas import (
    CreateNodeRequest,
    DataPointOut,
    NodeDetail,
    NodeSummary,
    ProcessResponse,
)


class _Entry:
    def __init__(self, node: InFactNode):
        self.node = node
        self.lock = threading.Lock()  # one data file processed at a time per node


def _summary(node_id: str, node: InFactNode) -> dict:
    return {
        "id": node_id,
        "hypothesis": node.hypothesis,
        "prior_log_odds": node.prior_log_odds,
        "current_posterior": node.current_posterior,
        "probability": node.probability,
        "interval": node._calculate_uncertainty(),
        "num_data_points": len(node.data_points),
    }


def create_router(settings: Settings | None = None) -> APIRouter:
    """Build a router that can be mounted into any FastAPI app.

    Example::

        app.include_router(create_router(), prefix="/infact")
    """
    settings = settings or Settings()
    nodes: dict[str, _Entry] = {}

    def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        if settings.api_key and not hmac.compare_digest(x_api_key or "", settings.api_key):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing API key")

    router = APIRouter(dependencies=[Depends(require_api_key)])

    def get_entry(node_id: str) -> _Entry:
        entry = nodes.get(node_id)
        if entry is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Node not found")
        return entry

    @router.post("/nodes", response_model=NodeSummary, status_code=status.HTTP_201_CREATED)
    def create_node(req: CreateNodeRequest):
        overrides = {
            k: v
            for k, v in (("llm_provider", req.llm_provider), ("llm_model", req.llm_model))
            if v
        }
        try:
            node = InFactNode(
                req.hypothesis,
                prior_log_odds=req.prior_log_odds,
                settings=dataclasses.replace(settings, **overrides),
            )
        except (ValueError, ImportError) as e:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
        node_id = uuid.uuid4().hex
        nodes[node_id] = _Entry(node)
        return _summary(node_id, node)

    @router.get("/nodes", response_model=list[NodeSummary])
    def list_nodes():
        return [_summary(i, e.node) for i, e in nodes.items()]

    @router.get("/nodes/{node_id}", response_model=NodeDetail)
    def get_node(node_id: str):
        node = get_entry(node_id).node
        return {
            **_summary(node_id, node),
            "data_points": [DataPointOut(**dp) for dp in node.data_points],
        }

    @router.delete("/nodes/{node_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_node(node_id: str):
        get_entry(node_id)
        del nodes[node_id]

    # Sync `def` so the blocking LLM calls run in FastAPI's threadpool.
    @router.post("/nodes/{node_id}/data", response_model=ProcessResponse)
    def process_data(node_id: str, file: UploadFile = File(...)):
        entry = get_entry(node_id)

        content = file.file.read(settings.max_upload_bytes + 1)
        if len(content) > settings.max_upload_bytes:
            raise HTTPException(413, "File too large")

        # Only the basename is used; the client-supplied path is never trusted.
        filename = Path(file.filename or "upload").name or "upload"

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / filename
            path.write_bytes(content)
            with entry.lock:
                before = len(entry.node.data_points)
                try:
                    entry.node.process_data(str(path))
                except Exception as e:
                    entry.node.logger.exception("process_data failed")
                    raise HTTPException(
                        status.HTTP_502_BAD_GATEWAY, f"Processing failed: {type(e).__name__}"
                    )
                redundant = len(entry.node.data_points) == before

        return {**_summary(node_id, entry.node), "redundant": redundant}

    return router


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="InFact Agents API")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    app.include_router(create_router(settings))
    return app
