"""Request/response models for the InFact API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CreateNodeRequest(BaseModel):
    hypothesis: str = Field(min_length=1)
    prior_log_odds: float = 0.0
    # Optional per-node backend override. API keys are never accepted here.
    llm_provider: str | None = None
    llm_model: str | None = None


class DataPointOut(BaseModel):
    metadata: dict[str, Any]
    raw_data: dict[str, Any]
    l_plus: float
    l_minus: float
    posterior: float
    confidence_assessment: dict[str, Any]
    analysis_rationale: str


class NodeSummary(BaseModel):
    id: str
    hypothesis: str
    prior_log_odds: float
    current_posterior: float
    probability: float
    interval: tuple[float, float]
    num_data_points: int


class NodeDetail(NodeSummary):
    data_points: list[DataPointOut]


class ProcessResponse(NodeSummary):
    redundant: bool
