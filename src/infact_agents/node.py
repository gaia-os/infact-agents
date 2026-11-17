"""Model-agnostic InFact node."""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from typing import Any

from .analyse_data import analyse_data
from .llm import LLMClient, create_llm_client
from .process_data import extract_metadata, is_redundant, parse_data
from .settings import Settings
from .utils import to_probability

_DEFAULT_CONFIDENCE = {
    "confidence_score": 0,
    "explanation": "No confidence assessment available",
    "key_strengths": [],
    "key_limitations": [],
}


class InFactNode:
    """Bayesian belief node for a single hypothesis, driven by any ``LLMClient``.

    If ``llm`` is omitted, one is built from ``settings`` (default: Ollama).
    """

    def __init__(
        self,
        hypothesis: str,
        llm: LLMClient | None = None,
        prior_log_odds: float = 0.0,
        settings: Settings | None = None,
    ):
        self.settings = settings or Settings()
        self.logger = self._setup_logger()
        self.llm = llm or create_llm_client(self.settings)

        self.hypothesis = hypothesis
        self.prior_log_odds = prior_log_odds
        self.current_posterior = prior_log_odds
        self.data_points: list[dict[str, Any]] = []

        self.logger.info(f"Initialized InFactNode with hypothesis: {hypothesis}")

    # ------------------------------------------------------------------ setup
    def _setup_logger(self) -> logging.Logger:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = logging.getLogger(f"infact.node.{timestamp}.{id(self)}")
        logger.setLevel(self.settings.log_level)

        self.settings.log_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(
            self.settings.log_dir / f"infact_{timestamp}_{id(self):x}.log", encoding="utf-8"
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
        return logger

    # ------------------------------------------------------------ persistence
    def save(self, filename: str) -> None:
        """Save node state to a JSON file."""
        data = {
            "hypothesis": self.hypothesis,
            "prior_log_odds": self.prior_log_odds,
            "current_posterior": self.current_posterior,
            "data_points": [
                {
                    "metadata": dp["metadata"],
                    "raw_data": dp["raw_data"],
                    "l_plus": dp["l_plus"],
                    "l_minus": dp["l_minus"],
                    "posterior": dp["posterior"],
                    "confidence_assessment": dp.get("confidence_assessment", {}),
                    "analysis_rationale": dp.get("analysis_rationale", ""),
                }
                for dp in self.data_points
            ],
        }
        self.logger.info(f"Saving node data to {filename}")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(
        cls,
        filename: str,
        llm: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> "InFactNode":
        """Load a node from a JSON file."""
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        node = cls(
            hypothesis=data["hypothesis"],
            llm=llm,
            prior_log_odds=data["prior_log_odds"],
            settings=settings,
        )
        node.current_posterior = data["current_posterior"]
        node.data_points = data["data_points"]
        node.logger.info(f"Loaded node data from {filename}")
        return node

    # ------------------------------------------------------------- processing
    def process_data(self, data_file: str) -> tuple[float, tuple[float, float]]:
        """Process a new data file and update beliefs.

        Returns the new posterior log-odds and a 95% interval on the probability.
        """
        self.logger.info(f"Processing data file: {data_file}")
        try:
            parsed = parse_data(data_file, self.hypothesis, self.llm, self.settings, self.logger)

            existing = [dp["raw_data"] for dp in self.data_points]
            if is_redundant(parsed, existing, self.llm, self.logger):
                self.logger.info("Data determined to be redundant, skipping")
                return self.current_posterior, self._calculate_uncertainty()

            l_plus, l_minus, code = analyse_data(
                parsed, self.hypothesis, self.llm, self.settings, self.logger
            )
            new_posterior = self.current_posterior + l_plus - l_minus
            self.logger.info(
                f"Updated posterior from {self.current_posterior} to {new_posterior}"
            )

            self.data_points.append(
                {
                    "raw_data": parsed,
                    "metadata": extract_metadata(data_file, self.logger),
                    "l_plus": l_plus,
                    "l_minus": l_minus,
                    "posterior": new_posterior,
                    "confidence_assessment": parsed.get(
                        "confidence_assessment", dict(_DEFAULT_CONFIDENCE)
                    ),
                    "analysis_rationale": code,
                }
            )
            self.current_posterior = new_posterior
            lower, upper = self._calculate_uncertainty()

            self.logger.info(
                f"Processing complete. Current probability: "
                f"{self.probability:.2%} ({lower:.2%}, {upper:.2%})"
            )
            return new_posterior, (lower, upper)

        except Exception as e:
            self.logger.error(f"Error processing {data_file}: {e}", exc_info=True)
            raise

    # ----------------------------------------------------------- probabilities
    @property
    def probability(self) -> float:
        return to_probability(self.current_posterior)

    def _calculate_uncertainty(self) -> tuple[float, float]:
        """95% interval on the posterior probability via a Beta approximation."""
        from scipy import stats

        p = self.probability
        total_evidence = sum(
            abs(math.exp(dp["l_plus"] - dp["l_minus"]) - 1) for dp in self.data_points
        )
        if total_evidence < 1e-6:
            return (0.0, 1.0)

        alpha = total_evidence * p
        beta = total_evidence * (1 - p)
        low, high = stats.beta.interval(0.95, alpha, beta)
        return max(0.0, min(1.0, float(low))), max(0.0, min(1.0, float(high)))
