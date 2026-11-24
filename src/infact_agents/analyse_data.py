"""LLM-generated likelihood analysis with a self-debugging execution loop.

WARNING: the generated code is run with ``exec`` in-process and is NOT
sandboxed. Only use with models/inputs you trust, or run inside a container.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import scipy

from .llm import LLMClient
from .prompts import analysis_prompt, debug_prompt
from .settings import Settings
from .utils import first_code_block

_PY = ("python", "py", "")


def analyse_data(
    data: dict,
    hypothesis: str,
    llm: LLMClient,
    settings: Settings,
    logger: logging.Logger,
) -> tuple[float, float, str]:
    """Generate and execute analysis code. Returns ``(l_plus, l_minus, code)``."""
    logger.info("Analysis stage: generating Python likelihood function from parsed evidence")
    prompt = analysis_prompt(data, hypothesis, settings.max_log_likelihood_ratio)
    logger.debug(f"Analysis prompt: {prompt}")
    logger.info("Analysis stage: requesting initial analysis code from LLM")
    response = llm.generate(prompt)
    logger.debug(f"LLM response: {response}")

    code = first_code_block(response, _PY)
    if not code:
        logger.error("No Python code block found in LLM response")
        raise ValueError("LLM response did not contain a Python code block")

    return execute_code_with_debug(code, data, llm, settings, logger)


def _clamp(l_plus: float, l_minus: float, cap: float) -> tuple[float, float]:
    """Enforce |l_plus - l_minus| <= cap, preserving the midpoint."""
    diff = l_plus - l_minus
    if abs(diff) <= cap:
        return l_plus, l_minus
    mid = (l_plus + l_minus) / 2
    half = math.copysign(cap, diff) / 2
    return mid + half, mid - half


def execute_code_with_debug(
    code: str,
    data: dict,
    llm: LLMClient,
    settings: Settings,
    logger: logging.Logger,
) -> tuple[float, float, str]:
    """Run ``calculate_log_likelihoods``; on failure ask the LLM to fix it."""
    max_attempts = settings.max_debug_attempts
    logger.info("Analysis stage: executing generated code with auto-debug fallback")

    for attempt in range(1, max_attempts + 1):
        logger.info(f"Code execution attempt {attempt}/{max_attempts}")
        logger.debug(f"Executing code:\n{code}")

        namespace = {"np": np, "numpy": np, "math": math, "scipy": scipy}
        try:
            exec(code, namespace)
            l_plus, l_minus = namespace["calculate_log_likelihoods"](data)

            if not all(isinstance(v, (int, float, np.number)) for v in (l_plus, l_minus)):
                raise ValueError("l_plus and l_minus must be numeric values")
            if not (math.isfinite(l_plus) and math.isfinite(l_minus)):
                raise ValueError("l_plus and l_minus must be finite")

            l_plus, l_minus = _clamp(float(l_plus), float(l_minus), settings.max_log_likelihood_ratio)
            logger.info(f"Code execution successful - l_plus: {l_plus}, l_minus: {l_minus}")
            return l_plus, l_minus, code

        except Exception as e:
            logger.warning(f"Code execution failed on attempt {attempt}: {e}")
            if attempt == max_attempts:
                raise RuntimeError(
                    f"Failed to generate working code after {max_attempts} attempts. "
                    f"Final error: {e}"
                ) from e

            fix_prompt = debug_prompt(code, data, e)
            logger.info("Requesting corrected code from LLM after execution failure")
            logger.debug(f"Sending debug prompt to LLM:\n{fix_prompt}")
            fixed = first_code_block(llm.generate(fix_prompt), _PY)
            if fixed:
                code = fixed
                logger.info("Received corrected Python code, retrying execution")
            else:
                logger.error("No Python code block found in debug response")

    raise RuntimeError("Unexpected error in debug loop")  # pragma: no cover
