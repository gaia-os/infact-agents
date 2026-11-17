"""All LLM prompts used by InFact nodes, in one importable place.

Each prompt is a function returning the rendered string, so callers never
build prompt text inline and prompts can be reused or overridden easily.
"""

from __future__ import annotations

import json

_PARSE_TEMPLATE = """
Extract relevant data points for evaluating the hypothesis:
"{hypothesis}"

Provide your response as a JSON code block, like this:
```json
{{
    "numerical_values": [],
    "metadata": {{}},
    "issues": [],
    "confidence_assessment": {{
        "confidence_score": 0.75,
        "explanation": "Detailed explanation of confidence level",
        "key_strengths": ["Strength 1", "Strength 2"],
        "key_limitations": ["Limitation 1", "Limitation 2"]
    }}
}}
```

The confidence_assessment should:
1. Include a confidence_score between 0 and 1
2. Provide a detailed explanation of the confidence level
3. List key strengths of the evidence
4. List key limitations or potential issues

The overall JSON should include:
1. Extracted numerical values and their uncertainties
2. Relevant metadata (source quality, methodology, etc.)
3. Any potential issues or biases in the data
"""

_ANALYSIS_TEMPLATE = """
Given this data:
{data}

Generate Python code to calculate log likelihoods for the hypothesis:
"{hypothesis}"

This should be a single function named `calculate_log_likelihoods`.
It should take a single argument, a dict with the format given above,
and output only the tuple of log-likelihoods
  l_plus = log P(data | hypothesis),
  l_minus = log P(data | not hypothesis).

The code should:
1. Calculate l_plus and l_minus (log likelihoods)
2. Handle uncertainties properly
3. Account for data quality and potential biases
4. Limit overconfidence by capping the absolute difference between l_plus and l_minus to {cap}.
5. Use the usual libraries such as numpy and scipy for calculations
6. Use print() to output intermediate results, as well as the final result before returning.

Return only executable Python code in a ```python code block with the function definition.
Do not include the function call itself.
"""

_DEBUG_TEMPLATE = """
The following code failed with error: {error}

Code:
```python
{code}
```

Input data:
```json
{data}
```

Please fix the code to:
1. Handle the error properly
2. Return numeric values for l_plus and l_minus
3. Include proper error checking
4. Handle edge cases in the input data

Return only the corrected Python code in a ```python code block.
"""

_REDUNDANCY_TEMPLATE = """
Compare the following new data:
{new_data}

With these existing data points:
{existing}

Is the new data redundant with any existing data points?
Consider:
1. Same source or study being cited
2. Same measurements within uncertainty
3. Derived results from already incorporated primary data

Return only "true" if redundant, "false" if novel information.
"""


def parse_prompt(hypothesis: str, document_text: str = "") -> str:
    prompt = _PARSE_TEMPLATE.format(hypothesis=hypothesis)
    if document_text:
        prompt = f"Document content:\n{document_text}\n\n{prompt}"
    return prompt


def analysis_prompt(data: dict, hypothesis: str, cap: float) -> str:
    return _ANALYSIS_TEMPLATE.format(
        data=json.dumps(data, indent=2), hypothesis=hypothesis, cap=cap
    )


def debug_prompt(code: str, data: dict, error: object) -> str:
    return _DEBUG_TEMPLATE.format(error=error, code=code, data=json.dumps(data, indent=2))


def redundancy_prompt(new_data: dict, existing: list[dict]) -> str:
    return _REDUNDANCY_TEMPLATE.format(
        new_data=json.dumps(new_data, indent=2), existing=json.dumps(existing, indent=2)
    )
