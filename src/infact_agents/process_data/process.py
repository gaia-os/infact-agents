"""Data ingestion: parsing, metadata extraction and redundancy checks."""

from __future__ import annotations

import base64
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..llm import Attachment, LLMClient
from ..settings import Settings
from ..utils import extract_code_blocks

IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

_PARSE_PROMPT = """
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


def _truncate(text: str, limit: int, logger: logging.Logger) -> str:
    if len(text) > limit:
        logger.warning(f"Input truncated from {len(text)} to {limit} characters")
        return text[:limit]
    return text


def _pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _load_input(
    path: Path, llm: LLMClient, settings: Settings, logger: logging.Logger
) -> tuple[str, list[Attachment]]:
    """Return (text, attachments) appropriate for the given LLM's capabilities."""
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return _truncate(pd.read_csv(path).to_string(), settings.max_input_chars, logger), []

    if suffix == ".pdf":
        if llm.supports_pdf:
            data = base64.b64encode(path.read_bytes()).decode("utf-8")
            return "", [Attachment("pdf", "application/pdf", data)]
        logger.info("LLM lacks native PDF support, extracting text with pypdf")
        return _truncate(_pdf_text(path), settings.max_input_chars, logger), []

    if suffix in IMAGE_TYPES:
        if not llm.supports_images:
            raise ValueError(f"The configured LLM does not support image input ({path.name})")
        data = base64.b64encode(path.read_bytes()).decode("utf-8")
        return "", [Attachment("image", IMAGE_TYPES[suffix], data)]

    text = path.read_text(encoding="utf-8", errors="replace")
    return _truncate(text, settings.max_input_chars, logger), []


def _parse_json_response(response_text: str) -> dict | None:
    for lang, block in extract_code_blocks(response_text):
        if lang.lower() in ("json", ""):
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    # Last resort: outermost braces
    start, end = response_text.find("{"), response_text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(response_text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


def parse_data(
    data_file: str,
    hypothesis: str,
    llm: LLMClient,
    settings: Settings,
    logger: logging.Logger,
) -> dict:
    """Parse a data file into structured evidence using the LLM."""
    logger.info(f"Parsing data file: {data_file}")
    text, attachments = _load_input(Path(data_file), llm, settings, logger)

    prompt = _PARSE_PROMPT.format(hypothesis=hypothesis)
    if text:
        prompt = f"Document content:\n{text}\n\n{prompt}"

    logger.info("Sending parse request to LLM")
    response_text = llm.generate(prompt, attachments=attachments)
    logger.debug(f"LLM response: {response_text}")

    parsed = _parse_json_response(response_text)
    if parsed is None:
        logger.error("Failed to parse response as JSON")
        return {
            "extraction_error": "Failed to parse LLM response",
            "raw_response": response_text,
        }
    return parsed


def extract_metadata(data_file: str, logger: logging.Logger) -> dict[str, Any]:
    """Extract file-level metadata (size, PDF info, image info, ...)."""
    logger.info(f"Extracting metadata from {data_file}")
    try:
        path = Path(data_file)
        stat = path.stat()
        suffix = path.suffix.lower()
        meta: dict[str, Any] = {
            "filename": path.name,
            "file_type": suffix,
            "file_size": stat.st_size,
            "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "source_path": str(path.absolute()),
        }

        if suffix == ".pdf":
            from pypdf import PdfReader

            pdf = PdfReader(str(path))
            info = pdf.metadata
            meta["page_count"] = len(pdf.pages)
            if info:
                meta.update(
                    {
                        "title": info.get("/Title", ""),
                        "author": info.get("/Author", ""),
                        "creator": info.get("/Creator", ""),
                        "producer": info.get("/Producer", ""),
                        "creation_date": info.get("/CreationDate", ""),
                        "modification_date": info.get("/ModDate", ""),
                    }
                )

        elif suffix in IMAGE_TYPES:
            from PIL import Image

            with Image.open(path) as img:
                meta.update(
                    {
                        "image_format": img.format,
                        "image_size": img.size,
                        "image_mode": img.mode,
                    }
                )

        elif suffix in (".html", ".htm"):
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
            meta_tags = {}
            for tag in soup.find_all("meta"):
                name = tag.get("name", tag.get("property", ""))
                content = tag.get("content", "")
                if name and content:
                    meta_tags[name] = content
            meta.update(
                {
                    "title": soup.title.string if soup.title else "",
                    "meta_tags": meta_tags,
                    "has_article": bool(soup.find("article")),
                    "has_main": bool(soup.find("main")),
                    "num_headers": len(soup.find_all(["h1", "h2", "h3"])),
                    "has_tables": bool(soup.find_all("table")),
                }
            )

        elif suffix == ".csv":
            df = pd.read_csv(path)
            meta.update(
                {
                    "num_rows": len(df),
                    "num_columns": len(df.columns),
                    "column_names": list(df.columns),
                    "data_types": {c: str(t) for c, t in df.dtypes.items()},
                    "has_nulls": bool(df.isnull().any().any()),
                }
            )

        logger.debug(f"Extracted metadata: {json.dumps(meta, indent=2, default=str)}")
        return meta

    except Exception as e:
        logger.error(f"Error in extract_metadata: {e}", exc_info=True)
        return {"filename": data_file, "error": str(e)}


def is_redundant(
    new_data: dict,
    existing: list[dict],
    llm: LLMClient,
    logger: logging.Logger,
) -> bool:
    """Ask the LLM whether ``new_data`` duplicates any existing raw data."""
    logger.info("Checking for data redundancy")
    if not existing:
        return False

    prompt = f"""
Compare the following new data:
{json.dumps(new_data, indent=2)}

With these existing data points:
{json.dumps(existing, indent=2)}

Is the new data redundant with any existing data points?
Consider:
1. Same source or study being cited
2. Same measurements within uncertainty
3. Derived results from already incorporated primary data

Return only "true" if redundant, "false" if novel information.
"""
    response = llm.generate(prompt, max_tokens=4096).strip().lower()
    logger.debug(f"Redundancy check response: {response}")
    result = response.strip("\"'`. \n").startswith("true")
    logger.info(f"Redundancy check result: {result}")
    return result
