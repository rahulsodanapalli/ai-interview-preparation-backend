"""Utilities for cleaning markdown code fences and parsing JSON returned by LLMs."""

import json
import re
from typing import Any


def strip_markdown_fences(raw_text: str) -> str:
    """Strip surrounding ``` or ```json markdown code fences from LLM output."""
    cleaned = (raw_text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()
    return cleaned


def parse_llm_json(
    raw_text: str,
    *,
    error_cls: type[Exception] = ValueError,
    decode_error_message: str = "Failed to parse JSON response from the local AI model.",
    expect_dict: bool = True,
    type_error_message: str = "The AI model returned an unexpected JSON structure.",
) -> Any:
    """Strip markdown fences, decode JSON, and optionally enforce a JSON object (dict)."""
    cleaned = strip_markdown_fences(raw_text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise error_cls(decode_error_message) from exc

    if expect_dict and not isinstance(parsed, dict):
        raise error_cls(type_error_message)

    return parsed

