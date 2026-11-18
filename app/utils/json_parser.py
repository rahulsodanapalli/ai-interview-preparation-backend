"""Utilities for cleaning markdown code fences and parsing JSON returned by LLMs."""

import json
import re
from typing import Any


_FENCE_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def strip_markdown_fences(raw_text: str) -> str:
    """Strip surrounding or embedded ``` / ```json markdown code fences from LLM output."""
    cleaned = (raw_text or "").strip()
    if not cleaned:
        return ""
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned.strip()
    match = _FENCE_BLOCK_RE.search(cleaned)
    if match:
        return match.group(1).strip()
    return cleaned


def _looks_truncated_json(text: str) -> bool:
    """Heuristic check for JSON that was cut off mid-stream (unbalanced delimiters or at EOF)."""
    stripped = text.strip()
    if not stripped or stripped[0] not in ("{", "["):
        return False

    depth = 0
    in_string = False
    escape = False
    for ch in stripped:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ("{", "["):
            depth += 1
        elif ch in ("}", "]"):
            depth -= 1

    return in_string or depth > 0


def parse_llm_json(
    raw_text: str,
    *,
    error_cls: type[Exception] = ValueError,
    decode_error_message: str = "Failed to parse JSON response from the local AI model.",
    empty_error_message: str = "The local AI model returned an empty JSON response.",
    truncated_error_message: str = "The local AI model returned truncated/incomplete JSON.",
    expect_dict: bool = True,
    type_error_message: str = "The AI model returned an unexpected JSON structure.",
) -> Any:
    """Extract, decode, and validate JSON from LLM output."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise error_cls(empty_error_message)

    cleaned = strip_markdown_fences(raw_text)
    if not cleaned:
        raise error_cls(empty_error_message)

    parsed: Any = None
    last_exc: Exception | None = None
    truncated: bool = False

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        last_exc = exc
        decoder = json.JSONDecoder()
        target_chars = ("{",) if expect_dict else ("{", "[")
        if cleaned[0] in target_chars:
            # Root JSON starts at index 0; allow trailing prose via raw_decode(0),
            # but never skip a broken root '{' to extract an inner nested object.
            try:
                candidate, _end = decoder.raw_decode(cleaned, 0)
                parsed = candidate
                last_exc = None
            except json.JSONDecodeError:
                truncated = _looks_truncated_json(cleaned)
        else:
            # Try decoding the first valid JSON object/array if preceded by explanatory text
            for idx, ch in enumerate(cleaned):
                if ch in target_chars:
                    try:
                        candidate, _end = decoder.raw_decode(cleaned, idx)
                        parsed = candidate
                        last_exc = None
                        break
                    except json.JSONDecodeError:
                        if _looks_truncated_json(cleaned[idx:]):
                            truncated = True
                            break
                        continue

    if last_exc is not None:
        if truncated or _looks_truncated_json(cleaned):
            raise error_cls(truncated_error_message) from last_exc
        raise error_cls(decode_error_message) from last_exc

    if expect_dict and not isinstance(parsed, dict):
        raise error_cls(type_error_message)

    return parsed

