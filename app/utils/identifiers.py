"""Helpers for normalizing identifiers and enum values."""

import inspect
from typing import Any, Callable, Optional


def normalize_document_id(document_id: Optional[str]) -> str:
    """Normalize a document or source ID by trimming whitespace and stripping chunk suffixes."""
    clean = (document_id or "").strip()
    if "#" in clean:
        clean = clean.split("#")[0].strip()
    return clean


def enum_to_value(value: Any) -> str:
    """Extract string value from an Enum instance or coerce non-enum values to str."""
    return value.value if hasattr(value, "value") else str(value)


def supports_kwarg(fn: Callable[..., Any], name: str) -> bool:
    """Return True if fn explicitly declares keyword parameter `name` or accepts **kwargs."""
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return False
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD or param.name == name:
            return True
    return False

