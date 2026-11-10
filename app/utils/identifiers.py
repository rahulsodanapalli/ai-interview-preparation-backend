"""Helpers for normalizing identifiers and enum values."""

from typing import Any, Optional


def normalize_document_id(document_id: Optional[str]) -> str:
    """Normalize a document or source ID by trimming whitespace and stripping chunk suffixes."""
    clean = (document_id or "").strip()
    if "#" in clean:
        clean = clean.split("#")[0].strip()
    return clean


def enum_to_value(value: Any) -> str:
    """Extract string value from an Enum instance or coerce non-enum values to str."""
    return value.value if hasattr(value, "value") else str(value)

