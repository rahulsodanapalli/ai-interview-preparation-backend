"""Shared helpers for Ollama /api/tags model discovery and matching."""

from typing import Any, Dict, List


def extract_available_models(tags_payload: Dict[str, Any]) -> List[str]:
    """Extract installed model names from an Ollama /api/tags JSON response."""
    raw_models = tags_payload.get("models", [])
    return [
        m.get("name")
        for m in raw_models
        if isinstance(m, dict) and m.get("name")
    ]


def is_model_available(configured_model: str, available_models: List[str]) -> bool:
    """Check whether a configured Ollama model matches any installed model tag."""
    return any(
        m == configured_model
        or m.startswith(f"{configured_model}:")
        or configured_model.startswith(f"{m}:")
        for m in available_models
    )

