"""Input normalization and validation helpers for Gemini providers."""

from __future__ import annotations

from typing import Optional, Any
import os


def normalize_model_name(model: Any) -> str:
    """Normalize model input into a stable model-id string."""
    if isinstance(model, str):
        model_name = model.strip()
        if model_name:
            return model_name
        raise ValueError("model must be a non-empty string")

    value = getattr(model, "value", None)
    if isinstance(value, str) and value.strip():
        return value.strip()

    model_name = str(model).strip()
    if model_name:
        return model_name
    raise ValueError("model must be a non-empty string")


def normalize_thinking_level(thinking_level: Any) -> Optional[str]:
    """Normalize thinking level input into supported values."""
    if thinking_level is None:
        return None

    value = getattr(thinking_level, "value", thinking_level)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"minimal", "low", "medium", "high"}:
            return normalized

    raise ValueError("thinking_level must be one of: minimal, low, medium, high")


def validate_thinking_level_support(model_name: str, thinking_level: Optional[str]) -> None:
    """Validate known model-specific thinking-level constraints."""
    if thinking_level is None:
        return

    model_name_normalized = model_name.strip().lower()
    thinking_supported_models = {
        "gemini-3.1-pro-preview",
        "gemini-3-pro-preview",
        "gemini-3.1-pro-preview-customtools",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview",
    }

    if model_name_normalized not in thinking_supported_models:
        raise ValueError(
            f"thinking_level is not supported for model '{model_name}'. "
            "Use thinking_level=None or switch to a Gemini 3 thinking-capable model."
        )

    minimal_unsupported_models = {
        "gemini-3.1-pro-preview",
        "gemini-3-pro-preview",
        "gemini-3.1-pro-preview-customtools",
    }

    if thinking_level == "minimal" and model_name_normalized in minimal_unsupported_models:
        raise ValueError(
            f"thinking_level='minimal' is not supported for model '{model_name}'. "
            "Use low, medium, or high."
        )


def resolve_api_key(api_key: Optional[str]) -> Optional[str]:
    """Resolve API key from explicit input and supported env variables."""
    return api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")


def build_generation_config(
    *,
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    max_tokens: Optional[int],
    thinking_level: Optional[str],
    media_resolution: Optional[str] = None,
) -> dict[str, Any]:
    """Build generation config dictionary for SDK calls."""
    generation_config: dict[str, Any] = {}
    if temperature is not None:
        generation_config["temperature"] = temperature
    if top_p is not None:
        generation_config["top_p"] = top_p
    if top_k is not None:
        generation_config["top_k"] = top_k
    if max_tokens is not None:
        generation_config["max_output_tokens"] = max_tokens
    if thinking_level is not None:
        generation_config["thinking_config"] = {"thinking_level": thinking_level}
    if media_resolution is not None:
        generation_config["media_resolution"] = media_resolution
    return generation_config
