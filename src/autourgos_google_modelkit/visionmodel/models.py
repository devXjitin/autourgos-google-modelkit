"""Typed model identifiers for Google vision model integrations."""

from enum import Enum
from typing import Any, Dict, Optional


class GOOGLE_VISION_MODEL_NAME(str, Enum):
    """Supported Google Gemini vision-capable model identifiers."""

    GEMINI_3_1_PRO_PREVIEW = "gemini-3.1-pro-preview"
    GEMINI_3_FLASH_PREVIEW = "gemini-3-flash-preview"
    GEMINI_3_1_FLASH_LITE_PREVIEW = "gemini-3.1-flash-lite-preview"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"


class GOOGLE_VISION_THINKING_LEVEL(str, Enum):
    """Supported Gemini thinking levels."""

    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class GOOGLE_VISION_MEDIA_RESOLUTION(str, Enum):
    """Supported media resolution levels for vision inputs."""

    LOW = "media_resolution_low"
    MEDIUM = "media_resolution_medium"
    HIGH = "media_resolution_high"


MODEL_PRICING_USD_PER_MILLION: Dict[str, Dict[str, Any]] = {
    "gemini-3.1-pro-preview": {
        "input": 2.00,
        "output": 12.00,
        "input_over_200k": 4.00,
        "output_over_200k": 18.00,
        "threshold_prompt_tokens": 200_000,
    },
    "gemini-3-flash-preview": {
        "input": 0.50,
        "output": 3.00,
    },
    "gemini-3.1-flash-lite-preview": {
        "input": 0.25,
        "output": 1.50,
    },
    "gemini-2.5-pro": {
        "input": 1.25,
        "output": 10.00,
        "input_over_200k": 2.50,
        "output_over_200k": 15.00,
        "threshold_prompt_tokens": 200_000,
    },
    "gemini-2.5-flash": {
        "input": 0.30,
        "output": 2.50,
    },
    "gemini-2.5-flash-lite": {
        "input": 0.10,
        "output": 0.40,
    },
}


def resolve_model_pricing(
    model_name: str,
    *,
    prompt_tokens: Optional[int] = None,
) -> Optional[Dict[str, float]]:
    """Resolve effective input/output token rates for a model."""
    key = model_name.strip().lower()
    entry = MODEL_PRICING_USD_PER_MILLION.get(key)
    if not entry:
        return None

    threshold = entry.get("threshold_prompt_tokens")
    if (
        isinstance(threshold, int)
        and isinstance(prompt_tokens, int)
        and prompt_tokens > threshold
    ):
        input_rate = float(entry.get("input_over_200k", entry["input"]))
        output_rate = float(entry.get("output_over_200k", entry["output"]))
    else:
        input_rate = float(entry["input"])
        output_rate = float(entry["output"])

    return {
        "input_rate_per_million": input_rate,
        "output_rate_per_million": output_rate,
    }


__all__ = [
    "GOOGLE_VISION_MODEL_NAME",
    "GOOGLE_VISION_THINKING_LEVEL",
    "GOOGLE_VISION_MEDIA_RESOLUTION",
    "MODEL_PRICING_USD_PER_MILLION",
    "resolve_model_pricing",
]
