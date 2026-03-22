"""Shared core utilities for Gemini-based model providers."""

from .runtime import (
    configure_runtime_environment,
    suppress_stderr,
)
from .sdk import (
    load_genai_module,
    configure_genai_client,
)
from .normalization import (
    normalize_model_name,
    normalize_thinking_level,
    validate_thinking_level_support,
    resolve_api_key,
    build_generation_config,
)
from .prompting import (
    extract_template_fields,
    coerce_prompt_variable,
)
from .response import (
    extract_text_from_response,
    extract_usage_metadata,
)
from .billing import (
    calculate_cost_usd,
    build_structured_output,
)

__all__ = [
    "configure_runtime_environment",
    "suppress_stderr",
    "load_genai_module",
    "normalize_model_name",
    "normalize_thinking_level",
    "validate_thinking_level_support",
    "resolve_api_key",
    "build_generation_config",
    "extract_template_fields",
    "coerce_prompt_variable",
    "configure_genai_client",
    "extract_text_from_response",
    "extract_usage_metadata",
    "calculate_cost_usd",
    "build_structured_output",
]
