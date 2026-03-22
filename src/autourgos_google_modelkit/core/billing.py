"""Token-cost and structured-output helpers."""

from __future__ import annotations

from typing import Any, Callable, Optional

from .response import extract_usage_metadata


def calculate_cost_usd(
    *,
    model_name: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    resolve_model_pricing: Callable[..., Optional[dict[str, float]]],
) -> dict[str, Any]:
    """Calculate request cost from token usage and model pricing function."""
    pricing = resolve_model_pricing(model_name, prompt_tokens=input_tokens)
    if pricing is None:
        return {
            "value": None,
            "display": "N/A",
            "input_rate_per_million": None,
            "output_rate_per_million": None,
        }

    safe_input_tokens = input_tokens or 0
    safe_output_tokens = output_tokens or 0

    input_cost = (safe_input_tokens / 1_000_000) * pricing["input_rate_per_million"]
    output_cost = (safe_output_tokens / 1_000_000) * pricing["output_rate_per_million"]
    total_cost = input_cost + output_cost

    return {
        "value": round(total_cost, 8),
        "display": f"${total_cost:.8f}",
        "input_rate_per_million": pricing["input_rate_per_million"],
        "output_rate_per_million": pricing["output_rate_per_million"],
    }


def build_structured_output(
    *,
    model_name: str,
    response_text: str,
    raw_response: Any,
    resolve_model_pricing: Callable[..., Optional[dict[str, float]]],
    extra_fields: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a normalized structured response payload with usage and cost."""
    usage = extract_usage_metadata(raw_response)
    input_tokens = usage["input_tokens"]
    output_tokens = usage["output_tokens"]
    total_tokens = usage["total_tokens"]

    cost = calculate_cost_usd(
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        resolve_model_pricing=resolve_model_pricing,
    )

    payload: dict[str, Any] = {
        "model": model_name,
        "response": response_text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "Total_tokens": total_tokens,
        "Cost": cost["display"],
        "cost_details": {
            "value_usd": cost["value"],
            "input_rate_per_million": cost["input_rate_per_million"],
            "output_rate_per_million": cost["output_rate_per_million"],
        },
    }

    if extra_fields:
        payload.update(extra_fields)

    return payload
