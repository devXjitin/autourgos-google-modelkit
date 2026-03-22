"""Prompt-template related helpers."""

from __future__ import annotations

from string import Formatter
from typing import Any


def extract_template_fields(template: str) -> set[str]:
    """Extract placeholder field names from a format template string."""
    fields: set[str] = set()
    for _, field_name, _, _ in Formatter().parse(template):
        if not field_name:
            continue
        base_name = field_name.split("!", 1)[0].split(":", 1)[0].strip()
        if base_name:
            fields.add(base_name)
    return fields


def coerce_prompt_variable(value: Any) -> str:
    """Convert prompt variables into strings for template rendering."""
    if value is None:
        return ""
    return str(value)
