"""Response parsing and usage extraction helpers."""

from __future__ import annotations

from typing import Any, Optional


def extract_text_from_response(resp: Any) -> Optional[str]:
    """Extract generated text from Gemini API response shapes."""
    if resp is None:
        return None

    if isinstance(resp, str) and resp.strip():
        return resp

    try:
        text = getattr(resp, "text", None)
        if text is not None:
            if callable(text):
                text = text()
            if isinstance(text, str) and text.strip():
                return text
    except Exception:
        pass

    candidates = getattr(resp, "candidates", None)
    if candidates and isinstance(candidates, (list, tuple)) and candidates:
        first = candidates[0]
        content = getattr(first, "content", None)
        if content:
            parts = getattr(content, "parts", None)
            if parts and isinstance(parts, (list, tuple)):
                joined_parts: list[str] = []
                for part in parts:
                    part_text = getattr(part, "text", None)
                    if isinstance(part_text, str) and part_text.strip():
                        joined_parts.append(part_text)
                if joined_parts:
                    return "".join(joined_parts)

            content_text = getattr(content, "text", None)
            if isinstance(content_text, str) and content_text.strip():
                return content_text

        candidate_text = getattr(first, "text", None)
        if isinstance(candidate_text, str) and candidate_text.strip():
            return candidate_text

    if isinstance(resp, dict):
        for key in ("text", "output_text", "delta", "content"):
            value = resp.get(key)
            if isinstance(value, str) and value.strip():
                return value

        candidates = resp.get("candidates")
        if candidates and isinstance(candidates, (list, tuple)) and candidates:
            first = candidates[0]
            if isinstance(first, dict):
                content = first.get("content")
                if isinstance(content, dict):
                    parts = content.get("parts")
                    if isinstance(parts, (list, tuple)):
                        joined_parts: list[str] = []
                        for part in parts:
                            if isinstance(part, dict):
                                text = part.get("text")
                                if isinstance(text, str) and text.strip():
                                    joined_parts.append(text)
                        if joined_parts:
                            return "".join(joined_parts)

                text = first.get("content") or first.get("text")
                if isinstance(text, str) and text.strip():
                    return text

        text = resp.get("text")
        if isinstance(text, str) and text.strip():
            return text

    return None


def extract_usage_metadata(resp: Any) -> dict[str, Optional[int]]:
    """Extract input/output/total token counts from response metadata."""
    usage = getattr(resp, "usage_metadata", None)
    if usage is None:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        }

    input_tokens = getattr(usage, "prompt_token_count", None)
    output_tokens = getattr(usage, "candidates_token_count", None)
    total_tokens = getattr(usage, "total_token_count", None)

    return {
        "input_tokens": int(input_tokens) if isinstance(input_tokens, int) else None,
        "output_tokens": int(output_tokens) if isinstance(output_tokens, int) else None,
        "total_tokens": int(total_tokens) if isinstance(total_tokens, int) else None,
    }
