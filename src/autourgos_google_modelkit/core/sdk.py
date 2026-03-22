"""Gemini SDK loading and client configuration helpers."""

from __future__ import annotations

from typing import Any, Optional

from .runtime import suppress_stderr


def load_genai_module() -> tuple[bool, Any, Optional[str]]:
    """Load google Gemini SDK from supported package paths."""
    try:
        with suppress_stderr():
            try:
                import google.generativeai as genai_module  # type: ignore

                return True, genai_module, None
            except Exception:
                from google import genai as genai_module  # type: ignore

                return True, genai_module, None
    except Exception as exc:
        return False, None, str(exc)


def configure_genai_client(genai: Any, api_key: str) -> Any:
    """Configure SDK auth state and optionally construct a client instance."""
    client = None

    try:
        with suppress_stderr():
            cfg = getattr(genai, "configure", None)
            if callable(cfg):
                cfg(api_key=api_key)
    except Exception:
        pass

    try:
        with suppress_stderr():
            client_cls = getattr(genai, "Client", None)
            if callable(client_cls):
                try:
                    client = client_cls(api_key=api_key)
                except TypeError:
                    client = client_cls()
    except Exception:
        pass

    return client
