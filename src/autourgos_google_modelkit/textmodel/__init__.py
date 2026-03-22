"""
Autourgos Google ModelKit - TextModel
==================================

Google Gemini text model integration for Autourgos.

Example:
    >>> from autourgos_google_modelkit.textmodel import GOOGLE_TEXT_MODEL_NAME, GOOGLE_TEXT_THINKING_LEVEL, GoogleTextModel
    >>> 
    >>> # Class-based
    >>> llm = GoogleTextModel(model=GOOGLE_TEXT_MODEL_NAME.GEMINI_3_FLASH_PREVIEW, api_key="your-key")
    >>> response = llm.invoke("What is AI?")

Author: Autourgos Developer
Version: 0.1.0
"""

from .models import (
    GOOGLE_TEXT_MODEL_NAME,
    GOOGLE_TEXT_THINKING_LEVEL,
    MODEL_PRICING_USD_PER_MILLION,
    resolve_model_pricing,
)
from .base import (
    GoogleTextModel,
    GoogleTextModelError,
    GoogleTextModelAPIError,
    GoogleTextModelImportError,
    GoogleTextModelResponseError,
)

__all__ = [
    "GOOGLE_TEXT_MODEL_NAME",
    "GOOGLE_TEXT_THINKING_LEVEL",
    "MODEL_PRICING_USD_PER_MILLION",
    "resolve_model_pricing",
    "GoogleTextModel",
    "GoogleTextModelError",
    "GoogleTextModelAPIError",
    "GoogleTextModelImportError",
    "GoogleTextModelResponseError",
]

