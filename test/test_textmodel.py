import pytest

from autourgos_google_modelkit.textmodel import (
    GoogleTextModel,
    GoogleTextModelImportError,
    GOOGLE_TEXT_MODEL_NAME,
    GOOGLE_TEXT_THINKING_LEVEL,
)


def test_textmodel_requires_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    model = GoogleTextModel(model=GOOGLE_TEXT_MODEL_NAME.GEMINI_3_FLASH_PREVIEW)
    with pytest.raises(GoogleTextModelImportError):
        model.invoke("Hello")


def test_textmodel_parameter_validation_top_p():
    model = GoogleTextModel(
        model=GOOGLE_TEXT_MODEL_NAME.GEMINI_3_FLASH_PREVIEW,
        top_p=1.5,
    )
    with pytest.raises(ValueError, match="top_p must be between 0.0 and 1.0"):
        model.invoke("Hello")


def test_textmodel_allows_default_thinking_none_for_unsupported_model(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    model = GoogleTextModel(model=GOOGLE_TEXT_MODEL_NAME.GEMINI_2_5_FLASH)
    with pytest.raises(GoogleTextModelImportError):
        model.invoke("Hello")


def test_textmodel_rejects_thinking_level_for_unsupported_model():
    model = GoogleTextModel(
        model=GOOGLE_TEXT_MODEL_NAME.GEMINI_2_5_FLASH,
        thinking_level=GOOGLE_TEXT_THINKING_LEVEL.LOW,
    )
    with pytest.raises(ValueError, match="thinking_level is not supported"):
        model.invoke("Hello")
