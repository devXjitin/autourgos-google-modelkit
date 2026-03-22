import pytest

from autourgos_google_modelkit.textmodel import (
    GoogleTextModel,
    GoogleTextModelImportError,
    GOOGLE_TEXT_MODEL_NAME,
)


def test_textmodel_requires_api_key():
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
