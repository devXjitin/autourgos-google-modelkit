import pytest
from pathlib import Path

from autourgos_google_modelkit.visionmodel import (
    GoogleVisionModel,
    GoogleVisionModelImportError,
    GOOGLE_VISION_MODEL_NAME,
    GOOGLE_VISION_THINKING_LEVEL,
)


def test_visionmodel_requires_image_input():
    model = GoogleVisionModel(model=GOOGLE_VISION_MODEL_NAME.GEMINI_3_FLASH_PREVIEW)
    with pytest.raises(ValueError, match="At least one image must be provided"):
        model.invoke(prompt="Describe this")


def test_visionmodel_requires_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    model = GoogleVisionModel(model=GOOGLE_VISION_MODEL_NAME.GEMINI_3_FLASH_PREVIEW)
    fake_jpeg_header = b"\xff\xd8\xff\xe0" + b"1234"
    with pytest.raises(GoogleVisionModelImportError):
        model.invoke(prompt="Describe this", image=fake_jpeg_header)


def test_visionmodel_accepts_real_image_path(monkeypatch):
    model = GoogleVisionModel(model=GOOGLE_VISION_MODEL_NAME.GEMINI_3_FLASH_PREVIEW)
    image_path = Path(__file__).with_name("image.png")
    assert image_path.exists()

    # Ensure we only validate local image-path handling, not external API calls.
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(GoogleVisionModelImportError):
        model.invoke(prompt="Describe this", image=str(image_path))


def test_visionmodel_rejects_thinking_level_for_unsupported_model():
    model = GoogleVisionModel(
        model=GOOGLE_VISION_MODEL_NAME.GEMINI_2_5_FLASH,
        thinking_level=GOOGLE_VISION_THINKING_LEVEL.LOW,
    )
    fake_jpeg_header = b"\xff\xd8\xff\xe0" + b"1234"
    with pytest.raises(ValueError, match="thinking_level is not supported"):
        model.invoke(prompt="Describe this", image=fake_jpeg_header)
