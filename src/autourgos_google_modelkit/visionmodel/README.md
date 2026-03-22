# Vision Model

This module provides a production-oriented wrapper for Gemini multimodal calls (image + text input, text output).

## Public Exports

- `GOOGLE_VISION_MODEL_NAME`
- `GOOGLE_VISION_THINKING_LEVEL`
- `GOOGLE_VISION_MEDIA_RESOLUTION`
- `MODEL_PRICING_USD_PER_MILLION`
- `resolve_model_pricing`
- `GoogleVisionModel`
- `GoogleVisionModelError`
- `GoogleVisionModelImportError`
- `GoogleVisionModelAPIError`
- `GoogleVisionModelResponseError`

## Quick Start

```python
from autourgos_google_modelkit.visionmodel import GoogleVisionModel, GOOGLE_VISION_MODEL_NAME

vision = GoogleVisionModel(model=GOOGLE_VISION_MODEL_NAME.GEMINI_3_FLASH_PREVIEW)

print(
    vision.invoke(
        prompt="Describe what is visible in this image.",
        image="./sample.jpg",
    )
)
```

## Supported Image Inputs

Single image via `image=` and multiple images via `images=` are supported.

Accepted image types:

- path (`str` or `pathlib.Path`)
- raw bytes (`bytes`)
- dict: `{"data": bytes, "mime_type": "image/png"}`
- tuple: `(bytes, "image/png")`

## Constructor Highlights

`GoogleVisionModel(...)` supports:

- `model` (enum or string)
- `api_key`
- `prompt_template`
- `temperature`, `top_p`, `top_k`, `max_tokens`
- `thinking_level`
- `media_resolution`
- `structured_output`
- `Stream`
- `max_retries`, `timeout`, `backoff_factor`

## Media Resolution

Use enum values from `GOOGLE_VISION_MEDIA_RESOLUTION`:

- `LOW`
- `MEDIUM`
- `HIGH`

## Structured Output Example

```python
vision = GoogleVisionModel(
    model=GOOGLE_VISION_MODEL_NAME.GEMINI_3_FLASH_PREVIEW,
    structured_output=True,
)

result = vision.invoke(
    prompt="List major UI elements and summarize layout.",
    image="./screen.png",
)
print(result)
```

When `structured_output=True`, return value includes response text, token usage, model name, estimated cost, and input image count.

## Validation Rules

- Prompt must resolve to a non-empty string
- At least one image must be provided via `image` or `images`
- `temperature` in `[0.0, 2.0]`
- `top_p` in `[0.0, 1.0]`
- `top_k >= 1` when provided
- `max_tokens >= 1` when provided
- `max_retries >= 1`
- `timeout > 0` when provided
- `backoff_factor >= 0`
- `structured_output=True` requires `Stream=False`

## Retry Behavior

Retries use exponential backoff:

`sleep = backoff_factor * (2 ** (attempt - 1))`

Defaults:

- `max_retries=3`
- `backoff_factor=0.5`

## Error Model

- `GoogleVisionModelImportError`: SDK import/configuration issues
- `GoogleVisionModelAPIError`: request failures after retries
- `GoogleVisionModelResponseError`: response parsing/empty output issues
