# Autourgos GoogleVisionModel Documentation

GoogleVisionModel is a production-ready wrapper for Google Gemini multimodal
requests where inputs include image(s) plus text and the output is always text.

## Module Scope

Location:

- src/autourgos_google_modelkit/visionmodel

Public exports:

- MODEL
- THINKING_LEVEL
- MEDIA_RESOLUTION
- MODEL_PRICING_USD_PER_MILLION
- resolve_model_pricing
- GoogleVisionModel
- GoogleVisionModelError
- GoogleVisionModelImportError
- GoogleVisionModelAPIError
- GoogleVisionModelResponseError

## Quick Start

```python
from autourgos_google_modelkit.visionmodel import GoogleVisionModel, MODEL

vision = GoogleVisionModel(model=MODEL.GEMINI_3_FLASH_PREVIEW)
result = vision.invoke(
    prompt="Describe all objects in this image.",
    image="./sample.jpg",
)
print(result)
```

## Supported Vision Inputs

- `image`: single image input
- `images`: list of image inputs

Accepted image value types:

- file path (`str` or `pathlib.Path`)
- raw image bytes (`bytes`)
- dict with keys `data` and optional `mime_type`
- tuple `(bytes, mime_type)`

## Notes

- Responses are always parsed as text output.
- `Stream=True` yields incremental text chunks.
- `structured_output=True` returns response text + token/cost metadata.
- `MEDIA_RESOLUTION` can be supplied to tune vision fidelity.
