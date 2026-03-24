# Autourgos Google VisionModel

GoogleVisionModel is the multimodal wrapper for Google Gemini vision workflows in the Autourgos Google Model Kit.

It is built for image + text input with text output, and follows the same package principles:
- clean invoke-based API
- strict validation before API calls
- retry + timeout resilience
- optional streaming output
- optional structured usage/cost metadata

## Installation and API Key Setup

Install package:

```bash
pip install autourgos-google-modelkit
```

Set API key (PowerShell):

```powershell
$env:GOOGLE_API_KEY = "your-api-key"
```

Set API key (Bash):

```bash
export GOOGLE_API_KEY="your-api-key"
```

## Basic Usage

```python
from autourgos_google_modelkit import GoogleVisionModel, GOOGLE_VISION_MODEL_NAME

vision = GoogleVisionModel(
    model=GOOGLE_VISION_MODEL_NAME.GEMINI_3_FLASH_PREVIEW,
)

response = vision.invoke(
    prompt="Describe what is visible in this image.",
    image="./sample.jpg",
)

print(response)
```

Example response:

```text
The image contains a laptop on a desk, a coffee mug, and a notebook.
The background is a white wall with soft daylight.
```

## Supported Image Inputs

You can pass images through:

- image (single input)
- images (multiple inputs)

Accepted image types:

- file path (str or pathlib.Path)
- raw bytes
- dict with data and mime_type
- tuple of (bytes, mime_type)

## Supported Parameters

GoogleVisionModel supports:

- model
- api_key
- prompt_template
- temperature
- top_p
- top_k
- max_tokens
- thinking_level
- media_resolution
- structured_output
- Stream
- max_retries
- timeout
- backoff_factor

## Parameter Examples

### Model

Use enums (recommended) or raw strings.

```python
from autourgos_google_modelkit import GoogleVisionModel, GOOGLE_VISION_MODEL_NAME

vision = GoogleVisionModel(model=GOOGLE_VISION_MODEL_NAME.GEMINI_3_FLASH_PREVIEW)
```

```python
from autourgos_google_modelkit import GoogleVisionModel

vision = GoogleVisionModel(model="gemini-3-flash-preview")
```

### Prompt Template

```python
from autourgos_google_modelkit import GoogleVisionModel

vision = GoogleVisionModel(
    model="gemini-3-flash-preview",
    prompt_template="Analyze this image for {goal}.",
)

print(vision.invoke(image="./sample.jpg", prompt_variables={"goal": "accessibility issues"}))
```

### Media Resolution

Use GOOGLE_VISION_MEDIA_RESOLUTION values:

- LOW
- MEDIUM
- HIGH

```python
from autourgos_google_modelkit import (
    GoogleVisionModel,
    GOOGLE_VISION_MODEL_NAME,
    GOOGLE_VISION_MEDIA_RESOLUTION,
)

vision = GoogleVisionModel(
    model=GOOGLE_VISION_MODEL_NAME.GEMINI_3_FLASH_PREVIEW,
    media_resolution=GOOGLE_VISION_MEDIA_RESOLUTION.HIGH,
)
```

Note:
- Higher media resolution can improve quality on complex images.
- Higher media resolution can also increase latency and cost.

### Structured Output

When structured_output=True, invoke returns a dictionary with response text and metadata.

```python
from autourgos_google_modelkit import GoogleVisionModel, GOOGLE_VISION_MODEL_NAME

vision = GoogleVisionModel(
    model=GOOGLE_VISION_MODEL_NAME.GEMINI_3_FLASH_PREVIEW,
    structured_output=True,
)

result = vision.invoke(
    prompt="List major UI elements and summarize the layout.",
    image="./screen.png",
)

print(result)
```

Expected metadata includes:

- model
- response
- token usage
- cost details
- input_image_count

### Stream

When Stream=True, invoke returns an iterator of text chunks.

```python
from autourgos_google_modelkit import GoogleVisionModel, GOOGLE_VISION_MODEL_NAME

vision = GoogleVisionModel(
    model=GOOGLE_VISION_MODEL_NAME.GEMINI_3_FLASH_PREVIEW,
    Stream=True,
)

stream = vision.invoke(
    prompt="Describe this image in short bullet points.",
    image="./sample.jpg",
)

for chunk in stream:
    print(chunk, end="", flush=True)
print()
```

### Retries and Timeouts

```python
from autourgos_google_modelkit import GoogleVisionModel

vision = GoogleVisionModel(
    model="gemini-3-flash-preview",
    max_retries=5,
    timeout=60.0,
    backoff_factor=1.5,
)
```

## Validation Rules

Important behavior:

- At least one image must be provided via image or images
- structured_output=True is only supported when Stream=False
- temperature must be in [0.0, 2.0]
- top_p must be in [0.0, 1.0]
- top_k must be >= 1
- max_tokens must be >= 1
- max_retries must be >= 1
- timeout must be > 0 (when provided)
- backoff_factor must be >= 0

## Error Hierarchy

- GoogleVisionModelError
- GoogleVisionModelImportError
- GoogleVisionModelAPIError
- GoogleVisionModelResponseError

## References

- [Google Gemini API Documentation](https://developers.generativeai.google/api/)
- [Google Gemini Pricing](https://developers.generativeai.google/pricing/)
- [Google Gemini Model Capabilities](https://developers.generativeai.google/models/)