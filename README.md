
# Autourgos Google Model Kit
![Gemini](./README/Image_Dark.png)

![Pypi](https://img.shields.io/badge/Pypi-0.1.0-blue?style=flat-square)
![Release](https://img.shields.io/badge/Release-Early%20Development-brown?style=flat-square)
![Framework](https://img.shields.io/badge/Framework-Autourgos-orange?style=flat-square)
![Wrapper](https://img.shields.io/badge/Wrapper-Google-brightgreen?style=flat-square)
![docs](https://img.shields.io/badge/docs-latest-blue?style=flat-square)
![Developed%20by](https://img.shields.io/badge/Developed%20by-DevxJitin-gold?style=flat-square)
![Documented%20by](https://img.shields.io/badge/Documented%20by-Sonia-silver?style=flat-square)


Developer-friendly Google Gemini Kit for Autourgos Framework.

Autourgos is a framework currently in development. It is designed in the same problem space as LangChain and AutoGen, with a stronger focus on developer-friendly APIs and clean code.

This package gives you two Model Wrappers for Google Gemini APIs:

- `GoogleTextModel` for text generation
- `GoogleVisionModel` for image + text prompts with text output

It focuses on clean API usage, validation, retries, and structured response metadata.

## Why Use This Package

- **Typed Model Enums**: Safer model selection using built-in enums (`GOOGLE_TEXT_MODEL_NAME`, etc.).
- **Consistent API**: One unified class interface (`invoke`) across both text and vision models.
- **Streaming Support**: Optional real-time streaming mode for token-by-token generation.
- **Structured Output**: Access response text alongside token usage and estimated cost metadata.
- **Prompt Templates**: Reusable templates with strict variable validation.
- **Advanced Capabilities**: Full support for Gemini 3 thinking levels and vision media resolution tuning.
- **Resilience**: Built-in retry mechanism with exponential backoff and timeout configurations.
- **Flexible Configuration**: API key resolution from explicit arguments or standard environment variables.

## Installation

Requirements:

- Python `>=3.13`
- `google-genai>=1.68.0`

Install in editable mode:

```bash
pip install -e .
```

Or install the runtime dependency directly:

```bash
pip install "google-genai>=1.68.0"
```

## API Key Setup

Resolution order:

1. `api_key` argument
2. `GOOGLE_API_KEY`
3. `GEMINI_API_KEY`

PowerShell:

```powershell
$env:GOOGLE_API_KEY = "your-api-key"
```

Bash:

```bash
export GOOGLE_API_KEY="your-api-key"
```

## Quick Start

### Text generation

```python
from autourgos_google_modelkit import GoogleTextModel, GOOGLE_TEXT_MODEL_NAME

llm = GoogleTextModel(
	model=GOOGLE_TEXT_MODEL_NAME.GEMINI_3_1_PRO,
	Stream=False,
)

print(llm.invoke("Explain RAG in simple terms."))
```

Example response:

```text
RAG (Retrieval-Augmented Generation) combines search and generation.
The model first retrieves relevant knowledge, then writes an answer using that context.
This improves factual accuracy and reduces hallucinations.
```

### Vision generation (image + prompt)

```python
from autourgos_google_modelkit import GoogleVisionModel, GOOGLE_VISION_MODEL_NAME

vision = GoogleVisionModel(model=GOOGLE_VISION_MODEL_NAME.GEMINI_3_FLASH_PREVIEW)

print(
	vision.invoke(
		prompt="Describe the main objects in this image.",
		image="./sample.jpg",
	)
)
```

Example response:

```text
The image contains a laptop on a desk, a coffee mug, and a notebook.
The main background is a white wall with soft daylight.
```

### Streaming mode

```python
from autourgos_google_modelkit import GoogleTextModel, GOOGLE_TEXT_MODEL_NAME

llm = GoogleTextModel(
	model=GOOGLE_TEXT_MODEL_NAME.GEMINI_3_1_PRO,
	Stream=True,
)

stream = llm.invoke("Write a short note on clean architecture.")
for chunk in stream:
	print(chunk, end="", flush=True)
print()
```

Debugging chunk boundaries (optional):

```python
stream = llm.invoke("Write a short note on clean architecture.")
for chunk in stream:
	print(repr(chunk))
```

Example streamed chunks (illustrative only):

```text
'Clean architecture '
'separates business logic '
'from framework details.'
```

Note: chunk boundaries are not fixed and can vary by SDK/model/network conditions.

Final assembled response:

```text
Clean architecture separates business logic from framework details.
It improves testability, long-term maintainability, and replacement of external dependencies.
```

## Public API Surface

Top-level exports:

- `GoogleTextModel`
- `GoogleVisionModel`
- `GOOGLE_TEXT_MODEL_NAME`
- `GOOGLE_VISION_MODEL_NAME`
- `GOOGLE_TEXT_THINKING_LEVEL`
- `GOOGLE_VISION_THINKING_LEVEL`
- `GOOGLE_VISION_MEDIA_RESOLUTION`

Module-level detailed exports are available under:

- `autourgos_google_modelkit.textmodel`
- `autourgos_google_modelkit.visionmodel`

## Common Constructor Options

Both classes support:

- `model`
- `api_key`
- `prompt_template`
- `temperature`, `top_p`, `top_k`, `max_tokens`
- `thinking_level`
- `structured_output`
- `Stream`
- `max_retries`, `timeout`, `backoff_factor`

Vision-only:

- `media_resolution`

## Parameter Explanation

### Core parameters (both text and vision)

- `model`: Model ID as enum or string. Prefer enum for autocomplete and safer selection.
  ```python
  model=GOOGLE_TEXT_MODEL_NAME.GEMINI_3_1_PRO
  ```
- `api_key`: Explicit API key. If omitted, keys are resolved from environment variables.
  ```python
  api_key="AIzaSy..."
  ```
- `prompt_template`: Reusable template string with placeholders like `{topic}`.
  ```python
  prompt_template="Explain {topic} in simple terms."
  ```
- `temperature`: Randomness control. Lower values are more deterministic; higher values are more creative.
  ```python
  temperature=0.7
  ```
- `top_p`: Nucleus sampling threshold in the range `[0.0, 1.0]`.
  ```python
  top_p=0.9
  ```
- `top_k`: Limits sampling to top-k token candidates.
  ```python
  top_k=40
  ```
- `max_tokens`: Maximum output token budget.
  ```python
  max_tokens=1024
  ```
- `thinking_level`: Controls reasoning depth for supported Gemini models.
  ```python
  thinking_level=GOOGLE_TEXT_THINKING_LEVEL.HIGH
  ```
- `structured_output`: Returns metadata-rich dictionary instead of plain text.
  ```python
  structured_output=True
  ```
- `Stream`: When `True`, `invoke()` returns an iterator of text chunks.
  ```python
  Stream=True
  ```
- `max_retries`: Total retry attempts on transient failures.
  ```python
  max_retries=5
  ```
- `timeout`: Request timeout in seconds.
  ```python
  timeout=60.0
  ```
- `backoff_factor`: Retry delay factor using exponential backoff.
  ```python
  backoff_factor=1.5
  ```

### Vision-only parameter

- `media_resolution`: Vision input quality hint. Supported enum values:
  - `GOOGLE_VISION_MEDIA_RESOLUTION.LOW`
  - `GOOGLE_VISION_MEDIA_RESOLUTION.MEDIUM`
  - `GOOGLE_VISION_MEDIA_RESOLUTION.HIGH`

## Structured Output

Set `structured_output=True` to receive response metadata (model, token usage, and estimated cost) instead of plain text.

```python
from autourgos_google_modelkit import GoogleTextModel, GOOGLE_TEXT_MODEL_NAME

llm = GoogleTextModel(
	model=GOOGLE_TEXT_MODEL_NAME.GEMINI_3_FLASH_PREVIEW,
	structured_output=True,
)

result = llm.invoke("Summarize observability in one paragraph.")
print(result)
```

Example response:

```json
{
	"model": "gemini-3-flash-preview",
	"response": "Observability is the ability to understand system state from outputs like logs, metrics, and traces.",
	"input_tokens": 10,
	"output_tokens": 24,
	"Total_tokens": 34,
	"Cost": "$0.00007700",
	"cost_details": {
		"value_usd": 0.000077,
		"input_rate_per_million": 0.5,
		"output_rate_per_million": 3.0
	}
}
```

## Prompt Templates

You can set a reusable template in the constructor and pass only variables at call time:

```python
from autourgos_google_modelkit import GoogleTextModel, GOOGLE_TEXT_MODEL_NAME

llm = GoogleTextModel(
	model=GOOGLE_TEXT_MODEL_NAME.GEMINI_2_5_FLASH,
	prompt_template="Summarize this in {tone} tone:\n\n{text}",
)

print(
	llm.invoke(
		prompt_variables={
			"tone": "concise",
			"text": "Autourgos provides model wrappers for Gemini APIs.",
		}
	)
)
```

Example response:

```text
Autourgos provides clean, reusable wrappers around Gemini APIs for text and vision workflows.
```

## Validation and Errors

The package validates prompt content, sampling parameters, retry settings, and type constraints before making API calls.

Important behavior:

- `structured_output=True` is only supported when `Stream=False`
- `top_p` must be in `[0.0, 1.0]`
- `temperature` must be in `[0.0, 2.0]`
- `max_retries` must be `>= 1`

Error hierarchy:

- Text: `GoogleTextModelError` and specialized subclasses
- Vision: `GoogleVisionModelError` and specialized subclasses

## Project Layout

```text
src/autourgos_google_modelkit/
  __init__.py
  core/
  textmodel/
	__init__.py
	base.py
	models.py
	README.md
  visionmodel/
	__init__.py
	base.py
	models.py
	README.md
```

## Development

Run tests:

```bash
pytest -q
```

## Notes

- Pricing constants in the package are convenience estimates and should be verified against current Google pricing docs before production billing decisions.
- Avoid committing API keys in source files.