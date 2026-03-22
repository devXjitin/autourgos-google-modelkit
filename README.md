# autourgos-google-kit

Developer-friendly Google Gemini wrappers for Autourgos projects.

Autourgos is a framework currently in development. It is designed in the same problem space as LangChain and AutoGen, with a stronger focus on developer-friendly APIs and clean code.

This package gives you two production-oriented clients:

- `GoogleTextModel` for text generation
- `GoogleVisionModel` for image + text prompts with text output

It focuses on clean API usage, validation, retries, and structured response metadata.

## Why Use This Package

- Typed model enums for safer model selection
- One consistent class API across text and vision
- Optional streaming mode
- Optional structured output with token and cost metadata
- Prompt templates with variable validation
- Built-in retry with exponential backoff
- API key resolution from args or environment variables

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
	model=GOOGLE_TEXT_MODEL_NAME.GEMINI_2_5_FLASH,
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
	model=GOOGLE_TEXT_MODEL_NAME.GEMINI_2_5_FLASH,
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
- `api_key`: Explicit API key. If omitted, keys are resolved from environment variables.
- `prompt_template`: Reusable template string with placeholders like `{topic}`.
- `temperature`: Randomness control. Lower values are more deterministic; higher values are more creative.
- `top_p`: Nucleus sampling threshold in the range `[0.0, 1.0]`.
- `top_k`: Limits sampling to top-k token candidates.
- `max_tokens`: Maximum output token budget.
- `thinking_level`: Controls reasoning depth for supported Gemini models.
- `structured_output`: Returns metadata-rich dictionary instead of plain text.
- `Stream`: When `True`, `invoke()` returns an iterator of text chunks.
- `max_retries`: Total retry attempts on transient failures.
- `timeout`: Request timeout in seconds.
- `backoff_factor`: Retry delay factor using exponential backoff.

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
