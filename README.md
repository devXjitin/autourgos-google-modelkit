# autourgos-google-kit

Production-focused Google Gemini text model toolkit for Autourgos.

This repository currently provides a complete text-model integration layer with:

- Typed model enum for IDE autosuggest
- Class-based API
- Stream mode toggle via class parameter
- Retry + exponential backoff behavior
- Defensive response parsing for SDK compatibility
- Clear custom exceptions for import, API, and response errors

## Features

- `MODEL` enum with supported Gemini model IDs
- `GoogleTextModel` class with `invoke()`
- `Stream=True/False` constructor parameter (default `False`)
- API key fallback resolution (`api_key` arg -> `GOOGLE_API_KEY` -> `GEMINI_API_KEY`)
- Input validation for prompt, sampling parameters, retry settings, and timeouts
- Multi-strategy SDK calls to support multiple Google client surfaces

## Project Layout

```text
autourgos-google-kit/
  main.py
  pyproject.toml
  README.md
  src/
	autourgos_google_modelkit/
	  textmodel/
		__init__.py
		base.py
		models.py
	  embedding/
	  imagemodel/
	  videomodel/
	  visionmodel/
```

Current implementation focus is `textmodel`. Other model folders are present as scaffolding and are currently empty.

## Requirements

- Python `>=3.13`
- Dependency: `google-genai>=1.68.0`

From `pyproject.toml`:

```toml
[project]
name = "autourgos-google-kit"
requires-python = ">=3.13"
dependencies = [
	"google-genai>=1.68.0",
]
```

## Installation

### Option 1: Install dependencies with your project workflow

Use your preferred tool (`pip`, `uv`, Poetry) to install project dependencies.

Example using `pip`:

```bash
pip install -e .
```

### Option 2: Install runtime dependency directly

```bash
pip install google-genai>=1.68.0
```

## Environment Setup

Set one of the following environment variables:

- `GOOGLE_API_KEY` (preferred)
- `GEMINI_API_KEY` (fallback)

Windows PowerShell:

```powershell
$env:GOOGLE_API_KEY = "your-api-key"
```

Bash:

```bash
export GOOGLE_API_KEY="your-api-key"
```

You can also pass `api_key="..."` directly to APIs.

## Quick Start

### Class-based usage

```python
from autourgos_google_modelkit import GoogleTextModel, MODEL

llm = GoogleTextModel(
	model=MODEL.GEMINI_2_5_FLASH_LITE,
	api_key="your-api-key",
	Stream=False,
)

response_text = llm.invoke("Explain agentic AI in simple terms.")
print(response_text)
```

### Stream mode usage

```python
from autourgos_google_modelkit import GoogleTextModel, MODEL

llm = GoogleTextModel(
	model=MODEL.GEMINI_2_5_FLASH,
	api_key="your-api-key",
	Stream=True,
)

response_text = llm.invoke("Write a short poem about software reliability.")
print(response_text)
```

## API Reference

### `MODEL` enum

Available typed values:

- `MODEL.GEMINI_3_1_PRO_PREVIEW`
- `MODEL.GEMINI_3_PRO_PREVIEW`
- `MODEL.GEMINI_3_FLASH_PREVIEW`
- `MODEL.GEMINI_2_5_PRO`
- `MODEL.GEMINI_2_5_FLASH`
- `MODEL.GEMINI_2_5_FLASH_LITE`

You may also pass raw model strings, but using `MODEL` provides better IDE autocomplete and safer model selection.

### `GoogleTextModel`

Constructor parameters mirror function options and store shared config:

- `model`, `api_key`
- `Stream` (bool, default `False`)
- `temperature`, `top_p`, `top_k`, `max_tokens`
- `max_retries`, `timeout`, `backoff_factor`

Methods:

- `invoke(prompt: str) -> str`

Behavior:

- `Stream=False` (default): non-streaming generation path
- `Stream=True`: streaming path is used internally and aggregated text is returned

## Validation Rules

The text-model APIs validate:

- `prompt` must be a non-empty string
- `temperature` must be in `[0.0, 2.0]`
- `top_p` must be in `[0.0, 1.0]`
- `top_k` must be `>= 1` when provided
- `max_tokens` must be `>= 1` when provided
- `max_retries` must be an integer `>= 1`
- `timeout` must be `> 0` when provided
- `backoff_factor` must be `>= 0`

## Error Handling

The module defines a clear exception hierarchy:

- `GoogleTextModelError` (base)
- `GoogleTextModelImportError`
- `GoogleTextModelAPIError`
- `GoogleTextModelResponseError`

Typical handling:

```python
from autourgos_google_modelkit import (
	GoogleTextModel,
	GoogleTextModelImportError,
	GoogleTextModelAPIError,
	GoogleTextModelResponseError,
	MODEL,
)

llm = GoogleTextModel(model=MODEL.GEMINI_2_5_FLASH)

try:
	print(llm.invoke("Hello"))
except GoogleTextModelImportError as exc:
	print(f"Import/config error: {exc}")
except GoogleTextModelAPIError as exc:
	print(f"Request failed: {exc}")
except GoogleTextModelResponseError as exc:
	print(f"Response parse error: {exc}")
```

## Retry and Backoff Behavior

- Default retries: `max_retries=3`
- Backoff uses exponential delay per attempt:

`sleep = backoff_factor * (2 ** (attempt - 1))`

Example with `backoff_factor=0.5`: `0.5s`, `1.0s`, `2.0s`, ...

## SDK Compatibility Notes

The provider is defensive across Google SDK variations and attempts multiple paths:

- Imports from both `google.generativeai` and `google.genai`
- Supports client-style calls when available
- Supports `GenerativeModel` flow and helper call fallbacks

This increases compatibility across SDK versions and deployment environments.

## Running the Example

`main.py` contains a simple interactive CLI loop using `GoogleTextModel`.

Run:

```bash
python main.py
```

Enter prompts until `exit`.

## Current Status

- Text model integration is implemented and typed.
- Embedding/image/video/vision folders are currently placeholders.

## License

Add your project license details here.
