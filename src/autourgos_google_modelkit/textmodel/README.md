# Text Model

This module provides a production-oriented wrapper for Gemini text generation.

## Public Exports

- `GOOGLE_TEXT_MODEL_NAME`
- `GOOGLE_TEXT_THINKING_LEVEL`
- `MODEL_PRICING_USD_PER_MILLION`
- `resolve_model_pricing`
- `GoogleTextModel`
- `GoogleTextModelError`
- `GoogleTextModelImportError`
- `GoogleTextModelAPIError`
- `GoogleTextModelResponseError`

## Quick Start

```python
from autourgos_google_modelkit.textmodel import GoogleTextModel, GOOGLE_TEXT_MODEL_NAME

llm = GoogleTextModel(model=GOOGLE_TEXT_MODEL_NAME.GEMINI_3_FLASH_PREVIEW)
print(llm.invoke("Explain vector databases in plain language."))
```

## Constructor Highlights

`GoogleTextModel(...)` supports:

- `model` (enum or string)
- `api_key`
- `prompt_template`
- `temperature`, `top_p`, `top_k`, `max_tokens`
- `thinking_level`
- `structured_output`
- `Stream`
- `max_retries`, `timeout`, `backoff_factor`

## Invoke Signatures

- `invoke(prompt="...")`
- `invoke(prompt_variables={...})` when `prompt_template` is configured

## Prompt Template Example

```python
llm = GoogleTextModel(
    model=GOOGLE_TEXT_MODEL_NAME.GEMINI_2_5_FLASH,
    prompt_template="Answer in {style} style: {question}",
)

print(llm.invoke(prompt_variables={"style": "brief", "question": "What is caching?"}))
```

## Structured Output Example

```python
llm = GoogleTextModel(
    model=GOOGLE_TEXT_MODEL_NAME.GEMINI_3_FLASH_PREVIEW,
    structured_output=True,
)

result = llm.invoke("Summarize software observability.")
print(result)
```

When `structured_output=True`, return value includes response text, token usage, model name, and estimated cost.

## Validation Rules

- Prompt must resolve to a non-empty string
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

Use specific exceptions for more precise handling:

- `GoogleTextModelImportError`: SDK import/configuration issues
- `GoogleTextModelAPIError`: request failures after retries
- `GoogleTextModelResponseError`: response parsing/empty output issues
