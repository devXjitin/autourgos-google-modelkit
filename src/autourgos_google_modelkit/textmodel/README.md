# Autourgos Google TextModel

`GoogleTextModel` is the text-generation wrapper for Google Gemini models in the Autourgos framework.

It provides:

- one class-based API
- prompt-template support
- streaming support
- structured token and cost metadata
- retries, timeout, and backoff controls
- Gemini thinking-level controls

## Quick Start

```python
from autourgos_google_modelkit.textmodel import GoogleTextModel, GOOGLE_TEXT_MODEL_NAME

llm = GoogleTextModel(model=GOOGLE_TEXT_MODEL_NAME.GEMINI_3_FLASH_PREVIEW)
print(llm.invoke("Explain vector databases in plain language."))
```

Example output:

```plaintext
Vector databases store data as numerical embeddings so semantic search can find meaning, not just exact keywords.
```

## Example Notes

- Unless `api_key` is passed explicitly, examples assume `GOOGLE_API_KEY` or `GEMINI_API_KEY` is already set.
- Model text is illustrative. Exact wording, token counts, and chunk boundaries vary by model and prompt.
- `Stream` is intentionally capitalized because that is the current public constructor argument.

## Constructor Signature

```python
GoogleTextModel(
    model,
    api_key=None,
    *,
    prompt_template=None,
    temperature=None,
    top_p=None,
    top_k=None,
    max_tokens=None,
    thinking_level=GOOGLE_TEXT_THINKING_LEVEL.MINIMAL,
    structured_output=False,
    Stream=False,
    max_retries=3,
    timeout=30.0,
    backoff_factor=0.5,
)
```

## Constructor Parameters At A Glance


| Parameter           | Type    | Required                   | Default |
| ------------------- | ------- | -------------------------- | ------- |
| `model`             | str     | GOOGLE_TEXT_MODEL_NAME`    | Yes     |
| `api_key`           | str     | None`                      | No      |
| `prompt_template`   | str     | None`                      | No      |
| `temperature`       | float   | None`                      | No      |
| `top_p`             | float   | None`                      | No      |
| `top_k`             | int     | None`                      | No      |
| `max_tokens`        | int     | None`                      | No      |
| `thinking_level`    | str     | GOOGLE_TEXT_THINKING_LEVEL | None    |
| `structured_output` | bool    | No                         | False   |
| `Stream`            | bool    | No                         | False   |
| `max_retries`       | int     | No                         | 3       |
| `timeout`           | float   | None`                      | No      |
| `backoff_factor`    | float   | No                         | 0.5     |


## Constructor Parameter Examples

### `model`

Selects the Gemini model. You can pass either an enum value or a raw string.

```python
from autourgos_google_modelkit.textmodel import GoogleTextModel, GOOGLE_TEXT_MODEL_NAME

llm = GoogleTextModel(model=GOOGLE_TEXT_MODEL_NAME.GEMINI_3_FLASH_PREVIEW)
print(llm.invoke("Answer in one sentence: what is caching?"))
```

Output:

```text
Caching stores frequently reused data so future reads are faster and cheaper.
```

### `api_key`

Overrides environment-variable lookup and uses the provided key directly.

```python
from autourgos_google_modelkit.textmodel import GoogleTextModel

llm = GoogleTextModel(
    model="gemini-2.5-flash",
    api_key="your-google-api-key",
)
print(llm.invoke("What is a queue in software?"))
```

Output:

```text
A queue is a first-in, first-out structure used to process work in arrival order.
```

### `prompt_template`

Defines a reusable prompt pattern that is filled later with `prompt_variables`.

```python
from autourgos_google_modelkit.textmodel import GoogleTextModel, GOOGLE_TEXT_MODEL_NAME

llm = GoogleTextModel(
    model=GOOGLE_TEXT_MODEL_NAME.GEMINI_2_5_FLASH,
    prompt_template="Explain {topic} in {style} style.",
)

print(
    llm.invoke(
        prompt_variables={
            "topic": "rate limiting",
            "style": "simple",
        }
    )
)
```

Output:

```text
Rate limiting controls how many requests can happen in a given time so systems stay stable and fair.
```

### `temperature`

Controls randomness. Lower values are usually more deterministic.

```python
from autourgos_google_modelkit.textmodel import GoogleTextModel

llm = GoogleTextModel(
    model="gemini-2.5-flash",
    temperature=0.2,
)
print(llm.invoke("Summarize unit testing in one sentence."))
```

Output:

```text
Unit testing checks small pieces of code in isolation so bugs are found early.
```

### `top_p`

Controls nucleus sampling in the range `0.0` to `1.0`.

```python
from autourgos_google_modelkit.textmodel import GoogleTextModel

llm = GoogleTextModel(
    model="gemini-2.5-flash",
    top_p=0.8,
)
print(llm.invoke("Give a short analogy for a load balancer."))
```

Output:

```text
A load balancer is like a traffic officer directing cars across open lanes so no single lane gets jammed.
```

### `top_k`

Limits sampling to the top `k` next-token candidates.

```python
from autourgos_google_modelkit.textmodel import GoogleTextModel

llm = GoogleTextModel(
    model="gemini-2.5-flash",
    top_k=20,
)
print(llm.invoke("Describe a REST API in one sentence."))
```

Output:

```text
A REST API exposes resources through standard HTTP methods such as GET, POST, PUT, and DELETE.
```

### `max_tokens`

Caps the output length.

```python
from autourgos_google_modelkit.textmodel import GoogleTextModel

llm = GoogleTextModel(
    model="gemini-2.5-flash",
    max_tokens=20,
)
print(llm.invoke("Explain event-driven architecture in two or three lines."))
```

Output:

```text
Event-driven architecture reacts to events instead of direct calls.
Services stay loosely coupled.
```

### `thinking_level`

Sets Gemini thinking depth using a string or `GOOGLE_TEXT_THINKING_LEVEL`.

For `gemini-3.1-pro-preview` and `gemini-3-pro-preview`, `minimal` is not supported. Use `low`, `medium`, or `high`.

```python
from autourgos_google_modelkit.textmodel import (
    GoogleTextModel,
    GOOGLE_TEXT_MODEL_NAME,
    GOOGLE_TEXT_THINKING_LEVEL,
)

llm = GoogleTextModel(
    model=GOOGLE_TEXT_MODEL_NAME.GEMINI_3_FLASH_PREVIEW,
    thinking_level=GOOGLE_TEXT_THINKING_LEVEL.HIGH,
)
print(llm.invoke("Compare threads and processes in two sentences."))
```

Output:

```text
Threads share the same process memory, which makes communication faster but isolation weaker.
Processes have separate memory spaces, which improves isolation but adds more overhead.
```

### `structured_output`

Returns a dictionary with response text, token usage, and cost metadata.

```python
from autourgos_google_modelkit.textmodel import GoogleTextModel, GOOGLE_TEXT_MODEL_NAME

llm = GoogleTextModel(
    model=GOOGLE_TEXT_MODEL_NAME.GEMINI_3_FLASH_PREVIEW,
    structured_output=True,
)

result = llm.invoke("Summarize observability in one sentence.")
print(result)
```

Output:

```python
{
    'model': 'gemini-3-flash-preview',
    'response': 'Observability uses logs, metrics, and traces to explain how a system behaves.',
    'input_tokens': 7,
    'output_tokens': 14,
    'Total_tokens': 21,
    'Cost': '$0.00004550',
    'cost_details': {
        'value_usd': 4.55e-05,
        'input_rate_per_million': 0.5,
        'output_rate_per_million': 3.0,
    },
}
```

### `Stream`

When `Stream=True`, `invoke()` returns an iterator of text chunks instead of one final string.

```python
from autourgos_google_modelkit.textmodel import GoogleTextModel, GOOGLE_TEXT_MODEL_NAME

llm = GoogleTextModel(
    model=GOOGLE_TEXT_MODEL_NAME.GEMINI_3_FLASH_PREVIEW,
    Stream=True,
)

for chunk in llm.invoke("Name three common database types."):
    print(repr(chunk))
```

Output:

```text
'Relational '
'databases, '
'document '
'databases, '
'and key-value '
'databases.'
```

### `max_retries`

Controls how many times the request is retried before a final API error is raised.

```python
from autourgos_google_modelkit.textmodel import GoogleTextModel

llm = GoogleTextModel(
    model="gemini-2.5-flash",
    max_retries=0,
)

try:
    llm.invoke("Hello")
except Exception as exc:
    print(type(exc).__name__)
    print(exc)
```

Output:

```text
ValueError
max_retries must be an integer >= 1
```

### `timeout`

Sets the request timeout in seconds.

```python
from autourgos_google_modelkit.textmodel import GoogleTextModel

llm = GoogleTextModel(
    model="gemini-2.5-flash",
    timeout=0,
)

try:
    llm.invoke("Hello")
except Exception as exc:
    print(type(exc).__name__)
    print(exc)
```

Output:

```text
ValueError
timeout must be > 0 when provided
```

### `backoff_factor`

Controls the exponential retry delay multiplier.

```python
from autourgos_google_modelkit.textmodel import GoogleTextModel

llm = GoogleTextModel(
    model="gemini-2.5-flash",
    backoff_factor=-1,
)

try:
    llm.invoke("Hello")
except Exception as exc:
    print(type(exc).__name__)
    print(exc)
```

Output:

```text
ValueError
backoff_factor must be >= 0
```

## Invoke Signature

```python
invoke(prompt=None, prompt_variables=None)
```

## Invoke Parameters At A Glance


| Parameter          | Type            | Required | Default |
| ------------------ | --------------- | -------- | ------- |
| `prompt`           | str             | None     | No      |
| `prompt_variables` | dict[str, Any]  | None     | No      |


## Invoke Parameter Examples

### `prompt`

Passes a direct prompt string to the model.

If `prompt` is provided, it is used directly even when `prompt_template` is configured.

```python
from autourgos_google_modelkit.textmodel import GoogleTextModel

llm = GoogleTextModel(model="gemini-2.5-flash")
print(llm.invoke(prompt="What is memoization?"))
```

Output:

```text
Memoization stores previous function results so repeated calls with the same input can return faster.
```

### `prompt_variables`

Supplies values for placeholders defined in `prompt_template`.

```python
from autourgos_google_modelkit.textmodel import GoogleTextModel

llm = GoogleTextModel(
    model="gemini-2.5-flash",
    prompt_template="Write a {tone} explanation of {subject}.",
)

print(
    llm.invoke(
        prompt_variables={
            "tone": "beginner-friendly",
            "subject": "dependency injection",
        }
    )
)
```

Output:

```text
Dependency injection means giving an object the tools it needs from the outside instead of letting it create them by itself.
```

## Return Types

- `str` when `Stream=False` and `structured_output=False`
- `Iterator[str]` when `Stream=True`
- `dict[str, Any]` when `structured_output=True`

## Validation Rules

- `model` must resolve to a non-empty string
- `prompt` is required when `prompt_template` is not configured
- `prompt` must be a non-empty string when provided
- `prompt_template` must be a non-empty string when provided
- `prompt_variables` must be a dictionary when provided
- all required template placeholders must be supplied
- `temperature` must be in `[0.0, 2.0]`
- `top_p` must be in `[0.0, 1.0]`
- `top_k` must be `>= 1`
- `max_tokens` must be `>= 1`
- `thinking_level` must be one of `minimal`, `low`, `medium`, `high`
- `Stream` must be a boolean
- `structured_output` must be a boolean
- `structured_output=True` requires `Stream=False`
- `max_retries` must be `>= 1`
- `timeout` must be `> 0` when provided
- `backoff_factor` must be `>= 0`

## Retry Behavior

Retry sleep uses exponential backoff:

```python
sleep_seconds = backoff_factor * (2 ** (attempt - 1))
```

Default retry settings:

- `max_retries=3`
- `backoff_factor=0.5`

## Error Model

- `GoogleTextModelImportError`: SDK import failure or missing API key
- `GoogleTextModelAPIError`: request failure after all retries
- `GoogleTextModelResponseError`: response parsing failure or empty response