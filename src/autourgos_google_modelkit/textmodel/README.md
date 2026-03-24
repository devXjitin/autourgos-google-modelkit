# Autourgos Google TextModel

GoogleTextModel is the text-generation wrapper for Google Gemini models in the Autourgos Google Model Kit.

It follows the same design goals as the main package README:
- simple invoke-based API
- input validation and clear errors
- retry + timeout resilience
- optional streaming output
- optional structured output with usage and cost metadata

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
from autourgos_google_modelkit import GoogleTextModel, GOOGLE_TEXT_MODEL_NAME

llm = GoogleTextModel(
    model=GOOGLE_TEXT_MODEL_NAME.GEMINI_3_1_PRO,
)

print(llm.invoke("Explain RAG in simple terms."))
```

Example response:

```text
RAG (Retrieval-Augmented Generation) combines search and generation.
The model first retrieves relevant knowledge, then writes an answer using that context.
```

## Supported Parameters

GoogleTextModel supports:

- model
- api_key
- prompt_template
- temperature
- top_p
- top_k
- max_tokens
- thinking_level
- structured_output
- Stream
- max_retries
- timeout
- backoff_factor

## Parameter Examples

### Model

Use enums (recommended) or raw strings.

```python
from autourgos_google_modelkit import GoogleTextModel, GOOGLE_TEXT_MODEL_NAME

llm = GoogleTextModel(model=GOOGLE_TEXT_MODEL_NAME.GEMINI_3_1_PRO)
print(llm.invoke("Explain Agentic AI in simple terms."))
```

```python
from autourgos_google_modelkit import GoogleTextModel

llm = GoogleTextModel(model="gemini-3.1-pro")
print(llm.invoke("Explain observability in software systems."))
```

### Prompt Template

```python
from autourgos_google_modelkit import GoogleTextModel

llm = GoogleTextModel(
    model="gemini-2.5-flash",
    prompt_template="Explain {topic} in simple terms.",
)

print(llm.invoke(prompt_variables={"topic": "rate limiting"}))
```

### Temperature, Top P, Top K, Max Tokens

```python
from autourgos_google_modelkit import GoogleTextModel

llm = GoogleTextModel(
    model="gemini-2.5-flash",
    temperature=0.7,
    top_p=0.9,
    top_k=40,
    max_tokens=512,
)
```

### Thinking Level

```python
from autourgos_google_modelkit import GoogleTextModel, GOOGLE_TEXT_THINKING_LEVEL

llm = GoogleTextModel(
    model="gemini-3-flash-preview",
    thinking_level=GOOGLE_TEXT_THINKING_LEVEL.HIGH,
)
```

Note:
- Higher thinking levels may improve reasoning quality.
- Higher thinking levels may also increase latency and cost.
- thinking_level is only supported by compatible Gemini models.

### Structured Output

When structured_output=True, invoke returns a dictionary with response text and metadata.

```python
from autourgos_google_modelkit import GoogleTextModel

llm = GoogleTextModel(
    model="gemini-3-flash-preview",
    structured_output=True,
)

result = llm.invoke("Summarize observability in one paragraph.")
print(result)
```

Example response:

```json
{
  "model": "gemini-3-flash-preview",
  "response": "Observability is the ability to understand system behavior from logs, metrics, and traces.",
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

### Stream

When Stream=True, invoke returns an iterator of text chunks.

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

### Retries and Timeouts

```python
from autourgos_google_modelkit import GoogleTextModel

llm = GoogleTextModel(
    model="gemini-2.5-flash",
    max_retries=5,
    timeout=60.0,
    backoff_factor=1.5,
)
```

## Validation Rules

Important behavior:

- structured_output=True is only supported when Stream=False
- temperature must be in [0.0, 2.0]
- top_p must be in [0.0, 1.0]
- top_k must be >= 1
- max_tokens must be >= 1
- max_retries must be >= 1
- timeout must be > 0 (when provided)
- backoff_factor must be >= 0

## Error Hierarchy

- GoogleTextModelError
- GoogleTextModelImportError
- GoogleTextModelAPIError
- GoogleTextModelResponseError

## References

- [Google Gemini API Documentation](https://developers.generativeai.google/api/)
- [Google Gemini Pricing](https://developers.generativeai.google/pricing/)
- [Google Gemini Model Capabilities](https://developers.generativeai.google/models/)
