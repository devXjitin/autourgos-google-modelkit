# Autourgos GoogleTextModel Documentation

**GoogleTextModel** is a production-ready wrapper for [Google Generative AI](https://ai.google.dev/) (Gemini) text generation APIs. It is designed to simplify the process of interacting with Gemini models and provide a consistent interface across different SDKs and platforms.

It offers a clean and extensible interface for generating text with support for streaming, retries, validation, and structured outputs while making it suitable for both rapid prototyping and scalable applications.

This module is part of the Autourgos ModelKit ecosystem and focuses specifically on text-based generation using Google models.

### Key Goals
- Simplify interaction with Gemini APIs
- Provide type-safe model selection
- Enable robust production usage with retries and validation
- Support structured outputs for AI-driven workflows
- Track token usage and pricing awareness

### References
- [Google Gemini API Documentation](https://ai.google.dev/)
- [Google Generative AI Python SDK](https://github.com/google/generative-ai-python)
- [Autourgos Framework (if applicable)](https://github.com/)

## Features

- **Typed Model Selection**  
  Use predefined enums for selecting supported Gemini models, reducing errors and improving clarity.

- **Streaming Support**  
  Generate responses in real-time using streaming mode for better user experience in chat or live applications.

- **Retry Mechanism**  
  Built-in retry logic for handling transient API failures and improving reliability in production systems.

- **Structured Output Support**  
  Enforce schema-based responses for consistent and machine-readable outputs.

- **Prompt Templates**  
  Easily manage and reuse prompts with templating support for dynamic inputs.

- **Validation Rules**  
  Validate inputs and outputs to ensure correctness and prevent runtime issues.

- **Token Usage Tracking**  
  Monitor token consumption for better cost control and optimization.

- **Thinking Level Control**  
  Adjust reasoning depth using configurable thinking levels for performance vs. quality trade-offs.

- **Flexible Configuration**  
  Customize parameters like temperature, max tokens, top-p, and top-k.

- **Production-Ready Design**  
  Designed for scalability, maintainability, and clean integration into larger systems.


## Module Scope

Location:

- src/autourgos_google_modelkit/textmodel

Public exports:

- MODEL
- THINKING_LEVEL
- MODEL_PRICING_USD_PER_MILLION
- resolve_model_pricing
- GoogleTextModel
- GoogleTextModelError
- GoogleTextModelImportError
- GoogleTextModelAPIError
- GoogleTextModelResponseError

## Quick Start

### 1) Set API key

The module resolves API keys in this order:

1. api_key argument passed at call time
2. GOOGLE_API_KEY environment variable
3. GEMINI_API_KEY environment variable

PowerShell:

```powershell
$env:GOOGLE_API_KEY = "your-api-key"
```

Bash:

```bash
export GOOGLE_API_KEY="your-api-key"
```

### 2) Class-based usage

```python
from autourgos_google_modelkit import GoogleTextModel, MODEL, THINKING_LEVEL

llm = GoogleTextModel(
    model=MODEL.GEMINI_3_FLASH_PREVIEW,
    temperature=0.2,
    max_tokens=300,
    thinking_level=THINKING_LEVEL.MINIMAL,
    Stream=False,
)

answer = llm.invoke("Explain retrieval augmented generation in simple terms.")
print(answer)
```

### 3) Stream mode usage

```python
from autourgos_google_modelkit import GoogleTextModel, MODEL

llm = GoogleTextModel(model=MODEL.GEMINI_2_5_FLASH, Stream=True)

print(llm.invoke("Write a short haiku about clean code."))
```

## API Reference

### MODEL enum

The MODEL enum gives type-safe model IDs and IDE autocomplete.

Available values:

- MODEL.GEMINI_3_1_PRO_PREVIEW
- MODEL.GEMINI_3_PRO_PREVIEW
- MODEL.GEMINI_3_FLASH_PREVIEW
- MODEL.GEMINI_3_1_FLASH_LITE_PREVIEW
- MODEL.GEMINI_2_5_PRO
- MODEL.GEMINI_2_5_FLASH
- MODEL.GEMINI_2_5_FLASH_LITE

You can also pass raw strings to APIs when needed.

### GoogleTextModel

Constructor mirrors the function parameters and stores shared defaults for repeated calls.

Important parameter:

- Stream: bool = False

Methods:

- invoke(prompt: str | None = None, prompt_variables: dict[str, Any] | None = None) -> str

Behavior:

- Stream=False: non-streaming generation mode
- Stream=True: streaming mode is used internally, and invoke returns aggregated text
- structured_output=True: returns a dictionary containing response, token usage, model info, and cost
- prompt is optional when prompt_template is configured in the constructor

## Parameter Validation

The module validates these rules before making API calls:

- prompt must be a non-empty string when provided
- if prompt is omitted, prompt_template must be configured and render to a non-empty string
- prompt_variables must be a dictionary when provided
- temperature must be between 0.0 and 2.0 when provided
- top_p must be between 0.0 and 1.0 when provided
- top_k must be >= 1 when provided
- max_tokens must be >= 1 when provided
- thinking_level must be one of: minimal, low, medium, high when provided
- structured_output must be a boolean
- structured_output=True is only supported when Stream=False
- max_retries must be an integer >= 1
- timeout must be > 0 when provided
- backoff_factor must be >= 0

## Thinking Controls

The wrapper supports Gemini 3 thinking level control.

- thinking_level: use THINKING_LEVEL enum values for typed usage

Enum values:

- THINKING_LEVEL.MINIMAL
- THINKING_LEVEL.LOW
- THINKING_LEVEL.MEDIUM
- THINKING_LEVEL.HIGH

Default behavior keeps thinking off by setting:

- thinking_level="minimal"

Gemini 3 support note from docs:

- minimal is not supported on Pro preview models
- minimal is supported on Flash/Flash-Lite

You can increase reasoning depth explicitly when needed:

```python
llm = GoogleTextModel(
    model=MODEL.GEMINI_3_FLASH_PREVIEW,
    thinking_level=THINKING_LEVEL.LOW,
)
```

## Structured Output

Enable structured output to receive a metadata-rich dictionary:

```python
from autourgos_google_modelkit import GoogleTextModel, MODEL, THINKING_LEVEL

llm = GoogleTextModel(
        model=MODEL.GEMINI_3_FLASH_PREVIEW,
        thinking_level=THINKING_LEVEL.MINIMAL,
        structured_output=True,
)

result = llm.invoke("How does AI work?")
print(result)
```

## Optional Prompt Templates

You can configure a prompt template and supply variables at call time.

```python
from autourgos_google_modelkit import GoogleTextModel, MODEL

llm = GoogleTextModel(
    model=MODEL.GEMINI_2_5_FLASH,
    prompt_template="Summarize this in {tone} tone:\n\n{text}",
)

result = llm.invoke(
    prompt_variables={
        "tone": "concise",
        "text": "Autourgos is an agentic AI framework...",
    }
)
print(result)
```

Direct prompts still override template usage when provided:

```python
result = llm.invoke("Write one-line summary of agentic AI.")
print(result)
```

Returned structure:

```json
{
    "model": "gemini-3-flash-preview",
    "response": "...",
    "input_tokens": 6,
    "output_tokens": 732,
    "Total_tokens": 738,
    "Cost": "$0.00219900",
    "cost_details": {
        "value_usd": 0.002199,
        "input_rate_per_million": 0.5,
        "output_rate_per_million": 3.0
    }
}
```

## Supported Text Models and Pricing

Only models supported by this textmodel package are listed below.

| model_name | input token limit | output token limit | input token pricing (USD / 1M) | output token pricing (USD / 1M) |
| --- | --- | --- | --- | --- |
| gemini-3.1-pro-preview | provider docs | provider docs | 2.00 (4.00 over 200k prompt tokens) | 12.00 (18.00 over 200k prompt tokens) |
| gemini-3-pro-preview | provider docs | provider docs | 2.00 (4.00 over 200k prompt tokens) | 12.00 (18.00 over 200k prompt tokens) |
| gemini-3-flash-preview | provider docs | provider docs | 0.50 | 3.00 |
| gemini-3.1-flash-lite-preview | provider docs | provider docs | 0.25 | 1.50 |
| gemini-2.5-pro | provider docs | provider docs | 1.25 (2.50 over 200k prompt tokens) | 10.00 (15.00 over 200k prompt tokens) |
| gemini-2.5-flash | provider docs | provider docs | 0.30 | 2.50 |
| gemini-2.5-flash-lite | provider docs | provider docs | 0.10 | 0.40 |

Notes:

- Pricing values are sourced from MODEL_PRICING_USD_PER_MILLION in this package.
- Input and output token limits are not hardcoded in this package and should be read from current Google model docs.

## Retry and Backoff

Generation retries use exponential backoff:

sleep = backoff_factor * (2 ** (attempt - 1))

Defaults:

- GoogleTextModel (invoke): max_retries = 3
- backoff_factor: 0.5

## Error Model

Exception hierarchy:

- GoogleTextModelError (base)
- GoogleTextModelImportError
- GoogleTextModelAPIError
- GoogleTextModelResponseError

Recommended handling:

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
    print(f"Setup or import error: {exc}")
except GoogleTextModelAPIError as exc:
    print(f"Request failed: {exc}")
except GoogleTextModelResponseError as exc:
    print(f"Response parsing issue: {exc}")
```

## SDK Compatibility Strategy

The implementation is defensive across SDK variations and tries multiple call paths.

Import paths attempted:

- google.generativeai
- google.genai

Generation strategies attempted:

- client.models.generate_content
- GenerativeModel.generate_content
- helper fallbacks (generate_text, generate, model_generate)

Streaming strategies attempted:

- GenerativeModel.generate_content with stream=True
- client.models.stream_generate_content
- GenerativeModel.generate_content_stream

This design improves runtime compatibility across changing SDK versions.

## Notes for Integration

- For agent frameworks, GoogleTextModel is usually the simplest integration surface.
- Use Stream=True when you want generation to run through streaming mode internally.

## Minimal End-to-End Example

```python
from autourgos_google_modelkit import GoogleTextModel, MODEL


def build_model() -> GoogleTextModel:
    return GoogleTextModel(
        model=MODEL.GEMINI_2_5_FLASH_LITE,
        temperature=0.2,
        max_tokens=256,
    )


if __name__ == "__main__":
    llm = build_model()
    print(llm.invoke("Say hello in one line."))
```
