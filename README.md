# Autourgos Google Model Kit

![Gemini](https://raw.githubusercontent.com/DevxJitin/autourgos-google-modelkit/main/README/Image_Dark.png)

![Pypi](https://img.shields.io/badge/Pypi-0.1.0-blue?style=flat-square)
![Release](https://img.shields.io/badge/Release-Early%20Development-brown?style=flat-square)
![Framework](https://img.shields.io/badge/Framework-Autourgos-orange?style=flat-square)
![Wrapper](https://img.shields.io/badge/Wrapper-Google-brightgreen?style=flat-square)
![Developed%20by](https://img.shields.io/badge/Developed%20by-DevxJitin-gold?style=flat-square)
![Documented%20by](https://img.shields.io/badge/Documented%20by-Sonia-blue?style=flat-square)

Autourgos is a Python framework that helps developers build AI agents, from simple bots to advanced systems. It will contains tools, autonomous tools, memory, llm agents, and many more features to make it easy to create intelligent Agents.

The Autourgos Google Model Kit is a package that connects this framework to Google's Gemini AI models. It provides simple, ready-to-use wrappers for handling both text and vision (image-based) tasks. Features like automatic retries, input validation, and cost tracking are built-in, making it straightforward to add Google's powerful AI capabilities to your projects in a reliable way.

This package gives you two Model Wrappers for Google Gemini APIs:

- `GoogleTextModel` for text generation
- `GoogleVisionModel` for image + text prompts with text output

It focuses on clean API usage, validation, retries, and structured response metadata.

## Table of Contents

- [Why Use This Package](#why-use-this-package)
- [Installation and API Key Setup](#installation-and-api-key-setup)
- [Text generation (GoogleTextModel)](#text-generation-googletextmodel)
  - [Basic usage](#basic-usage)
  - [Model Initialization and Configuration](#model-initialization-and-configuration)
  - [Base Setup](#base-setup)
  - [Parameter: Model](#parameter-model)
  - [Parameter: API Key](#parameter-api-key)
  - [Parameter: Prompt Template](#parameter-prompt-template)
  - [Parameter: Temperature](#parameter-temperature)
  - [Parameter: Top P](#parameter-top-p)
  - [Parameter: Top K](#parameter-top-k)
  - [Parameter: Max Tokens](#parameter-max-tokens)
  - [Parameter: Thinking Level](#parameter-thinking-level)
  - [Parameter: Structured Output](#parameter-structured-output)
  - [Parameter: Stream](#parameter-stream)
  - [Parameter: Retries and Timeouts](#parameter-retries-and-timeouts)
- [Vision generation (GoogleVisionModel)](#vision-generation-googlevisionmodel)
  - [Basic usage](#basic-usage-1)
  - [Streaming mode](#streaming-mode)
  - [Supported Parameters for Vision Model Initialization and Configuration](#supported-parameters-for-vision-model-initialization-and-configuration)
  - [Parameter: Media Resolution](#parameter-media-resolution)
- [Validation and Errors](#validation-and-errors)
- [References](#references)
- [Credits](#credits)
- [Social Media](#social-media)
- [Contributing](#contributing)

## Why Use This Package

- **Typed Model Enums**: Safer model selection using built-in enums (`GOOGLE_TEXT_MODEL_NAME`, etc.).
- **Consistent API**: One unified class interface (`invoke`) across both text and vision models.
- **Streaming Support**: Optional real-time streaming mode for token-by-token generation.
- **Structured Output**: Access response text alongside token usage and estimated cost metadata.
- **Prompt Templates**: Reusable templates with strict variable validation.
- **Advanced Capabilities**: Full support for Gemini 3 thinking levels and vision media resolution tuning.
- **Resilience**: Built-in retry mechanism with exponential backoff and timeout configurations.
- **Flexible Configuration**: API key resolution from explicit arguments or standard environment variables.

## Installation and API Key Setup

#### Install the package:
```command
pip install autourgos-google-modelkit
```

#### Set API key in environment variables: PowerShell:

```powershell
$env:GOOGLE_API_KEY = "your-api-key"
```

#### Set API key in environment variables: Bash:

```bash
export GOOGLE_API_KEY="your-api-key"
```



## Text generation (```GoogleTextModel```)

### Basic usage
```python
from autourgos_google_modelkit import GoogleTextModel, GOOGLE_TEXT_MODEL_NAME

llm = GoogleTextModel(
	model=GOOGLE_TEXT_MODEL_NAME.GEMINI_3_1_PRO,
)

print(llm.invoke("Explain RAG in simple terms."))
```

### Example response

```text
RAG (Retrieval-Augmented Generation) combines search and generation.
The model first retrieves relevant knowledge, then writes an answer using that context.
This improves factual accuracy and reduces hallucinations.
```
### Model Initialization and Configuration
### Base Setup
```python
from autourgos_google_modelkit import GoogleTextModel
llm = GoogleTextModel()
```
### Parameter: ```Model```
Model selection can be done using enums or strings. Enums provide better safety and autocomplete, while strings offer flexibility.

Supported models include:
- `GOOGLE_TEXT_MODEL_NAME.GEMINI_3_1_PRO` (string: "gemini-3.1-pro")
- `GOOGLE_TEXT_MODEL_NAME.GEMINI_3_FLASH_PREVIEW` (string: "gemini-3-flash-preview")
- `GOOGLE_TEXT_MODEL_NAME.GEMINI_2_5_FLASH` (string: "gemini-2.5-flash")
- `GOOGLE_TEXT_MODEL_NAME.GEMINI_2_5_PRO` (string: "gemini-2.5-pro")

Using enums (recommended) for better safety and autocomplete:
```python
from autourgos_google_modelkit import GOOGLE_TEXT_MODEL_NAME, GoogleTextModel
llm = GoogleTextModel(
  model=GOOGLE_TEXT_MODEL_NAME.GEMINI_3_1_PRO
  )
print(llm.invoke("Explain RAG in simple terms."))
```
Using strings:
```python
from autourgos_google_modelkit import GoogleTextModel
llm = GoogleTextModel(
  model="gemini-3.1-pro"
  )
print(llm.invoke("Explain Agentic AI in simple terms."))
```
### Parameter: ```API Key```

Explicitly setting the API key:
```python
from autourgos_google_modelkit import GoogleTextModel
llm = GoogleTextModel(
  api_key="AIzaSy..."
  )
```
Relying on environment variable (recommended):
```python
from autourgos_google_modelkit import GoogleTextModel
llm = GoogleTextModel()
```

### Parameter: ```Prompt Template```

Setting a reusable prompt template with variables
```python
from autourgos_google_modelkit import GoogleTextModel
llm = GoogleTextModel(
  prompt_template="Explain {topic} in simple terms."
  )
print(llm.invoke(prompt_variables={"topic": "RAG"}))
```

### Parameter: ```Temperature```
Temperature controls randomness in generation. Google suggests values between 0.0 and 2.0.
- Lower values (e.g., 0.0) produce more deterministic output.
- Moderate values (e.g., 1.0) balance coherence and creativity.
- Higher values (e.g., 2.0) produce more creative and varied output.

```python
from autourgos_google_modelkit import GoogleTextModel
llm = GoogleTextModel(
  temperature=0.7
  )
```

### Parameter: ```Top P```
Top-p (nucleus sampling) controls diversity by limiting token selection to a cumulative probability threshold. Valid values are between 0.0 and 1.0.
- Lower values (e.g., 0.0) produce more deterministic output.
- Moderate values (e.g., 0.9) allow for more diversity while maintaining coherence.
- Higher values (e.g., 1.0) produce more diverse and creative output.

```python
from autourgos_google_modelkit import GoogleTextModel
llm = GoogleTextModel(
  top_p=0.9
  )
```

### Parameter: ```Top K```
Top-k (top-k sampling) limits token selection to the top-k most likely tokens. Valid values are between 1 and 40.
- Lower values (e.g., 1) produce more deterministic output.
- Moderate values (e.g., 20) allow for more diversity while maintaining coherence.
- Higher values (e.g., 40) produce more diverse and creative output.

```python
from autourgos_google_modelkit import GoogleTextModel
llm = GoogleTextModel(
  top_k=40
  )
```

### Parameter: ```Max Tokens```
Max tokens sets the maximum output token budget for the response.
```python
from autourgos_google_modelkit import GoogleTextModel
llm = GoogleTextModel(
  max_tokens=1024
  )
```

### Parameter: ```Thinking Level```
Thinking level controls the depth of reasoning for supported Gemini models. Valid values are:
- `GOOGLE_TEXT_THINKING_LEVEL.LOW`
- `GOOGLE_TEXT_THINKING_LEVEL.MEDIUM`
- `GOOGLE_TEXT_THINKING_LEVEL.HIGH`
```python
from autourgos_google_modelkit import GoogleTextModel, GOOGLE_TEXT_THINKING_LEVEL
llm = GoogleTextModel(
  thinking_level=GOOGLE_TEXT_THINKING_LEVEL.HIGH
  )
```
> Note: Higher thinking levels may improve reasoning quality but can also increase latency and cost.

> Note: The `thinking_level` parameter is only supported by Gemini 3.1 Pro, Gemini 3 Flash Preview models and Gemini 3.1 Flash Lite models. Using it with unsupported models will raise a validation error.

### Parameter: ```Structured Output```
Setting `structured_output=True` returns a dictionary with response text and metadata instead of plain text.
```python
from autourgos_google_modelkit import GoogleTextModel
llm = GoogleTextModel(
  structured_output=True
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

### Parameter: ```Stream```
Setting `Stream=True` enables real-time streaming of generated text chunks. The `invoke()` method will return an iterator that yields text chunks as they are generated.
```python
from autourgos_google_modelkit import GoogleTextModel
llm = GoogleTextModel(
  model=GOOGLE_TEXT_MODEL_NAME.GEMINI_3_1_PRO,
  Stream=True,
)
stream = llm.invoke("Write a short note on clean architecture.")
for chunk in stream:
  print(chunk, end="", flush=True)
print()
```

### Parameter: ```Retries and Timeouts```
The package includes built-in retry logic with exponential backoff for transient errors. You can configure the retry behavior using the following parameters:
- `max_retries`: Total number of retry attempts (default: 3)
- `timeout`: Request timeout in seconds (default: 30.0)
- `backoff_factor`: Multiplier for calculating delay between retries (default: 1.0)
```python
from autourgos_google_modelkit import GoogleTextModel
llm = GoogleTextModel(
  max_retries=5,
  timeout=60.0,
  backoff_factor=1.5
)
```


## Vision generation (```GoogleVisionModel```)

### Basic usage

```python
from autourgos_google_modelkit import GoogleVisionModel, GOOGLE_VISION_MODEL_NAME

vision = GoogleVisionModel(
  model=GOOGLE_VISION_MODEL_NAME.GEMINI_3_FLASH_PREVIEW
  )
response = vision.invoke(
  prompt="Describe what is visible in this image.",
  image="./sample.jpg"
  )
print(response)
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

> Note: The exact chunk boundaries are not fixed and can vary by SDK/model/network conditions

Final assembled response:

```text
Clean architecture separates business logic from framework details.
It improves testability, long-term maintainability, and replacement of external dependencies.
```

### Supported Parameters for Vision Model Initialization and Configuration
- `model`: Model selection using enums or strings.
- `api_key`: Explicit API key or environment variable resolution.
- `prompt_template`: Reusable prompt templates with variable validation.
- `temperature`, `top_p`, `top_k`, `max_tokens`: Sampling parameters for text generation.
- `thinking_level`: Reasoning depth control for supported Gemini models.
- `structured_output`: Option to receive response metadata instead of plain text.
- `Stream`: Enable real-time streaming of generated text chunks.
- `media_resolution`: Vision input quality hint (enum values: LOW, MEDIUM, HIGH).
- `max_retries`, `timeout`, `backoff_factor`: Retry and timeout configurations for API calls.

### Parameter: ```Media Resolution```
The `media_resolution` parameter allows you to specify the quality of the vision input. Supported enum values are:
- `GOOGLE_VISION_MEDIA_RESOLUTION.LOW`
- `GOOGLE_VISION_MEDIA_RESOLUTION.MEDIUM`
- `GOOGLE_VISION_MEDIA_RESOLUTION.HIGH`
```python
from autourgos_google_modelkit import GoogleVisionModel, GOOGLE_VISION_MEDIA_RESOLUTION
vision = GoogleVisionModel(
  model=GOOGLE_VISION_MODEL_NAME.GEMINI_3_FLASH_PREVIEW,
  media_resolution=GOOGLE_VISION_MEDIA_RESOLUTION.HIGH
)
```
> Note: Higher media resolution may improve model performance on complex images but can also increase latency and cost.


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

## References
- [Google Gemini API Documentation](https://developers.generativeai.google/api/)
- [Google Gemini Pricing](https://developers.generativeai.google/pricing/)
- [Google Gemini Model Capabilities](https://developers.generativeai.google/models/)

## Credits
Developed and maintained by [DevxJitin](https://github.com/DevxJitin)  
Documented by [Sonia](https://github.com/SoniaDahiya)

## Social Media
[![GitHub](https://img.shields.io/badge/GitHub-DevxJitin-green?style=flat-square&logo=github)](https://github.com/DevxJitin)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-DevxJitin-green?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/devxjitin/)
[![Whatsapp](https://img.shields.io/badge/WhatsApp-DevxJitin-green?style=flat-square&logo=whatsapp)](https://wa.me/7078710389)  
[![GitHub Sonia](https://img.shields.io/badge/GitHub-SoniaDahiya-blue?style=flat-square&logo=github)](https://github.com/SoniaDahiya)
[![LinkedIn Sonia](https://img.shields.io/badge/LinkedIn-SoniaDahiya-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/sonia-59193a318/)

## Contributing
Contributions are welcome! Please open issues for bugs or feature requests, and submit pull requests for improvements.