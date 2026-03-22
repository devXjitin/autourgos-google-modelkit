"""
Google Gemini Text Model Provider

Production-ready wrapper around Google's Generative AI client.

Author: DevxJitin & QueenSonia
Version: 0.1.0
"""

from typing import Optional, Any, Dict, Iterator
import os
import time
import warnings
import sys
import re
from string import Formatter
from contextlib import contextmanager

from .models import MODEL, THINKING_LEVEL, resolve_model_pricing


"""Environment Configuration.

This section reduces noisy runtime diagnostics from underlying gRPC/logging
libraries. ``setdefault`` preserves deployment-provided environment values.
"""
os.environ.setdefault('GRPC_VERBOSITY', 'ERROR')
os.environ.setdefault('GLOG_minloglevel', '2')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

"""Suppress Python warnings emitted by gRPC internals."""
warnings.filterwarnings('ignore', category=UserWarning, module='.*grpc.*')

@contextmanager
def suppress_stderr():
    """Temporarily suppress stderr noise produced by SDK internals.

    This context manager suppresses both Python-level ``sys.stderr`` writes and,
    when possible, low-level file descriptor writes to stderr. It is primarily
    used around Google SDK import/configuration and request calls to avoid
    verbose gRPC diagnostics leaking into user logs.

    Yields:
        None. Control returns to the wrapped block while stderr is muted.

    Notes:
        - The low-level redirection may fail on some runtimes; in that case,
          Python-level stderr suppression still applies.
        - Original stderr state is always restored in ``finally``.
    """
    import io
    
    original_stderr = sys.stderr
    original_stderr_fd = None
    
    try:
        """Save and redirect the low-level stderr file descriptor."""
        try:
            original_stderr_fd = os.dup(2)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 2)
            os.close(devnull)
        except Exception:
            pass
        
        """Redirect Python-level stderr writes during the protected block."""
        sys.stderr = io.StringIO()
        yield
    finally:
        """Restore stderr state for both low-level and Python-level streams."""
        if original_stderr_fd is not None:
            try:
                os.dup2(original_stderr_fd, 2)
                os.close(original_stderr_fd)
            except Exception:
                pass
        sys.stderr = original_stderr

"""Module-Level Client Import.

Importing at module load avoids repeated import overhead and supports both the
legacy and current packaging paths used by Google Gemini SDK releases.
"""
_GOOGLE_GENAI_AVAILABLE = False
genai_module = None
_GOOGLE_IMPORT_ERROR: Optional[str] = None
try:
    with suppress_stderr():
        try:
            """Preferred import path for the older google-generativeai SDK."""
            import google.generativeai as genai_module  # type: ignore
            _GOOGLE_GENAI_AVAILABLE = True
        except Exception:
            """Fallback import path used by newer google-genai style packaging."""
            from google import genai as genai_module  # type: ignore
            _GOOGLE_GENAI_AVAILABLE = True
except Exception as exc:
    _GOOGLE_GENAI_AVAILABLE = False
    genai_module = None
    _GOOGLE_IMPORT_ERROR = str(exc)


"""Custom Exception Hierarchy used by this provider."""
class GoogleTextModelError(Exception):
    """
    Base exception class for all Google Gemini LLM-related errors.
    
    All custom exceptions in this module inherit from this class,
    allowing users to catch any module-specific error with a single
    except clause.
    """


class GoogleTextModelImportError(GoogleTextModelError):
    """
    Raised when the Google generative AI client library cannot be imported or initialized.
    
    Common causes:
        - google-generativeai package not installed (pip install google-generativeai)
        - Missing or invalid API key
        - Client initialization failure
        - Incompatible SDK version
    """


class GoogleTextModelAPIError(GoogleTextModelError):
    """
    Raised when the API request fails after all retry attempts.
    
    Common causes:
        - Network connectivity issues
        - API service unavailable
        - Rate limiting or quota exceeded
        - Invalid model name or parameters
        - Authentication failures
    """


class GoogleTextModelResponseError(GoogleTextModelError):
    """
    Raised when the API response cannot be interpreted or is malformed.
    
    Common causes:
        - Empty response from API
        - Missing expected fields in response structure
        - Unexpected response format from SDK version
        - Content safety filters blocked the response
    """


def _normalize_model_name(model: str | MODEL) -> str:
    """Normalize model input into a stable model-id string.

    Args:
        model: Either a raw model name string or an enum-like object whose
            ``value`` contains the model identifier.

    Returns:
        A non-empty model identifier string.

    Raises:
        ValueError: If the model cannot be resolved to a non-empty string.
    """
    if isinstance(model, str):
        model_name = model.strip()
        if model_name:
            return model_name
        raise ValueError("model must be a non-empty string")

    """Support enum-like objects whose .value stores the model identifier."""
    value = getattr(model, "value", None)
    if isinstance(value, str) and value.strip():
        return value.strip()

    model_name = str(model).strip()
    if model_name:
        return model_name
    raise ValueError("model must be a non-empty string")


def _normalize_thinking_level(thinking_level: str | THINKING_LEVEL | None) -> Optional[str]:
    """Normalize thinking level input into one of the supported string values."""
    if thinking_level is None:
        return None

    if isinstance(thinking_level, THINKING_LEVEL):
        return thinking_level.value

    if isinstance(thinking_level, str):
        normalized = thinking_level.strip().lower()
        if normalized in {"minimal", "low", "medium", "high"}:
            return normalized
    raise ValueError("thinking_level must be one of: minimal, low, medium, high")


def _validate_thinking_level_support(model_name: str, thinking_level: Optional[str]) -> None:
    """Validate model-specific thinking level support from Gemini 3 docs."""
    if thinking_level is None:
        return

    model_name_normalized = model_name.strip().lower()
    minimal_unsupported_models = {
        "gemini-3.1-pro-preview",
        "gemini-3-pro-preview",
        "gemini-3.1-pro-preview-customtools",
    }

    if thinking_level == "minimal" and model_name_normalized in minimal_unsupported_models:
        raise ValueError(
            f"thinking_level='minimal' is not supported for model '{model_name}'. "
            "Use low, medium, or high."
        )


def _resolve_api_key(api_key: Optional[str]) -> Optional[str]:
    """Resolve API key from explicit input and supported environment fallbacks.

    Resolution order:
        1. Explicit ``api_key`` argument
        2. ``GOOGLE_API_KEY`` environment variable
        3. ``GEMINI_API_KEY`` environment variable

    Args:
        api_key: Optional explicit key passed by caller.

    Returns:
        Resolved API key string, or ``None`` if no source is available.
    """
    return api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")


def _build_generation_config(
    *,
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    max_tokens: Optional[int],
    thinking_level: Optional[str],
) -> Dict[str, Any]:
    """Build a generation configuration dictionary for Gemini SDK calls.

    Only explicitly provided values are included so the SDK can apply its own
    defaults for omitted fields.

    Args:
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability mass.
        top_k: Top-k sampling bound.
        max_tokens: Maximum output token count.

    Returns:
        Configuration dictionary compatible with supported SDK call patterns.
    """
    generation_config: Dict[str, Any] = {}
    if temperature is not None:
        generation_config["temperature"] = temperature
    if top_p is not None:
        generation_config["top_p"] = top_p
    if top_k is not None:
        generation_config["top_k"] = top_k
    if max_tokens is not None:
        generation_config["max_output_tokens"] = max_tokens
    if thinking_level is not None:
        generation_config["thinking_config"] = {"thinking_level": thinking_level}
    return generation_config


def _extract_template_fields(template: str) -> set[str]:
    """Extract placeholder field names from a format template string."""
    fields: set[str] = set()
    for _, field_name, _, _ in Formatter().parse(template):
        if not field_name:
            continue
        base_name = field_name.split("!", 1)[0].split(":", 1)[0].strip()
        if base_name:
            fields.add(base_name)
    return fields


def _coerce_prompt_variable(value: Any) -> str:
    """Convert prompt variables into strings for template rendering."""
    if value is None:
        return ""
    return str(value)


def _configure_genai_client(genai: Any, api_key: str) -> Any:
    """Configure SDK auth state and optionally construct a client instance.

    This helper centralizes SDK setup used by both non-streaming and streaming
    APIs, while tolerating version differences where a ``Client`` class may not
    exist or may require different initialization signatures.

    Args:
        genai: Imported Google SDK module.
        api_key: Resolved API key.

    Returns:
        Client object when available, otherwise ``None``.
    """
    client = None

    """Configure API key in SDK-global state when supported by SDK version."""
    try:
        with suppress_stderr():
            cfg = getattr(genai, "configure", None)
            if callable(cfg):
                cfg(api_key=api_key)
    except Exception:
        pass

    """Initialize client object for SDK versions exposing a Client class."""
    try:
        with suppress_stderr():
            ClientCls = getattr(genai, "Client", None)
            if callable(ClientCls):
                try:
                    client = ClientCls(api_key=api_key)
                except TypeError:
                    client = ClientCls()
    except Exception:
        pass

    return client


def _extract_text_from_response(resp: Any) -> Optional[str]:
    """
    Extract generated text from Google API response using defensive parsing.
    
    Optimized with early returns and minimal overhead for common response formats.
    Handles multiple SDK versions and response structures for maximum compatibility.
    
    Args:
        resp: Response object from Google Gemini API
    
    Returns:
        Extracted text string if found, None if extraction fails
    """
    if resp is None:
        return None

    """Strategy 0: Raw string chunks from some stream adapters."""
    if isinstance(resp, str) and resp.strip():
        return resp

    """Strategy 1: Direct .text attribute (most common SDK response shape)."""
    try:
        text = getattr(resp, "text", None)
        if text is not None:
            if callable(text):
                text = text()
            if isinstance(text, str) and text.strip():
                return text
    except Exception:
        pass

    """Strategy 2: Structured candidates format used by multiple SDK releases."""
    candidates = getattr(resp, "candidates", None)
    if candidates and isinstance(candidates, (list, tuple)) and candidates:
        first = candidates[0]
        
        """Attempt candidate.content.parts[*].text and join all text fragments."""
        content = getattr(first, "content", None)
        if content:
            parts = getattr(content, "parts", None)
            if parts and isinstance(parts, (list, tuple)) and parts:
                joined_parts: list[str] = []
                for part in parts:
                    part_text = getattr(part, "text", None)
                    if isinstance(part_text, str) and part_text.strip():
                        joined_parts.append(part_text)
                if joined_parts:
                    return "".join(joined_parts)
            
            """Fallback to content.text when parts are unavailable."""
            content_text = getattr(content, "text", None)
            if isinstance(content_text, str) and content_text.strip():
                return content_text
        
        """Fallback to candidate.text."""
        candidate_text = getattr(first, "text", None)
        if isinstance(candidate_text, str) and candidate_text.strip():
            return candidate_text

    """Strategy 3: Dictionary-based responses from alternate helpers/adapters."""
    if isinstance(resp, dict):
        """Fast-path common flat text keys in dictionary responses."""
        for key in ("text", "output_text", "delta", "content"):
            value = resp.get(key)
            if isinstance(value, str) and value.strip():
                return value

        candidates = resp.get("candidates")
        if candidates and isinstance(candidates, (list, tuple)) and candidates:
            first = candidates[0]
            if isinstance(first, dict):
                content = first.get("content")
                if isinstance(content, dict):
                    parts = content.get("parts")
                    if isinstance(parts, (list, tuple)) and parts:
                        joined_parts: list[str] = []
                        for part in parts:
                            if isinstance(part, dict):
                                text = part.get("text")
                                if isinstance(text, str) and text.strip():
                                    joined_parts.append(text)
                        if joined_parts:
                            return "".join(joined_parts)
                
                """Try simpler structures where content/text is directly present."""
                text = first.get("content") or first.get("text")
                if isinstance(text, str) and text.strip():
                    return text
        
        """Try top-level text field as a final fallback."""
        text = resp.get("text")
        if isinstance(text, str) and text.strip():
            return text

    return None


def _extract_usage_metadata(resp: Any) -> Dict[str, Optional[int]]:
    """Extract input/output/total token counts from SDK response metadata."""
    usage = getattr(resp, "usage_metadata", None)
    if usage is None:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        }

    input_tokens = getattr(usage, "prompt_token_count", None)
    output_tokens = getattr(usage, "candidates_token_count", None)
    total_tokens = getattr(usage, "total_token_count", None)

    return {
        "input_tokens": int(input_tokens) if isinstance(input_tokens, int) else None,
        "output_tokens": int(output_tokens) if isinstance(output_tokens, int) else None,
        "total_tokens": int(total_tokens) if isinstance(total_tokens, int) else None,
    }


def _calculate_cost_usd(model_name: str, input_tokens: Optional[int], output_tokens: Optional[int]) -> Dict[str, Any]:
    """Calculate request cost from model pricing and token usage."""
    pricing = resolve_model_pricing(model_name, prompt_tokens=input_tokens)
    if pricing is None:
        return {
            "value": None,
            "display": "N/A",
            "input_rate_per_million": None,
            "output_rate_per_million": None,
        }

    safe_input_tokens = input_tokens or 0
    safe_output_tokens = output_tokens or 0

    input_cost = (safe_input_tokens / 1_000_000) * pricing["input_rate_per_million"]
    output_cost = (safe_output_tokens / 1_000_000) * pricing["output_rate_per_million"]
    total_cost = input_cost + output_cost

    return {
        "value": round(total_cost, 8),
        "display": f"${total_cost:.8f}",
        "input_rate_per_million": pricing["input_rate_per_million"],
        "output_rate_per_million": pricing["output_rate_per_million"],
    }


def _build_structured_output(
    *,
    model_name: str,
    response_text: str,
    raw_response: Any,
) -> Dict[str, Any]:
    """Build structured response payload with tokens and cost."""
    usage = _extract_usage_metadata(raw_response)
    input_tokens = usage["input_tokens"]
    output_tokens = usage["output_tokens"]
    total_tokens = usage["total_tokens"]

    cost = _calculate_cost_usd(model_name, input_tokens, output_tokens)

    return {
        "model": model_name,
        "response": response_text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "Total_tokens": total_tokens,
        "Cost": cost["display"],
        "cost_details": {
            "value_usd": cost["value"],
            "input_rate_per_million": cost["input_rate_per_million"],
            "output_rate_per_million": cost["output_rate_per_million"],
        },
    }


class GoogleTextModel:
    """
    Class-based wrapper for Google Gemini text generation.
    
    Class-only interface for Google Gemini text generation.
    
    Example:
        >>> from autourgos_google_modelkit.textmodel import MODEL
        >>> llm = GoogleTextModel(model=MODEL.GEMINI_3_FLASH_PREVIEW, api_key="your-key")
        >>> response = llm.invoke("What is Python?")
        >>> print(response)
    """
    
    def __init__(
        self,
        model: str | MODEL,
        api_key: Optional[str] = None,
        *,
        prompt_template: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        thinking_level: str | THINKING_LEVEL | None = THINKING_LEVEL.MINIMAL,
        structured_output: bool = False,
        Stream: bool = False,
        max_retries: int = 3,
        timeout: Optional[float] = 30.0,
        backoff_factor: float = 0.5,
    ):
        """
        Initialize Google Gemini text model wrapper.
        
        Args:
            model: Model identifier (e.g. MODEL.GEMINI_3_FLASH_PREVIEW or "gemini-2.5-pro")
            api_key: API key (optional if GOOGLE_API_KEY env var is set)
            prompt_template: Optional default prompt template supporting
                Python format placeholders such as ``{task}``.
            temperature: Controls randomness (0.0-2.0)
            top_p: Nucleus sampling threshold (0.0-1.0)
            top_k: Top-k sampling
            max_tokens: Maximum tokens to generate
            thinking_level: Gemini 3 thinking level (``THINKING_LEVEL`` enum or
                string: ``minimal``, ``low``, ``medium``, ``high``).
                Defaults to ``THINKING_LEVEL.MINIMAL``.
            structured_output: When ``True``, ``invoke`` returns a dictionary
                containing response text, token usage, and estimated cost.
                Defaults to ``False``.
            Stream: When True, generation uses streaming internally and returns
                aggregated text. When False, standard non-streaming generation
                is used. Defaults to False.
            max_retries: Number of retry attempts on failure
            timeout: Request timeout in seconds
            backoff_factor: Exponential backoff factor for retries
        """
        self.model = model
        self.api_key = api_key
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.thinking_level = thinking_level
        self.structured_output = structured_output
        self.Stream = Stream
        self.max_retries = max_retries
        self.timeout = timeout
        self.backoff_factor = backoff_factor

    def _resolve_prompt(self, prompt: Optional[str], prompt_variables: Optional[Dict[str, Any]]) -> str:
        """Resolve final prompt from direct input or template + variables."""
        if prompt is not None:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("prompt must be a non-empty string when provided")
            return prompt

        if self.prompt_template is None:
            raise ValueError(
                "prompt is required when prompt_template is not configured"
            )

        if not isinstance(self.prompt_template, str) or not self.prompt_template.strip():
            raise ValueError("prompt_template must be a non-empty string when provided")

        if prompt_variables is not None and not isinstance(prompt_variables, dict):
            raise ValueError("prompt_variables must be a dictionary when provided")

        merged_vars: Dict[str, Any] = dict(prompt_variables or {})

        required_fields = _extract_template_fields(self.prompt_template)
        missing = sorted(
            field for field in required_fields
            if field not in merged_vars or merged_vars[field] is None or str(merged_vars[field]).strip() == ""
        )
        if missing:
            missing_fields = ", ".join(missing)
            raise ValueError(
                f"Missing prompt template variables: {missing_fields}"
            )

        render_vars = {k: _coerce_prompt_variable(v) for k, v in merged_vars.items()}
        try:
            rendered = self.prompt_template.format(**render_vars)
        except KeyError as exc:
            raise ValueError(f"Missing prompt template variable: {exc.args[0]}") from exc

        if not rendered.strip():
            raise ValueError("Resolved prompt must be a non-empty string")
        return rendered

    def _validate_request(self, prompt: Optional[str], prompt_variables: Optional[Dict[str, Any]] = None) -> tuple[str, Dict[str, Any], str, Any, Any]:
        """Validate request inputs and return normalized runtime dependencies."""
        resolved_prompt = self._resolve_prompt(prompt, prompt_variables)

        model_name = _normalize_model_name(self.model)
        if not isinstance(self.Stream, bool):
            raise ValueError("Stream must be a boolean")
        if not isinstance(self.structured_output, bool):
            raise ValueError("structured_output must be a boolean")
        if self.Stream and self.structured_output:
            raise ValueError("structured_output=True is only supported when Stream=False")
        if not isinstance(self.max_retries, int) or self.max_retries < 1:
            raise ValueError("max_retries must be an integer >= 1")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("timeout must be > 0 when provided")
        if self.backoff_factor < 0:
            raise ValueError("backoff_factor must be >= 0")
        if self.temperature is not None and not (0.0 <= self.temperature <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        if self.top_p is not None and not (0.0 <= self.top_p <= 1.0):
            raise ValueError("top_p must be between 0.0 and 1.0")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        resolved_thinking_level = _normalize_thinking_level(self.thinking_level)
        _validate_thinking_level_support(model_name, resolved_thinking_level)

        generation_config = _build_generation_config(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
            thinking_level=resolved_thinking_level,
        )

        api_key = _resolve_api_key(self.api_key)
        if not api_key:
            raise GoogleTextModelImportError(
                "No API key provided and environment variables GOOGLE_API_KEY / GEMINI_API_KEY are not set"
            )

        if not _GOOGLE_GENAI_AVAILABLE or genai_module is None:
            details = f" Details: {_GOOGLE_IMPORT_ERROR}" if _GOOGLE_IMPORT_ERROR else ""
            raise GoogleTextModelImportError(
                "Failed to import or initialize Google Gemini SDK (google-generativeai or google-genai)."
                + details
            )

        genai = genai_module
        client = _configure_genai_client(genai, api_key)
        return model_name, generation_config, resolved_prompt, genai, client

    def _invoke_non_stream(self, *, model_name: str, generation_config: Dict[str, Any], prompt: str, genai: Any, client: Any) -> tuple[str, Any]:
        """Execute non-streaming generation and return model text + raw response."""
        last_exc: Optional[BaseException] = None
        legacy_generation_config = {k: v for k, v in generation_config.items() if k != "thinking_config"}
        for attempt in range(1, self.max_retries + 1):
            try:
                with suppress_stderr():
                    if client is not None:
                        models_attr = getattr(client, "models", None)
                        gen_fn = getattr(models_attr, "generate_content", None) if models_attr else None
                        if callable(gen_fn):
                            kwargs: Dict[str, Any] = {}
                            if generation_config:
                                kwargs["config"] = generation_config
                            resp = gen_fn(model=model_name, contents=prompt, **kwargs)
                            text = _extract_text_from_response(resp)
                            if text:
                                return text, resp

                    GenerativeModel = getattr(genai, "GenerativeModel", None)
                    if callable(GenerativeModel):
                        try:
                            if legacy_generation_config:
                                model_obj = GenerativeModel(model_name, generation_config=legacy_generation_config)
                            else:
                                model_obj = GenerativeModel(model_name)
                            gen_fn = getattr(model_obj, "generate_content", None)
                            if callable(gen_fn):
                                kwargs = {"request_options": {"timeout": self.timeout}} if self.timeout else {}
                                resp = gen_fn(prompt, **kwargs)
                                text = _extract_text_from_response(resp)
                                if text:
                                    return text, resp
                        except Exception:
                            pass

                    for helper_name in ("generate_text", "generate", "model_generate"):
                        helper = getattr(genai, helper_name, None)
                        if callable(helper):
                            try:
                                resp = helper(model=model_name, prompt=prompt)
                                text = _extract_text_from_response(resp)
                                if text:
                                    return text, resp
                            except Exception:
                                pass

                raise GoogleTextModelResponseError("No text could be extracted from the API response")
            except Exception as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    raise GoogleTextModelAPIError(
                        f"Google text model request failed after {self.max_retries} attempts: {exc}"
                    ) from exc
                time.sleep(self.backoff_factor * (2 ** (attempt - 1)))

        raise GoogleTextModelAPIError("Google text model request failed") from last_exc

    def _invoke_stream_mode(self, *, model_name: str, generation_config: Dict[str, Any], prompt: str, genai: Any, client: Any) -> Iterator[str]:
        """Execute streaming generation and yield chunks as they arrive."""
        last_exc: Optional[BaseException] = None
        legacy_generation_config = {k: v for k, v in generation_config.items() if k != "thinking_config"}
        for attempt in range(1, self.max_retries + 1):
            emitted = False
            pending_word = ""

            def emit_word_chunks(fragment: str) -> Iterator[str]:
                """Emit word-level chunks while preserving spacing between words."""
                nonlocal pending_word
                pending_word += fragment

                while True:
                    match = re.search(r"\s+", pending_word)
                    if not match:
                        break

                    token = pending_word[:match.start()]
                    spacing = match.group(0)
                    pending_word = pending_word[match.end():]

                    if token:
                        yield token + spacing
                    else:
                        yield spacing

            try:
                with suppress_stderr():
                    GenerativeModel = getattr(genai, "GenerativeModel", None)
                    if callable(GenerativeModel):
                        try:
                            if legacy_generation_config:
                                model_obj = GenerativeModel(model_name, generation_config=legacy_generation_config)
                            else:
                                model_obj = GenerativeModel(model_name)
                            gen_fn = getattr(model_obj, "generate_content", None)
                            if callable(gen_fn):
                                kwargs: Dict[str, Any] = {"stream": True}
                                if self.timeout:
                                    kwargs["request_options"] = {"timeout": self.timeout}
                                response_stream = gen_fn(prompt, **kwargs)
                                for chunk in response_stream:  # type: ignore
                                    text = _extract_text_from_response(chunk)
                                    if text:
                                        emitted = True
                                        for word_chunk in emit_word_chunks(text):
                                            yield word_chunk
                                if emitted:
                                    if pending_word:
                                        yield pending_word
                                    return
                                final_text = _extract_text_from_response(response_stream)
                                if final_text:
                                    for word_chunk in emit_word_chunks(final_text):
                                        yield word_chunk
                                    if pending_word:
                                        yield pending_word
                                    return
                        except Exception:
                            pass

                    if client is not None:
                        models_attr = getattr(client, "models", None)
                        for stream_name in ("generate_content_stream", "stream_generate_content"):
                            stream_fn = getattr(models_attr, stream_name, None) if models_attr else None
                            if callable(stream_fn):
                                kwargs: Dict[str, Any] = {"timeout": self.timeout} if self.timeout else {}
                                if generation_config:
                                    kwargs["config"] = generation_config
                                try:
                                    response_stream = stream_fn(model=model_name, contents=prompt, **kwargs)
                                except TypeError:
                                    kwargs.pop("timeout", None)
                                    response_stream = stream_fn(model=model_name, contents=prompt, **kwargs)
                                for chunk in response_stream:  # type: ignore
                                    text = _extract_text_from_response(chunk)
                                    if text:
                                        emitted = True
                                        for word_chunk in emit_word_chunks(text):
                                            yield word_chunk
                                if emitted:
                                    if pending_word:
                                        yield pending_word
                                    return
                                final_text = _extract_text_from_response(response_stream)
                                if final_text:
                                    for word_chunk in emit_word_chunks(final_text):
                                        yield word_chunk
                                    if pending_word:
                                        yield pending_word
                                    return

                    if callable(GenerativeModel):
                        try:
                            if legacy_generation_config:
                                model_obj = GenerativeModel(model_name, generation_config=legacy_generation_config)
                            else:
                                model_obj = GenerativeModel(model_name)
                            stream_fn = getattr(model_obj, "generate_content_stream", None)
                            if callable(stream_fn):
                                response_stream = stream_fn(prompt)
                                for chunk in response_stream:  # type: ignore
                                    text = _extract_text_from_response(chunk)
                                    if text:
                                        emitted = True
                                        for word_chunk in emit_word_chunks(text):
                                            yield word_chunk
                                if emitted:
                                    if pending_word:
                                        yield pending_word
                                    return
                                final_text = _extract_text_from_response(response_stream)
                                if final_text:
                                    for word_chunk in emit_word_chunks(final_text):
                                        yield word_chunk
                                    if pending_word:
                                        yield pending_word
                                    return
                        except Exception:
                            pass

                raise GoogleTextModelResponseError("No text could be extracted from streaming response")
            except Exception as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    raise GoogleTextModelAPIError(
                        f"Google text model request failed after {self.max_retries} attempts: {exc}"
                    ) from exc
                time.sleep(self.backoff_factor * (2 ** (attempt - 1)))

        raise GoogleTextModelAPIError("Google text model request failed") from last_exc
    
    def invoke(self, prompt: Optional[str] = None, prompt_variables: Optional[Dict[str, Any]] = None) -> str | Iterator[str] | Dict[str, Any]:
        """
        Generate a response from the Google Gemini model.
        
        Args:
            prompt: Optional direct input prompt text.
            prompt_variables: Optional template variables used when no direct
                ``prompt`` is provided and ``prompt_template`` is configured.
            
        Returns:
            Generated response text when ``Stream=False``.
            Stream iterator of text chunks when ``Stream=True``.
            Structured dictionary when ``structured_output=True``.
            
        Raises:
            ValueError: If prompt is invalid
            GoogleTextModelImportError: If Google client not available
            GoogleTextModelAPIError: If API request fails
            GoogleTextModelResponseError: If response is invalid

        Note:
            If ``Stream=True`` in the constructor, this method yields chunks
            from the model stream.
        """
        model_name, generation_config, normalized_prompt, genai, client = self._validate_request(prompt, prompt_variables)
        if self.Stream:
            return self._invoke_stream_mode(
                model_name=model_name,
                generation_config=generation_config,
                prompt=normalized_prompt,
                genai=genai,
                client=client,
            )
        response_text, raw_response = self._invoke_non_stream(
            model_name=model_name,
            generation_config=generation_config,
            prompt=normalized_prompt,
            genai=genai,
            client=client,
        )
        if self.structured_output:
            return _build_structured_output(
                model_name=model_name,
                response_text=response_text,
                raw_response=raw_response,
            )
        return response_text

__all__ = [
    "GoogleTextModel",
    "GoogleTextModelError",
    "GoogleTextModelAPIError",
    "GoogleTextModelImportError",
    "GoogleTextModelResponseError",
]

