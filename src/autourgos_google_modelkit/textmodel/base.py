"""
Google Gemini Text Model Provider

Production-ready wrapper around Google's Generative AI client.

Author: DevxJitin & QueenSonia
Version: 0.1.1
"""

from typing import Optional, Any, Dict, Iterator
import time
import re

from .models import (
    GOOGLE_TEXT_MODEL_NAME,
    GOOGLE_TEXT_THINKING_LEVEL,
    resolve_model_pricing,
)
from ..core import (
    configure_runtime_environment,
    suppress_stderr,
    load_genai_module,
    normalize_model_name,
    normalize_thinking_level,
    validate_thinking_level_support,
    resolve_api_key,
    build_generation_config,
    extract_template_fields,
    coerce_prompt_variable,
    configure_genai_client,
    extract_text_from_response,
    build_structured_output,
)


configure_runtime_environment()
_GOOGLE_GENAI_AVAILABLE, genai_module, _GOOGLE_IMPORT_ERROR = load_genai_module()


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


_normalize_model_name = normalize_model_name
_normalize_thinking_level = normalize_thinking_level
_validate_thinking_level_support = validate_thinking_level_support
_resolve_api_key = resolve_api_key
_build_generation_config = build_generation_config
_extract_template_fields = extract_template_fields
_coerce_prompt_variable = coerce_prompt_variable
_configure_genai_client = configure_genai_client
_extract_text_from_response = extract_text_from_response


def _build_structured_output(
    *,
    model_name: str,
    response_text: str,
    raw_response: Any,
) -> Dict[str, Any]:
    """Build structured response payload with tokens and cost."""
    return build_structured_output(
        model_name=model_name,
        response_text=response_text,
        raw_response=raw_response,
        resolve_model_pricing=resolve_model_pricing,
    )


class GoogleTextModel:
    """
    Class-based wrapper for Google Gemini text generation.
    
    Class-only interface for Google Gemini text generation.
    
    Example:
        >>> from autourgos_google_modelkit.textmodel import GOOGLE_TEXT_MODEL_NAME
        >>> llm = GoogleTextModel(model=GOOGLE_TEXT_MODEL_NAME.GEMINI_3_FLASH_PREVIEW, api_key="your-key")
        >>> response = llm.invoke("What is Python?")
        >>> print(response)
    """
    
    def __init__(
        self,
        model: str | GOOGLE_TEXT_MODEL_NAME,
        api_key: Optional[str] = None,
        *,
        prompt_template: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        thinking_level: str | GOOGLE_TEXT_THINKING_LEVEL | None = None,
        structured_output: bool = False,
        Stream: bool = False,
        max_retries: int = 3,
        timeout: Optional[float] = 30.0,
        backoff_factor: float = 0.5,
    ):
        """
        Initialize Google Gemini text model wrapper.
        
        Args:
            model: Model identifier (e.g. GOOGLE_TEXT_MODEL_NAME.GEMINI_3_FLASH_PREVIEW or "gemini-2.5-pro")
            api_key: API key (optional if GOOGLE_API_KEY env var is set)
            prompt_template: Optional default prompt template supporting
                Python format placeholders such as ``{task}``.
            temperature: Controls randomness (0.0-2.0)
            top_p: Nucleus sampling threshold (0.0-1.0)
            top_k: Top-k sampling
            max_tokens: Maximum tokens to generate
            thinking_level: Gemini 3 thinking level (``GOOGLE_TEXT_THINKING_LEVEL`` enum or
                string: ``minimal``, ``low``, ``medium``, ``high``).
                Defaults to ``None`` (disabled).
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

