"""Google Gemini Vision Model Provider."""

from __future__ import annotations

from typing import Optional, Any, Dict, Iterator, Sequence
import mimetypes
import os
import re
import time
from pathlib import Path

from .models import (
    GOOGLE_VISION_MODEL_NAME,
    GOOGLE_VISION_THINKING_LEVEL,
    GOOGLE_VISION_MEDIA_RESOLUTION,
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


class GoogleVisionModelError(Exception):
    """Base exception for all Google vision model errors."""


class GoogleVisionModelImportError(GoogleVisionModelError):
    """Raised when Google SDK import/configuration fails."""


class GoogleVisionModelAPIError(GoogleVisionModelError):
    """Raised when API requests fail after all retries."""


class GoogleVisionModelResponseError(GoogleVisionModelError):
    """Raised when API responses cannot be parsed as text."""


_normalize_model_name = normalize_model_name
_normalize_thinking_level = normalize_thinking_level
_validate_thinking_level_support = validate_thinking_level_support
_resolve_api_key = resolve_api_key
_build_generation_config = build_generation_config
_extract_template_fields = extract_template_fields
_coerce_prompt_variable = coerce_prompt_variable
_configure_genai_client = configure_genai_client
_extract_text_from_response = extract_text_from_response


def _normalize_media_resolution(
    media_resolution: str | GOOGLE_VISION_MEDIA_RESOLUTION | None,
) -> Optional[str]:
    """Normalize media resolution to Gemini API values."""
    if media_resolution is None:
        return None

    if isinstance(media_resolution, GOOGLE_VISION_MEDIA_RESOLUTION):
        return media_resolution.value

    if isinstance(media_resolution, str):
        normalized = media_resolution.strip().lower()
        if normalized in {
            "media_resolution_low",
            "media_resolution_medium",
            "media_resolution_high",
            "media_resolution_ultra_high",
        }:
            return normalized

    raise ValueError(
        "media_resolution must be one of: media_resolution_low, "
        "media_resolution_medium, media_resolution_high, media_resolution_ultra_high"
    )


def _detect_mime_from_bytes(data: bytes) -> Optional[str]:
    """Detect common image mime types from file signatures."""
    if not data:
        return None

    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"BM"):
        return "image/bmp"
    if data.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"

    return None


def _normalize_image_item(item: Any) -> Dict[str, Any]:
    """Normalize one image input into a dictionary of bytes + mime type."""
    if isinstance(item, tuple) and len(item) == 2:
        data, mime_type = item
        if not isinstance(data, (bytes, bytearray)):
            raise ValueError("tuple image data must be bytes")
        if not isinstance(mime_type, str) or not mime_type.strip():
            raise ValueError("tuple mime_type must be a non-empty string")
        return {"data": bytes(data), "mime_type": mime_type.strip()}

    if isinstance(item, dict):
        data = item.get("data")
        mime_type = item.get("mime_type")
        if not isinstance(data, (bytes, bytearray)):
            raise ValueError("dict image input must include bytes under 'data'")
        if mime_type is not None and not isinstance(mime_type, str):
            raise ValueError("dict mime_type must be a string when provided")
        resolved_mime = mime_type.strip() if isinstance(mime_type, str) and mime_type.strip() else None
        if resolved_mime is None:
            resolved_mime = _detect_mime_from_bytes(bytes(data)) or "image/jpeg"
        return {"data": bytes(data), "mime_type": resolved_mime}

    if isinstance(item, (bytes, bytearray)):
        data = bytes(item)
        mime_type = _detect_mime_from_bytes(data) or "image/jpeg"
        return {"data": data, "mime_type": mime_type}

    if isinstance(item, (str, os.PathLike, Path)):
        path = Path(item)
        if not path.exists() or not path.is_file():
            raise ValueError(f"image path does not exist or is not a file: {path}")
        data = path.read_bytes()
        mime_type = mimetypes.guess_type(path.name)[0] or _detect_mime_from_bytes(data) or "image/jpeg"
        return {"data": data, "mime_type": mime_type}

    raise ValueError(
        "Unsupported image input type. Use path, bytes, dict(data,mime_type), or tuple(bytes,mime_type)."
    )


def _normalize_images(
    image: Optional[Any],
    images: Optional[Sequence[Any]],
) -> list[Dict[str, Any]]:
    """Normalize single and list image inputs into a non-empty image list."""
    items: list[Any] = []
    if image is not None:
        items.append(image)
    if images is not None:
        if not isinstance(images, Sequence) or isinstance(images, (str, bytes, bytearray)):
            raise ValueError("images must be a sequence of image inputs")
        items.extend(list(images))

    if not items:
        raise ValueError("At least one image must be provided via image or images")

    return [_normalize_image_item(item) for item in items]


def _build_multimodal_contents(
    *,
    prompt: str,
    image_items: list[Dict[str, Any]],
    media_resolution: Optional[str],
    genai: Any,
) -> list[Any]:
    """Build multimodal contents payload compatible with multiple SDK shapes."""
    contents: list[Any] = [prompt]

    types_module = getattr(genai, "types", None)
    part_cls = getattr(types_module, "Part", None) if types_module else None
    blob_cls = getattr(types_module, "Blob", None) if types_module else None

    for image_item in image_items:
        data = image_item["data"]
        mime_type = image_item["mime_type"]

        part = None
        if callable(part_cls) and callable(blob_cls):
            try:
                kwargs: Dict[str, Any] = {
                    "inline_data": blob_cls(mime_type=mime_type, data=data)
                }
                if media_resolution is not None:
                    kwargs["media_resolution"] = {"level": media_resolution}
                part = part_cls(**kwargs)
            except Exception:
                part = None

        if part is None and part_cls is not None:
            from_bytes = getattr(part_cls, "from_bytes", None)
            if callable(from_bytes):
                try:
                    part = from_bytes(data=data, mime_type=mime_type)
                except Exception:
                    part = None

        if part is None:
            part = {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": data,
                }
            }

        contents.append(part)

    return contents


def _build_structured_output(
    *,
    model_name: str,
    response_text: str,
    raw_response: Any,
    image_count: int,
) -> Dict[str, Any]:
    """Build structured response payload with text + token usage + cost."""
    return build_structured_output(
        model_name=model_name,
        response_text=response_text,
        raw_response=raw_response,
        resolve_model_pricing=resolve_model_pricing,
        extra_fields={"input_image_count": image_count},
    )


class GoogleVisionModel:
    """Class-based wrapper for Gemini multimodal input with text-only output."""

    def __init__(
        self,
        model: str | GOOGLE_VISION_MODEL_NAME,
        api_key: Optional[str] = None,
        *,
        prompt_template: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        thinking_level: str | GOOGLE_VISION_THINKING_LEVEL | None = None,
        media_resolution: str | GOOGLE_VISION_MEDIA_RESOLUTION | None = GOOGLE_VISION_MEDIA_RESOLUTION.HIGH,
        structured_output: bool = False,
        Stream: bool = False,
        max_retries: int = 3,
        timeout: Optional[float] = 30.0,
        backoff_factor: float = 0.5,
    ):
        self.model = model
        self.api_key = api_key
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.thinking_level = thinking_level
        self.media_resolution = media_resolution
        self.structured_output = structured_output
        self.Stream = Stream
        self.max_retries = max_retries
        self.timeout = timeout
        self.backoff_factor = backoff_factor

    def _resolve_prompt(
        self,
        prompt: Optional[str],
        prompt_variables: Optional[Dict[str, Any]],
    ) -> str:
        """Resolve final prompt from direct prompt or template + variables."""
        if prompt is not None:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("prompt must be a non-empty string when provided")
            return prompt

        if self.prompt_template is None:
            raise ValueError("prompt is required when prompt_template is not configured")

        if not isinstance(self.prompt_template, str) or not self.prompt_template.strip():
            raise ValueError("prompt_template must be a non-empty string when provided")

        if prompt_variables is not None and not isinstance(prompt_variables, dict):
            raise ValueError("prompt_variables must be a dictionary when provided")

        merged_vars: Dict[str, Any] = dict(prompt_variables or {})

        required_fields = _extract_template_fields(self.prompt_template)
        missing = sorted(
            field
            for field in required_fields
            if field not in merged_vars
            or merged_vars[field] is None
            or str(merged_vars[field]).strip() == ""
        )
        if missing:
            missing_fields = ", ".join(missing)
            raise ValueError(f"Missing prompt template variables: {missing_fields}")

        render_vars = {k: _coerce_prompt_variable(v) for k, v in merged_vars.items()}
        try:
            rendered = self.prompt_template.format(**render_vars)
        except KeyError as exc:
            raise ValueError(f"Missing prompt template variable: {exc.args[0]}") from exc

        if not rendered.strip():
            raise ValueError("Resolved prompt must be a non-empty string")

        return rendered

    def _validate_request(
        self,
        prompt: Optional[str],
        prompt_variables: Optional[Dict[str, Any]],
        image: Optional[Any],
        images: Optional[Sequence[Any]],
    ) -> tuple[str, Dict[str, Any], list[Any], str, int, Any, Any]:
        """Validate request and return normalized runtime dependencies."""
        resolved_prompt = self._resolve_prompt(prompt, prompt_variables)
        normalized_images = _normalize_images(image, images)

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

        resolved_media_resolution = _normalize_media_resolution(self.media_resolution)

        generation_config = _build_generation_config(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
            thinking_level=resolved_thinking_level,
            media_resolution=resolved_media_resolution,
        )

        api_key = _resolve_api_key(self.api_key)
        if not api_key:
            raise GoogleVisionModelImportError(
                "No API key provided and environment variables GOOGLE_API_KEY / GEMINI_API_KEY are not set"
            )

        if not _GOOGLE_GENAI_AVAILABLE or genai_module is None:
            details = f" Details: {_GOOGLE_IMPORT_ERROR}" if _GOOGLE_IMPORT_ERROR else ""
            raise GoogleVisionModelImportError(
                "Failed to import or initialize Google Gemini SDK (google-generativeai or google-genai)."
                + details
            )

        genai = genai_module
        client = _configure_genai_client(genai, api_key)

        contents = _build_multimodal_contents(
            prompt=resolved_prompt,
            image_items=normalized_images,
            media_resolution=resolved_media_resolution,
            genai=genai,
        )

        return model_name, generation_config, contents, resolved_prompt, len(normalized_images), genai, client

    def _invoke_non_stream(
        self,
        *,
        model_name: str,
        generation_config: Dict[str, Any],
        contents: list[Any],
        genai: Any,
        client: Any,
    ) -> tuple[str, Any]:
        """Execute non-streaming generation and return text + raw response."""
        last_exc: Optional[BaseException] = None
        legacy_generation_config = {
            k: v for k, v in generation_config.items() if k not in {"thinking_config", "media_resolution"}
        }

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
                            resp = gen_fn(model=model_name, contents=contents, **kwargs)
                            text = _extract_text_from_response(resp)
                            if text:
                                return text, resp

                    GenerativeModel = getattr(genai, "GenerativeModel", None)
                    if callable(GenerativeModel):
                        try:
                            if legacy_generation_config:
                                model_obj = GenerativeModel(
                                    model_name,
                                    generation_config=legacy_generation_config,
                                )
                            else:
                                model_obj = GenerativeModel(model_name)
                            gen_fn = getattr(model_obj, "generate_content", None)
                            if callable(gen_fn):
                                kwargs = {"request_options": {"timeout": self.timeout}} if self.timeout else {}
                                resp = gen_fn(contents, **kwargs)
                                text = _extract_text_from_response(resp)
                                if text:
                                    return text, resp
                        except Exception:
                            pass

                raise GoogleVisionModelResponseError("No text could be extracted from the API response")
            except Exception as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    raise GoogleVisionModelAPIError(
                        f"Google vision model request failed after {self.max_retries} attempts: {exc}"
                    ) from exc
                time.sleep(self.backoff_factor * (2 ** (attempt - 1)))

        raise GoogleVisionModelAPIError("Google vision model request failed") from last_exc

    def _invoke_stream_mode(
        self,
        *,
        model_name: str,
        generation_config: Dict[str, Any],
        contents: list[Any],
        genai: Any,
        client: Any,
    ) -> Iterator[str]:
        """Execute streaming generation and yield text chunks."""
        last_exc: Optional[BaseException] = None
        legacy_generation_config = {
            k: v for k, v in generation_config.items() if k not in {"thinking_config", "media_resolution"}
        }

        for attempt in range(1, self.max_retries + 1):
            emitted = False
            pending_word = ""

            def emit_word_chunks(fragment: str) -> Iterator[str]:
                nonlocal pending_word
                pending_word += fragment

                while True:
                    match = re.search(r"\s+", pending_word)
                    if not match:
                        break

                    token = pending_word[: match.start()]
                    spacing = match.group(0)
                    pending_word = pending_word[match.end() :]

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
                                model_obj = GenerativeModel(
                                    model_name,
                                    generation_config=legacy_generation_config,
                                )
                            else:
                                model_obj = GenerativeModel(model_name)

                            gen_fn = getattr(model_obj, "generate_content", None)
                            if callable(gen_fn):
                                kwargs: Dict[str, Any] = {"stream": True}
                                if self.timeout:
                                    kwargs["request_options"] = {"timeout": self.timeout}
                                response_stream = gen_fn(contents, **kwargs)
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
                        except Exception:
                            pass

                    if client is not None:
                        models_attr = getattr(client, "models", None)
                        for stream_name in ("generate_content_stream", "stream_generate_content"):
                            stream_fn = getattr(models_attr, stream_name, None) if models_attr else None
                            if callable(stream_fn):
                                kwargs = {"timeout": self.timeout} if self.timeout else {}
                                if generation_config:
                                    kwargs["config"] = generation_config
                                try:
                                    response_stream = stream_fn(
                                        model=model_name,
                                        contents=contents,
                                        **kwargs,
                                    )
                                except TypeError:
                                    kwargs.pop("timeout", None)
                                    response_stream = stream_fn(
                                        model=model_name,
                                        contents=contents,
                                        **kwargs,
                                    )

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

                raise GoogleVisionModelResponseError("No text could be extracted from streaming response")
            except Exception as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    raise GoogleVisionModelAPIError(
                        f"Google vision model request failed after {self.max_retries} attempts: {exc}"
                    ) from exc
                time.sleep(self.backoff_factor * (2 ** (attempt - 1)))

        raise GoogleVisionModelAPIError("Google vision model request failed") from last_exc

    def invoke(
        self,
        prompt: Optional[str] = None,
        *,
        image: Optional[Any] = None,
        images: Optional[Sequence[Any]] = None,
        prompt_variables: Optional[Dict[str, Any]] = None,
    ) -> str | Iterator[str] | Dict[str, Any]:
        """Generate a text response from image + text multimodal inputs."""
        model_name, generation_config, contents, _, image_count, genai, client = self._validate_request(
            prompt,
            prompt_variables,
            image,
            images,
        )

        if self.Stream:
            return self._invoke_stream_mode(
                model_name=model_name,
                generation_config=generation_config,
                contents=contents,
                genai=genai,
                client=client,
            )

        response_text, raw_response = self._invoke_non_stream(
            model_name=model_name,
            generation_config=generation_config,
            contents=contents,
            genai=genai,
            client=client,
        )

        if self.structured_output:
            return _build_structured_output(
                model_name=model_name,
                response_text=response_text,
                raw_response=raw_response,
                image_count=image_count,
            )

        return response_text
