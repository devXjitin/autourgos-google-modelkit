# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [0.1.1] - 2026-03-24

### Changed
- `thinking_level` now defaults to `None` for both `GoogleTextModel` and `GoogleVisionModel` to avoid sending unsupported thinking configuration by default.
- Validation now rejects explicit `thinking_level` values for models that do not support Gemini thinking levels, with a clear local error.

### Fixed
- Resolved API failures where unsupported models could return `400 INVALID_ARGUMENT` due to `thinking_level` configuration.
- Stabilized tests that relied on missing API keys by isolating `GOOGLE_API_KEY` and `GEMINI_API_KEY` environment variables.

### Added
- Regression tests for unsupported thinking-level scenarios in text and vision model wrappers.

## [0.1.0] - 2026-03-24

### Added
- Initial release of `autourgos-google-modelkit`.
- `GoogleTextModel` wrapper for Gemini text generation.
- `GoogleVisionModel` wrapper for multimodal image+text generation.
- Input validation, retries with backoff, streaming support, and structured output metadata.
- Typed enums for model names, thinking levels, and vision media resolution.
