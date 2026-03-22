from autourgos_google_modelkit.core import (
    configure_runtime_environment,
    suppress_stderr,
    load_genai_module,
    normalize_model_name,
)


def test_core_symbols_are_importable():
    assert callable(configure_runtime_environment)
    assert callable(suppress_stderr)
    assert callable(load_genai_module)
    assert callable(normalize_model_name)
