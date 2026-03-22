"""Runtime environment and stderr control helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator
import os
import sys
import warnings


def configure_runtime_environment() -> None:
    """Set process-level defaults that reduce noisy SDK diagnostics."""
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    warnings.filterwarnings("ignore", category=UserWarning, module=".*grpc.*")


@contextmanager
def suppress_stderr() -> Iterator[None]:
    """Temporarily suppress stderr noise produced by SDK internals."""
    import io

    original_stderr = sys.stderr
    original_stderr_fd = None

    try:
        try:
            original_stderr_fd = os.dup(2)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 2)
            os.close(devnull)
        except Exception:
            pass

        sys.stderr = io.StringIO()
        yield
    finally:
        if original_stderr_fd is not None:
            try:
                os.dup2(original_stderr_fd, 2)
                os.close(original_stderr_fd)
            except Exception:
                pass
        sys.stderr = original_stderr
