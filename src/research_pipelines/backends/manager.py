"""Global backend management."""

import builtins
import tempfile
import sys
from typing import Optional

from research_pipelines.backends.base import Backend
from research_pipelines.backends.pickle_backend import PickleBackend

_BACKEND_STATE_KEY = "_research_pipelines_backend"


def _get_backend_state() -> Optional[Backend]:
    """Read the process-global backend state."""
    return getattr(builtins, _BACKEND_STATE_KEY, None)


def _set_backend_state(backend: Optional[Backend]) -> None:
    """Write the process-global backend state."""
    setattr(builtins, _BACKEND_STATE_KEY, backend)


# Legacy import alias support: some environments may still import this module via
# build.lib.research_pipelines.backends.manager. Point that name at the same module
# object so the backend singleton stays shared.
sys.modules.setdefault("build.lib.research_pipelines.backends.manager", sys.modules[__name__])


def get_backend() -> Backend:
    """
    Get the currently active backend.
    """
    backend = _get_backend_state()
    if backend is None:
        # is wandb available? If so, use WandBBackend, otherwise default to PickleBackend
        try:
            from research_pipelines.backends.wandb_backend import WandBBackend
            backend = WandBBackend()
            _set_backend_state(backend)
        except ImportError:
            print("WandB not found, PickleBackend needs manual setup.")
            error_msg = """
            No backend set and WandB not available.
            To use PickleBackend, call set_backend(PickleBackend(directory='your_directory')) before tracing.
            """
            raise ValueError(
                error_msg
            )
    return backend


def set_backend(backend: Backend) -> None:
    """Set the active backend."""
    _set_backend_state(backend)


def reset_backend() -> None:
    """Reset backend to None (will use default on next get_backend call)."""
    _set_backend_state(None)
