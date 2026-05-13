"""Global backend management."""

import builtins
from pathlib import Path
import tempfile
import sys
from typing import Any, Optional

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
sys.modules.setdefault(
    "build.lib.research_pipelines.backends.manager", sys.modules[__name__]
)


def get_backend(no_error: bool = False) -> Backend:
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
            if not no_error:
                print("WandB not found, PickleBackend needs manual setup.")
                error_msg = """
                No backend set and WandB not available.
                To use PickleBackend, call set_backend(PickleBackend(directory='your_directory')) before tracing.
                """
                raise ValueError(error_msg)
            else:
                return None  # type: ignore
    return backend


def read(object: Any) -> None:
    """
    Best effort generic method to read a traced run object from the backend.
    """
    if isinstance(object, str) or isinstance(object, Path):
        dir = str(object)
        backend = PickleBackend(directory=dir)
        set_backend(backend)
    # detect whether it's a wandb run but without importing wandb
    elif (
        type(object).__module__.startswith("wandb")
        and type(object).__name__ == "Run"
    ):
        from research_pipelines.backends.wandb_backend import WandBBackend

        backend = WandBBackend(run=object)
        set_backend(backend)
    else:
        raise ValueError(
            f"Unsupported object type for read: {type(object)}. Expected a directory path or a wandb Run object."
        )


def set_backend(backend: Backend) -> None:
    """Set the active backend."""
    _set_backend_state(backend)


def reset_backend() -> None:
    """Reset backend to None (will use default on next get_backend call)."""
    _set_backend_state(None)
