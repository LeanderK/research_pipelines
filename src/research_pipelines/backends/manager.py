"""Global backend management."""

from typing import Optional

from research_pipelines.backends.base import Backend
from research_pipelines.backends.pickle_backend import PickleBackend

# Global backend instance
_backend: Optional[Backend] = None


def get_backend() -> Backend:
    """
    Get the currently active backend.
    """
    global _backend
    if _backend is None:
        # is wandb available? If so, use WandBBackend, otherwise default to PickleBackend
        try:
            from research_pipelines.backends.wandb_backend import WandBBackend
            _backend = WandBBackend()
        except ImportError:
            print("WandB not found, PickleBackend needs manual setup.")
            error_msg = """
            No backend set and WandB not available.
            To use PickleBackend, call set_backend(PickleBackend(directory='your_directory')) before tracing.
            """
            raise ValueError(
                error_msg
            )
    return _backend


def set_backend(backend: Backend) -> None:
    """Set the active backend."""
    global _backend
    _backend = backend


def reset_backend() -> None:
    """Reset backend to None (will use default on next get_backend call)."""
    global _backend
    _backend = None
