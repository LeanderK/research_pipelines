"""Global backend management."""

from typing import Optional

from research_pipelines.backends.base import Backend
from research_pipelines.backends.pickle_backend import PickleBackend

# Global backend instance
_backend: Optional[Backend] = None


def get_backend() -> Backend:
    """
    Get the currently active backend.

    Returns default PickleBackend if no backend has been set.
    """
    global _backend
    if _backend is None:
        _backend = PickleBackend()
    return _backend


def set_backend(backend: Backend) -> None:
    """Set the active backend."""
    global _backend
    _backend = backend


def reset_backend() -> None:
    """Reset backend to None (will use default on next get_backend call)."""
    global _backend
    _backend = None
