"""Pickle-based backend for storing traced configurations."""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

from research_pipelines.backends.base import Backend


class PickleBackend(Backend):
    """Simple pickle-based backend for testing and development."""

    def __init__(self, directory, recording_enabled: bool = False):
        """
        Initialize PickleBackend.

        Args:
            directory: Directory to store pickle files (default: .traced_configs)
            recording_enabled: Whether this backend should record traces
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._recording_enabled = recording_enabled

    def is_recording_enabled(self) -> bool:
        """Return whether this backend currently records traces."""
        return self._recording_enabled

    def set_recording_enabled(self, enabled: bool) -> None:
        """Enable or disable recording for this backend."""
        self._recording_enabled = enabled

    def _get_pickle_path(self, object_id: str) -> Path:
        """Get the pickle file path for an object_id."""
        # Sanitize object_id to be a valid filename
        safe_id = object_id.replace("/", "_").replace("\\", "_")
        return self.directory / f"{safe_id}.pkl"

    def log_config(
        self,
        object_id: str,
        callable: str,
        config_dict: Dict[str, Any],
        dependencies: Dict[str, str],
        object_type: str = "object",
        parent_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Log configuration for a traced object."""
        if not self.is_recording_enabled():
            return

        if tags is None:
            tags = []

        data = {
            "type": object_type,
            "config": config_dict,
            "dependencies": dependencies,
            "callable": callable,
            "parent_id": parent_id,
            "tags": tags,
        }
        with open(self._get_pickle_path(object_id), "wb") as f:
            pickle.dump(data, f)

    def get_config(self, object_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve configuration for a traced object."""
        path = self._get_pickle_path(object_id)
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def load_all(self) -> Dict[str, Dict[str, Any]]:
        """Load all configurations from pickle files."""
        result = {}
        for pickle_file in self.directory.glob("*.pkl"):
            object_id = pickle_file.stem
            with open(pickle_file, "rb") as f:
                result[object_id] = pickle.load(f)
        return result

    def clear(self) -> None:
        """Clear all stored configurations."""
        for pickle_file in self.directory.glob("*.pkl"):
            pickle_file.unlink()
