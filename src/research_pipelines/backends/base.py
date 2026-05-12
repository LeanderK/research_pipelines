"""Abstract backend interface for storing traced configurations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class Backend(ABC):
    """Abstract base class for configuration storage backends."""

    def is_recording_enabled(self) -> bool:
        """Return whether this backend should record traces right now."""
        return True

    @abstractmethod
    def log_config(
        self,
        object_id: str,
        callable: str,
        config_dict: Dict[str, Any],
        dependencies: Dict[str, str],
        object_type: str = "object",
        parent_id: Optional[str] = None,
    ) -> None:
        """
        Log configuration for a traced object.

        Args:
            object_id: Unique identifier for the traced object
            callable: The callable that created the object
            config_dict: Dictionary of serializable configuration (str, int, float)
            dependencies: Dictionary mapping argument names to object_ids this object depends on
            object_type: Type of traced object (dataset, model, evaluation, ...)
        """
        pass

    @abstractmethod
    def get_config(self, object_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve configuration for a traced object.

        Args:
            object_id: Unique identifier for the traced object

        Returns:
            Dictionary containing config and dependencies, or None if not found
        """
        pass

    @abstractmethod
    def load_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all configurations.

        Returns:
            Dictionary mapping object_id to config+dependencies dicts
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored configurations."""
        pass
