"""Abstract backend interface for storing traced configurations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class Backend(ABC):
    """Abstract base class for configuration storage backends."""

    @abstractmethod
    def log_config(
        self,
        object_id: str,
        config_dict: Dict[str, Any],
        dependencies: List[str],
        object_type: str = "object",
    ) -> None:
        """
        Log configuration for a traced object.

        Args:
            object_id: Unique identifier for the traced object
            config_dict: Dictionary of serializable configuration (str, int, float)
            dependencies: List of object_ids that this object depends on
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
