"""WandB-based backend for storing traced configurations."""

import importlib
from typing import Any, Dict, List, Optional

from research_pipelines.backends.base import Backend


class WandBBackend(Backend):
    """Backend that stores configurations in wandb.run.config."""

    def __init__(self):
        """
        Initialize WandBBackend.

        Requires an active wandb run (wandb.init() should be called first).
        """
        try:
            self.wandb = importlib.import_module("wandb")
        except ImportError:
            raise ImportError(
                "wandb is required for WandBBackend. "
                "Install it with: pip install wandb"
            )

    def is_recording_enabled(self) -> bool:
        """Return whether there is an active wandb run to record into."""
        return self.wandb.run is not None

    def log_config(
        self,
        object_id: str,
        callable: str,
        config_dict: Dict[str, Any],
        dependencies: Dict[str, str],
        object_type: str = "object",
        parent_id: Optional[str] = None,
    ) -> None:
        """Log configuration for a traced object to wandb.run.config."""
        if not self.is_recording_enabled():
            return

        data = {
            "type": object_type,
            "config": config_dict,
            "dependencies": dependencies,
            "callable": callable,
            "parent_id": parent_id,
        }

        self.wandb.run.config[object_id] = data

    def get_config(self, object_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve configuration for a traced object from wandb.run.config."""
        if not self.is_recording_enabled():
            return None

        return self.wandb.run.config.get(object_id)

    def load_all(self) -> Dict[str, Dict[str, Any]]:
        """Load all configurations from wandb.run.config."""
        if not self.is_recording_enabled():
            return {}

        result = {}
        for key, value in self.wandb.run.config.items():
            # Only include entries that have our structure (config + dependencies)
            if isinstance(value, dict) and "config" in value and "dependencies" in value:
                result[key] = value

        return result

    def clear(self) -> None:
        """Clear all stored configurations from wandb.run.config."""
        if not self.is_recording_enabled():
            return

        # Get all keys that are our traced objects
        keys_to_remove = []
        for key, value in self.wandb.run.config.items():
            if isinstance(value, dict) and "config" in value and "dependencies" in value:
                keys_to_remove.append(key)

        # Remove them
        for key in keys_to_remove:
            del self.wandb.run.config[key]
