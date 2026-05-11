"""WandB-based backend for storing traced configurations."""

from typing import Any, Dict, List, Optional

from research_pipelines.backends.base import Backend


class WandBBackend(Backend):
    """Backend that stores configurations in wandb.run.config."""

    def __init__(self):
        """
        Initialize WandBBackend.

        Requires an active wandb run (wandb.init() should be called first).
        """
        # Lazy import of wandb
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            raise ImportError(
                "wandb is required for WandBBackend. "
                "Install it with: pip install wandb"
            )

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
        if self.wandb.run is None:
            raise RuntimeError(
                "No active wandb run. Call wandb.init() before using WandBBackend."
            )

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
        if self.wandb.run is None:
            return None

        return self.wandb.run.config.get(object_id)

    def load_all(self) -> Dict[str, Dict[str, Any]]:
        """Load all configurations from wandb.run.config."""
        if self.wandb.run is None:
            return {}

        result = {}
        for key, value in self.wandb.run.config.items():
            # Only include entries that have our structure (config + dependencies)
            if isinstance(value, dict) and "config" in value and "dependencies" in value:
                result[key] = value

        return result

    def clear(self) -> None:
        """Clear all stored configurations from wandb.run.config."""
        if self.wandb.run is None:
            return

        # Get all keys that are our traced objects
        keys_to_remove = []
        for key, value in self.wandb.run.config.items():
            if isinstance(value, dict) and "config" in value and "dependencies" in value:
                keys_to_remove.append(key)

        # Remove them
        for key in keys_to_remove:
            del self.wandb.run.config[key]
