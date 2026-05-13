"""WandB-based backend for storing traced configurations."""

import importlib
from typing import Any, Dict, List, Optional
import copy

from research_pipelines.backends.base import Backend


class WandBBackend(Backend):
    """Backend that stores configurations in wandb.run.config."""

    def __init__(self, run=None):
        """
        Initialize WandBBackend.

        Requires an active wandb run (wandb.init() should be called first).
        """
        try:
            self.wandb = importlib.import_module("wandb")
            if run is not None:
                self.run = run
            else:
                self.run = self.wandb.run
        except ImportError:
            raise ImportError(
                "wandb is required for WandBBackend. "
                "Install it with: pip install wandb"
            )

    def is_recording_enabled(self) -> bool:
        """Return whether there is an active wandb run to record into."""
        if self.run is not None:
            return isinstance(self.run, self.wandb.sdk.wandb_run.Run)
        return False

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
        """Log configuration for a traced object to wandb.run.config."""
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

        self.run.config[f"research_pipelines:{object_id}"] = data

    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize by adding missing fields"""
        config = copy.deepcopy(config)
        if "config" not in config:
            config["config"] = {}
        if "dependencies" not in config:
            config["dependencies"] = {}
        if "parent_id" not in config:
            config["parent_id"] = None
        if "tags" not in config:
            config["tags"] = []
        return config

    def get_config(self, object_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve configuration for a traced object from wandb.run.config."""

        return self._normalize_config(self.run.config.get(f"research_pipelines:{object_id}"))

    def load_all(self) -> Dict[str, Dict[str, Any]]:
        """Load all configurations from wandb.run.config."""
        # if not self.is_recording_enabled():
        #     return {}

        result = {}
        for key, value in self.run.config.items():
            if "research_pipelines:" not in key:
                continue
            # Only include entries that have our structure (config + dependencies)
            if isinstance(value, dict) and "callable" in value:
                real_key = key.split("research_pipelines:")[1]
                result[real_key] = self._normalize_config(value)

        return result

    def clear(self) -> None:
        """Clear all stored configurations from wandb.run.config."""
        if not self.is_recording_enabled():
            raise RuntimeError("No active wandb run to clear configurations from.")

        # Get all keys that are our traced objects
        keys_to_remove = []
        for key, value in self.run.config.items():
            if "research_pipelines:" not in key:
                continue
            if isinstance(value, dict) and "config" in value and "dependencies" in value:
                keys_to_remove.append(key)

        # Remove them
        for key in keys_to_remove:
            del self.run.config[key]
