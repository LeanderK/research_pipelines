"""Core tracing functionality for research pipelines."""

import sys
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, get_args, get_origin, get_type_hints
import builtins

# Global registry of traced objects (in-memory, separate from backend).
# Store it on `builtins` so different import paths share the same process-wide
# singleton (avoids duplicate module instances creating separate registries).
_TRACED_REGISTRY_KEY = "_research_pipelines_traced_registry"


def _get_registry() -> Dict[str, Dict[str, Any]]:
    reg = getattr(builtins, _TRACED_REGISTRY_KEY, None)
    if reg is None:
        reg = {}
        setattr(builtins, _TRACED_REGISTRY_KEY, reg)
    return reg


class IgnoreArg:
    """
    Marker class for arguments that should be ignored during tracing.
    
    Can be used in two ways:
    1. As a type wrapper: artifact_root: IgnoreArg[str]
    2. As an annotation marker with typing.Annotated: artifact_root: Annotated[str, Ignore()]
    
    Example:
        @dataset()
        def create_splits(artifact_root: IgnoreArg[str], samples: int):
            # artifact_root won't appear in traced config
            pass
    """

    def __class_getitem__(cls, item):
        """Allow IgnoreArg[str] syntax."""
        # Simply return a marker that can be detected
        return cls

    def __repr__(self) -> str:
        return "Ignore()"


# Convenience alias for use with typing.Annotated
Ignore = IgnoreArg


def _is_ignore_marker(annotation: Any) -> bool:
    """
    Check if an annotation is an IgnoreArg marker.
    
    Handles both:
    - Direct IgnoreArg[T] usage
    - Annotated[T, Ignore()] usage (via get_args)
    """
    if annotation is IgnoreArg or annotation is Ignore:
        return True
    
    # Check if it's IgnoreArg[T] - when we use __class_getitem__, it returns IgnoreArg
    if get_origin(annotation) is IgnoreArg:
        return True
    
    # Check for Annotated[T, Ignore()] or Annotated[T, IgnoreArg()]
    if sys.version_info >= (3, 9):
        try:
            from typing import Annotated, get_args
            origin = get_origin(annotation)
            if origin is Annotated:
                args = get_args(annotation)
                # args[0] is the actual type, args[1:] are the metadata
                for metadata in args[1:]:
                    if isinstance(metadata, IgnoreArg) or metadata is IgnoreArg:
                        return True
        except (ImportError, TypeError):
            pass
    
    return False


def extract_ignored_args_from_signature(func_or_class: Any) -> Set[str]:
    """
    Extract argument names that are marked with IgnoreArg in the function signature.
    
    Args:
        func_or_class: Function or class to inspect
        
    Returns:
        Set of parameter names that should be ignored
    """
    try:
        import inspect
        
        # For classes, get the __init__ signature
        if inspect.isclass(func_or_class):
            sig = inspect.signature(func_or_class.__init__)
        else:
            sig = inspect.signature(func_or_class)

        hints = get_type_hints(func_or_class, include_extras=True)
        
        ignored = set()
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            # check the type also
            if _is_ignore_marker(param.annotation):
                ignored.add(param_name)
            annotation = hints.get(param_name, inspect._empty)
            if _is_ignore_marker(annotation):
                ignored.add(param_name)
        
        return ignored
    except (ValueError, TypeError):
        return set()


def generate_object_id() -> str:
    """
    Generate a unique object ID.

    Returns:
        A unique string identifier (UUID v4)
    """
    return str(uuid.uuid4())


def is_basic_type(value: Any) -> bool:
    """
    Check if a value is a basic serializable type.

    Basic types are: str, int, float, bool, None

    Args:
        value: The value to check

    Returns:
        True if value is a basic type, False otherwise
    """
    return isinstance(value, (str, int, float, bool, type(None)))


def filter_arguments(
    args: Dict[str, Any],
    traced_objects: Optional[Set[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Filter arguments into basic types and non-basic types.

    Args:
        args: Dictionary of arguments to filter
        traced_objects: Optional set of traced object IDs to recognize

    Returns:
        Tuple of (basic_args, non_basic_args) dictionaries
    """
    if traced_objects is None:
        traced_objects = set()

    basic_args = {}
    non_basic_args = {}

    for key, value in args.items():
        if is_basic_type(value):
            basic_args[key] = value
        else:
            non_basic_args[key] = value

    return basic_args, non_basic_args


def register_traced_object(
    object_id: str,
    object_type: str,
    callable: str,
    config: Dict[str, Any],
    dependencies: Optional[Dict[str, str]] = None,
    parent_id: Optional[str] = None
) -> None:
    """
    Register a traced object in the in-memory registry.

    Args:
        object_id: Unique identifier for the object
        object_type: Type of object (e.g., "dataset", "model", "evaluation")
        callable: The callable that created the object
        config: Configuration dictionary (basic types only)
        dependencies: Dictionary mapping argument names to object_ids this object depends on
        parent_id: Optional ID of the parent object if this is a nested trace
    """
    if dependencies is None:
        dependencies = {}

    _get_registry()[object_id] = {
        "id": object_id,
        "type": object_type,
        "callable": callable,
        "config": config,
        "dependencies": dependencies,
        "parent_id": parent_id,
    }


def get_traced_object(object_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a traced object from the registry.

    Args:
        object_id: Unique identifier for the object

    Returns:
        The traced object metadata, or None if not found
    """
    return _get_registry().get(object_id)


def get_traced_registry() -> Dict[str, Dict[str, Any]]:
    """
    Get the entire traced object registry.

    Returns:
        Dictionary mapping object_ids to their metadata
    """
    return dict(_get_registry())


def clear_traced_registry() -> None:
    """Clear all entries from the traced object registry."""
    setattr(builtins, _TRACED_REGISTRY_KEY, {})
