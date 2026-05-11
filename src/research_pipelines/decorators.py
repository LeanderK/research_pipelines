"""Decorators for tracing dataset, model, and evaluation dependencies."""

import functools
import inspect
from typing import Any, Callable, Dict, Iterable, Optional, Set

from research_pipelines.core import (
    filter_arguments,
    generate_object_id,
    is_basic_type,
    register_traced_object,
    get_traced_registry,
    extract_ignored_args_from_signature,
)
from research_pipelines.backends.manager import get_backend

# Global mapping of returned objects to their traced object IDs
# This allows us to detect when a traced object is used as a dependency
_object_to_traced_id: Dict[int, str] = {}


def _mark_object_as_traced(obj: Any, traced_id: str) -> None:
    """Mark an object as traced by storing its id() -> traced_id mapping."""
    _object_to_traced_id[id(obj)] = traced_id


def _get_traced_id_from_object(obj: Any) -> Optional[str]:
    """Get the traced ID for an object, if it was marked as traced."""
    return _object_to_traced_id.get(id(obj))


def _find_traced_dependencies(args: Dict[str, Any]) -> Dict[str, str]:
    """
    Find traced object dependencies in arguments.

    Returns a set of traced object IDs that are referenced in the arguments.
    """
    dependencies = {}
    
    def extract_ids(name: str, value: Any) -> None:
        """Recursively extract traced IDs from a value."""
        traced_id = _get_traced_id_from_object(value)
        if traced_id:
            dependencies[name] = traced_id
        elif isinstance(value, list):
            for i, item in value:
                extract_ids(f"{name}:l{i}", item)
        elif isinstance(value, tuple):
            for i, item in value:
                extract_ids(f"{name}:t{i}", item)
        elif isinstance(value, dict):
            for k, v in value.items():
                assert isinstance(k, str), "Only string keys are supported in argument dicts for tracing."
                extract_ids(f"{name}:{k}", v)

    for name, value in args.items():
        extract_ids(name, value)

    return dependencies


def traced(traced_type: str = "object", ignore_args: Optional[Iterable[str]] = None) -> Callable:
    """
    Generic decorator for tracing callable objects (functions or classes).

    Args:
        traced_type: Type of traced object (e.g., "dataset", "model", "evaluation")
        ignore_args: Optional list of argument names to ignore during tracing

    Returns:
        Decorator function
        
    Supports two ways to ignore arguments:
    1. Via ignore_args parameter: @dataset(ignore_args=["artifact_root"])
    2. Via IgnoreArg annotation: def func(artifact_root: IgnoreArg[str], ...):
    
    Both approaches can be combined and will be merged.
    """

    explicit_ignored = set(ignore_args or [])

    def decorator(func_or_class: Callable) -> Callable:
        """The actual decorator."""
        # Extract ignored args from function signature annotations
        annotation_ignored = extract_ignored_args_from_signature(func_or_class)
        # Merge both sources of ignored args
        ignored_names = explicit_ignored | annotation_ignored

        is_class = inspect.isclass(func_or_class)

        qualname = f"{func_or_class.__module__}:{func_or_class.__qualname__}"

        if is_class:
            # Decorator on a class - wrap the __init__ method
            original_init = func_or_class.__init__

            @functools.wraps(original_init)
            def new_init(self, *args, **kwargs):
                # Capture the __init__ arguments
                sig = inspect.signature(original_init)
                bound_args = sig.bind(self, *args, **kwargs)
                bound_args.apply_defaults()

                # Convert to dict, excluding 'self'
                call_args = {
                    k: v
                    for k, v in bound_args.arguments.items()
                    if k != "self" and k not in ignored_names
                }

                # Call original __init__
                result = original_init(self, *args, **kwargs)

                # Now log the trace
                _trace_return_value(
                    traced_type=traced_type,
                    call_args=call_args,
                    return_value=self,
                    callable=qualname
                )

                return result

            func_or_class.__init__ = new_init
            return func_or_class
        else:
            # Decorator on a function
            @functools.wraps(func_or_class)
            def wrapper(*args, **kwargs):
                # Capture the function arguments
                sig = inspect.signature(func_or_class)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                call_args = dict(bound_args.arguments)
                call_args = {
                    key: value
                    for key, value in call_args.items()
                    if key not in ignored_names
                }

                # Call original function
                result = func_or_class(*args, **kwargs)

                # Log the trace
                _trace_return_value(
                    traced_type=traced_type,
                    call_args=call_args,
                    return_value=result,
                    callable=qualname
                )

                return result

            return wrapper

    return decorator


def _trace_return_value(
    traced_type: str,
    call_args: Dict[str, Any],
    return_value: Any,
    callable: str,
    parent_id: Optional[str] = None,
) -> None:
    """Trace a return value, recursively handling tuples and lists."""
    # if is_basic_type(return_value):
    #     return
    
    object_id = _log_trace(
        traced_type=traced_type,
        call_args=call_args,
        return_value=return_value,
        callable=callable,
        parent_id=parent_id,
    )

    if isinstance(return_value, list):
        for i, item in enumerate(return_value):
            callable_item = f"{callable}[{i}]"
            _trace_return_value(traced_type, {}, item, callable=callable_item, parent_id=object_id)
        return

    if isinstance(return_value, tuple):
        for i, item in enumerate(return_value):
            callable_item = f"{callable}[{i}]"
            _trace_return_value(traced_type, {}, item, callable=callable_item, parent_id=object_id)
        return


def _log_trace(
    traced_type: str,
    call_args: Dict[str, Any],
    return_value: Any,
    callable: str,
    parent_id: Optional[str] = None
) -> str:
    """
    Log a trace: register object and persist to backend.

    Args:
        traced_type: Type of traced object
        call_args: Arguments passed to the function/constructor
        return_value: The returned value to mark as traced
        callable: The callable that created the object
        parent_id: Optional ID of the parent object if this is a nested trace
    Returns:
        The generated object_id for the traced object
    """
    # Generate unique ID
    object_id = generate_object_id()

    # Filter arguments
    config, non_basic = filter_arguments(call_args)

    # Find dependencies (traced objects used as arguments)
    dependencies = _find_traced_dependencies(call_args)

    traced_args = set(
        v.split(":")[0] for v in dependencies
    )

    if non_basic and not traced_args:
        raise ValueError(
            f"Traced object of type '{traced_type}' has non-basic arguments that are not traced objects. "
            f"Please either trace the non-basic arguments or remove them from the trace using ignore_args. "
            f"Non-basic args: {non_basic}"
        )

    # Register in in-memory registry
    register_traced_object(
        object_id=object_id,
        object_type=traced_type,
        config=config,
        dependencies=dependencies,
        callable=callable,
        parent_id=parent_id
    )

    # Mark the returned object as traced
    _mark_object_as_traced(return_value, object_id)

    # Persist to backend
    backend = get_backend()
    backend.log_config(
        object_id=object_id,
        config_dict=config,
        dependencies=dependencies,
        object_type=traced_type,
        callable=callable,
        parent_id=parent_id
    )

    return object_id


def dataset(ignore_args: Optional[Iterable[str]] = None) -> Callable:
    """
    Decorator for tracing dataset creation.

    Usage:
        @dataset()
        def load_data(path: str, split: str):
            return data
    """
    return traced(traced_type="dataset", ignore_args=ignore_args)


def model(ignore_args: Optional[Iterable[str]] = None) -> Callable:
    """
    Decorator for tracing model creation.

    Usage:
        @model()
        def create_model(architecture: str, lr: float):
            return model
    """
    return traced(traced_type="model", ignore_args=ignore_args)


def evaluation(ignore_args: Optional[Iterable[str]] = None) -> Callable:
    """
    Decorator for tracing evaluation metrics.

    Usage:
        @evaluation()
        def evaluate(metric: str, threshold: float):
            return results
    """
    return traced(traced_type="evaluation", ignore_args=ignore_args)


def training(ignore_args: Optional[Iterable[str]] = None) -> Callable:
    """
    Decorator for tracing training runs.

    Usage:
        @training()
        def train(model, data, epochs: int):
            return trained_model
    """
    return traced(traced_type="training", ignore_args=ignore_args)
