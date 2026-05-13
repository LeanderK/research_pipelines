"""Decorators for tracing dataset, model, and evaluation dependencies."""

import __main__
import functools
import inspect
import pathlib
import sys
from typing import Any, Callable, Dict, Iterable, Optional, Set

import inspect
import pathlib
import sys

from research_pipelines.core import (
    filter_arguments,
    generate_object_id,
    is_basic_type,
    register_traced_object,
    get_traced_registry,
    extract_ignored_args_from_signature,
)
from research_pipelines.backends.manager import get_backend
import research_pipelines.dag as dag_tools

# Global mapping of returned objects to their traced object IDs
# This allows us to detect when a traced object is used as a dependency
_object_to_traced_id: Dict[int, str] = {}

# Global tag stack for context-manager-based tagging
# Tracks active tags in nested contexts, allowing tags to accumulate
_tag_stack: list[str] = []

# Global disabled flag for tracing, can be used to temporarily disable tracing in certain contexts
_tracing_disabled: bool = False


def _mark_object_as_traced(obj: Any, traced_id: str) -> None:
    """Mark an object as traced by storing its id() -> traced_id mapping."""
    _object_to_traced_id[id(obj)] = traced_id


def _get_traced_id_from_object(obj: Any) -> Optional[str]:
    """Get the traced ID for an object, if it was marked as traced."""
    return _object_to_traced_id.get(id(obj))


def tag(name: str):
    """
    Context manager for tagging traced function calls.

    Tags allow disambiguating multiple calls to the same function by associating
    them with a string label. Tags accumulate in nested contexts.

    Example:
        with tag("final-validation"):
            result = traced_evaluate_fn(model, val_dataset)

        # Later, reconstruct by tag
        val_result = query.build(traced_evaluate_fn, tag="final-validation")

    Args:
        name: String tag to associate with traced calls in this context

    Yields:
        None
    """
    import contextlib

    @contextlib.contextmanager
    def tag_context():
        global _tag_stack
        _tag_stack.append(name)
        try:
            yield
        finally:
            _tag_stack.pop()

    return tag_context()

def no_tracing():
    """
    Context manager to temporarily disable tracing.

    This can be useful to avoid tracing certain calls that are not relevant or that
    would cause issues if traced (e.g., calls that create non-serializable objects).

    Example:
        with no_tracing():
            # This call will not be traced
            obj = create_non_serializable_object()

    Args:
        None

    Yields:
        None
    """
    import contextlib

    @contextlib.contextmanager
    def no_tracing_context():
        global _tracing_disabled
        _tracing_disabled = True
        try:
            yield
        finally:
            _tracing_disabled = False

    return no_tracing_context()


def _get_current_tags() -> list[str]:
    """Get a copy of the current tag stack."""
    global _tag_stack
    return list(_tag_stack)

def _is_tracing_enabled() -> bool:
    """Check if tracing is currently enabled (not globally disabled)."""
    global _tracing_disabled
    return not _tracing_disabled


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
            for i, item in enumerate(value):
                extract_ids(f"{name}:l{i}", item)
        elif isinstance(value, tuple):
            for i, item in enumerate(value):
                extract_ids(f"{name}:t{i}", item)
        elif isinstance(value, dict):
            for k, v in value.items():
                assert isinstance(
                    k, str
                ), "Only string keys are supported in argument dicts for tracing."
                extract_ids(f"{name}:{k}", v)

    for name, value in args.items():
        extract_ids(name, value)

    return dependencies


def _stable_qualname(obj):
    module = inspect.getmodule(obj)

    # --- Case 1: normal import ---
    if module and module.__name__ != "__main__":
        return f"{module.__name__}:{obj.__qualname__}"

    # --- Case 2: executed as script ---
    file_path = pathlib.Path(
        getattr(module, "__file__", None) or getattr(obj, "__code__", None).co_filename
    ).resolve()

    # --- Walk upwards to find package root (__init__.py) ---
    parts = file_path.with_suffix("").parts

    # find last directory that is inside a package
    current = file_path.parent
    package_root = None

    while current != current.parent:
        if (current / "__init__.py").exists():
            package_root = current
        current = current.parent

    # --- Reconstruct module path ---
    if package_root:
        rel = file_path.relative_to(package_root)
        module_name = ".".join(package_root.parts[-1:] + rel.with_suffix("").parts)
    else:
        # fallback: sys.path heuristic
        for base in sorted(sys.path, key=len, reverse=True):
            try:
                base_path = pathlib.Path(base).resolve()
                rel = file_path.relative_to(base_path)
                module_name = ".".join(rel.with_suffix("").parts)
                break
            except Exception:
                continue
        else:
            module_name = file_path.stem

    return f"{module_name}:{obj.__qualname__}"


def _sanity_check_dag(my_id: str) -> None:
    """Check for cycles in the DAG of traced objects."""
    dag = dag_tools.build_dag()
    if dag_tools.detect_circular_dependencies(dag):
        raise ValueError(
            f"Circular dependency detected in DAG after tracing object {my_id}. DAG: {dag}"
        )
    # nothing should depend on training as it might
    # lead to accidentially starting the training again when rebuilding a traced object
    leaves = dag_tools.get_leaf_objects(dag)
    # if we find a training object that is not a leaf
    # it means something depends on it, which is not allowed
    for k, v in dag.items():
        if v["type"] == "training" and k not in leaves:
            error_msg = f"""
            Traced training object {k} is not a leaf in the DAG, which means something depends on it. 
            This is not allowed as it might lead to accidentially starting the training again when 
            rebuilding a traced object. DAG: {dag}
            """
            raise ValueError(error_msg)


def traced(
    traced_type: str = "object", ignore_args: Optional[Iterable[str]] = None
) -> Callable:
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

        qualname = _stable_qualname(func_or_class)
        # print(f"Decorating {qualname} as traced {traced_type} with ignored args: {ignored_names}")

        if is_class:
            # Decorator on a class - wrap the __init__ method
            original_init = func_or_class.__init__

            @functools.wraps(original_init)
            def new_init(self, *args, **kwargs):
                __tracebackhide__ = True

                if not _is_tracing_enabled():
                    return original_init(self, *args, **kwargs)
               
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
                object_id = _trace_return_value(
                    traced_type=traced_type,
                    call_args=call_args,
                    return_value=self,
                    callable=qualname,
                )

                if object_id:
                    _sanity_check_dag(object_id)

                return result

            func_or_class.__init__ = new_init
            return func_or_class
        else:
            # Decorator on a function
            @functools.wraps(func_or_class)
            def wrapper(*args, **kwargs):
                __tracebackhide__ = True

                if not _is_tracing_enabled():
                    return func_or_class(*args, **kwargs)

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
                object_id = _trace_return_value(
                    traced_type=traced_type,
                    call_args=call_args,
                    return_value=result,
                    callable=qualname,
                )

                if object_id:
                    _sanity_check_dag(object_id)

                return result

            return wrapper

    return decorator


def _trace_return_value(
    traced_type: str,
    call_args: Dict[str, Any],
    return_value: Any,
    callable: str,
    parent_id: Optional[str] = None,
) -> Optional[str]:
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

    if not object_id:
        return None

    if isinstance(return_value, list):
        for i, item in enumerate(return_value):
            callable_item = f"{callable}[{i}]"
            _trace_return_value(
                traced_type, {}, item, callable=callable_item, parent_id=object_id
            )
        return object_id

    if isinstance(return_value, tuple):
        for i, item in enumerate(return_value):
            callable_item = f"{callable}[{i}]"
            _trace_return_value(
                traced_type, {}, item, callable=callable_item, parent_id=object_id
            )
        return object_id
    return object_id


def _log_trace(
    traced_type: str,
    call_args: Dict[str, Any],
    return_value: Any,
    callable: str,
    parent_id: Optional[str] = None,
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
    backend = get_backend()
    if not backend.is_recording_enabled():
        return ""

    # Generate unique ID
    object_id = generate_object_id()

    # Filter arguments
    config, non_basic = filter_arguments(call_args)

    # Capture current tags and store separately (not in config to keep config clean)
    current_tags = _get_current_tags()

    # Find dependencies (traced objects used as arguments)
    dependencies = _find_traced_dependencies(call_args)

    traced_args = set(v.split(":")[0] for v in dependencies)

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
        parent_id=parent_id,
        tags=current_tags,
    )

    # Mark the returned object as traced
    _mark_object_as_traced(return_value, object_id)

    # Persist to backend
    backend.log_config(
        object_id=object_id,
        config_dict=config,
        dependencies=dependencies,
        object_type=traced_type,
        callable=callable,
        parent_id=parent_id,
        tags=current_tags,
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
