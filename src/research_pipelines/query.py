from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    ParamSpec,
    TypeVar,
    TypeVarTuple,
    overload,
)

import research_pipelines.backends.base as base
import research_pipelines.decorators as decorators
from research_pipelines.backends.manager import get_backend

# format of config
# {'type': 'dataset',
#  'config': {},
#  'dependencies': {},
#  'callable': '__main__:create_classification_splits[2]',
#  'parent_id': '70ee0ea8-efbc-4ec1-9b03-48a4905957f3'}


def _build_args_from_config(
    config: dict[str, dict[str, Any]],
    kwargs_manual: dict[str, Any],
    dependency_objects: dict[str, Any],
    import_translation: dict[str, str],
) -> dict[str, Any]:
    callable_raw_str: str = config["callable"]  # type: ignore
    if "[" in callable_raw_str:
        callable_str, selection = callable_raw_str.split("[")
        selection = int(selection[:-1])
    else:
        callable_str = callable_raw_str
        selection = None
    module_str, func_str = callable_str.split(":")
    if module_str in import_translation:
        module_str = import_translation[module_str]
    kwargs = config["config"]

    list_tuple_dependencies = []

    for arg_name, dep_id in config["dependencies"].items():
        if dep_id not in dependency_objects:
            raise ValueError(
                f"Dependency {dep_id} not found for argument {arg_name} of {callable_str}."
            )
        if ":" in arg_name:
            # can be either :l{i} for lists or :t{i} for tuples
            arg_base, arg_selector = arg_name.split(":")
            type_arg = arg_selector[0]
            index_arg = int(arg_selector[1:])
            if arg_base not in kwargs:
                kwargs[arg_base] = {}
                list_tuple_dependencies.append((arg_base, type_arg))
            existing = kwargs[arg_base]
            if existing["type"] != type_arg:
                raise ValueError(
                    f"Conflicting types for argument {arg_base}: {existing['type']} vs {type_arg}."
                )
            existing[index_arg] = dependency_objects[dep_id]
        else:
            kwargs[arg_name] = dependency_objects[dep_id]

    for arg_name, arg_value in kwargs_manual.items():
        if arg_name in kwargs:
            print(
                f"Warning: Manual argument {arg_name} is overriding traced argument with value {kwargs[arg_name]}."
            )
        kwargs[arg_name] = arg_value

    for arg_base, type_arg in list_tuple_dependencies:
        arg = kwargs[arg_base]
        if type_arg == "l":
            indices = sorted(arg.keys())
            if indices != list(range(len(indices))):
                raise ValueError(
                    f"Missing indices for list argument {arg_base}: found {indices}."
                )
            kwargs[arg_base] = [arg[i] for i in indices]
        elif type_arg == "t":
            indices = sorted(arg.keys())
            if indices != list(range(len(indices))):
                raise ValueError(
                    f"Missing indices for tuple argument {arg_base}: found {indices}."
                )
            kwargs[arg_base] = tuple(arg[i] for i in indices)
        else:
            raise ValueError(
                f"Unknown type selector {type_arg} for argument {arg_base}."
            )

    return kwargs


def _get_function_from_callable_str(
    callable_raw_str: str, import_translation: dict[str, str]
) -> tuple[Callable[..., Any], Optional[int]]:
    if "[" in callable_raw_str:
        callable_str, selection = callable_raw_str.split("[")
        selection = int(selection[:-1])
    else:
        callable_str = callable_raw_str
        selection = None
    module_str, func_str = callable_str.split(":")
    if module_str in import_translation:
        module_str = import_translation[module_str]

    try:
        module = __import__(module_str, fromlist=[func_str])
    except ImportError as e:
        msg = f"""Error importing module {module_str} for callable {callable_str}: {e}.
        Consider adding an import translation for this module if it cannot be imported directly,
        or add the module to sys.path if it is a local module."""
        raise ImportError(msg) from e
    func = getattr(module, func_str)
    return func, selection


def build_from_config(
    config: dict[str, dict[str, Any]],
    kwargs_manual: dict[str, Any],
    dependency_objects: dict[str, Any],
    import_translation: dict[str, str],
) -> Any:
    func, selection = _get_function_from_callable_str(config["callable"], import_translation)  # type: ignore
    kwargs = _build_args_from_config(
        config, kwargs_manual, dependency_objects, import_translation
    )
    assert get_backend().is_recording_enabled() is False, "Backends should not be recording during build."
    value = func(**kwargs)
    if selection is not None:
        value = value[selection]
    return value


T = TypeVar("T")


def _prepare_for_build(
    to_build: Callable[..., T],
    backend: base.Backend,
    manual_kwargs: Optional[dict[str, Any]] = None,
    manual_import_translation: Optional[dict[str, str]] = None,
) -> tuple[str, dict[str, Any], dict[str, dict[str, Any]]]:
    module = to_build.__module__
    manual_import_translation_backwards = {
        v: k for k, v in (manual_import_translation or {}).items()
    }
    if module in manual_import_translation_backwards:
        module = manual_import_translation_backwards[module]
    if isinstance(to_build, Callable):
        callable_str = f"{module}:{to_build.__qualname__}"
        callable_inspection = callable_str
    elif isinstance(to_build, type):
        # we need the __init__ method for the callable string, but we want to return the class type
        callable_str = f"{module}:{to_build.__qualname__}"
        callable_inspection = f"{callable_str}.__init__"
    configs = backend.load_all()
    target_configs = [
        (v, config)
        for v, config in configs.items()
        if config["callable"] == callable_str and config["parent_id"] is None
    ]
    if len(target_configs) == 0:
        msg = f"""No configuration found for {callable_str}.
        Consider adding an import translation for the module {module} if it cannot be imported directly,
        or check if the callable was traced with a different module name (e.g., __main__)."""
        raise ValueError(msg)
    elif len(target_configs) > 1:
        raise ValueError(
            f"Multiple configurations found for {callable_str}, cannot disambiguate."
        )
    config_id, config = target_configs[0]
    return config_id, config, configs


def _build_recursive(
    object_id: str,
    config: dict[str, Any],
    object_cache: dict[str, Any],
    configs: dict[str, dict[str, Any]],
    manual_kwargs: Optional[dict[str, Any]] = None,
    manual_import_translation: Optional[dict[str, str]] = None,
) -> Any:
    if object_id in object_cache:
        return object_cache[object_id]
    if config["parent_id"] is not None:
        parent_config = configs[config["parent_id"]]
        parent_object = _build_recursive(
            config["parent_id"],
            parent_config,
            object_cache,
            configs,
            manual_kwargs,
            manual_import_translation,
        )
        object_cache[object_id] = parent_object
        callable_raw_str: str = config["callable"]  # type: ignore
        if "[" in callable_raw_str:
            _, selection = callable_raw_str.split("[")
            selection = int(selection[:-1])
        else:
            _ = callable_raw_str
            selection = None
        if selection is not None:
            return parent_object[selection]
        return parent_object
    else:
        for dep_id in config["dependencies"].values():
            if dep_id not in object_cache:
                dep_config = configs[dep_id]
                _build_recursive(
                    dep_id,
                    dep_config,
                    object_cache,
                    configs,
                    manual_kwargs,
                    manual_import_translation,
                )
        value = build_from_config(
            config, manual_kwargs or {}, object_cache, manual_import_translation or {}
        )
        object_cache[object_id] = value
        return value


def build(
    to_build: Callable[..., T],
    manual_kwargs: Optional[dict[str, Any]] = None,
    backend: Optional[base.Backend] = None,
    manual_import_translation: Optional[dict[str, str]] = None,
    persistent_cache: Optional[dict[str, Any]] = None,
) -> T:
    """
    Build an object from its traced configuration.

    Args:
        to_build: The callable or class type to build
        backend: The backend to retrieve configurations from
        manual_kwargs: Optional dictionary of arguments to override traced config values
        manual_import_translation: Optional dictionary to translate module names in callables
            (e.g., for handling __main__ cases or renamed modules)
        persistent_cache: Optional dictionary to cache built objects
    Returns:
        The built object
    """
    if backend is None:
        backend = get_backend()
    config_id, config, configs = _prepare_for_build(
        to_build, backend, manual_kwargs, manual_import_translation
    )

    object_cache = {}
    if persistent_cache is not None:
        object_cache = persistent_cache

    return _build_recursive(
        config_id,
        config,
        object_cache,
        configs,
        manual_kwargs,
        manual_import_translation,
    )


def build_arguments_kwargs(
    to_build: Callable,
    manual_kwargs: Optional[dict[str, Any]] = None,
    backend: Optional[base.Backend] = None,
    manual_import_translation: Optional[dict[str, str]] = None,
    persistent_cache: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Untyped version of build_arguments that returns a dictionary of arguments instead of a tuple.
    Is more flexible in case we can't properly type the tuple arguments as a tuple (varargs/kwrags)
    """
    if backend is None:
        backend = get_backend()
    config_id, config, configs = _prepare_for_build(
        to_build, backend, manual_kwargs, manual_import_translation
    )

    object_cache = {}
    if persistent_cache is not None:
        object_cache = persistent_cache

    dependency_objects = {}
    for dep_id in config["dependencies"].values():
        if dep_id not in object_cache:
            dep_config = configs[dep_id]
            _build_recursive(
                dep_id,
                dep_config,
                object_cache,
                configs,
                manual_kwargs,
                manual_import_translation,
            )
        dependency_objects[dep_id] = object_cache[dep_id]

    kwargs = _build_args_from_config(
        config, manual_kwargs or {}, dependency_objects, manual_import_translation or {}
    )
    return kwargs


Ts = TypeVarTuple("Ts")


class ArgsTuple(tuple, Generic[T]):
    __names__: tuple[str, ...]

    def to_kwargs(self) -> dict[str, Any]:
        return dict(zip(self.__names__, self))


def build_arguments(
    to_build: Callable[[*Ts], Any],
    manual_kwargs: Optional[dict[str, Any]] = None,
    backend: Optional[base.Backend] = None,
    manual_import_translation: Optional[dict[str, str]] = None,
    persistent_cache: Optional[dict[str, Any]] = None,
) -> tuple[*Ts]:
    """
    Build arguments for a callable from its traced configuration.

    This is similar to build(), but instead of returning the object itself, it returns the arguments
    that would be used to call the object. This can be useful for inspecting or modifying the arguments
    before instantiating the object.

    Args:
        to_build: The callable to build arguments for
        backend: The backend to retrieve configurations from
        manual_kwargs: Optional dictionary of arguments to override traced config values
        manual_import_translation: Optional dictionary to translate module names in callables
            (e.g., for handling __main__ cases or renamed modules)
        persistent_cache: Optional dictionary to cache built objects
    Returns:
        Tuple of arguments to call the object with, also supports getting the kwrags as a dictionary via the to_kwargs() method of the returned tuple
    """
    if backend is None:
        backend = get_backend()

    config_id, config, configs = _prepare_for_build(
        to_build, backend, manual_kwargs, manual_import_translation
    )

    kwargs = build_arguments_kwargs(
        to_build, manual_kwargs, backend, manual_import_translation, persistent_cache
    )

    func, selection = _get_function_from_callable_str(
        config["callable"], manual_import_translation or {}
    )
    assert selection is None, "Selection is not supported for build_arguments"

    # now we need to inspect the function signature to get the argument names and order
    from inspect import signature
    sig = signature(func)
    arg_names = []
    arg_values = []
    for param in sig.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            # we don't support *args or **kwargs in the traced function signature for now
            raise ValueError(
                f"""build_arguments does not support *args or **kwargs in the traced function signature, but found {param} in {func}.
                Please use build_arguments_kwargs to get a dictionary of arguments instead, or refactor the traced function to not use *args or **kwargs."""
            )
        arg_names.append(param.name)
        if param.name in kwargs:
            arg_values.append(kwargs[param.name])
        elif param.default is not param.empty:
            arg_values.append(param.default)
        else:
            raise ValueError(
                f"Missing value for argument {param.name} of {func} and no default value provided."
            )
    args = ArgsTuple(arg_values)
    args.__names__ = tuple(arg_names)
    return args