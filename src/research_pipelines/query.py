from typing import Any

import research_pipelines.backends.base as base
import research_pipelines.decorators as decorators

# format of config
# {'type': 'dataset',
#  'config': {},
#  'dependencies': {},
#  'callable': '__main__:create_classification_splits[2]',
#  'parent_id': '70ee0ea8-efbc-4ec1-9b03-48a4905957f3'}

def load(
        config: dict[str, dict[str, Any]],
        kwargs_manual: dict[str, Any],
        dependency_objects: dict[str, Any],
    ) -> Any:
    callable_raw_str: str = config['callable'] # type: ignore
    if '[' in callable_raw_str:
        callable_str, selection = callable_raw_str.split('[')
        selection = int(selection[:-1])
    else:
        callable_str = callable_raw_str
        selection = None
    module_str, func_str = callable_str.split(':')
    
    kwargs = config['config']

    list_tuple_dependencies = []

    for arg_name, dep_id in config['dependencies'].items():
        if dep_id not in dependency_objects:
            raise ValueError(f"Dependency {dep_id} not found for argument {arg_name} of {callable_str}.")
        if ":" in arg_name:
            # can be either :l{i} for lists or :t{i} for tuples
            arg_base, arg_selector = arg_name.split(':')
            type_arg = arg_selector[0]
            index_arg = int(arg_selector[1:])
            if arg_base not in kwargs:
                kwargs[arg_base] = {}
                list_tuple_dependencies.append((arg_base, type_arg))
            existing = kwargs[arg_base]
            if existing['type'] != type_arg:
                raise ValueError(f"Conflicting types for argument {arg_base}: {existing['type']} vs {type_arg}.")
            existing[index_arg] = dependency_objects[dep_id]
        else:
            kwargs[arg_name] = dependency_objects[dep_id]

    for arg_name, arg_value in kwargs_manual.items():
        if arg_name in kwargs:
            print(f"Warning: Manual argument {arg_name} is overriding traced argument with value {kwargs[arg_name]}.")
        kwargs[arg_name] = arg_value

    for arg_base, type_arg in list_tuple_dependencies:
        arg = kwargs[arg_base]
        if type_arg == 'l':
            indices = sorted(arg.keys())
            if indices != list(range(len(indices))):
                raise ValueError(f"Missing indices for list argument {arg_base}: found {indices}.")
            kwargs[arg_base] = [arg[i] for i in indices]
        elif type_arg == 't':
            indices = sorted(arg.keys())
            if indices != list(range(len(indices))):
                raise ValueError(f"Missing indices for tuple argument {arg_base}: found {indices}.")
            kwargs[arg_base] = tuple(arg[i] for i in indices)
        else:
            raise ValueError(f"Unknown type selector {type_arg} for argument {arg_base}.")
        
    module = __import__(module_str, fromlist=[func_str])
    func = getattr(module, func_str)
    value = func(**kwargs)
    if selection is not None:
        value = value[selection]
    return value