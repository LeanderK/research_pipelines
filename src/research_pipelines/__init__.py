"""
Research Pipelines: Lightweight DAG tracing framework for research projects.

Provides decorators to automatically track dataset, model, and evaluation dependencies,
with pluggable backends for persisting configuration.
"""

from typing import TYPE_CHECKING

__version__ = "0.1.0"

__all__ = [
    "dataset",
    "model",
    "evaluation",
    "traced",
    "get_backend",
    "set_backend",
    "IgnoreArg",
    "Ignore",
    "read",
]

# Type hints for IDE/type checker support
if TYPE_CHECKING:
    from research_pipelines.decorators import dataset, model, evaluation, traced
    from research_pipelines.backends.manager import get_backend, set_backend, read
    from research_pipelines.core import IgnoreArg, Ignore
    from research_pipelines import dag, query, visualize
    from research_pipelines import backends


# Lazy imports - only import when accessed
def __getattr__(name):
    if name == "dataset":
        from research_pipelines.decorators import dataset

        return dataset
    elif name == "model":
        from research_pipelines.decorators import model

        return model
    elif name == "evaluation":
        from research_pipelines.decorators import evaluation

        return evaluation
    elif name == "traced":
        from research_pipelines.decorators import traced

        return traced
    elif name == "get_backend":
        from research_pipelines.backends.manager import get_backend

        return get_backend
    elif name == "set_backend":
        from research_pipelines.backends.manager import set_backend

        return set_backend
    elif name == "IgnoreArg":
        from research_pipelines.core import IgnoreArg

        return IgnoreArg
    elif name == "Ignore":
        from research_pipelines.core import Ignore

        return Ignore
    elif name == "read":
        from research_pipelines.backends.manager import read

        return read
    elif name == "query":
        import research_pipelines.query

        return research_pipelines.query
    elif name == "dag":
        import research_pipelines.dag

        return research_pipelines.dag
    elif name == "visualize":
        import research_pipelines.visualize

        return research_pipelines.visualize
    elif name == "backends":
        import research_pipelines.backends

        return research_pipelines.backends
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
