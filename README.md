# Research Pipelines
![PyPI - Version](https://img.shields.io/pypi/v/research-pipelines)

A lightweight Python framework for tracing the components of research experiments. Automatically track datasets, models, and evaluations-function arguments and function-dependencies, then persist everything to wandb or local storage. This is especially useful for plotting or further evaluation of a trained model, as we can recreate the a function call or just the arguments of a traced function. By design, it is a pickle-free solution that relies on recording primitve arguments. It does not track mutation, so we assume a more functional-stype at the top-level.

Just decorate function during training like this, which automatically records the value of the arguments:
```python
@evaluation()
def evaluate(model_obj, test_set, full_evaluation=False):
    return ...
```

It turns a huge, messy notebook into something simple like:

```python
# (no pictured: select a traced run and load its saved configurations)
# rebuild the arguments such that we can call evaluate ourselves
# no pickle!
model_obj, test_set, _ = query.build_arguments(
    target=evaluate
)
# load saved weights
model_obj.load_state_dict(state_dict)
# call evaluate, but now with everything!
evaluate(model_obj, test_set, full_evaluation=True)
# do some plotting
```

This is done through computing the dependency-graph between the function calls, which can look like this:
![img](/figures/dependencies.png)

## Install
```bash
pip install research-pipelines
```

## Example
Compare the example in `./examples`. We first trace a run in `examples/simple_pipeline.py` and can then rebuild our model (or our dataset) in `examples/load_and_predict.ipynb`.

## Features

- **Automatic DAG Tracing**: Decorators automatically detect when traced objects are used as dependencies
- **Configuration Persistence**: Basic types (str, int, float, bool, None) are automatically captured and stored
- **Flexible Rebuilding**: The query backend allows for calling the traced functions again, even if they depend on other traced functions
- **Pluggable Backends**: Use PickleBackend for testing or WandBBackend for production wandb integration
- **Zero Boilerplate**: Apply decorators and your functions/classes are automatically traced
- **Recursive Dependency Resolution**: Full transitive closure of all dependencies

## Quick Start

```python
from research_pipelines.decorators import dataset, model, evaluation
from research_pipelines.dag import build_dag

# Decorate your functions
@dataset()
def load_data(path: str, split: str):
    # Load your data...
    return {"data": [...], "metadata": {...}}

@model()
def build_model(architecture: str):
    # Basic args (architecture) become config
    return trained_model

@training()
def train_model(train_data, model, lr: float, epochs: int):
    # Non-basic args (train_data, model) become dependencies
    # Basic args (lr, epochs) become config
    # here we train the model
    for epoch in range(epochs):
        ...

@evaluation()
def evaluate(model_obj, metric: str):
    return {"score": 0.95}

# Execute your pipeline
data = load_data(path="/data/train.csv", split="train")
model = build_model(architecture="bert")
train_model(data, model)
results = evaluate(model_obj=model, metric="accuracy")
```

### Rebuild the traced object
The traced objects are not pickled, instead the arguments the functions are called with are saved.

```python
import research_pipelines.query as query

# we can now easily call the functions with the recorded arguments via build(fn_to_call)
dataset = query.build(
    load_data
)

# or just get the arguments such that we can call it ourselves
model_obj, metric = query.build_arguments(
    evaluate
)
model_obj.load_state_dict(state_dict)
evaluate(model_obj, metric)
```

## Installation (Dev)

```bash
# Clone or create the project
cd research_pipelines

# Create conda environment
conda create -n research_pipelines python=3.11

# Activate environment
conda activate research_pipelines

# Install package in editable mode
pip install -e .

# Optional: Install the Torch example extra
pip install ".[example]"

# Optional: Install wandb backend
pip install ".[wandb]"
```


## How It Works

### 1. Decoration

Apply `@dataset()`, `@model()`, `@evaluation()`, or generic `@traced(traced_type="...")` to your functions or class constructors:

```python
@dataset()
def load_data(path: str, split: str):
    return load_from_disk(path)

@model()
class MyModel:
    def __init__(self, layers: int, dataset_input):
        self.layers = layers
        self.data = dataset_input
```

### 2. Automatic Tracing

When you call a decorated function/constructor:
- **Arguments are classified**:
  - **Basic types** (str, int, float, bool, None): stored as configuration
  - **Traced objects** (returned from other @traced functions): become dependencies
  - **Other types**: ignored (can be supplied manually later)
- **Unique ID** is generated for this object
- **Configuration** (basic args + type) is persisted to backend
- **Dependencies** (other traced object IDs) are recorded

### 3. DAG Structure

The framework automatically builds a DAG:
```
dataset_1 (config: path="/data/train.csv", split="train")
  ↓
model_1 (config: architecture="bert", lr=0.001, depends_on: [dataset_1])
  ↓
eval_1 (config: metric="accuracy", depends_on: [model_1])
```

### 4. Backend Persistence

Choose a backend to persist configurations:

**PickleBackend** (default for testing):
```python
from research_pipelines.backends.pickle_backend import PickleBackend
from research_pipelines.backends.manager import set_backend

backend = PickleBackend(directory=".traced_configs")
set_backend(backend)
```

**WandBBackend** (for wandb integration):
```python
import wandb
from research_pipelines.backends.wandb_backend import WandBBackend
from research_pipelines.backends.manager import set_backend

wandb.init(project="my_project")
backend = WandBBackend()
set_backend(backend)

# Configs are automatically logged to wandb.run.config
```

## API Reference

### Decorators

```python
from research_pipelines.decorators import dataset, model, evaluation, traced

@dataset()
def load_data(...):
    """Traces a dataset creation function/class."""
    pass

@model()
def train(...):
    """Traces a model creation function/class."""
    pass

@evaluation()
def eval(...):
    """Traces an evaluation function/class."""
    pass

@traced(traced_type="custom")
def my_function(...):
    """Generic tracer with custom type."""
    pass
```

### DAG Operations

```python
from research_pipelines.dag import (
    build_dag,
    get_dependencies_recursive,
    detect_circular_dependencies,
    export_dag,
    get_root_objects,
    get_leaf_objects,
    get_objects_by_type,
    get_dependents,
)

# Build full DAG
dag = build_dag()

# Get all transitive dependencies
deps = get_dependencies_recursive(object_id)

# Check for cycles
has_cycles = detect_circular_dependencies()

# Export for serialization
dag_export = export_dag()

# Find roots (datasets with no dependencies)
roots = get_root_objects()

# Find leaves (objects nothing depends on)
leaves = get_leaf_objects()

# Filter by type
datasets = get_objects_by_type("dataset")
models = get_objects_by_type("model")

# Find what depends on an object
dependents = get_dependents(object_id)
```

### Backends

```python
from research_pipelines.backends.manager import get_backend, set_backend

# Get active backend
backend = get_backend()

# Set custom backend
set_backend(my_backend)

# Backend interface
class Backend(ABC):
    def log_config(object_id, config_dict, dependencies):
        """Persist config for an object."""
        pass
    
    def get_config(object_id):
        """Retrieve config for an object."""
        pass
    
    def load_all():
        """Load all configs."""
        pass
    
    def clear():
        """Clear all configs."""
        pass
```

## Configuration Format

Configurations are stored as dictionaries with the following structure:

```python
{
    "object_id_1": {
        "callable": "examples.simple_pipeline:load_dataset"
        "config": {
            "path": "/data/train.csv",
            "split": "train",
            "batch_size": 32,
        },
        "dependencies": [],
    },
    "object_id_2": {
        "callable": "examples.simple_pipeline:create_model"
        "config": {
            "architecture": "bert",
            "learning_rate": 0.001,
        },
        "dependencies": ["object_id_1"],
    },
}
```

When using WandBBackend, this is stored directly in `wandb.run.config`.

## Examples

See [examples/simple_pipeline.py](examples/simple_pipeline.py) for a complete end-to-end example.

Run it:
```bash
conda activate research_pipelines
python examples/simple_pipeline.py
```

## Testing

All tests use PickleBackend and are fully isolated:

```bash
conda activate research_pipelines
pytest tests/ -v
```

## Development

The framework is organized into modules:

- `src/research_pipelines/core.py` - Core tracing logic
- `src/research_pipelines/decorators.py` - @dataset, @model, @evaluation decorators
- `src/research_pipelines/backends/` - Backend implementations
  - `base.py` - Abstract Backend interface
  - `pickle_backend.py` - PickleBackend (testing)
  - `wandb_backend.py` - WandBBackend (wandb integration)
  - `manager.py` - Global backend management
- `src/research_pipelines/dag.py` - DAG utilities
- `tests/` - Test suite (61 tests, all passing)

## Key Design Decisions

1. **Lazy Imports**: wandb is only imported when WandBBackend is used
2. **Automatic Dependency Detection**: Uses Python's `id()` to track object identity
3. **In-Memory Registry**: Separate from backend storage, enables DAG operations
4. **UUID v4 IDs**: Unique, collision-free object identifiers
5. **Type-Based Filtering**: Basic types automatically detected and persisted
6. **Pluggable Backends**: Easy to add custom storage implementations

## Limitations & Future Work

- No support for custom object serialization (by design)
- No execution timing/profiling (configuration-only tracking)
- No automatic versioning/hashing of objects

## License

MIT
