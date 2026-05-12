# Research Pipelines

A lightweight Python framework for tracing the DAG (directed acyclic graph) of research experiments. Automatically track datasets, models, and evaluations with their configurations and dependencies, then persist everything to wandb or local storage.

## Features

- **Automatic DAG Tracing**: Decorators automatically detect when traced objects are used as dependencies
- **Configuration Persistence**: Basic types (str, int, float, bool, None) are automatically captured and stored
- **Pluggable Backends**: Use PickleBackend for testing or WandBBackend for production wandb integration
- **Zero Boilerplate**: Apply decorators and your functions/classes are automatically traced
- **Circular Dependency Detection**: Validates DAG structure to catch mistakes early
- **Recursive Dependency Resolution**: Full transitive closure of all dependencies

## Installation

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
def train_model(train_data, architecture: str, lr: float):
    # Non-basic args (train_data) become dependencies
    # Basic args (architecture, lr) become config
    return trained_model

@evaluation()
def evaluate(model_obj, metric: str):
    return {"score": 0.95}

# Execute your pipeline
data = load_data(path="/data/train.csv", split="train")
model = train_model(train_data=data, architecture="bert", lr=0.001)
results = evaluate(model_obj=model, metric="accuracy")

# Print the DAG
dag = build_dag()
for obj_id, obj in dag.items():
    print(f"{obj['type']}: {obj['config']}, depends on: {obj['dependencies']}")
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

### Core Functions

```python
from research_pipelines.core import (
    generate_object_id,
    is_basic_type,
    filter_arguments,
    register_traced_object,
    get_traced_object,
    get_traced_registry,
    clear_traced_registry,
)

# Generate unique ID (UUID v4)
obj_id = generate_object_id()

# Check if value is serializable
if is_basic_type(value):
    pass  # Can be stored in config

# Separate basic from non-basic args
basic_args, non_basic_args = filter_arguments(args_dict)

# Register object in in-memory registry
register_traced_object(obj_id, "dataset", {"key": "value"}, dependencies=[])

# Retrieve object
obj = get_traced_object(obj_id)

# Get full registry
registry = get_traced_registry()

# Clear registry (testing)
clear_traced_registry()
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
        "config": {
            "path": "/data/train.csv",
            "split": "train",
            "batch_size": 32,
        },
        "dependencies": [],
    },
    "object_id_2": {
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
- Circular dependencies are detected but not prevented
- No automatic versioning/hashing of objects

## License

MIT
