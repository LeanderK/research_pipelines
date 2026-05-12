"""Tests for decorator functionality."""

import tempfile
import shutil
from typing import Annotated

import pytest

from research_pipelines.core import (
    clear_traced_registry,
    get_traced_object,
    generate_object_id,
    register_traced_object,
    Ignore,
)
from research_pipelines.backends.base import Backend
from research_pipelines.backends.pickle_backend import PickleBackend
from research_pipelines.backends.manager import set_backend, reset_backend
from research_pipelines.decorators import traced, dataset, model, evaluation


class TestGenericTracedDecorator:
    """Tests for the generic @traced decorator."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear registry and setup backend before each test."""
        clear_traced_registry()
        reset_backend()
        temp_dir = tempfile.mkdtemp()
        backend = PickleBackend(directory=temp_dir, recording_enabled=True)
        set_backend(backend)
        yield
        shutil.rmtree(temp_dir)

    def test_traced_on_function(self):
        """Test @traced decorator on a simple function."""
        @traced(traced_type="dataset")
        def create_dataset(name: str, size: int):
            return {"data": [1, 2, 3], "name": name, "size": size}

        result = create_dataset(name="test", size=100)

        # Function should still return the original result
        assert result == {"data": [1, 2, 3], "name": "test", "size": 100}

        # Check registry has the traced object
        registry = __import__("research_pipelines.core", fromlist=["get_traced_registry"]).get_traced_registry()
        assert len(registry) > 0

    def test_traced_captures_config(self):
        """Test that @traced captures basic arguments as config."""
        @traced(traced_type="model")
        def train_model(learning_rate: float, num_layers: int):
            return {"trained": True}

        train_model(learning_rate=0.001, num_layers=5)

        registry = __import__("research_pipelines.core", fromlist=["get_traced_registry"]).get_traced_registry()
        assert len(registry) == 1

        obj_id = list(registry.keys())[0]
        obj = registry[obj_id]
        assert obj["config"]["learning_rate"] == 0.001
        assert obj["config"]["num_layers"] == 5

    def test_traced_ignores_non_basic_args(self):
        """Test that @traced ignores non-basic arguments."""
        class CustomDataLoader:
            pass

        loader = CustomDataLoader()

        @traced(traced_type="dataset")
        def create_with_loader(name: str, loader_obj):
            return {"name": name}

        with pytest.raises(ValueError):
            create_with_loader(name="data", loader_obj=loader)

    def test_traced_detects_dependencies(self):
        """Test that @traced detects other traced objects as dependencies."""
        # Create a traced dataset
        @traced(traced_type="dataset")
        def create_dataset():
            return {"data": [1, 2, 3]}

        dataset_obj = create_dataset()

        # Get the dataset's traced object ID
        registry = __import__("research_pipelines.core", fromlist=["get_traced_registry"]).get_traced_registry()
        dataset_id = list(registry.keys())[0]

        # Now create a model that depends on the dataset
        # We need to pass the actual traced object reference
        @traced(traced_type="model")
        def create_model(dataset_input):
            return {"model": "bert"}

        model_obj = create_model(dataset_input=dataset_obj)

        # Check that the model has the dataset as a dependency
        registry = __import__("research_pipelines.core", fromlist=["get_traced_registry"]).get_traced_registry()
        model_entry = [obj for obj in registry.values() if obj["type"] == "model"][0]

        # Dependencies should contain the dataset_id
        assert dataset_id in model_entry["dependencies"].values()

    def test_traced_stores_in_backend(self):
        """Test that @traced stores config in the backend."""
        backend = __import__("research_pipelines.backends.manager", fromlist=["get_backend"]).get_backend()

        @traced(traced_type="dataset")
        def create_dataset(name: str):
            return {"name": name}

        create_dataset(name="test_data")

        # Check that backend has the config
        all_configs = backend.load_all()
        assert len(all_configs) > 0

    def test_traced_unique_ids(self):
        """Test that each traced object gets a unique ID."""
        @traced(traced_type="dataset")
        def create_dataset(version: int):
            return {"version": version}

        create_dataset(version=1)
        create_dataset(version=2)
        create_dataset(version=3)

        registry = __import__("research_pipelines.core", fromlist=["get_traced_registry"]).get_traced_registry()
        assert len(registry) == 3

        ids = set(registry.keys())
        assert len(ids) == 3  # All IDs should be unique

    def test_traced_skips_when_backend_inactive(self):
        """Tracing should no-op when backend recording is disabled."""

        class InactiveBackend(Backend):
            def is_recording_enabled(self) -> bool:
                return False

            def log_config(self, object_id, callable, config_dict, dependencies, object_type="object", parent_id=None):
                raise AssertionError("log_config should not be called when recording is disabled")

            def get_config(self, object_id):
                return None

            def load_all(self):
                return {}

            def clear(self):
                return None

        set_backend(InactiveBackend())

        @traced(traced_type="dataset")
        def create_dataset(name: str):
            return {"name": name}

        result = create_dataset(name="test")
        assert result == {"name": "test"}

        registry = __import__("research_pipelines.core", fromlist=["get_traced_registry"]).get_traced_registry()
        assert len(registry) == 0


class TestSpecializedDecorators:
    """Tests for specialized decorators."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear registry and setup backend before each test."""
        clear_traced_registry()
        reset_backend()
        temp_dir = tempfile.mkdtemp()
        backend = PickleBackend(directory=temp_dir, recording_enabled=True)
        set_backend(backend)
        yield
        shutil.rmtree(temp_dir)

    def test_dataset_decorator(self):
        """Test @dataset decorator."""
        @dataset()
        def load_dataset(path: str, split: str):
            return {"data": [1, 2, 3]}

        load_dataset(path="/data/train.csv", split="train")

        registry = __import__("research_pipelines.core", fromlist=["get_traced_registry"]).get_traced_registry()
        obj = list(registry.values())[0]
        assert obj["type"] == "dataset"

    def test_model_decorator(self):
        """Test @model decorator."""
        @model()
        def create_model(architecture: str, hidden_size: int):
            return {"model": "created"}

        create_model(architecture="transformer", hidden_size=512)

        registry = __import__("research_pipelines.core", fromlist=["get_traced_registry"]).get_traced_registry()
        obj = list(registry.values())[0]
        assert obj["type"] == "model"
    
    def test_evaluation_decorator(self):
        """Test @evaluation decorator."""
        @evaluation()
        def evaluate(metric_name: str, threshold: float):
            return {"score": 0.95}

        evaluate(metric_name="accuracy", threshold=0.8)

        registry = __import__("research_pipelines.core", fromlist=["get_traced_registry"]).get_traced_registry()
        obj = list(registry.values())[0]
        assert obj["type"] == "evaluation"


class TestDecoratorOnClassConstructor:
    """Tests for using decorators on class constructors."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear registry and setup backend before each test."""
        clear_traced_registry()
        reset_backend()
        temp_dir = tempfile.mkdtemp()
        backend = PickleBackend(directory=temp_dir, recording_enabled=True)
        set_backend(backend)
        yield
        shutil.rmtree(temp_dir)

    def test_traced_on_class_init(self):
        """Test @traced on a class __init__ method."""

        @dataset()
        class MyDataset:
            def __init__(self, path: str, cache: bool = True):
                self.path = path
                self.cache = cache
                self.data = [1, 2, 3]

        ds = MyDataset(path="/data/train.csv", cache=True)

        # Instance should still work normally
        assert ds.path == "/data/train.csv"
        assert ds.cache is True

        # And should be traced
        registry = __import__("research_pipelines.core", fromlist=["get_traced_registry"]).get_traced_registry()
        assert len(registry) > 0

    def test_class_init_captures_config(self):
        """Test that @traced on class captures __init__ args."""

        @model()
        class MyModel:
            def __init__(self, architecture: str, lr: float):
                self.architecture = architecture
                self.lr = lr

        model_instance = MyModel(architecture="bert", lr=0.001)

        registry = __import__("research_pipelines.core", fromlist=["get_traced_registry"]).get_traced_registry()
        obj = list(registry.values())[0]

        assert obj["config"]["architecture"] == "bert"
        assert obj["config"]["lr"] == 0.001


class TestIterableReturnTracing:
    """Tests for tracing tuple/list return values individually."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear registry and setup backend before each test."""
        clear_traced_registry()
        reset_backend()
        temp_dir = tempfile.mkdtemp()
        backend = PickleBackend(directory=temp_dir, recording_enabled=True)
        set_backend(backend)
        yield
        shutil.rmtree(temp_dir)

    def test_tuple_return_traces_each_item(self):
        """Test that tuple return values are traced item by item."""

        @dataset()
        def get_dataset(prefix: str):
            return (
                {"name": f"{prefix}_train"},
                {"name": f"{prefix}_val"},
                {"name": f"{prefix}_test"},
            )

        train, val, test = get_dataset(prefix="split")

        registry = __import__("research_pipelines.core", fromlist=["get_traced_registry"]).get_traced_registry()
        assert len(registry) == 4
        assert all(entry["type"] == "dataset" for entry in registry.values())

        @evaluation()
        def evaluate(sample):
            return {"ok": True}

        evaluate(sample=test)

        registry = __import__("research_pipelines.core", fromlist=["get_traced_registry"]).get_traced_registry()
        evaluation_entry = [entry for entry in registry.values() if entry["type"] == "evaluation"][0]
        assert len(evaluation_entry["dependencies"]) == 1


class TestIgnoredArguments:
    """Tests for ignored tracing arguments."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear registry and setup backend before each test."""
        clear_traced_registry()
        reset_backend()
        temp_dir = tempfile.mkdtemp()
        backend = PickleBackend(directory=temp_dir, recording_enabled=True)
        set_backend(backend)
        yield
        shutil.rmtree(temp_dir)

    def test_ignored_args_not_in_config(self):
        """Ignored arguments should not be persisted in the config."""

        @dataset(ignore_args=["artifact_root", "cache_dir"])
        def create_dataset(artifact_root: str, cache_dir: str, seed: int):
            return {"seed": seed}

        create_dataset(artifact_root="/tmp/a", cache_dir="/tmp/b", seed=7)

        registry = __import__("research_pipelines.core", fromlist=["get_traced_registry"]).get_traced_registry()
        obj = list(registry.values())[0]

        assert "artifact_root" not in obj["config"]
        assert "cache_dir" not in obj["config"]
        assert obj["config"]["seed"] == 7

    def test_ignored_args_do_not_create_dependencies(self):
        """Ignored arguments that reference traced objects should not become dependencies."""

        @dataset()
        def make_source():
            return {"value": 1}

        source = make_source()

        @model(ignore_args=["artifact_root", "source_path"])
        def build_model(artifact_root: str, source_path: str, source):
            return {"ok": True}

        build_model(artifact_root="/tmp/run", source_path="/tmp/source", source=source)

        registry = __import__("research_pipelines.core", fromlist=["get_traced_registry"]).get_traced_registry()
        model_entry = [entry for entry in registry.values() if entry["type"] == "model"][0]

        assert "artifact_root" not in model_entry["config"]
        assert "source_path" not in model_entry["config"]
        assert len(model_entry["dependencies"]) == 1

    def test_list_return_traces_each_item(self):
        """Test that list return values are traced item by item."""

        @dataset()
        def get_dataset(prefix: str):
            return [
                {"name": f"{prefix}_train"},
                {"name": f"{prefix}_val"},
                {"name": f"{prefix}_test"},
            ]

        splits = get_dataset(prefix="split")

        registry = __import__("research_pipelines.core", fromlist=["get_traced_registry"]).get_traced_registry()
        assert len(registry) == 4
        assert all(entry["type"] == "dataset" for entry in registry.values())

        @evaluation()
        def evaluate(sample):
            return {"ok": True}

        evaluate(sample=splits[1])

        registry = __import__("research_pipelines.core", fromlist=["get_traced_registry"]).get_traced_registry()
        evaluation_entry = [entry for entry in registry.values() if entry["type"] == "evaluation"][0]
        assert len(evaluation_entry["dependencies"]) == 1

    def test_ignore_arg_with_annotated(self):
        """Ignored arguments using Annotated[T, Ignore()] should not be traced."""
        @dataset()
        def create_dataset(artifact_root: Annotated[str, Ignore()], seed: int):
            return {"seed": seed}

        create_dataset(artifact_root="/tmp/data", seed=42)

        registry = __import__("research_pipelines.core", fromlist=["get_traced_registry"]).get_traced_registry()
        obj = list(registry.values())[0]

        assert "artifact_root" not in obj["config"]
        assert obj["config"]["seed"] == 42

    def test_ignore_arg_with_annotated_model(self):
        """Annotated ignore markers should work for model decorators too."""

        @model()
        def build_model(artifact_root: Annotated[str, Ignore()], hidden_dim: int):
            return {"ok": True}

        build_model(artifact_root="/tmp/model", hidden_dim=256)

        registry = __import__("research_pipelines.core", fromlist=["get_traced_registry"]).get_traced_registry()
        obj = list(registry.values())[0]

        assert "artifact_root" not in obj["config"]
        assert obj["config"]["hidden_dim"] == 256

    def test_ignore_arg_mixed_with_explicit_ignore_args(self):
        """Test combining Annotated ignore markers with explicit ignore_args parameter."""

        @dataset(ignore_args=["cache_dir"])
        def create_dataset(artifact_root: Annotated[str, Ignore()], cache_dir: str, seed: int):
            return {"seed": seed}

        create_dataset(artifact_root="/tmp/data", cache_dir="/tmp/cache", seed=99)

        registry = __import__("research_pipelines.core", fromlist=["get_traced_registry"]).get_traced_registry()
        obj = list(registry.values())[0]

        # Both should be ignored
        assert "artifact_root" not in obj["config"]
        assert "cache_dir" not in obj["config"]
        assert obj["config"]["seed"] == 99

    def test_ignore_arg_with_class_constructor(self):
        """Test Annotated ignore markers on class constructor parameters."""

        @model()
        class MyModel:
            def __init__(self, artifact_root: Annotated[str, Ignore()], hidden_dim: int):
                self.hidden_dim = hidden_dim

        m = MyModel(artifact_root="/tmp/model", hidden_dim=512)

        registry = __import__("research_pipelines.core", fromlist=["get_traced_registry"]).get_traced_registry()
        obj = list(registry.values())[0]

        assert "artifact_root" not in obj["config"]
        assert obj["config"]["hidden_dim"] == 512
