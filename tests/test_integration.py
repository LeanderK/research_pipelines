"""Integration tests for the full pipeline."""

import tempfile
import shutil

import pytest

from research_pipelines.core import clear_traced_registry
from research_pipelines.backends.pickle_backend import PickleBackend
from research_pipelines.backends.manager import set_backend, reset_backend
from research_pipelines.decorators import dataset, model, evaluation
from research_pipelines.dag import build_dag, get_dependencies_recursive


class TestEndToEndPipeline:
    """End-to-end integration tests."""

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

    def test_simple_pipeline(self):
        """Test a simple dataset -> model pipeline."""

        @dataset()
        def load_data(path: str, split: str):
            return {"data": [1, 2, 3]}

        @model()
        def create_model(architecture: str, dataset_input):
            return {"model": "created"}

        # Execute pipeline
        data = load_data(path="/data/train.csv", split="train")
        model_obj = create_model(architecture="bert", dataset_input=data)

        # Verify DAG
        dag = build_dag()
        assert len(dag) == 2

        # Find dataset and model in DAG
        dataset_obj = [obj for obj in dag.values() if obj["type"] == "dataset"][0]
        model_obj_dag = [obj for obj in dag.values() if obj["type"] == "model"][0]

        # Model should depend on dataset
        assert dataset_obj["id"] in model_obj_dag["dependencies"]

    def test_full_research_pipeline(self):
        """Test full dataset -> model -> evaluation pipeline."""

        @dataset()
        def load_dataset(split: str):
            return {"split": split, "size": 1000}

        @model()
        def train_model(lr: float, dataset_input):
            return {"lr": lr}

        @evaluation()
        def evaluate(metric: str, model_input):
            return {"score": 0.95}

        # Execute pipeline
        dataset_obj = load_dataset(split="train")
        model_obj = train_model(lr=0.001, dataset_input=dataset_obj)
        eval_obj = evaluate(metric="accuracy", model_input=model_obj)

        # Verify DAG structure
        dag = build_dag()
        assert len(dag) == 3

        dataset_id = [obj["id"] for obj in dag.values() if obj["type"] == "dataset"][0]
        model_id = [obj["id"] for obj in dag.values() if obj["type"] == "model"][0]
        eval_id = [obj["id"] for obj in dag.values() if obj["type"] == "evaluation"][0]

        # Check dependencies
        assert dag[model_id]["dependencies"] == [dataset_id]
        assert dag[eval_id]["dependencies"] == [model_id]

    def test_pipeline_with_multiple_datasets(self):
        """Test pipeline with multiple datasets."""

        @dataset()
        def load_train(path: str):
            return {"path": path}

        @dataset()
        def load_test(path: str):
            return {"path": path}

        @model()
        def train_model(train_data, test_data, lr: float):
            return {"lr": lr}

        # Execute
        train_obj = load_train(path="/data/train")
        test_obj = load_test(path="/data/test")
        model_obj = train_model(train_data=train_obj, test_data=test_obj, lr=0.01)

        # Verify DAG
        dag = build_dag()
        assert len(dag) == 3

        model_entry = [obj for obj in dag.values() if obj["type"] == "model"][0]
        # Both datasets should be in dependencies
        assert len(model_entry["dependencies"]) == 2

    def test_backend_persistence(self):
        """Test that configs are persisted to backend."""
        backend = __import__("research_pipelines.backends.manager", fromlist=["get_backend"]).get_backend()

        @dataset()
        def load_data():
            return {"data": [1, 2, 3]}

        load_data()

        # Check backend has the config
        all_configs = backend.load_all()
        assert len(all_configs) > 0

        # Verify config structure
        config_entry = list(all_configs.values())[0]
        assert "config" in config_entry
        assert "dependencies" in config_entry

    def test_recursive_dependencies(self):
        """Test getting recursive dependencies."""

        @dataset()
        def load_data():
            return {}

        @dataset()
        def preprocess(data_input):
            return {}

        @model()
        def train(preprocessed_input):
            return {}

        # Execute
        data = load_data()
        preprocessed = preprocess(data_input=data)
        model_obj = train(preprocessed_input=preprocessed)

        # Get DAG
        dag = build_dag()
        model_id = [obj["id"] for obj in dag.values() if obj["type"] == "model"][0]

        # Get recursive dependencies
        deps = get_dependencies_recursive(model_id)

        # Should include both preprocessing and original dataset
        assert len(deps) == 2
