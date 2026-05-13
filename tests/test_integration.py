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


class TestTagging:
    """Tests for the tagging system."""

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

    def test_single_tag_context(self):
        """Test that a tag context properly tags traced objects."""
        from research_pipelines.decorators import tag
        from research_pipelines.core import get_traced_registry

        @model()
        def create_model():
            return {"name": "model"}

        @dataset()
        def load_dataset():
            return {"name": "dataset"}

        @evaluation()
        def evaluate(model, dataset):
            return {"score": 0.95}

        # Create traced objects
        model_obj = create_model()
        dataset_obj = load_dataset()

        # Trace with a tag
        with tag("final-validation"):
            result = evaluate(model=model_obj, dataset=dataset_obj)

        # Verify the tag was stored in the registry
        registry = get_traced_registry()
        eval_obj = [obj for obj in registry.values() if obj["type"] == "evaluation"][0]
        assert "final-validation" in eval_obj["tags"]
        assert eval_obj["tags"] == ["final-validation"]

    def test_multiple_calls_with_different_tags(self):
        """Test that multiple calls to the same function can be disambiguated with different tags."""
        from research_pipelines.decorators import tag
        from research_pipelines.core import get_traced_registry

        @model()
        def create_model():
            return {"name": "model"}

        @dataset()
        def load_validation_data():
            return {"name": "val"}

        @dataset()
        def load_test_data():
            return {"name": "test"}

        @evaluation()
        def evaluate(model, dataset):
            return {"score": 0.95}

        # Create traced objects
        model_obj = create_model()
        val_dataset = load_validation_data()
        test_dataset = load_test_data()

        # Trace with different tags
        with tag("final-validation"):
            val_result = evaluate(model=model_obj, dataset=val_dataset)

        with tag("final-test"):
            test_result = evaluate(model=model_obj, dataset=test_dataset)

        # Verify both calls are tagged correctly
        registry = get_traced_registry()
        eval_objs = [obj for obj in registry.values() if obj["type"] == "evaluation"]
        assert len(eval_objs) == 2
        assert any(obj["tags"] == ["final-validation"] for obj in eval_objs)
        assert any(obj["tags"] == ["final-test"] for obj in eval_objs)

    def test_build_without_tag_on_single_call(self):
        """Test that build() works without a tag if there's only one call."""
        from research_pipelines.decorators import tag
        from research_pipelines.core import get_traced_registry

        @model()
        def create_model():
            return {"name": "model"}

        @dataset()
        def load_dataset():
            return {"name": "dataset"}

        @evaluation()
        def evaluate(model, dataset):
            return {"score": 0.95}

        model_obj = create_model()
        dataset_obj = load_dataset()

        with tag("some-tag"):
            result = evaluate(model=model_obj, dataset=dataset_obj)

        # Verify tag is stored even when there's only one call
        registry = get_traced_registry()
        eval_obj = [obj for obj in registry.values() if obj["type"] == "evaluation"][0]
        assert eval_obj["tags"] == ["some-tag"]

    def test_build_without_tag_on_multiple_calls_raises_error(self):
        """Test that build() raises an error without a tag if there are multiple calls."""
        from research_pipelines.decorators import tag
        import research_pipelines.query as query

        @model()
        def create_model():
            return {"name": "model"}

        @dataset()
        def load_validation_data():
            return {"name": "val"}

        @dataset()
        def load_test_data():
            return {"name": "test"}

        @evaluation()
        def evaluate(model, dataset):
            return {"score": 0.95}

        model_obj = create_model()
        val_dataset = load_validation_data()
        test_dataset = load_test_data()

        with tag("val-tag"):
            val_result = evaluate(model=model_obj, dataset=val_dataset)

        with tag("test-tag"):
            test_result = evaluate(model=model_obj, dataset=test_dataset)

        # Should raise ValueError because there are multiple calls and no tag specified
        with pytest.raises(ValueError, match="Multiple configurations found"):
            query.build(evaluate)

    def test_build_with_nonexistent_tag_raises_error(self):
        """Test that build() raises an error with a nonexistent tag."""
        from research_pipelines.decorators import tag
        import research_pipelines.query as query

        @model()
        def create_model():
            return {"name": "model"}

        @dataset()
        def load_dataset():
            return {"name": "dataset"}

        @evaluation()
        def evaluate(model, dataset):
            return {"score": 0.95}

        model_obj = create_model()
        dataset_obj = load_dataset()

        with tag("existing-tag"):
            result = evaluate(model=model_obj, dataset=dataset_obj)

        # Should raise KeyError for nonexistent tag
        with pytest.raises(KeyError, match="No configuration found"):
            query.build(evaluate, tag="nonexistent-tag")

    def test_nested_tags_accumulate(self):
        """Test that nested tags accumulate in the tag list."""
        from research_pipelines.decorators import tag
        import research_pipelines.query as query
        from research_pipelines.core import get_traced_registry

        @model()
        def create_model():
            return {"name": "model"}

        @dataset()
        def load_dataset():
            return {"name": "dataset"}

        @evaluation()
        def evaluate(model, dataset):
            return {"score": 0.95}

        model_obj = create_model()
        dataset_obj = load_dataset()

        # Nested tags should accumulate
        with tag("outer-tag"):
            with tag("inner-tag"):
                result = evaluate(model=model_obj, dataset=dataset_obj)

        # Check that both tags are in the registry
        registry = get_traced_registry()
        eval_obj = [obj for obj in registry.values() if obj["type"] == "evaluation"][0]
        assert "outer-tag" in eval_obj["tags"]
        assert "inner-tag" in eval_obj["tags"]
        assert eval_obj["tags"] == ["outer-tag", "inner-tag"]

    def test_build_by_tag_single_match(self):
        """Test that build_by_tag() finds tags in the backend registry."""
        from research_pipelines.decorators import tag
        from research_pipelines.backends.manager import get_backend

        @dataset()
        def load_data(path: str):
            return {"path": path}

        @model()
        def create_model():
            return {"name": "model"}

        @evaluation()
        def evaluate(model, dataset):
            return {"score": 0.95}

        # Trace with tags
        with tag("my-dataset"):
            data = load_data(path="/data/train.csv")

        model_obj = create_model()

        with tag("my-evaluation"):
            result = evaluate(model=model_obj, dataset=data)

        # Verify tags are persisted to backend
        backend = get_backend()
        all_configs = backend.load_all()
        eval_configs = [c for c in all_configs.values() if c["type"] == "evaluation"]
        assert len(eval_configs) > 0
        assert any("my-evaluation" in c.get("tags", []) for c in eval_configs)

    def test_build_by_tag_no_match_raises_error(self):
        """Test that build_by_tag() raises KeyError when no match is found."""
        import research_pipelines.query as query

        with pytest.raises(KeyError, match="No configuration found with tag"):
            query.build_by_tag("nonexistent-tag")

    def test_build_by_tag_multiple_matches_raises_error(self):
        """Test that build_by_tag() detects duplicate tags in backend."""
        from research_pipelines.decorators import tag
        from research_pipelines.backends.manager import get_backend

        @model()
        def create_model():
            return {"name": "model"}

        @dataset()
        def load_dataset1():
            return {"name": "dataset1"}

        @dataset()
        def load_dataset2():
            return {"name": "dataset2"}

        @evaluation()
        def evaluate(model, dataset):
            return {"score": 0.95}

        model_obj = create_model()
        dataset1 = load_dataset1()
        dataset2 = load_dataset2()

        # Both calls have the same tag
        with tag("duplicate-tag"):
            result1 = evaluate(model=model_obj, dataset=dataset1)

        with tag("duplicate-tag"):
            result2 = evaluate(model=model_obj, dataset=dataset2)

        # Verify both tags are in the backend
        backend = get_backend()
        all_configs = backend.load_all()
        configs_with_tag = [c for c in all_configs.values() if "duplicate-tag" in c.get("tags", [])]
        assert len(configs_with_tag) == 2  # Both should have the duplicate tag

    def test_build_arguments_by_tag(self):
        """Test that tags are properly stored with arguments."""
        from research_pipelines.decorators import tag
        from research_pipelines.backends.manager import get_backend

        @dataset()
        def load_data(path: str):
            return {"path": path, "size": 1000}

        @evaluation()
        def evaluate(metric: str, dataset_input):
            return {"score": 0.95}

        # Trace with tags
        with tag("training-data"):
            data = load_data(path="/data/train.csv")

        with tag("final-eval"):
            result = evaluate(metric="accuracy", dataset_input=data)

        # Verify tags are in backend with correct configuration
        backend = get_backend()
        all_configs = backend.load_all()
        eval_configs = [c for c in all_configs.values() if c["type"] == "evaluation" and "final-eval" in c.get("tags", [])]
        assert len(eval_configs) == 1
        assert eval_configs[0]["config"]["metric"] == "accuracy"

    def test_tag_isolation_between_contexts(self):
        """Test that tags don't leak between different context managers."""
        from research_pipelines.decorators import tag
        from research_pipelines.core import get_traced_registry

        @model()
        def create_model():
            return {"name": "model"}

        @dataset()
        def load_dataset1():
            return {"name": "dataset1"}

        @dataset()
        def load_dataset2():
            return {"name": "dataset2"}

        @evaluation()
        def evaluate(model, dataset):
            return {"score": 0.95}

        model_obj = create_model()
        dataset1 = load_dataset1()
        dataset2 = load_dataset2()

        # First tagged context
        with tag("tag-1"):
            result1 = evaluate(model=model_obj, dataset=dataset1)

        # Second tagged context - tag-1 should not be active
        with tag("tag-2"):
            result2 = evaluate(model=model_obj, dataset=dataset2)

        # Verify tags are isolated
        registry = get_traced_registry()
        results = [obj for obj in registry.values() if obj["type"] == "evaluation"]
        assert len(results) == 2
        assert any(obj["tags"] == ["tag-1"] for obj in results)
        assert any(obj["tags"] == ["tag-2"] for obj in results)
        # Ensure no cross-contamination
        assert not any("tag-1" in obj["tags"] and "tag-2" in obj["tags"] for obj in results)
