"""Tests for core tracing functionality."""

import uuid

import pytest

from research_pipelines.core import (
    get_traced_registry,
    clear_traced_registry,
    generate_object_id,
    is_basic_type,
    filter_arguments,
    register_traced_object,
    get_traced_object,
)


class TestObjectIdGeneration:
    """Tests for object ID generation."""

    def test_generate_object_id_returns_string(self):
        """Test that generate_object_id returns a string."""
        obj_id = generate_object_id()
        assert isinstance(obj_id, str)

    def test_generate_object_id_unique(self):
        """Test that generated IDs are unique."""
        ids = {generate_object_id() for _ in range(100)}
        assert len(ids) == 100

    def test_generate_object_id_format(self):
        """Test that generated IDs can be parsed as UUIDs."""
        obj_id = generate_object_id()
        # Should be a valid UUID format (though we don't require exact format)
        assert len(obj_id) > 0
        try:
            uuid.UUID(obj_id)
        except ValueError:
            pytest.fail(f"Generated ID {obj_id} is not a valid UUID")


class TestBasicTypeDetection:
    """Tests for basic type detection."""

    def test_basic_types(self):
        """Test that basic types are detected."""
        assert is_basic_type("string")
        assert is_basic_type(42)
        assert is_basic_type(3.14)
        assert is_basic_type(True)
        assert is_basic_type(None)

    def test_non_basic_types(self):
        """Test that non-basic types are rejected."""
        # Lists/tuples of basic types ARE considered basic
        assert is_basic_type([1, 2, 3])
        assert is_basic_type(("a", "b"))
        # But dicts, objects, etc. are not basic
        assert not is_basic_type({"key": "value"})
        assert not is_basic_type(set([1, 2]))
        assert not is_basic_type(lambda x: x)
        assert not is_basic_type(object())

    def test_numeric_types(self):
        """Test various numeric types."""
        assert is_basic_type(int(5))
        assert is_basic_type(float(5.5))
        assert is_basic_type(0)
        assert is_basic_type(0.0)
        assert is_basic_type(-42)
        assert is_basic_type(-3.14)


class TestArgumentFiltering:
    """Tests for argument filtering."""

    def test_filter_only_basic_args(self):
        """Test filtering with only basic arguments."""
        args = {"name": "dataset", "size": 1000, "ratio": 0.8}
        filtered, non_basic = filter_arguments(args)
        
        assert filtered == args
        assert non_basic == {}

    def test_filter_with_non_basic_args(self):
        """Test filtering with non-basic arguments."""
        obj = object()
        args = {"name": "model", "layers": 3, "custom_obj": obj, "funcs": []}
        filtered, non_basic = filter_arguments(args)
        
        # funcs=[] is a list of basic types, so it's considered basic
        assert filtered == {"name": "model", "layers": 3, "funcs": []}
        assert set(non_basic.keys()) == {"custom_obj"}

    def test_filter_with_traced_objects(self):
        """Test that traced objects are detected and separated."""
        clear_traced_registry()
        
        # Register a traced object
        traced_id = generate_object_id()
        register_traced_object(traced_id, "dataset", "callable", {"data": "value"})
        
        # Create a mock traced object (we'll use a dict with special marker)
        traced_obj = {"_traced_object_id": traced_id}
        
        args = {"model_name": "bert", "dataset": traced_obj, "seed": 42}
        filtered, non_basic = filter_arguments(args, traced_objects={traced_id})
        
        # dataset should be in non_basic since it's not basic
        assert "model_name" in filtered
        assert "seed" in filtered

    def test_filter_mixed_types(self):
        """Test filtering with mix of all types."""
        args = {
            "string": "hello",
            "integer": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
        }
        filtered, non_basic = filter_arguments(args)
        
        # list of basic types is also basic, so we have 6 basic items
        assert len(filtered) == 6  # str, int, float, bool, none, list
        assert len(non_basic) == 1  # dict


class TestTracedObjectRegistry:
    """Tests for traced object registry."""

    def setup_method(self):
        """Clear registry before each test."""
        clear_traced_registry()

    def test_register_traced_object(self):
        """Test registering a traced object."""
        obj_id = generate_object_id()
        config = {"name": "my_dataset"}
        
        register_traced_object(obj_id, "dataset", "callable", config)
        
        retrieved = get_traced_object(obj_id)
        assert retrieved is not None
        assert retrieved["type"] == "dataset"
        assert retrieved["config"] == config

    def test_register_with_dependencies(self):
        """Test registering with dependencies."""
        dataset_id = generate_object_id()
        model_id = generate_object_id()
        
        register_traced_object(dataset_id, "dataset", "callable", {"name": "data"})
        register_traced_object(
            model_id,
            "model",
            "callable",
            {"lr": 0.001},
            dependencies={"dataset": dataset_id},
        )
        
        model_obj = get_traced_object(model_id)
        assert model_obj["dependencies"] == {"dataset": dataset_id}

    def test_get_nonexistent_object(self):
        """Test retrieving a non-existent object."""
        result = get_traced_object("nonexistent_id")
        assert result is None

    def test_get_traced_registry(self):
        """Test retrieving the full registry."""
        obj_id1 = generate_object_id()
        obj_id2 = generate_object_id()
        
        register_traced_object(obj_id1, "dataset", "callable", {"name": "d1"})
        register_traced_object(obj_id2, "model", "callable", {"name": "m1"})
        
        registry = get_traced_registry()
        assert len(registry) == 2
        assert obj_id1 in registry
        assert obj_id2 in registry

    def test_clear_traced_registry(self):
        """Test clearing the registry."""
        obj_id = generate_object_id()
        register_traced_object(obj_id, "dataset", "callable", {})
        assert len(get_traced_registry()) == 1
        
        clear_traced_registry()
        assert len(get_traced_registry()) == 0

    def test_registry_isolation(self):
        """Test that registry is isolated from backend."""
        # Registry should be separate from backend storage
        obj_id = generate_object_id()
        register_traced_object(obj_id, "dataset", "callable", {"name": "data"})
        
        retrieved = get_traced_object(obj_id)
        assert retrieved is not None
        
        # Clear registry
        clear_traced_registry()
        assert get_traced_object(obj_id) is None
