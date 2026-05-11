"""Tests for backend implementations."""

import shutil
import tempfile
from pathlib import Path

import pytest

from research_pipelines.backends.pickle_backend import PickleBackend
from research_pipelines.backends.manager import get_backend, set_backend, reset_backend


class TestPickleBackend:
    """Tests for PickleBackend implementation."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for backend storage."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)

    @pytest.fixture
    def backend(self, temp_dir):
        """Create a PickleBackend instance with temporary directory."""
        return PickleBackend(directory=temp_dir)

    def test_init_creates_directory(self):
        """Test that __init__ creates the storage directory."""
        temp = tempfile.mkdtemp()
        try:
            storage_dir = Path(temp) / "new_storage"
            assert not storage_dir.exists()

            backend = PickleBackend(directory=str(storage_dir))

            assert storage_dir.exists()
            assert storage_dir.is_dir()
        finally:
            shutil.rmtree(temp)

    def test_log_and_get_config(self, backend):
        """Test logging and retrieving a configuration."""
        object_id = "test_dataset_1"
        config = {"name": "my_dataset", "size": 1000, "version": 1.5}
        dependencies = []

        backend.log_config(object_id, config, dependencies)

        result = backend.get_config(object_id)
        assert result is not None
        assert result["config"] == config
        assert result["dependencies"] == dependencies

    def test_get_nonexistent_config(self, backend):
        """Test retrieving a config that doesn't exist."""
        result = backend.get_config("nonexistent_id")
        assert result is None

    def test_log_with_dependencies(self, backend):
        """Test logging a config with dependencies."""
        dataset_id = "dataset_1"
        model_id = "model_1"

        backend.log_config("dataset_1", {"name": "data"}, [])
        backend.log_config("model_1", {"lr": 0.001}, [dataset_id])

        model_config = backend.get_config(model_id)
        assert model_config["dependencies"] == [dataset_id]

    def test_load_all_empty(self, backend):
        """Test load_all with no configurations."""
        result = backend.load_all()
        assert result == {}

    def test_load_all_multiple_configs(self, backend):
        """Test load_all with multiple configurations."""
        backend.log_config("obj1", {"x": 1}, [])
        backend.log_config("obj2", {"y": 2}, ["obj1"])
        backend.log_config("obj3", {"z": 3}, ["obj1", "obj2"])

        result = backend.load_all()
        assert len(result) == 3
        assert "obj1" in result
        assert "obj2" in result
        assert "obj3" in result
        assert result["obj2"]["dependencies"] == ["obj1"]
        assert result["obj3"]["dependencies"] == ["obj1", "obj2"]

    def test_clear(self, backend):
        """Test clearing all configurations."""
        backend.log_config("obj1", {"x": 1}, [])
        backend.log_config("obj2", {"y": 2}, [])
        assert len(backend.load_all()) == 2

        backend.clear()

        assert len(backend.load_all()) == 0

    def test_config_with_various_types(self, backend):
        """Test that various basic types are preserved."""
        config = {
            "string": "hello",
            "integer": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "list": [1, "two", 3.0],
            "dict": {"nested": "value"},
        }
        backend.log_config("test_id", config, [])

        result = backend.get_config("test_id")
        assert result["config"] == config

    def test_sanitizes_object_id_in_filename(self, backend):
        """Test that special characters in object_id are sanitized for filenames."""
        # Object IDs with special characters should be sanitized
        object_id = "uuid-like/id\\with:special-chars"
        backend.log_config(object_id, {"data": "test"}, [])

        result = backend.get_config(object_id)
        assert result is not None
        assert result["config"] == {"data": "test"}


class TestBackendManager:
    """Tests for backend manager (global state)."""

    def test_get_backend_default(self):
        """Test that get_backend returns default PickleBackend."""
        reset_backend()
        backend = get_backend()
        assert isinstance(backend, PickleBackend)

    def test_set_backend(self):
        """Test setting a custom backend."""
        reset_backend()
        custom_backend = PickleBackend(directory=tempfile.mkdtemp())

        set_backend(custom_backend)
        retrieved = get_backend()

        assert retrieved is custom_backend

    def test_get_backend_singleton(self):
        """Test that get_backend returns same instance."""
        reset_backend()
        backend1 = get_backend()
        backend2 = get_backend()

        assert backend1 is backend2

    def test_backend_persists_after_set(self):
        """Test that backend persists after calling set_backend."""
        reset_backend()
        backend = get_backend()  # Get default

        set_backend(backend)
        retrieved = get_backend()

        assert retrieved is backend

    def test_reset_backend(self):
        """Test resetting backend."""
        reset_backend()
        backend1 = get_backend()
        
        reset_backend()
        backend2 = get_backend()

        # Should be different instances after reset
        assert backend1 is not backend2
