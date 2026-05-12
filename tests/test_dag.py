"""Tests for DAG utilities."""

import pytest

from research_pipelines.core import (
    clear_traced_registry,
    register_traced_object,
    generate_object_id,
)
from research_pipelines.dag import (
    build_dag,
    get_dependencies_recursive,
    detect_circular_dependencies,
    export_dag,
)


class TestDAGBuilding:
    """Tests for building DAG from registry."""

    def setup_method(self):
        """Clear registry before each test."""
        clear_traced_registry()

    def test_build_dag_empty_registry(self):
        """Test building DAG from empty registry."""
        dag = build_dag()
        assert dag == {}

    def test_build_dag_single_object(self):
        """Test building DAG with single object."""
        obj_id = generate_object_id()
        register_traced_object(obj_id, "dataset", "callable", {"name": "data"})

        dag = build_dag()

        assert obj_id in dag
        assert dag[obj_id]["type"] == "dataset"
        assert dag[obj_id]["config"] == {"name": "data"}
        assert dag[obj_id]["dependencies"] == []

    def test_build_dag_with_dependencies(self):
        """Test building DAG with dependencies."""
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

        dag = build_dag()

        assert len(dag) == 2
        assert dag[model_id]["dependencies"] == [dataset_id]

    def test_build_dag_chain(self):
        """Test building DAG with chain of dependencies."""
        dataset_id = generate_object_id()
        model_id = generate_object_id()
        eval_id = generate_object_id()

        register_traced_object(dataset_id, "dataset", "callable", {"name": "data"})
        register_traced_object(
            model_id,
            "model",
            "callable",
            {"lr": 0.001},
            dependencies={"dataset": dataset_id},
        )
        register_traced_object(
            eval_id,
            "evaluation",
            "callable",
            {"metric": "accuracy"},
            dependencies={"model": model_id},
        )

        dag = build_dag()

        assert len(dag) == 3
        assert dag[eval_id]["dependencies"] == [model_id]
        assert dag[model_id]["dependencies"] == [dataset_id]


class TestRecursiveDependencies:
    """Tests for recursive dependency resolution."""

    def setup_method(self):
        """Clear registry before each test."""
        clear_traced_registry()

    def test_get_dependencies_recursive_no_deps(self):
        """Test getting recursive dependencies for object with no deps."""
        obj_id = generate_object_id()
        register_traced_object(obj_id, "dataset", "callable", {})

        deps = get_dependencies_recursive(obj_id)

        assert deps == set()

    def test_get_dependencies_recursive_direct(self):
        """Test getting recursive dependencies (direct deps only)."""
        dataset_id = generate_object_id()
        model_id = generate_object_id()

        register_traced_object(dataset_id, "dataset", "callable", {})
        register_traced_object(
            model_id,
            "model",
            "callable",
            {},
            dependencies={"dataset": dataset_id},
        )

        deps = get_dependencies_recursive(model_id)

        assert deps == {dataset_id}

    def test_get_dependencies_recursive_transitive(self):
        """Test getting recursive dependencies (transitive)."""
        dataset_id = generate_object_id()
        model_id = generate_object_id()
        eval_id = generate_object_id()

        register_traced_object(dataset_id, "dataset", "callable", {})
        register_traced_object(
            model_id,
            "model",
            "callable",
            {},
            dependencies={"dataset": dataset_id},
        )
        register_traced_object(
            eval_id,
            "evaluation",
            "callable",
            {},
            dependencies={"model": model_id},
        )

        deps = get_dependencies_recursive(eval_id)

        # Should include both direct dep (model) and transitive dep (dataset)
        assert deps == {model_id, dataset_id}

    def test_get_dependencies_recursive_diamond(self):
        """Test recursive dependencies with diamond graph."""
        # Diamond: eval depends on model_a and model_b, both depend on dataset
        dataset_id = generate_object_id()
        model_a_id = generate_object_id()
        model_b_id = generate_object_id()
        eval_id = generate_object_id()

        register_traced_object(dataset_id, "dataset", "callable", {})
        register_traced_object(model_a_id, "model", "callable", {}, dependencies={"dataset": dataset_id})
        register_traced_object(model_b_id, "model", "callable", {}, dependencies={"dataset": dataset_id})
        register_traced_object(
            eval_id,
            "evaluation",
            "callable",
            {},
            dependencies={"model_a": model_a_id, "model_b": model_b_id},
        )

        deps = get_dependencies_recursive(eval_id)

        # Should have model_a, model_b, and dataset (deduplicated)
        assert deps == {model_a_id, model_b_id, dataset_id}


class TestCircularDependencyDetection:
    """Tests for circular dependency detection."""

    def setup_method(self):
        """Clear registry before each test."""
        clear_traced_registry()

    def test_no_circular_dependencies(self):
        """Test that linear chain has no circular dependencies."""
        dataset_id = generate_object_id()
        model_id = generate_object_id()
        eval_id = generate_object_id()

        register_traced_object(dataset_id, "dataset", "callable", {})
        register_traced_object(
            model_id,
            "model",
            "callable",
            {},
            dependencies={"dataset": dataset_id},
        )
        register_traced_object(
            eval_id,
            "evaluation",
            "callable",
            {},
            dependencies={"model": model_id},
        )

        # Should raise no errors
        has_circular = detect_circular_dependencies()
        assert has_circular is False

    def test_self_loop_circular_dependency(self):
        """Test detection of self-loop circular dependency."""
        obj_id = generate_object_id()
        register_traced_object(obj_id, "object", "callable", {}, dependencies={"self": obj_id})

        has_circular = detect_circular_dependencies()
        assert has_circular is True

    def test_cycle_two_nodes(self):
        """Test detection of cycle between two nodes."""
        obj_a_id = generate_object_id()
        obj_b_id = generate_object_id()

        register_traced_object(obj_a_id, "object", "callable", {}, dependencies={"b": obj_b_id})
        register_traced_object(obj_b_id, "object", "callable", {}, dependencies={"a": obj_a_id})

        has_circular = detect_circular_dependencies()
        assert has_circular is True

    def test_cycle_three_nodes(self):
        """Test detection of cycle among three nodes."""
        obj_a_id = generate_object_id()
        obj_b_id = generate_object_id()
        obj_c_id = generate_object_id()

        register_traced_object(obj_a_id, "object", "callable", {}, dependencies={"b": obj_b_id})
        register_traced_object(obj_b_id, "object", "callable", {}, dependencies={"c": obj_c_id})
        register_traced_object(obj_c_id, "object", "callable", {}, dependencies={"a": obj_a_id})

        has_circular = detect_circular_dependencies()
        assert has_circular is True


class TestDAGExport:
    """Tests for exporting DAG structure."""

    def setup_method(self):
        """Clear registry before each test."""
        clear_traced_registry()

    def test_export_dag_simple(self):
        """Test exporting simple DAG."""
        dataset_id = generate_object_id()
        model_id = generate_object_id()

        register_traced_object(dataset_id, "dataset", "callable", {"path": "/data"})
        register_traced_object(
            model_id,
            "model",
            "callable",
            {"lr": 0.001},
            dependencies={"dataset": dataset_id},
        )

        exported = export_dag()

        assert exported[dataset_id]["config"] == {"path": "/data"}
        assert exported[model_id]["config"] == {"lr": 0.001}
        assert exported[model_id]["dependencies"] == [dataset_id]

    def test_export_dag_includes_metadata(self):
        """Test that exported DAG includes type and dependencies."""
        obj_id = generate_object_id()
        register_traced_object(obj_id, "dataset", "callable", {"name": "data"})

        exported = export_dag()

        assert exported[obj_id]["type"] == "dataset"
        assert exported[obj_id]["id"] == obj_id
        assert "config" in exported[obj_id]
        assert "dependencies" in exported[obj_id]

    def test_export_dag_complex(self):
        """Test exporting complex DAG with multiple dependencies."""
        dataset_id = generate_object_id()
        preprocess_id = generate_object_id()
        model_id = generate_object_id()
        eval_id = generate_object_id()

        register_traced_object(dataset_id, "dataset", "callable", {"split": "train"})
        register_traced_object(
            preprocess_id,
            "dataset",
            "callable",
            {"method": "normalize"},
            dependencies={"dataset": dataset_id},
        )
        register_traced_object(
            model_id,
            "model",
            "callable",
            {"arch": "bert"},
            dependencies={"preprocess": preprocess_id},
        )
        register_traced_object(
            eval_id,
            "evaluation",
            "callable",
            {"metric": "f1"},
            dependencies={"model": model_id},
        )

        exported = export_dag()

        assert len(exported) == 4
        assert exported[preprocess_id]["dependencies"] == [dataset_id]
        assert exported[model_id]["dependencies"] == [preprocess_id]
        assert exported[eval_id]["dependencies"] == [model_id]
