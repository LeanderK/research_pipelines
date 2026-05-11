"""DAG utilities for analyzing and exporting traced dependencies."""

from typing import Any, Dict, Set

from research_pipelines.core import get_traced_registry


def build_dag() -> Dict[str, Dict[str, Any]]:
    """
    Build DAG from the traced object registry.

    Returns:
        Dictionary mapping object_id to DAG node info
    """
    registry = get_traced_registry()
    dag = {}

    for obj_id, obj_info in registry.items():
        dag[obj_id] = {
            "id": obj_id,
            "type": obj_info["type"],
            "config": obj_info["config"],
            "dependencies": obj_info.get("dependencies", []),
        }

    return dag


def get_dependencies_recursive(object_id: str) -> Set[str]:
    """
    Get all recursive dependencies for an object.

    Performs depth-first traversal to find all transitive dependencies.

    Args:
        object_id: The object to get dependencies for

    Returns:
        Set of all object_ids that this object depends on (transitively)
    """
    dag = build_dag()

    if object_id not in dag:
        return set()

    visited = set()
    to_visit = list(dag[object_id]["dependencies"])

    while to_visit:
        current = to_visit.pop()
        if current in visited:
            continue

        visited.add(current)

        # Add this node's dependencies to the queue
        if current in dag:
            for dep in dag[current]["dependencies"]:
                if dep not in visited:
                    to_visit.append(dep)

    return visited


def detect_circular_dependencies() -> bool:
    """
    Detect if there are any circular dependencies in the DAG.

    Uses DFS with a recursion stack to detect back edges (cycles).

    Returns:
        True if any circular dependencies are found, False otherwise
    """
    dag = build_dag()

    # For each node, track: white (unvisited), gray (visiting), black (visited)
    color = {obj_id: "white" for obj_id in dag}

    def has_cycle_dfs(node: str) -> bool:
        """DFS to detect cycles."""
        if color[node] == "gray":
            # Back edge found - cycle detected
            return True
        if color[node] == "black":
            # Already fully processed
            return False

        color[node] = "gray"

        for neighbor in dag[node]["dependencies"]:
            if neighbor in dag and has_cycle_dfs(neighbor):
                return True

        color[node] = "black"
        return False

    # Check each node for cycles
    for obj_id in dag:
        if color[obj_id] == "white":
            if has_cycle_dfs(obj_id):
                return True

    return False


def export_dag() -> Dict[str, Dict[str, Any]]:
    """
    Export the full DAG structure suitable for serialization.

    Returns:
        Dictionary mapping object_id to complete node information
    """
    return build_dag()


def get_root_objects() -> Set[str]:
    """
    Get all root objects (objects with no dependencies).

    Returns:
        Set of object_ids that have no dependencies
    """
    dag = build_dag()
    return {obj_id for obj_id, node in dag.items() if not node["dependencies"]}


def get_leaf_objects() -> Set[str]:
    """
    Get all leaf objects (objects that nothing depends on).

    Returns:
        Set of object_ids that are not dependencies of any other object
    """
    dag = build_dag()

    # Collect all objects that are dependencies
    depended_upon = set()
    for node in dag.values():
        depended_upon.update(node["dependencies"])

    # Leaf objects are those not depended upon by anyone
    return set(dag.keys()) - depended_upon


def get_objects_by_type(object_type: str) -> Set[str]:
    """
    Get all objects of a specific type.

    Args:
        object_type: The type to filter by (e.g., "dataset", "model", "evaluation")

    Returns:
        Set of object_ids with that type
    """
    dag = build_dag()
    return {obj_id for obj_id, node in dag.items() if node["type"] == object_type}


def get_dependents(object_id: str) -> Set[str]:
    """
    Get all objects that depend on the given object (direct dependents).

    Args:
        object_id: The object to find dependents for

    Returns:
        Set of object_ids that directly depend on the given object
    """
    dag = build_dag()
    dependents = set()

    for obj_id, node in dag.items():
        if object_id in node["dependencies"]:
            dependents.add(obj_id)

    return dependents
