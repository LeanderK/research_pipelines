from typing import Any

import research_pipelines.backends.base as base
try:
    import matplotlib.pyplot as plt
    import networkx as nx
except ImportError:
    raise ImportError(
        "matplotlib is required for visualization. Please install it using 'pip install matplotlib'."
        )

def visualize_dag(
    dag: base.Backend | dict[str, dict[str, Any]],
) -> None:
    """
    Visualize the DAG of traced configurations.

    Args:
        dag: A Backend instance or a dictionary of configurations to visualize.
    """
    if isinstance(dag, base.Backend):
        configs = dag.load_all()
    elif isinstance(dag, dict):
        configs = dag
    else:
        raise ValueError("Input must be a Backend instance or a dictionary of configurations.")
    
    # Build the graph
    graph = {}
    for object_id, config in configs.items():
        dependencies = config.get("dependencies", [])
        graph[object_id] = dependencies

    g = nx.DiGraph()

    def follow_parent(key) -> str:
        config = configs[key]
        if config['parent_id'] is not None:
            return follow_parent(config['parent_id'])
        return key

    def human_readable_name(key) -> str:
        config = configs[key]
        name = config['callable'].split(':')[-1]
        return name

    for node, dependencies in graph.items():
        config = configs[node]
        if config['parent_id'] is not None:
            continue
        for dep_name, dep_key in dependencies.items():
            real_key = follow_parent(dep_key)
            g.add_edge(real_key, node)

    labels = {
        n: human_readable_name(n)
        for n in g.nodes
    }

    pos = nx.shell_layout(g)

    plt.figure(figsize=(10, 7))

    # nx.draw_networkx_edges(g, pos, arrows=True)
    nx.draw_networkx_edges(
        g,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=15,
        min_source_margin=15,
        min_target_margin=25
    )

    for node, (x, y) in pos.items():
        plt.text(
            x, y,
            labels[node],
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.6",
                fc="white",
                ec="black"
            )
        )

    plt.axis("off")
    plt.show()