from ortools.graph.python.min_cost_flow import SimpleMinCostFlow
import numpy as np
import pandas as pd

def solve_min_cost_flow_graph(edges, costs, capacities, supplies, n_C, n_X):
    # Instantiate a SimpleMinCostFlow solver.
    min_cost_flow = SimpleMinCostFlow()

    if (
        (edges.dtype != "int32")
        or (costs.dtype != "int32")
        or (capacities.dtype != "int32")
        or (supplies.dtype != "int32")
    ):
        raise ValueError(
            "`edges`, `costs`, `capacities`, `supplies` must all be int dtype"
        )

    N_edges = edges.shape[0]
    N_nodes = len(supplies)

    print(edges)
    # Add each edge with associated capacities and cost
    min_cost_flow.add_arcs_with_capacity_and_unit_cost(
        edges[:, 0], edges[:, 1], capacities, costs
    )

    # Add node supplies
    min_cost_flow.set_nodes_supplies(np.arange(len(supplies)), supplies)
    
    print(min_cost_flow.solve())

    # Find the minimum cost flow between node 0 and node 4.
    if min_cost_flow.solve() != min_cost_flow.OPTIMAL:
        raise Exception("There was an issue with the min cost flow input.")

    # Assignment
    labels_M = (
        np.array([min_cost_flow.flow(i) for i in range(n_X * n_C)])
        .reshape(n_X, n_C)
        .astype("int32")
    )

    labels = labels_M.argmax(axis=1)
    return labels