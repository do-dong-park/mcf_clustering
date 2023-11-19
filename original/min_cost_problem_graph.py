import numpy as np
def minimum_cost_flow_problem_graph(X, C, D, size_min, size_max):
    # Setup minimum cost flow formulation graph
    # Vertices indexes:
    # X-nodes: [0, n(x)-1], C-nodes: [n(X), n(X)+n(C)-1], C-dummy nodes:[n(X)+n(C), n(X)+2*n(C)-1],
    # Artificial node: [n(X)+2*n(C), n(X)+2*n(C)+1-1]

    # Create indices of nodes
    n_X = X.shape[0]
    print(type(C))
    n_C = C.shape[0]
    X_ix = np.arange(n_X)
    C_dummy_ix = np.arange(X_ix[-1] + 1, X_ix[-1] + 1 + n_C)
    C_ix = np.arange(C_dummy_ix[-1] + 1, C_dummy_ix[-1] + 1 + n_C)
    art_ix = C_ix[-1] + 1

    # Edges
    edges_X_C_dummy = cartesian(
        [X_ix, C_dummy_ix]
    )  # All X's connect to all C dummy nodes (C')
    edges_C_dummy_C = np.stack(
        [C_dummy_ix, C_ix], axis=1
    )  # Each C' connects to a corresponding C (centroid)
    edges_C_art = np.stack(
        [C_ix, art_ix * np.ones(n_C)], axis=1
    )  # All C connect to artificial node

    edges = np.concatenate([edges_X_C_dummy, edges_C_dummy_C, edges_C_art])

    # Costs
    costs_X_C_dummy = D.reshape(D.size)
    costs = np.concatenate(
        [costs_X_C_dummy, np.zeros(edges.shape[0] - len(costs_X_C_dummy))]
    )

    # Capacities - can set for max-k
    capacities_C_dummy_C = size_max * np.ones(n_C)
    cap_non = n_X  # The total supply and therefore wont restrict flow
    capacities = np.concatenate(
        [
            np.ones(edges_X_C_dummy.shape[0]),
            capacities_C_dummy_C,
            cap_non * np.ones(n_C),
        ]
    )

    # Sources and sinks
    supplies_X = np.ones(n_X)
    supplies_C = -1 * size_min * np.ones(n_C)  # Demand node
    supplies_art = -1 * (n_X - n_C * size_min)  # Demand node
    supplies = np.concatenate(
        [supplies_X, np.zeros(n_C), supplies_C, [supplies_art]]  # C_dummies
    )

    # All arrays must be of int dtype for `SimpleMinCostFlow`
    edges = edges.astype("int32")
    costs = np.around(costs * 1000, 0).astype(
        "int32"
    )  # Times by 1000 to give extra precision
    capacities = capacities.astype("int32")
    supplies = supplies.astype("int32")

    return edges, costs, capacities, supplies, n_C, n_X