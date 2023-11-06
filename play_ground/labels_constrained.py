def _labels_constrained(X, centers, area_tol, required_areas, areas, distances):
    
    C = centers

    # Distances to each centre C. (the `distances` parameter is the distance to the closest centre)
    # K-mean original uses squared distances but this equivalent for constrained k-means
    D = euclidean_distances(X, C, squared=False)

    edges, costs, capacities, supplies, n_C, n_X = minimum_cost_flow_problem_graph(
        X, C, D, area_tol, required_areas, areas )
    labels = solve_min_cost_flow_graph(edges, costs, capacities, supplies, n_C, n_X)

    # cython k-means M step code assumes int32 inputs
    labels = labels.astype(np.int32)

    # Change distances in-place
    distances[:] = D[np.arange(D.shape[0]), labels] ** 2  # Square for M step of EM
    inertia = distances.sum()

    return labels, inertia