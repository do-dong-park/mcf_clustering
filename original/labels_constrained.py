def _labels_constrained(X, centers, size_min, size_max, distances):
    """Compute labels using the min and max cluster size constraint

    This will overwrite the 'distances' array in-place.

    Parameters
    ----------
    X : numpy array, shape (n_sample, n_features)
        Input data.

    size_min : int
        Minimum size for each cluster

    size_max : int
        Maximum size for each cluster

    centers : numpy array, shape (n_clusters, n_features)
        Cluster centers which data is assigned to.

    distances : numpy array, shape (n_samples,)
        Pre-allocated array in which distances are stored.

    Returns
    -------
    labels : numpy array, dtype=np.int, shape (n_samples,)
        Indices of clusters that samples are assigned to.

    inertia : float
        Sum of squared distances of samples to their closest cluster center.

    """
    C = centers

    # Distances to each centre C. (the `distances` parameter is the distance to the closest centre)
    # K-mean original uses squared distances but this equivalent for constrained k-means
    D = euclidean_distances(X, C, squared=False)

    edges, costs, capacities, supplies, n_C, n_X = minimum_cost_flow_problem_graph(
        X, C, D, size_min, size_max
    )
    labels = solve_min_cost_flow_graph(edges, costs, capacities, supplies, n_C, n_X)

    # cython k-means M step code assumes int32 inputs
    labels = labels.astype(np.int32)

    # Change distances in-place
    distances[:] = D[np.arange(D.shape[0]), labels] ** 2  # Square for M step of EM
    inertia = distances.sum()

    return labels, inertia