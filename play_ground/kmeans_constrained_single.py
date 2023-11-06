def kmeans_constrained_single(
    X,
    n_clusters,
    area_tol,
    required_areas,
    areas,
    max_iter=300,
    init="k-means++",
    verbose=False,
    x_squared_norms=None,
    random_state=None,
    tol=1e-4,
):
    
    if sp.issparse(X):
        raise NotImplementedError("Not implemented for sparse X")

    random_state = check_random_state(random_state)
    n_samples = X.shape[0]

    best_labels, best_inertia, best_centers = None, None, None
    # init
    centers = _init_centroids(
        X, n_clusters, init, random_state=random_state, x_squared_norms=x_squared_norms
    )
    if verbose:
        print("Initialization complete")

    # Allocate memory to store the distances for each sample to its
    # closer center for reallocation in case of ties
    distances = np.zeros(shape=(n_samples,), dtype=X.dtype)

    # iterations
    for i in range(max_iter):
        centers_old = centers.copy()
        # labels assignment is also called the E-step of EM
        labels, inertia = _labels_constrained(
            X, centers, area_tol, required_areas,areas, distances=distances
        )

        # computation of the means is also called the M-step of EM
        if sp.issparse(X):
            centers = _centers_sparse(X, labels, n_clusters, distances)
        else:
            centers = _centers_dense(X, labels, n_clusters, distances)

        if verbose:
            print("Iteration %2d, inertia %.3f" % (i, inertia))

        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia

        center_shift_total = squared_norm(centers_old - centers)
        if center_shift_total <= tol:
            if verbose:
                print(
                    "Converged at iteration %d: "
                    "center shift %e within tolerance %e" % (i, center_shift_total, tol)
                )
            break

    if center_shift_total > 0:
        # rerun E-step in case of non-convergence so that predicted labels
        # match cluster centers
        best_labels, best_inertia = _labels_constrained(
            X, centers, area_tol, required_areas, areas, distances=distances
        )

    return best_labels, best_inertia, best_centers, i + 1