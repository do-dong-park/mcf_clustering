def kmeans_constrained_single(
    X,
    n_clusters,
    size_min=None,
    size_max=None,
    max_iter=300,
    init="k-means++",
    verbose=False,
    x_squared_norms=None,
    random_state=None,
    tol=1e-4,
):
    """A single run of k-means constrained, assumes preparation completed prior.

    Parameters
    ----------
    X : array-like of floats, shape (n_samples, n_features)
        The observations to cluster.

    size_min : int, optional, default: None
        Constrain the label assignment so that each cluster has a minimum
        size of size_min. If None, no constrains will be applied

    size_max : int, optional, default: None
        Constrain the label assignment so that each cluster has a maximum
        size of size_max. If None, no constrains will be applied

    n_clusters : int
        The number of clusters to form as well as the number of
        centroids to generate.

    max_iter : int, optional, default 300
        Maximum number of iterations of the k-means algorithm to run.

    init : {'k-means++', 'random', or ndarray, or a callable}, optional
        Method for initialization, default to 'k-means++':

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'random': generate k centroids from a Gaussian with mean and
        variance estimated from the data.

        If an ndarray is passed, it should be of shape (k, p) and gives
        the initial centers.

        If a callable is passed, it should take arguments X, k and
        and a random state and return an initialization.

    tol : float, optional
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.

    verbose : boolean, optional
        Verbosity mode

    x_squared_norms : array
        Precomputed x_squared_norms.

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    centroid : float ndarray with shape (k, n_features)
        Centroids found at the last iteration of k-means.

    label : integer ndarray with shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    n_iter : int
        Number of iterations run.
    """
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

    # Determine min and max sizes if non given
    if size_min is None:
        size_min = 0
    if size_max is None:
        size_max = n_samples  # Number of data points

    # Check size min and max
    if not (
        (size_min >= 0)
        and (size_min <= n_samples)
        and (size_max >= 0)
        and (size_max <= n_samples)
    ):
        raise ValueError(
            "size_min and size_max must be a positive number smaller "
            "than the number of data points or `None`"
        )
    if size_max < size_min:
        raise ValueError("size_max must be larger than size_min")
    if size_min * n_clusters > n_samples:
        raise ValueError(
            "The product of size_min and n_clusters cannot exceed the number of samples (X)"
        )
    if size_max * n_clusters < n_samples:
        raise ValueError(
            "The product of size_max and n_clusters must be larger than or equal the number of samples (X)"
        )

    # iterations
    for i in range(max_iter):
        centers_old = centers.copy()
        # labels assignment is also called the E-step of EM
        labels, inertia = _labels_constrained(
            X, centers, size_min, size_max, distances=distances
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
            X, centers, size_min, size_max, distances=distances
        )

    return best_labels, best_inertia, best_centers, i + 1