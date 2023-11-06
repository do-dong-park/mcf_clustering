def k_means_constrained(
    X,
    n_clusters,
    
    area_tol,
    required_areas,
    areas,
    
    init="k-means++",
    n_init=10,
    max_iter=300,
    verbose=False,
    tol=1e-4,
    random_state=None,
    copy_x=True,
    n_jobs=1,
    return_n_iter=False,
):
    
    if sp.issparse(X):
        raise NotImplementedError("Not implemented for sparse X")

    if n_init <= 0:
        raise ValueError(
            "Invalid number of initializations."
            " n_init=%d must be bigger than zero." % n_init
        )
    random_state = check_random_state(random_state)

    if max_iter <= 0:
        raise ValueError(
            "Number of iterations should be a positive number,"
            " got %d instead" % max_iter
        )

    X = as_float_array(X, copy=copy_x)
    tol = _tolerance(X, tol)

    # Validate init array
    if hasattr(init, "__array__"):
        init = check_array(init, dtype=X.dtype.type, copy=True)
        _validate_center_shape(X, n_clusters, init)

        if n_init != 1:
            warnings.warn(
                "Explicit initial center position passed: "
                "performing only one init in k-means instead of n_init=%d" % n_init,
                RuntimeWarning,
                stacklevel=2,
            )
            n_init = 1

    # subtract of mean of x for more accurate distance computations
    if not sp.issparse(X):
        X_mean = X.mean(axis=0)
        # The copy was already done above
        X -= X_mean

        if hasattr(init, "__array__"):
            init -= X_mean

    # precompute squared norms of data points
    x_squared_norms = row_norms(X, squared=True)

    best_labels, best_inertia, best_centers = None, None, None

    if n_jobs == 1:
        # For a single thread, less memory is needed if we just store one set
        # of the best results (as opposed to one set per run per thread).
        for it in range(n_init):
            # run a k-means once
            labels, inertia, centers, n_iter_ = kmeans_constrained_single(
                X,
                n_clusters,
                
                area_tol,
                required_areas,
                areas,
                
                max_iter=max_iter,
                init=init,
                verbose=verbose,
                tol=tol,
                x_squared_norms=x_squared_norms,
                random_state=random_state,
            )
            # determine if these results are the best so far
            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.copy()
                best_centers = centers.copy()
                best_inertia = inertia
                best_n_iter = n_iter_
    else:
        # parallelisation of k-means runs
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(kmeans_constrained_single)(
                X,
                n_clusters,
                
                area_tol,
                required_areas,
                areas,
                
                max_iter=max_iter,
                init=init,
                verbose=verbose,
                tol=tol,
                x_squared_norms=x_squared_norms,
                # Change seed to ensure variety
                random_state=seed,
            )
            for seed in seeds
        )
        # Get results with the lowest inertia
        labels, inertia, centers, n_iters = zip(*results)
        best = np.argmin(inertia)
        best_labels = labels[best]
        best_inertia = inertia[best]
        best_centers = centers[best]
        best_n_iter = n_iters[best]

    if not sp.issparse(X):
        if not copy_x:
            X += X_mean
        best_centers += X_mean

    if return_n_iter:
        return best_centers, best_labels, best_inertia, best_n_iter
    else:
        return best_centers, best_labels, best_inertia