class KMeansConstrained(KMeans):
    
    def __init__(
        self,
        n_clusters=8,
        areas = None,
        required_areas = None,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-4,
        verbose=False,
        random_state=None,
        copy_x=True,
        n_jobs=1,
    ):
        self.areas = areas
        self.required_areas = required_areas
        self.area_tol = 5000

        super().__init__(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            copy_x=copy_x,
            n_jobs=n_jobs,
        )

    def fit(self, X, y=None):
        # 값이 대부분 0이면, 안돌아감
        if sp.issparse(X):
            raise NotImplementedError("Not implemented for sparse X")

        random_state = check_random_state(self.random_state)
        
        # k값이 Sample 수 보다 작은지 판별
        X = self._check_fit_data(X)

        (
            self.cluster_centers_,
            self.labels_,
            self.inertia_,
            self.n_iter_,
        ) = k_means_constrained(
            X,
            n_clusters=self.n_clusters,
            area_tol = self.area_tol,
            required_areas = self.required_areas,
            areas = self.areas,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            verbose=self.verbose,
            tol=self.tol,
            random_state=random_state,
            copy_x=self.copy_x,
            n_jobs=self.n_jobs,
            return_n_iter=True,
        )
        return self

    def predict(self, X, size_min="init", size_max="init"):
        """
        Predict the closest cluster each sample in X belongs to given the provided constraints.
        The constraints can be temporally overridden when determining which cluster each datapoint is assigned to.

        Only computes the assignment step. It does not re-fit the cluster positions.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            New data to predict.

        size_min : int, optional, default: size_min provided with initialisation
            Constrain the label assignment so that each cluster has a minimum
            size of size_min. If None, no constrains will be applied.
            If 'init' the value provided during initialisation of the
            class will be used.

        size_max : int, optional, default: size_max provided with initialisation
            Constrain the label assignment so that each cluster has a maximum
            size of size_max. If None, no constrains will be applied.
            If 'init' the value provided during initialisation of the
            class will be used.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """

        if sp.issparse(X):
            raise NotImplementedError("Not implemented for sparse X")
        
        if size_min == "init":
            size_min = self.size_min
        if size_max == "init":
            size_max = self.size_max

        n_clusters = self.n_clusters
        n_samples = X.shape[0]

        # data를 집어 넣고 예측 돌리는지 판단
        check_is_fitted(self, "cluster_centers_")
        # feature 수 점검
        X = self._check_test_data(X)

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

        labels, inertia = _labels_constrained(
            X, self.cluster_centers_, size_min, size_max, distances=distances
        )

        return labels

    def fit_predict(self, X, y=None):
        """Compute cluster centers and predict cluster index for each sample.

        Equivalent to calling fit(X) followed by predict(X) but also more efficient.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        return self.fit(X).labels_
