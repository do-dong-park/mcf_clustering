import numpy as np
from sklearn_import.metrics.pairwise import euclidean_distances
from sklearn_import.utils.validation import (
    check_random_state,
    check_is_fitted,
    as_float_array
)
from k_means_constrained.sklearn_import.cluster._k_means import _centers_dense, _centers_sparse
from k_means_constrained.sklearn_import.cluster.k_means_ import (
    _validate_center_shape,
    _tolerance,
    KMeans,
    _init_centroids,
)
from sklearn_import.utils.extmath import row_norms, squared_norm, cartesian
from joblib import Parallel, delayed
import scipy.sparse as sp
from mcf import mcf_solver



class KMeansConstrained(KMeans):
    def __init__(
        self,
        areas,
        required_areas,
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
        
        super().__init__(
            n_clusters=len(self.required_areas),
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
        random_state = check_random_state(self.random_state)
        X = self._check_fit_data(X)
        (
            self.cluster_centers_,
            self.labels_,
            self.inertia_,
            self.n_iter_,
        ) = k_means_constrained(
            X,
            areas = self.areas,
            
            required_areas=self.required_areas,
            n_clusters=self.n_clusters,
            
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

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_

def k_means_constrained(
    X,
    areas,
    n_clusters,
    required_areas,
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
    X = as_float_array(X, copy=copy_x)
    tol = _tolerance(X, tol)
    
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
    if not sp.issparse(X):
        X_mean = X.mean(axis=0)
        X -= X_mean
        if hasattr(init, "__array__"):
            init -= X_mean
    x_squared_norms = row_norms(X, squared=True)
    
    best_labels, best_inertia, best_centers = None, None, None
    
    if n_jobs == 1:
        for it in range(n_init):
            labels, inertia, centers, n_iter_ = kmeans_constrained_single(
                X,
                areas,
                n_clusters,
                required_areas,
                max_iter=max_iter,
                init=init,
                verbose=verbose,
                tol=tol,
                x_squared_norms=x_squared_norms,
                random_state=random_state,
            )
            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.copy()
                best_centers = centers.copy()
                best_inertia = inertia
    else:
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(kmeans_constrained_single)(
                X,
                areas,
                n_clusters,
                required_areas,
                max_iter=max_iter,
                init=init,
                verbose=verbose,
                tol=tol,
                x_squared_norms=x_squared_norms,
                random_state=seed,
            )
            for seed in seeds
        )
        labels, inertia, centers, n_iters = zip(*results)
        print("n_iters")
        print(n_iters)
        best = np.argmin(inertia)
        best_labels = labels[best]
        best_inertia = inertia[best]
        best_centers = centers[best]
    if not sp.issparse(X):
        if not copy_x:
            X += X_mean
        best_centers += X_mean
    if return_n_iter:
        return best_centers, best_labels, best_inertia, n_iters[best]
    else:
        return best_centers, best_labels, best_inertia

def kmeans_constrained_single(
    X,
    areas,
    n_clusters,
    required_areas,
    max_iter=300,
    init="k-means++",
    verbose=False,
    x_squared_norms=None,
    random_state=None,
    tol=1e-4,
):
    random_state = check_random_state(random_state)
    n_samples = X.shape[0]
    best_labels, best_inertia, best_centers = None, None, None
    
    centers = _init_centroids(
        X, n_clusters, init, random_state=random_state, x_squared_norms=x_squared_norms
    )
    
    distances = np.zeros(shape=(n_samples,), dtype=X.dtype)
    
    for i in range(max_iter):
        centers_old = centers.copy()
        labels, inertia = _labels_constrained(
            X, centers, areas, required_areas, distances=distances
        )
        
        if sp.issparse(X):
            centers = _centers_sparse(X, labels, n_clusters, distances)
        
        else:
            centers = _centers_dense(X, labels, n_clusters, distances)
            
        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia
            
        center_shift_total = squared_norm(centers_old - centers)
        
        if center_shift_total <= tol:
            break
        
    if center_shift_total > 0:
        best_labels, best_inertia = _labels_constrained(
            X, centers, areas, required_areas, distances=distances
        )
    return best_labels, best_inertia, best_centers, i + 1

def _labels_constrained(X, centers, areas, required_areas, distances):
    
    D = euclidean_distances(X, centers, squared=False)
    
    labels = mcf_solver(areas=areas, requested_areas=required_areas, costs=D)
    
    labels = labels.astype(np.int32)
    
    # Change distances in-place
    distances[:] = D[np.arange(D.shape[0]), labels] ** 2  # Square for M step of EM
    inertia = distances.sum()
    
    return labels, inertia