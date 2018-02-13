# cython: profile=True
# Profiling is enabled by default as the overhead does not seem to be measurable
# on this specific use case.

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck
#
# License: BSD 3 clause

from libc.math cimport sqrt
import numpy as np
import scipy.sparse as sp
cimport numpy as np
cimport cython
from cython cimport floating

from sklearn.utils.sparsefuncs_fast import assign_rows_csr

ctypedef np.float64_t DOUBLE
ctypedef np.int32_t INT

ctypedef floating (*DOT)(int N, floating *X, int incX, floating *Y,
                         int incY)

cdef extern from "cblas.h":
    double ddot "cblas_ddot"(int N, double *X, int incX, double *Y, int incY)
    float sdot "cblas_sdot"(int N, float *X, int incX, float *Y, int incY)

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef DOUBLE _assign_labels_csr(X, np.ndarray[floating, ndim=2] centers,
                                np.ndarray[INT, ndim=1] labels):
    """Compute label assignment and inertia for a CSR input in Spherical k-means
    Return the inertia (sum of squared distances to the centers).
    """
    cdef:
        np.ndarray[floating, ndim=1] X_data = X.data
        np.ndarray[INT, ndim=1] X_indices = X.indices
        np.ndarray[INT, ndim=1] X_indptr = X.indptr
        unsigned int n_clusters = centers.shape[0]
        unsigned int n_features = centers.shape[1]
        unsigned int n_samples = X.shape[0]
        unsigned int store_distances = 0
        unsigned int sample_idx, center_idx, feature_idx
        unsigned int k
        # the following variables are always double cause make them floating
        # does not save any memory, but makes the code much bigger
        DOUBLE inertia = 0.0
        DOUBLE min_dist
        DOUBLE dist
        DOT dot

    if floating is float:
        center_squared_norms = np.zeros(n_clusters, dtype=np.float32)
        dot = sdot
    else:
        center_squared_norms = np.zeros(n_clusters, dtype=np.float64)
        dot = ddot

    for sample_idx in range(n_samples):
        max_prod = 0
        for center_idx in range(n_clusters):
            prod = 0.0
            # hardcoded: minimize cosine distance to cluster center:
            # cos(a,b) = 1 - <a,b> if a and b are unit vector
            for k in range(X_indptr[sample_idx], X_indptr[sample_idx + 1]):
                prod += centers[center_idx, X_indices[k]] * X_data[k]
            if max_prod == 0 or max_prod < prod:
                max_prod = prod
                labels[sample_idx] = center_idx
        inertia += (1 - max_prod)

    return inertia