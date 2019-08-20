import numpy as np
from sklearn.utils.extmath import safe_sparse_dot

def inner_product(X, Y):
    """
    Arguments
    --------
    X, Y: numpy.ndarray or scipy.sparse.csr_matrix
        One of both must be sparse matrix
        shape of X = (n,p)
        shape of Y = (p,m)

    Returns
    -------
    Z : scipy.sparse.csr_matrix
        shape of Z = (n,m)
    """

    return safe_sparse_dot(X, Y, dense_output=False)

def check_sparsity(x):
    """
    It calculates sparsity of centroid vectors

    Argument
    --------
    x : numpy.ndarray

    Returns
    -------
    sparsity : float
        1 - proportion of nonzero elements
    """
    n,m = x.shape
    return sum(len(np.where(x[c] != 0)[0]) for c in range(n)) / (n*m)

