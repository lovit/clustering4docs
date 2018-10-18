import numpy as np
from sklearn.utils.extmath import safe_sparse_dot

def inner_product(X, Y):
    """X: shape=(n,p)
    Y: shape=(p,m)
    It returns (n,m)"""
    return safe_sparse_dot(X, Y, dense_output=False)

def check_sparsity(mat):
    n,m = mat.shape
    return sum(len(np.where(mat[c] != 0)[0]) for c in range(n)) / (n*m)