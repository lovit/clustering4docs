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

def proportion_keywords(centers, labels, topk=200, index2word=None):
    l1_normalize = lambda x:x/x.sum()

    n_clusters, n_features = centers.shape
    n_samples_in_cluster = np.bincount(labels, minlength=n_clusters)
    n_samples_in_cluster.shape
    total_frequency = np.zeros(n_features)

    for c, n_docs in enumerate(n_samples_in_cluster):
        total_frequency += (centers[c] * n_docs)
    total_sum = total_frequency.sum()

    keywords = []
    for c, n_docs in enumerate(n_samples_in_cluster):
        if n_docs == 0:
            keywords.append([])
            continue

        n_prop = l1_normalize(total_frequency - (centers[c] * n_docs))
        p_prop = l1_normalize(centers[c])

        indices = np.where(p_prop > 0)[0]
        scores = [(idx, p_prop[idx] / (p_prop[idx] + n_prop[idx])) for idx in indices]
        scores = sorted(scores, key=lambda x:-x[1])[:topk]
        keywords.append(scores)

    if index2word is not None:
        keywords = [[(index2word[idx], score) for idx, score in keyword] for keyword in keywords]

    return keywords