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

def proportion_keywords(centers, labels=None, min_score=0.5, topk=200,
                        candidates_topk=300, index2word=None, passwords=None):
    
    l1_normalize = lambda x:x/x.sum()

    index_type = passwords and isinstance(list(passwords)[0], int)

    n_clusters, n_features = centers.shape
    total_frequency = np.zeros(n_features)

    if labels is not None:
        n_samples_in_cluster = np.bincount(labels, minlength=n_clusters)
    else:
        n_samples_in_cluster = np.asarray([1] * n_clusters)

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
        indices = sorted(indices, key=lambda idx:-p_prop[idx])[:candidates_topk]
        scores = [(idx, p_prop[idx] / (p_prop[idx] + n_prop[idx])) for idx in indices]
        scores = [t for t in scores if t[1] >= min_score]
        scores = sorted(scores, key=lambda x:-x[1])
        keywords.append(scores)

    if passwords and index_type:
        scores = [t for t in scores if t[0] in passwords]

    if index2word is not None:
        keywords = [[(index2word[idx], score) for idx, score in keyword] for keyword in keywords]
        if passwords and not index_type:
            keywords = [t for t in keywords if t[0] in passwords]

    if topk > 0:
        keywords = [keyword[:topk] for keyword in keywords]

    return keywords