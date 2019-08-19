import numpy as np

def proportion_keywords(centers, labels=None, min_score=0.5, topk=200,
    candidates_topk=300, index2word=None, passwords=None):
    """
    Arguments
    ---------
    centers : numpy.ndarray
        Shape = (k, p)
        k : num of clusters
        p : num of features
    labels : numpy.ndarray or None
        Shape = (n,)
        n : num of documents in the dataset
        It is used as cluster size weight. If the value is None,
        it assumes that all size of clusters are equal.
    min_score : float
        0 < min_score < 1
        Minimum keyword score. The words corresponding less than the score will be
        removed from set of keywords.
    topk : int
        The number of keywords
    candidates_topk : int
        The number of keyword candidates
    index2word : list of str or None
        If the value is not None, the length must be p (num of features)
        If the value is None, it returns keywords as integer index
    passwords : set of str or None
        The set of words that will be removed from keyword set by force

    Returns
    -------
    keywords_in_clusters : nested list
        The length of list equals with num of clusters
        [[(keyword, score), (keyword, score), ... ],
         [(keyword, score), (keyword, score), ... ]
         ...
        ]
    """

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
