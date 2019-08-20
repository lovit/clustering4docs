import numpy as np
from sklearn.metrics import pairwise_distances

def merge_close_clusters(centers, labels, max_dist=0.7):
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
    max_dist : float
        Maximum Cosine distance between base cluster and other clusters
        The smaller the value, it groups closer clusters to a group.

    Returns
    -------
    group_centers : numpy.ndarray
        Merged cluster centroid vectors
        Shape = (l, p)
        l : num of merged clusters. l <= k
        p : num of features
    groups : list of list of int
        The index of first index corresponds the index of merged clusters.
        And the value in the inner lists corresponds the index of original clusters.
    """

    n_clusters, n_terms = centers.shape
    cluster_size = np.bincount(labels, minlength=n_clusters)
    sorted_indices, _ = zip(*sorted(enumerate(cluster_size), key=lambda x:-x[1]))

    groups = _grouping_with_centers(centers, max_dist, sorted_indices)
    centers_ = np.dot(np.diag(cluster_size), centers)

    n_groups = len(groups)
    group_centers = np.zeros((n_groups, n_terms))
    for g, idxs in enumerate(groups):
        sum_ = centers_[idxs].sum(axis=0)
        mean = sum_ / cluster_size[idxs].sum()
        group_centers[g] = mean
    return group_centers, groups

def _closest_group(groups, c, pdist, max_dist):
    """
    It finds the most closest merged clusters with the given cluster c

    Arguments
    ---------
    groups : list of list of int
        The index of first index corresponds the index of merged clusters.
        And the value in the inner lists corresponds the index of original clusters.
    c : int
        Index of the cluster which this function finds the most closest merged group
    pdist : numpy.ndarray
        Pairwise distance matrix.
        Shape is (k, k) where k is num of clusters.
    max_dist : float
        Maximum Cosine distance between base cluster and other clusters.
        The smaller the value, it groups closer clusters to a group.

    Returns
    -------
    closest : int or None
        The index of most closest merged group.
        If the closest one and target cluster c is distants than max_dist,
        it returns None
    """

    dist_ = 1
    closest = None
    for g, idxs in enumerate(groups):
        dist = pdist[idxs, c].mean()
        if dist > max_dist:
            continue
        if dist_ > dist:
            dist_ = dist
            closest = g
    return closest

def _grouping_with_centers(centers, max_dist, sorted_indices):
    """
    Arguments
    ---------
    centers : numpy.ndarray
        Shape = (k, p)
        k : num of clusters
        p : num of features
    max_dist : float
        Maximum Cosine distance between base cluster and other clusters.
        The smaller the value, it groups closer clusters to a group.
    sorted_index : list of int
        Cluster index sorted by the size of each cluster.

    Returns
    -------
    groups : list of list of int
        The index of first index corresponds the index of merged clusters.
        And the value in the inner lists corresponds the index of original clusters.
    """

    pdist = pairwise_distances(centers, metric='cosine')
    return _grouping_with_pdist(pdist, max_dist, sorted_indices)

def _grouping_with_pdist(pdist, max_dist, sorted_indices):
    """
    Arguments
    ---------
    pdist : numpy.ndarray
        Pairwise distance matrix.
        Shape is (k, k) where k is num of clusters.
    max_dist : float
        Maximum Cosine distance between base cluster and other clusters.
        The smaller the value, it groups closer clusters to a group.
    sorted_index : list of int
        Cluster index sorted by the size of each cluster.

    Returns
    -------
    groups : list of list of int
        The index of first index corresponds the index of merged clusters.
        And the value in the inner lists corresponds the index of original clusters.
    """
    groups = [[sorted_indices[0]]]
    for c in sorted_indices[1:]:
        g = _closest_group(groups, c, pdist, max_dist)
        # create new group
        if g is None:
            groups.append([c])
        # assign c to g
        else:
            groups[g].append(c)
    return groups
