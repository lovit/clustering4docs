import numpy as np
from sklearn.metrics import pairwise_distances

def merge_close_clusters(centers, labels, max_dist=0.7):
    n_clusters, n_terms = centers.shape
    cluster_size = np.bincount(labels, minlength=n_clusters)
    sorted_indices, _ = zip(*sorted(enumerate(cluster_size), key=lambda x:-x[1]))

    groups = _grouping_with_centers(centers, labels, max_dist, sorted_indices)
    centers_ = np.dot(np.diag(cluster_size), centers)

    n_groups = len(groups)
    group_centers = np.zeros((n_groups, n_terms))
    for g, idxs in enumerate(groups):
        sum_ = centers_[idxs].sum(axis=0)
        mean = sum_ / cluster_size[idxs].sum()
        group_centers[g] = mean
    return group_centers, groups

def _closest_group(groups, c, pdist, max_dist):
    dist_ = 1
    closest = None
    for g, idxs in enumerate(groups):
        dist = pdist[idxs, c].mean()
        if dist > max_dist:
            continue
        if dist_ > dist:
            closest = g
    return closest

def _grouping_with_centers(centers, max_dist, sorted_indices):
    pdist = pairwise_distances(centers, metric='cosine')
    return _grouping_with_pdist(pdist, max_dist, sorted_indices)

def _grouping_with_pdist(pdist, max_dist, sorted_indices):
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