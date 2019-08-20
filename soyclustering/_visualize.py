from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import numpy as np

from ._postprocess import _grouping_with_pdist

def visualize_pairwise_distance(centers, labels=None, max_dist=0.7, 
    sort=False, show=True, title=None, figsize=(15,15), cmap='gray',
    clim=None, dpi=50, facecolor=None, edgecolor=None, frameon=True):
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
        Maximum Cosine distance between base cluster and other clusters.
        The smaller the value, it groups closer clusters to a group.
    sort : Boolean
        It True, it first groups nearby clusters into one group
        then draw pairwise distance matrix
    title : str or None
        The title of the figure
    figsize : tuple of int
        Size of the figure. Default is (15,15)
    cmap : str
        Color map in matplotlib. Default is gray
    clim : tuple of int or None
        Color limit in matplotlib. 
    dpi : int
        DPI of the figure
    facecolor : tuple of float or None
        RBG color of background
        For example (0.15, 0.5, 0.5)
    edgecolor : tuple of float or None
        RBG color of edge for the figure
        For example (0.15, 0.5, 0.5)
    frameon : Boolean
        Ignored

    Returns
    -------
    figure : matplotlib.pyplot.figure
        Figure of pairwise distance
    """

    n_clusters = centers.shape[0]

    if labels is None:
        labels = [i for i in range(n_clusters)]

    pdist = pairwise_distances(centers, metric='cosine')

    if sort:
        # sort clusters by cluster size
        cluster_size = np.bincount(labels, minlength=n_clusters)
        sorted_indices, _ = zip(*sorted(enumerate(cluster_size), key=lambda x:-x[1]))

        # grouping clusters
        groups = _grouping_with_pdist(pdist, max_dist, sorted_indices)

        # sort groups by group size
        groups = sorted(groups, key=lambda x:-len(x))

        # create revised index
        sorted_indices = [idx for group in groups for idx in group]
        indices_revised = np.ix_(sorted_indices, sorted_indices)

        # create origin index
        indices_orig = list(range(n_clusters))

        # revise pairwise distance matrix
        pdist_revised = np.empty_like(pdist)
        pdist_revised[np.ix_(indices_orig,indices_orig)] = pdist[indices_revised]

    else:
        pdist_revised = pdist

    figure = pairwise_distance_to_matplotlib_figure(pdist_revised, title,
        figsize, cmap, clim, dpi, facecolor, edgecolor, frameon)
    return figure

def pairwise_distance_to_matplotlib_figure(pdist, title=None,
    figsize=(15,15), cmap='gray', clim=None, dpi=50,
    facecolor=None, edgecolor=None, frameon=True):
    """
    pdist : numpy.ndarray
        Pairwise distance matrix. Shape is (k, k) when k is num of clusters
    title : str or None
        The title of the figure
    figsize : tuple of int
        Size of the figure. Default is (15,15)
    cmap : str
        Color map in matplotlib. Default is gray
    clim : tuple of int or None
        Color limit in matplotlib. 
    dpi : int
        DPI of the figure
    facecolor : tuple of float or None
        RBG color of background
        For example (0.15, 0.5, 0.5)
    edgecolor : tuple of float or None
        RBG color of edge for the figure
        For example (0.15, 0.5, 0.5)
    frameon : Boolean
        Ignored

    Returns
    -------
    figure : matplotlib.pyplot.figure
        The figure of pairwise distance matrix
    """

    n_clusters = pdist.shape[0]

    figure = plt.figure(
        figsize = figsize,
        dpi = dpi,
        facecolor = facecolor,
        edgecolor = edgecolor,
        frameon = frameon
    )

    if title:
        plt.title(title)

    plt.xlim((0, n_clusters))
    plt.ylim((0, n_clusters))

    if clim:
        plt.imshow(pdist, cmap=cmap, clim=clim)
    else:
        plt.imshow(pdist, cmap=cmap)

    return figure
