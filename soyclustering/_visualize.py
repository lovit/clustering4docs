from bokeh.models import HoverTool
from bokeh.models import ColumnDataSource
from bokeh.palettes import Greys256
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import numpy as np

from ._postprocess import _grouping_with_pdist

def visualize_pairwise_distance(centers, labels=None,
    max_dist=0.7, sort=False, use_bokeh=False, show=True, **kargs):
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
    use_bokeh : Boolean
        In this version, this argument is ignored.
    show : Boolean
        If True, it shows the figure of pairwise distance and returns figure.
        Else, it just returns the figure.

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

    return pairwise_distance_to_matplotlib_figure(pdist_revised, show=show)

def pairwise_distance_to_bokeh_heatmap(pdist, cluster_idx, palettes=None):

    def get_color(dist, palettes, max_=1):
        idx = int( (len(palettes)-1) * dist / max_ )
        idx = min(idx, len(palettes)-1)
        return palettes[idx]

    if palettes is None:
        palettes = Greys256

    n_clusters = pdist.shape[0]

    xname = []
    yname = []
    color = []
    names = ['cluster {}'.format(cluster_idx[i]) for i in range(n_clusters)]

    for i in range(n_clusters):
        for j in range(n_clusters):
            xname.append('cluster {}'.format(cluster_idx[i]))
            yname.append('cluster {}'.format(cluster_idx[j]))
            color.append(get_color(pdist[i,j], palettes))

    source = ColumnDataSource(data=dict(
        xname=xname,
        yname=yname,
        colors=color,
        cos=pdist.flatten()
    ))

    p = figure(title="Cluster center pairwise distance",
           x_axis_location="above", x_range=names, y_range=names,
           tools="hover,save,wheel_zoom,box_zoom,reset")

    p.plot_width = 800
    p.plot_height = 800
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "5pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = np.pi/3

    p.rect('xname', 'yname', 0.9, 0.9, source=source,
           color='colors', line_color=None,
           hover_line_color='black', hover_color='colors')

    p.select_one(HoverTool).tooltips = [
        ('names', '@yname, @xname'),
        ('cos', '@cos'),
    ]

    return p

def pairwise_distance_to_matplotlib_figure(pdist, title=None,
    figsize=(15,15), cmap='gray', clim=None, dpi=50,
    facecolor=None, edgecolor=None, frameon=True, show=True):
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
    show : Boolean
        If True, it shows the figure of pairwise distance and returns figure.
        Else, it just returns the figure.
    """

    n_clusters = pdist.shape[0]

    figure = plt.figure(
        figsize = figsize,
        dpi = dpi,
        facecolor = facecolor,
        edgecolor = edgecolor,
        frameon = frameon
    )

    plt.xlim((0, n_clusters))
    plt.ylim((0, n_clusters))

    if clim:
        plt.imshow(pdist, cmap=cmap, clim=clim)
    else:
        plt.imshow(pdist, cmap=cmap)

    if title:
        plt.title(title)

    if show:
        plt.show()

    return figure
