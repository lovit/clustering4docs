__title__ = 'soyclustering'
__version__ = '0.0.4'
__author__ = 'Lovit'
__license__ = 'GPL v3'
__copyright__ = 'Copyright 2017 Lovit'

from ._kmeans import SphericalKMeans
from ._utils import proportion_keywords
from ._postprocess import merge_close_clusters