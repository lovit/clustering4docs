import os
import pytest
import sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)

from soyclustering import SphericalKMeans
from lovit_textmining_dataset.navernews_10days import get_bow


# pytest execution with verbose
# $ pytest tests/test_kmeans.py -s -v


@pytest.fixture
def test_data():
    x, idx_to_vocab, vocab_to_idx = get_bow(date='2016-10-20', tokenize='noun')
    return {'x': x}


def test_kmeans(test_data):
    x = test_data['x']
    n_features = x.shape[1]
    n_clusters = 100
    print('\nk-means test with various configuration')

    configs = [('k-means++', None), ('similar_cut', None)]
    for config in configs:
        init, sparsity = config
        print(f'\nConfig\n  - init={init}\n  - sparsity={sparsity}')
        kmeans = SphericalKMeans(n_clusters=n_clusters, init=init,
            sparsity=sparsity, max_iter=5, tol=0.0001, verbose = True, random_state=0)
        labels = kmeans.fit_predict(x)
        centers = kmeans.cluster_centers_
        assert centers.shape == (n_clusters, n_features)
